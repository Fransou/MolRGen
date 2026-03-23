import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import GRPOConfig

import wandb
from mol_gen_docking.baselines.reinvent.trainers import (
    N_REPEAT_TEST,
    EvalMolMetrics,
    ReinventGRPOTrainer,
    get_reward_fn,
)
from mol_gen_docking.data.pydantic_dataset import read_jsonl

os.environ["WANDB_PROJECT"] = "REINVENT_HF-RL"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rl_opt_rpo.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/molgendata/eval_data/eval_prompts.jsonl",
        help="Dataset name",
    )
    parser.add_argument("--datasets-path", type=str, default="data/molgendata")

    parser.add_argument(
        "--model_name",
        type=str,
        default="Franso/reinvent_171M_prior",
        help="Name of the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Sigma parameter for reinvent",
    )

    parser.add_argument(
        "--loss_type", type=str, default="dapo", choices=["grpo", "dr_grpo", "dapo"]
    )
    parser.add_argument("--train_on_beams", type=int, default=0)
    parser.add_argument(
        "--num_beams",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--generation_config",
        type=json.loads,
        default={},
    )

    parser.add_argument(
        "--remote_rm_url",
        type=str,
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--rewards_to_pick",
        type=str,  # Literal["docking_only", "std_only", "all"]
        default="all",
    )
    parser.add_argument(
        "--id_obj",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--custom_metadata",
        type=json.loads,
        default=None,
        help="Custom metadata as JSON string for reward definition",
    )
    parser.add_argument(
        "--custom_metadata_file",
        type=str,
        default=None,
        help="Path to JSON file containing custom metadata for reward definition",
    )
    args = parser.parse_args()
    args.train_on_beams = bool(args.train_on_beams)
    args.output_dir = os.path.join(
        args.output_dir, args.model_name.replace("/", "-") + "_rl"
    )
    return args


def run_training(
    metadata: dict,
    args: argparse.Namespace,
    reward_fn: Callable[[str | list[str]], float | list[float]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    out_dir: str,
) -> ReinventGRPOTrainer:
    """
    Set up and run training for a given task.

    Args:
        metadata: Task metadata containing objectives and properties
        args: Command line arguments
        reward_fn: Reward function for the task
        model: The language model to train
        tokenizer: Tokenizer for the model
        out_dir: Output directory for checkpoints

    Returns:
        Trained ReinventGRPOTrainer instance
    """
    logger.info("Setting up training configuration...")

    generation_config = {k: v for k, v in args.generation_config.items()}

    if not args.train_on_beams:
        generation_config["num_beams"] = 1
        generation_config["num_beam_groups"] = 1
        generation_config["penalty_alpha"] = None

    os.makedirs(out_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=out_dir,
        max_steps=args.num_train_epochs,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=int(args.lr_warmup_ratio * args.num_train_epochs),
        warmup_ratio=args.lr_warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_generations=args.batch_size,
        num_generations_eval=args.eval_batch_size,
        dataloader_num_workers=0,
        max_completion_length=256,
        max_grad_norm=1,
        bf16=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=1,
        beta=args.sigma,
        generation_kwargs=generation_config,
        batch_eval_metrics=False,
        log_completions=False,
        loss_type=args.loss_type,
        metric_for_best_model="reward",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    train_dataset = Dataset.from_dict({"prompt": ["<s>"] * args.num_train_epochs})
    eval_dataset = Dataset.from_dict({"prompt": ["<s>"] * args.eval_batch_size})

    trainer = ReinventGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        compute_metrics=EvalMolMetrics(tokenizer=tokenizer, reward_fn=reward_fn),
        n_repeat_test=N_REPEAT_TEST,
    )

    logger.info("Starting training...")
    trainer.train()

    return trainer


def run_final_generation(
    trainer: ReinventGRPOTrainer,
    tokenizer: AutoTokenizer,
    reward_fn: Callable[[str | list[str]], float | list[float]],
    args: argparse.Namespace,
    out_dir: str,
    metadata: dict,
    task_id: int,
) -> None:
    """
    Generate molecules using the trained model and log results.

    Args:
        trainer: Trained ReinventGRPOTrainer instance
        tokenizer: Tokenizer for decoding predictions
        reward_fn: Reward function for evaluating molecules
        args: Command line arguments
        out_dir: Output directory for results
        metadata: Task metadata
        task_id: Task ID for logging
    """
    logger.info("=" * 50)
    logger.info(f"Generating {args.eval_batch_size} molecules with trained model...")
    logger.info("=" * 50)

    if args.eval_batch_size > 0:
        trainer.compute_metrics = None
        final_gen_dataset = Dataset.from_dict(
            {"prompt": ["<s>"] * args.eval_batch_size}
        )
        output = trainer.predict(final_gen_dataset)
        logger.debug(f"Raw prediction output: {output}")

        smiles_predicted = tokenizer.batch_decode(
            output.predictions, skip_special_tokens=True
        )
        logger.info(f"Generated {len(smiles_predicted)} molecules")

        # Get reward for each molecule
        rewards = reward_fn(smiles_predicted)
        if not isinstance(rewards, list):
            rewards = [rewards]

        # Log results to CSV
        results_df = pd.DataFrame({"SMILES": smiles_predicted, "Reward": rewards})

        results_csv_path = os.path.join(out_dir, "generated_molecules.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"Saved generated molecules to {results_csv_path}")

        # Log top 5 molecules
        sorted_indices = np.argsort(rewards)
        logger.info("Generated Molecules:")
        for i, idx in enumerate(sorted_indices, 1):
            logger.info(
                f"  {i}. SMILES: {smiles_predicted[idx]} | Reward: {rewards[idx]:.4f}"
            )

    logger.info(f"Completed training and evaluation for task {task_id}")


if __name__ == "__main__":
    args = get_args()
    if args.num_beams >= 0:
        if args.num_beams == 0:
            args.generation_config = {}
        else:
            args.generation_config["num_beams"] = args.num_beams

    # Handle custom metadata or use dataset
    if args.custom_metadata:
        # Use custom metadata directly
        metadata_list = [args.custom_metadata]
    elif args.custom_metadata_file:
        # Load metadata from file
        with open(args.custom_metadata_file) as f:
            metadata_content = json.load(f)
        if isinstance(metadata_content, list):
            metadata_list = metadata_content
        else:
            metadata_list = [metadata_content]
    else:
        # Use existing dataset logic
        dataset = read_jsonl(Path(args.dataset))
        with open(os.path.join(args.datasets_path, "docking_targets.json")) as f:
            docking_targets = json.load(f)

        metadata_list = []
        for sample in dataset:
            metadata = sample.conversations[0].meta
            has_docking = any(
                [prop in docking_targets for prop in metadata["properties"]]
            )
            if args.rewards_to_pick == "std_only" and has_docking:
                continue
            elif args.rewards_to_pick == "docking_only" and not has_docking:
                continue
            metadata_list.append(metadata)

    id = 0
    for metadata in metadata_list:
        logger.info("=" * 80)
        logger.info(f"[{id}] Task : {metadata}")
        logger.info("=" * 80)

        if args.id_obj == -1 or args.id_obj == id:
            reward_fn = get_reward_fn(metadata, args.datasets_path, args.remote_rm_url)
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            args.generation_config["do_sample"] = True
            out_dir = os.path.join(args.output_dir, f"task_{id}")
            trainer = run_training(
                metadata=metadata,
                args=args,
                reward_fn=reward_fn,
                model=model,
                tokenizer=tokenizer,
                out_dir=out_dir,
            )
            wandb.config.update(  # type: ignore
                {
                    "id_obj": args.id_obj,
                    "objectives": metadata["objectives"],
                    "properties": metadata["properties"],
                    "target": metadata["target"],
                    "algo": "GRPOReinvent",
                }
            )
            # Run final generation
            run_final_generation(
                trainer=trainer,
                tokenizer=tokenizer,
                reward_fn=reward_fn,
                args=args,
                out_dir=out_dir,
                metadata=metadata,
                task_id=id,
            )
        id += 1
    logger.info("=" * 80)
    logger.info("Training completed for all tasks")
    logger.info("=" * 80)
