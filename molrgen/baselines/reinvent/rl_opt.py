import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import GRPOConfig

from molrgen.baselines.reinvent.trainers import (
    N_REPEAT_TEST,
    EvalMolMetrics,
    VanillaReinventTrainer,
    get_reward_fn,
)
from molrgen.data.pydantic_dataset import read_jsonl

os.environ["WANDB_PROJECT"] = "REINVENT_HF-RL"


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
        default="Franso/reinvent_42M",
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
        default=1024,
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
        default="http://0.0.0.0:5001",
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

    args = parser.parse_args()
    args.train_on_beams = bool(args.train_on_beams)
    args.output_dir = os.path.join(
        args.output_dir, args.model_name.replace("/", "-") + "_rl"
    )
    return args


if __name__ == "__main__":
    args = get_args()
    if args.num_beams >= 0:
        if args.num_beams == 0:
            args.generation_config = {}
        else:
            args.generation_config["num_beams"] = args.num_beams
    dataset = read_jsonl(Path(args.dataset))
    with open(os.path.join(args.datasets_path, "docking_targets.json")) as f:
        docking_targets = json.load(f)

    id = 0
    for sample in dataset:
        metadata = sample.conversations[0].meta
        has_docking = any([prop in docking_targets for prop in metadata["properties"]])
        if args.rewards_to_pick == "std_only" and has_docking:
            continue
        elif args.rewards_to_pick == "docking_only" and not has_docking:
            continue

        print("=#=#=#=#" * 15)
        print("-#-#-#-#" * 5, f"[{id}] Task : {metadata}", "-#-#-#-#" * 5)
        print("=#=#=#=#" * 15)

        if args.id_obj == -1 or args.id_obj == id:
            reward_fn = get_reward_fn(metadata, args.datasets_path, args.remote_rm_url)

            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)

            args.generation_config["do_sample"] = True
            generation_config = {k: v for k, v in args.generation_config.items()}

            if not args.train_on_beams:
                generation_config["num_beams"] = 1
                generation_config["num_beam_groups"] = 1
                generation_config["penalty_alpha"] = None
            out_dir = os.path.join(args.output_dir, f"task_{id}")
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
                dataloader_num_workers=0,
                max_completion_length=256,
                max_grad_norm=1,
                bf16=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                save_total_limit=1,
                beta=args.sigma,
                generation_kwargs=generation_config,
                batch_eval_metrics=False,
                log_completions=False,
                metric_for_best_model="reward",
                greater_is_better=True,
                load_best_model_at_end=True,
            )
            train_dataset = Dataset.from_dict(
                {"prompt": ["<s>"] * args.num_train_epochs}
            )
            eval_dataset = Dataset.from_dict({"prompt": ["<s>"] * 16})

            trainer = VanillaReinventTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                reward_funcs=reward_fn,
                compute_metrics=EvalMolMetrics(
                    tokenizer=tokenizer, reward_fn=reward_fn
                ),
                n_repeat_test=N_REPEAT_TEST,
            )
            trainer.train()
            wandb.config.update(  # type: ignore
                {
                    "id_obj": args.id_obj,
                    "objectives": metadata["objectives"],
                    "properties": metadata["properties"],
                    "target": metadata["target"],
                    "algo": "VanillaReinvent",
                }
            )

            ## Evaluate
            trainer.generation_config.num_beams = args.generation_config.get(
                "num_beams", 1
            )
            trainer.generation_config.num_beam_groups = args.generation_config.get(
                "num_beam_groups", 1
            )
            trainer.generation_config.penalty_alpha = args.generation_config.get(
                "penalty_alpha", None
            )
            eval_datasets = {
                f"@{n}": Dataset.from_dict({"prompt": ["<s>"] * N_REPEAT_TEST * n})
                for n in [1, 10, 100]
            }
            metrics = trainer.evaluate(eval_datasets)
            rows = []
            for k in metrics:
                n = int(k.split("@")[1].split("_")[0])
                metric = k.split("_")[-1]
                if metric in ["reward", "Validity", "Uniqueness", "Diversity"]:
                    v = metrics[k]
                    if np.isnan(v):
                        v = 0
                    rows.append([n, metric, v])

            metrics_df = pd.DataFrame(rows, columns=["n", "metric", "value"])
            metrics_df = pd.pivot_table(
                metrics_df, columns="metric", values="value", index="n"
            ).reset_index(names="n")
            table = wandb.Table(dataframe=metrics_df)
            wandb.log({"Final_Evaluation": table})
            for k, v in args.generation_config.items():
                wandb.log({f"eff_{k}": v})
        id += 1
