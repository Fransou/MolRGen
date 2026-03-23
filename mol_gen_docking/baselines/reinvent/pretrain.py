"""
Pretraining simply corresponds to SFT
"""

import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "REINVENT_HF"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="jarod0411/zinc10M", help="Dataset name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
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
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine_with_restarts",
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
        default=1024,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push to HuggingFace Hub",
        default=False,
    )

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "-"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    dataset = load_dataset(args.dataset).rename_column("smiles", "text")
    # Split dataset for train/eval
    train_dataset = dataset["train"]
    eval_dataset = (
        dataset["validation"]
        if "validation" in dataset
        else dataset["train"].select(range(1000))
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.lr_warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        packing=True,
        bf16=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=3,
        dataset_num_proc=8,
        dataloader_prefetch_factor=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    tuned_model = trainer.model

    if args.push_to_hub:
        tuned_model.push_to_hub(args.model_name + "_prior")
        tokenizer.push_to_hub(args.model_name + "_prior")
