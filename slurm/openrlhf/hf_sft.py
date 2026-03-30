"""
SFT fine-tuning script using TRL / Transformers.

Mirrors the behaviour of sft_model.sh (OpenRLHF-based) but can be launched
directly with:

    python hf_sft.py \\
        --dataset /path/to/data.jsonl \\
        --save_path /path/to/output \\
        --config  slurm/openrlhf/configs/sft_config/Qwen3-4B/0.0-0.4-none/train.yaml

The YAML config format is the same as the one consumed by train_sft in
OpenRLHF, so existing config files can be re-used without modification.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from mol_gen_docking.data.pydantic_dataset import read_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl_dataset(path: str, input_key: str = "messages") -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset.

    Each line must be a JSON object that contains a field whose name is given
    by *input_key* (default: ``"messages"``).  The field value must be a list
    of chat-message dicts (``[{"role": ..., "content": ...}, ...]``).

    Lines that cannot be parsed, or that do not contain the expected key, are
    skipped with a warning.
    """
    samples = read_jsonl(Path(path))

    records = []
    for sample in samples:
        for conversation in sample.conversations:
            # Extract messages from the conversation
            messages = [msg.model_dump() for msg in conversation.messages]
            records.append({input_key: messages})

    if not records:
        raise ValueError(f"No valid records found in dataset '{path}'.")

    logger.info("Loaded %d records from '%s'.", len(records), path)
    return Dataset.from_list(records)


def compute_gradient_accumulation_steps(
    train_batch_size: int,
    micro_train_batch_size: int,
) -> int:
    """Derive gradient-accumulation steps from the global and per-device batch sizes.

    The number of data-parallel processes is detected automatically from the
    ``WORLD_SIZE`` environment variable (set by torchrun / accelerate).
    Falls back to the number of visible CUDA devices, and ultimately to 1.
    """
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            max(1, torch.cuda.device_count()),
        )
    )
    accum = max(1, train_batch_size // (micro_train_batch_size * world_size))
    logger.info(
        "Gradient accumulation steps: %d  "
        "(train_batch_size=%d, micro_train_batch_size=%d, world_size=%d)",
        accum,
        train_batch_size,
        micro_train_batch_size,
        world_size,
    )
    return accum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning with TRL / Transformers (drop-in for sft_model.sh)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the training JSONL dataset.",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Directory where the final model (and intermediate checkpoints) are saved.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Optional checkpoint directory (defaults to <save_path>/ckpt).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML training-config file (same format as OpenRLHF train.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load YAML config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    pretrain: str = cfg["pretrain"]
    input_key: str = cfg.get("input_key", "messages")
    max_len: int = cfg.get("max_len", 4096)
    micro_train_batch_size: int = cfg.get("micro_train_batch_size", 1)
    train_batch_size: int = cfg.get("train_batch_size", 128)
    max_epochs: int = cfg.get("max_epochs", 1)
    learning_rate: float = float(cfg.get("learning_rate", 5e-5))
    lr_warmup_ratio: float = float(cfg.get("lr_warmup_ratio", 0.05))
    packing_samples: bool = bool(cfg.get("packing_samples", False))
    apply_chat_template: bool = bool(cfg.get("apply_chat_template", True))
    use_wandb: bool = bool(cfg.get("use_wandb", False))
    bf16: bool = bool(cfg.get("bf16", True))
    flash_attn: bool = bool(cfg.get("flash_attn", False))
    save_steps: int = cfg.get("save_steps", 50)
    max_ckpt_num: int = cfg.get("max_ckpt_num", 3)

    save_path = args.save_path
    ckpt_path = args.ckpt_path or os.path.join(save_path, "ckpt")

    # ------------------------------------------------------------------
    # Gradient accumulation
    # ------------------------------------------------------------------
    gradient_accumulation_steps = compute_gradient_accumulation_steps(
        train_batch_size, micro_train_batch_size
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer from '%s' …", pretrain)
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    raw_dataset = load_jsonl_dataset(args.dataset, input_key=input_key)

    if apply_chat_template:
        # Convert the list-of-dicts conversation to a plain string via the
        # tokenizer's chat template so that SFTTrainer receives a text column.
        def format_example(example: dict) -> dict:
            try:
                text = tokenizer.apply_chat_template(
                    example[input_key],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "apply_chat_template failed for an example (messages=%r): %s",
                    example.get(input_key),
                    exc,
                )
                text = ""
            return {"text": text}

        dataset = raw_dataset.map(
            format_example, remove_columns=raw_dataset.column_names
        )
    else:
        # Assume the field already contains a formatted string, or join content fields.
        def join_content(example: dict) -> dict:
            messages = example[input_key]
            text = "\n".join(m.get("content", "") for m in messages)
            return {"text": text}

        dataset = raw_dataset.map(join_content, remove_columns=raw_dataset.column_names)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Loading model from '%s' …", pretrain)
    model_kwargs = dict(trust_remote_code=True)
    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"  # type: ignore
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(pretrain, **model_kwargs)

    # ------------------------------------------------------------------
    # Reporting / W&B
    # ------------------------------------------------------------------
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "use_wandb is enabled in the config but 'wandb' is not installed. "
                "Install it with:  pip install wandb"
            ) from exc
    report_to = "wandb" if use_wandb else "none"

    # ------------------------------------------------------------------
    # SFTConfig / TrainingArguments
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=ckpt_path,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=micro_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=lr_warmup_ratio,
        bf16=bf16,
        fp16=False,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=max_ckpt_num,
        report_to=report_to,
        max_length=max_len,
        packing=packing_samples,
        dataset_text_field="text",
        remove_unused_columns=True,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )

    logger.info("Starting SFT training …")
    trainer.train()

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    logger.info("Saving final model to '%s' …", save_path)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
