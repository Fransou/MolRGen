import argparse
import logging
import os
from pathlib import Path

from molrgen.data.pydantic_dataset import write_jsonl

from .gen_dataset import (
    DatasetConfig,
    MolGenerationInstructionsDatasetGenerator,
    data_dict_to_pydantic,
)

logger = logging.getLogger(__name__)
# Set up logging to INFO level
logging.basicConfig()


def generate_prompts(config: DatasetConfig, args: argparse.Namespace) -> None:
    dataset = MolGenerationInstructionsDatasetGenerator(config)

    n_train_prompts = int(args.n_prompts * 0.98)
    n_valid_prompts = int(args.n_prompts * 0.01)  # 10% of the training set
    n_test_prompts = int(args.n_prompts * 0.01)  # 20% of the training set

    variables = [
        ("train_prompts", 0, n_train_prompts),
        ("eval_data/eval_prompts", 0, n_valid_prompts),
        ("test_data/test_prompts", 0, n_test_prompts),
        ("eval_data/eval_prompts_ood", 1, n_valid_prompts * 2),
        ("test_data/test_prompts_ood", 2, n_test_prompts * 2),
    ]

    for name, docking_split, n_prompts in variables:
        if "ood" in name:
            dataset.needs_dock = True
        data_dict = dataset.generate_dataset(n_prompts, docking_split=docking_split)
        pydantic_dataset = data_dict_to_pydantic(data_dict, origin=name)
        write_jsonl(
            Path(os.path.join(args.data_path, name + ".jsonl")), pydantic_dataset
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the RL dataset for molecular generation with docking instructions"
    )

    parser.add_argument(
        "--n-prompts",
        "-n",
        type=int,
        default=50000,
        help="The number of prompts to generate",
    )
    parser.add_argument(
        "--max-dock-prop-per-prompt",
        type=int,
        default=2,
        help="The maximum number of docking properties per prompt",
    )

    parser.add_argument(
        "--n-props-probs",
        nargs="+",
        type=float,
        default=[0.4, 0.3, 0.3],
        help="Probabilities for the number of properties per prompt",
    )

    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/molgendata",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--split-docking",
        nargs="+",
        default=[0.8, 0.1, 0.1],
    )
    parser.add_argument("--max-occ", type=int, default=8)
    args = parser.parse_args()

    config = DatasetConfig(
        data_path=args.data_path,
        max_n_props=len(args.n_props_probs),
        max_occ=args.max_occ,
        props_prob=args.n_props_probs,
        split_docking=args.split_docking,
        max_docking_per_prompt=args.max_dock_prop_per_prompt,
        chat_template={"user": "role", "content": "content"},
    )

    print("Generating prompts...")
    generate_prompts(config, args)
