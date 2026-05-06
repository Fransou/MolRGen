import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_metadata_from_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Extract metadata from dataset JSONL file.

    Returns list of dicts with keys: prompt_id, properties, objectives, target
    """
    dataset = read_jsonl(dataset_path)
    metadata_list = []

    for sample in dataset:
        # Extract metadata from conversations
        if sample.get("conversations"):
            meta = sample["conversations"][0].get("meta", {})
            metadata_list.append(
                {
                    "prompt_id": meta.get("prompt_id", sample.get("identifier", "")),
                    "properties": meta.get("properties", []),
                    "objectives": meta.get("objectives", []),
                    "target": meta.get("target", []),
                }
            )

    return metadata_list


def find_generated_molecules_files(reinvent_dir: str) -> List[tuple]:
    """
    Find all generated_molecules.csv files in the reinvent directory.

    Returns list of tuples: (file_path, task_id)
    """
    pattern = os.path.join(reinvent_dir, "*", "*", "task_*", "generated_molecules.csv")
    files = glob.glob(pattern)

    results = []
    for file_path in files:
        # Extract task_id from directory path
        parts = file_path.split(os.sep)
        for i, part in enumerate(parts):
            if part.startswith("task_"):
                task_id = int(part.replace("task_", ""))
                results.append((file_path, task_id))
                break

    return results


def read_generated_molecules(file_path: str) -> List[Dict[str, str]]:
    """Read generated_molecules.csv and return list of {smiles, reward} dicts."""
    data = []
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "smiles": row.get("SMILES", row.get("smiles", "")),
                    "reward": row.get("Reward", row.get("reward", "")),
                }
            )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate REINVENT RL optimization results into a single CSV"
    )
    parser.add_argument(
        "--reinvent_dir",
        type=str,
        default="MolGenOutput/reinvent",
        help="Path to the reinvent output directory (e.g., MolGenOutput/reinvent)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/molgendata/test_data/test_prompts_ood.jsonl",
        help="Path to the dataset JSONL file (e.g., data/molgendata/test_data/test_prompts_ood.jsonl)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output CSV file path"
    )
    args = parser.parse_args()
    metadata_list = extract_metadata_from_dataset(args.dataset)
    csv_files = find_generated_molecules_files(args.reinvent_dir)
    output_rows = []

    for csv_path, task_id in csv_files:
        if task_id >= len(metadata_list):
            print(
                f"Warning: task_id {task_id} exceeds metadata list length ({len(metadata_list)})"
            )
            continue

        metadata = metadata_list[task_id]

        # Read generated molecules
        molecules = read_generated_molecules(csv_path)

        for mol in molecules:
            output_rows.append(
                {
                    "model": "reinvent",
                    "prompt_id": metadata["prompt_id"],
                    "smiles": mol["smiles"],
                    "properties": ";".join(metadata["properties"]),
                    "objectives": ";".join(metadata["objectives"]),
                    "target": ";".join(str(t) for t in metadata["target"]),
                    "reward": mol["reward"],
                }
            )

    df = pd.DataFrame(output_rows)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
