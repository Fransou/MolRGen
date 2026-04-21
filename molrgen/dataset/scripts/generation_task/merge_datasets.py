import argparse
import json
import os
import shutil
from typing import Any, Dict, Tuple

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preference", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pdb_files"), exist_ok=True)

    pockets_info = {}
    docking_targets = {}
    final_docking_targets: list[str] = []
    names_mapping = {}
    final_names_mapping = {}
    for existing_dataset in os.listdir(args.input_dir):
        with open(
            os.path.join(args.input_dir, existing_dataset, "pockets_info.json")
        ) as f:
            json_data = json.load(f)
        pockets_info[existing_dataset] = json_data
    for existing_dataset in os.listdir(args.input_dir):
        with open(
            os.path.join(args.input_dir, existing_dataset, "docking_targets.json")
        ) as f:
            json_data = json.load(f)
        docking_targets[existing_dataset] = json_data
    for existing_dataset in os.listdir(args.input_dir):
        with open(
            os.path.join(args.input_dir, existing_dataset, "names_mapping.json")
        ) as f:
            json_data = json.load(f)
        names_mapping[existing_dataset] = json_data
    # Merge pocket infos
    final_pockets_info: Dict[str, Any] = {}
    prot_ids_to_dataset_key: Dict[str, Tuple[str, str]] = {}
    n_overlap = 0
    for existing_dataset in pockets_info:
        for key in docking_targets[existing_dataset]:
            if key.endswith("_docking"):
                continue
            prot_id = pockets_info[existing_dataset][key]["metadata"]["prot_id"]
            if prot_id in prot_ids_to_dataset_key:
                n_overlap += 1
            if (
                prot_id not in prot_ids_to_dataset_key
                or existing_dataset == args.preference
            ):
                prot_ids_to_dataset_key[prot_id] = (existing_dataset, key)
    print(f"Total overlap: {n_overlap} (/{len(prot_ids_to_dataset_key)})")

    for prot_id in prot_ids_to_dataset_key:
        dataset, key = prot_ids_to_dataset_key[prot_id]
        final_pockets_info[key] = pockets_info[dataset][key]

    # Merge docking targets
    for prot_id in prot_ids_to_dataset_key:
        dataset, key = prot_ids_to_dataset_key[prot_id]
        if key in docking_targets[dataset]:
            final_docking_targets.append(key)
            shutil.copy(
                os.path.join(args.input_dir, dataset, "pdb_files", key + ".pdb"),
                os.path.join(args.output_dir, "pdb_files", key + ".pdb"),
            )
    final_docking_targets = final_docking_targets + [
        "3pbl_docking",
        "1iep_docking",
        "2rgp_docking",
        "3eml_docking",
        "3ny8_docking",
        "4rlu_docking",
        "4unn_docking",
        "5mo4_docking",
        "7l11_docking",
    ]

    # Merge name mappings
    for prot_id in prot_ids_to_dataset_key:
        dataset, key = prot_ids_to_dataset_key[prot_id]
        if key in final_docking_targets:
            for k, v in names_mapping[dataset].items():
                if v == key:
                    final_names_mapping[k] = v
                    break

    # Save all final files
    with open(os.path.join(args.output_dir, "pockets_info.json"), "w") as f:
        json.dump(final_pockets_info, f)
    with open(os.path.join(args.output_dir, "docking_targets.json"), "w") as f:
        json.dump(final_docking_targets, f)
    with open(os.path.join(args.output_dir, "names_mapping.json"), "w") as f:
        json.dump(final_names_mapping, f)
