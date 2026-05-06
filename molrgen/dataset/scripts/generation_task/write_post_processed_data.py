import argparse
import json
import os
from subprocess import DEVNULL, STDOUT, check_call
from typing import Any

import pandas as pd
import ray
from ray.experimental import tqdm_ray

from molrgen.dataset.pdb_uniprot.target_naming import get_names_mapping_uniprot

# After all structures have been downloaded, and the pockets have been extracted, performs the last steps of the data processing:
# 1. Ensures all pdb files can be read byt the prepare_receptor script from AutoDockTools
# 2. Finds a textual description for each protein.


def check_pdb_file(path: str) -> bool:
    """Check that all pdb files in the given directory can be read by the prepare_receptor script from AutoDockTools.

    Args:
        pdb_dir (str): The directory containing the pdb files to check.
    """
    try:
        check_call(
            ["prepare_receptor", "-r", path, "-o", "tmp.pdbqt"],
            stdout=DEVNULL,
            stderr=STDOUT,
            timeout=60 * 5,
        )
        return True
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return False


@ray.remote(num_cpus=1)
def check_and_copy(
    structure: str,
    args: argparse.Namespace,
    pocket_info: dict,
    pbar: Any,
    do_copy: bool = False,
) -> str | None:
    if do_copy:
        pdb_path = os.path.join(args.data_path_csv, "pdb_files", f"{structure}.pdb")
    else:
        pdb_path = os.path.join(args.data_path, "pdb_files", f"{structure}.pdb")
    if not check_pdb_file(pdb_path):
        print(f"[Warning] {structure} could not be processed by prepare_receptor.")
        return None
    pocket_info[structure]["pdb_path"] = pdb_path

    # Copy pdb file to the new location (pdb_path)
    if do_copy:
        os.system(
            f"cp {pdb_path} {os.path.join(args.data_path, f'{structure}_processed.pdb')}"
        )
    if pbar is not None:
        pbar.update.remote(1)

    return structure


def get_final_df_and_pocket_info(
    data_path_csv: str,
) -> dict[str, dict[str, Any]]:
    final_df = pd.concat(
        [
            pd.read_csv(os.path.join(data_path_csv, f))
            for f in os.listdir(data_path_csv)
            if f.endswith(".csv") and f.startswith("sair_pockets")
        ],
        ignore_index=True,
    )
    final_df = (
        final_df.sort_values(by="n_ligand_poses", ascending=False)
        .drop_duplicates(subset=["prot_id"])
        .reset_index(drop=True)
    )
    # Save a pocket_infos.json file with the pocket information
    pocket_info: dict[str, dict[str, Any]] = {}

    for _, row in final_df.iterrows():
        pocket_info[row["id"]] = {
            "size": [row["width_x"], row["width_y"], row["width_z"]],
            "center": [row["center_x"], row["center_y"], row["center_z"]],
            "metadata": {
                "cluster": int(row["cluster"]),
                "center": [row["center_x"], row["center_y"], row["center_z"]],
                "size": [row["width_x"], row["width_y"], row["width_z"]],
                "n_ligand_poses": int(row["n_ligand_poses"]),
                "sequence": row["sequence"],
                "prot_id": row["prot_id"],
                "avg_pIC50": float(row["avg_pIC50"]),
                "avg_confidence": float(row["avg_confidence"]),
            },
        }
    return pocket_info


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser(
        description="Post-process the SAIR dataset after downloading and pocket extraction."
    )
    parser.add_argument(
        "--data-path", type=str, default="data/sair_rl", help="Path to the dataset"
    )
    parser.add_argument(
        "--data_path_csv",
        type=str,
        default="",
        help="Path to the csv data extracted (if necessary).",
    )
    parser.add_argument(
        "--do_copy",
        action="store_true",
    )

    args = parser.parse_args()

    data_path = args.data_path
    data_pdb_path = os.path.join(data_path, "pdb_files")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(data_pdb_path, exist_ok=True)

    if args.data_path_csv != "":
        pocket_info = get_final_df_and_pocket_info(args.data_path_csv)
        with open(os.path.join(data_path, "pockets_info.json"), "w") as f:
            json.dump(pocket_info, f)
    else:
        with open(os.path.join(data_path, "pockets_info.json")) as f:
            pocket_info = json.load(f)

    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(total=len(pocket_info))

    kept_pockets_jobs = []

    for structure in pocket_info:
        kept_pockets_jobs.append(
            check_and_copy.remote(structure, args, pocket_info, pbar, args.do_copy)
        )

    kept_pockets = ray.get(kept_pockets_jobs)
    kept_pockets = [p for p in kept_pockets if p is not None]

    pbar.close.remote()  # type: ignore

    kept_pockets_uniprots = [
        pocket_info[p]["metadata"]["prot_id"] for p in kept_pockets
    ]
    print(f"Number of pockets kept {len(kept_pockets)}")
    names_mapping_uniprot = get_names_mapping_uniprot(
        kept_pockets_uniprots, names_override=kept_pockets
    )

    docking_targets = kept_pockets + [
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

    with open(os.path.join(data_path, "docking_targets.json"), "w") as f:
        json.dump(docking_targets, f, indent=4)
    with open(os.path.join(data_path, "names_mapping.json"), "w") as f:
        json.dump(names_mapping_uniprot, f, indent=4)
    print(f"Kept {len(kept_pockets)} pockets after filtering.")
