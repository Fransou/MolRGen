import argparse
import json
import logging
import os
import pickle
import shutil

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
# Set up logging to INFO level
logging.basicConfig(level=logging.INFO)


def get_pockets_info(data_path: str) -> dict:
    with open(os.path.join(data_path, "final_dic.pkl"), "rb") as f:
        siu_data = pickle.load(f)
    pockets_info = {}
    assert os.path.exists(os.path.join(data_path, "result_structures"))
    os.makedirs(os.path.join(data_path, "pdb_files"), exist_ok=True)
    for uni_prot_id in tqdm(siu_data.keys()):
        for ligand_pocket in siu_data[uni_prot_id]:
            dir = ligand_pocket["source_data"].split(",")[1]
            file = os.path.join(
                data_path,
                "result_structures",
                uni_prot_id,
                "PDB_" + dir,
                dir + "_rmW.pdb",
            )
            if not os.path.exists(file):
                continue

            pocket_coordinates = np.concatenate(
                [pocket.reshape(1, 3) for pocket in ligand_pocket["pocket_coordinates"]]
            )
            file_name = f"{dir}.pdb"
            break
        if not os.path.exists(file):
            continue
        shutil.move(file, os.path.join(data_path, "pdb_files", file_name))

        average_pIC50 = np.nanmean(
            [
                ligand_pocket["label"].get("ic50", np.nan)
                for ligand_pocket in siu_data[uni_prot_id]
            ]
        )
        average_pKd = np.nanmean(
            [
                ligand_pocket["label"].get("kd", np.nan)
                for ligand_pocket in siu_data[uni_prot_id]
            ]
        )
        pocket_size = (
            pocket_coordinates.max(axis=0) - pocket_coordinates.min(axis=0)
        ).tolist()
        pocket_center = (
            (pocket_coordinates.max(axis=0) + pocket_coordinates.min(axis=0)) / 2
        ).tolist()
        metadata = {
            "size": pocket_size,
            "center": pocket_center,
            "sequence": "",
            "prot_id": uni_prot_id,
            "origin": "SIU",
        }
        if not np.isnan(average_pIC50):
            metadata["avg_pIC50"] = average_pIC50
        if not np.isnan(average_pKd):
            metadata["avg_pKd"] = average_pKd

        pockets_info[dir] = {
            "size": pocket_size,
            "center": pocket_center,
            "metadata": metadata,
        }

    return pockets_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the RL dataset for molecular generation with docking instructions"
    )

    parser.add_argument(
        "--data-path", type=str, default="data/SIU", help="Path to the dataset"
    )
    args = parser.parse_args()

    os.makedirs(args.data_path, exist_ok=True)

    pockets_info = get_pockets_info(data_path=args.data_path)

    with open(os.path.join(args.data_path, "pockets_info.json"), "w") as f:
        json.dump(pockets_info, f, indent=4)

    print(len(pockets_info), " prots extracted.")
