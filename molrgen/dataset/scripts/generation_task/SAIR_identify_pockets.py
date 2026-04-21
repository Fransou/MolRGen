import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from Bio.PDB import PDBIO, MMCIFParser, Superimposer
from ray.experimental import tqdm_ray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)

if not ray.is_initialized():
    ray.init()


def cif_file_to_entry_id(cif_file: str) -> int:
    """Extracts the entry ID from a CIF file name.
    Args:
        cif_file: Path to the CIF file
    Returns:
        Entry ID as a string
    """
    return int(cif_file.split("/")[-1].split("_")[1])


def filter_from_parquet(
    df_parquet: pd.DataFrame, cif_files: List[str]
) -> Tuple[Dict[str, List[str]], Dict[str, pd.DataFrame]]:
    """
    Filters the list of CIF files to only include those that are in the top 50% most potent ligands
    for each sequence, and have a confidence score in the top 50%.
    Args:
        df_parquet: DataFrame containing the SAIR parquet data
        cif_files: List of paths to CIF files
    Returns:
        Filtered list of CIF files
        Kept entry IDs
    """
    entry_ids = [cif_file_to_entry_id(f) for f in cif_files]
    df_filtered = df_parquet[df_parquet["entry_id"].isin(entry_ids)]

    protein_cif_files: Dict[str, List[str]] = {}
    protein_dfs = {}
    for protein in tqdm(
        df_filtered["protein"].unique(),
        desc="Filtering CIF files based on potency and confidence",
    ):
        new_cif_files: List[str] = []

        df_prot = df_filtered[df_filtered["protein"] == protein]
        if len(df_prot) == 0:
            continue
        pIC50_threshold = df_prot["pIC50"].quantile(0.5)
        df_prot_filtered = df_prot[(df_prot["pIC50"] >= pIC50_threshold)]
        confidence_threshold = df_prot_filtered["confidence_score"].quantile(0.5)
        df_prot_filtered = df_prot_filtered[
            df_prot_filtered["confidence_score"] >= confidence_threshold
        ]
        entry_ids_prot = df_prot_filtered["entry_id"].unique().tolist()

        for entry_id in entry_ids_prot:
            matching_files = [
                f for f in cif_files if cif_file_to_entry_id(f) == entry_id
            ]
            new_cif_files.extend(matching_files)

        protein_dfs[protein] = df_prot_filtered
        protein_cif_files[protein] = new_cif_files

    return protein_cif_files, protein_dfs


@ray.remote(num_cpus=1)
def get_pocket_cif(cif_path: str, topk: int = 3) -> Tuple[List[Any], Any]:
    """
    Opens a CIF file annd identifies the position of the ligand.
    Returns the list of residues in the pocket, defined as the top-k closest residues to the ligand + a padding.
    Args:
        cif_path: Path to the cif file
        topk: Number of closest residues to the ligand to include in the pocket

    Returns:
        List of residues in the pocket
        Structure object from Biopython (can be None if not needed)
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_path)
    model = structure[0]  # Get the first model
    ligand_residue = None

    # Identify the ligand residue (assuming it's the first HETATM found)
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " and residue.id[0] != "W":  # HETATM and not water
                ligand_residue = residue
                break
        if ligand_residue:
            break

    if not ligand_residue:
        raise ValueError("No ligand found in the CIF file.")

    # Get the coordinates of the ligand atoms
    ligand_coords = [atom.coord for atom in ligand_residue]

    # Find the closest residues to each ligand atom
    pocket_residues_set = set()
    for ligand_coord in ligand_coords:
        distances = []
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    for atom in residue:
                        dist = np.linalg.norm(atom.coord - ligand_coord)
                        distances.append((dist, residue))
        distances.sort(key=lambda x: x[0])
        closest_residues = [distances[i][1] for i in range(min(topk, len(distances)))]
        pocket_residues_set.update(closest_residues)
    pocket_residues = list(set(pocket_residues_set))

    return pocket_residues, structure


def get_seq_to_pocket_residues(
    cif_files: List[str], df_prot: pd.DataFrame, topk: int = 3
) -> pd.DataFrame:
    """
    Converts a list of residues to a list of strings in the format "CHAIN:RESIDUE_NUMBER:RESIDUE_NAME"
    Args:
        pocket_residues: List of residues
        topk: Number of closest residues to the ligand to include in the pocket
    Returns:
        Dictionary mapping each CIF file path to a list of residue strings
    """
    df = pd.DataFrame(
        columns=[
            "id",
            "pocket_residues",
            "structure",
            "prot_id",
            "sequence",
            "avg_pIC50",
            "avg_confidence",
        ]
    )
    res = []
    for cif_file in cif_files:
        res.append(get_pocket_cif.remote(cif_file, topk))
    results = ray.get(res)

    for idx, cif_file in enumerate(cif_files):
        pocket_residues, structure = results[idx]
        entry_id = cif_file_to_entry_id(cif_file)
        df_entry = df_prot[df_prot["entry_id"] == entry_id]
        df.loc[len(df)] = [
            cif_file.split("/")[-1].replace(".cif", ""),
            pocket_residues,
            structure,
            df_entry["protein"].values[0],
            df_entry["sequence"].values[0],
            df_prot["pIC50"].mean(),
            df_prot["confidence_score"].mean(),
        ]
    return df


def intersection_over_union(pocketA: List[Any], pocketB: List[Any]) -> float:
    """
    Computes the Intersection over Union (IoU) between two lists of residues.
    Args:
        pocketA: List of residues in pocket A
        pocketB: List of residues in pocket B

    Returns:
        IoU value as a float
    """
    setA = set(pocketA)
    setB = set(pocketB)
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    if union == 0:
        return 0.0
    return intersection / union


def aggregate_pocket(seq_df: pd.DataFrame, iou_threshold: float = 0.4) -> List[Any]:
    if len(seq_df) == 1:
        return [1.0]

    num_pockets = len(seq_df)
    iou_matrix = pd.DataFrame(index=seq_df.index, columns=seq_df.index, dtype=float)

    for i in range(num_pockets):
        for j in range(i, num_pockets):
            pocketA = seq_df.iloc[i]["pocket_residues"]
            pocketB = seq_df.iloc[j]["pocket_residues"]
            iou = intersection_over_union(pocketA, pocketB)
            iou_matrix.iat[i, j] = iou
            iou_matrix.iat[j, i] = iou

    # Cluster pockets based on IoU threshold

    distance_matrix = 1 - iou_matrix.fillna(0).values
    condensed_distance = squareform(distance_matrix)
    Z = linkage(condensed_distance, method="single")
    clusters = fcluster(Z, t=1 - iou_threshold, criterion="distance")
    return clusters.tolist()  # type: ignore


def process_df_and_aggregate_pockets(
    df: pd.DataFrame, iou_threshold: float = 0.4
) -> Tuple[pd.DataFrame, Dict[int, List[Any]]]:
    """
    Processes the DataFrame to aggregate pocket residues by sequence, and give them a cluster ID based on IoU.
    Args:
        df: DataFrame with columns "id", "pocket_residues", and "sequence"

    Returns:
        DataFrame with an additional column "cluster" indicating the cluster ID for each pocket
        Dictionary mapping each sequence to a dictionary of cluster IDs and their corresponding aggregated pocket residues
    """
    # Get the IoU matrix for each sequence of the pockets
    clusters = aggregate_pocket(df, iou_threshold=iou_threshold)
    df["cluster"] = clusters
    pocketid_pocket: Dict[int, List[Any]] = {}
    # For each cluster, aggregate the pocket residues by taking the residues that appear in at least 50% of the pockets
    for cluster_id in set(clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        all_residues: List[Any] = sum(list(cluster_df["pocket_residues"].values), [])
        residue_counts = pd.Series(all_residues).value_counts()
        threshold = len(cluster_df) * 7 / 10
        aggregated_residues = residue_counts[residue_counts >= threshold].index.tolist()
        if len(aggregated_residues) > 0:
            pocketid_pocket[cluster_id] = aggregated_residues
    return df, pocketid_pocket


def find_best_conf_pocket(df: pd.DataFrame, pocket: List[Any]) -> str:
    """
    Finds the best conformation of a pocket based on the conformation with the lowest RMSD for the concerned residues.
    Args:
        df: DataFrame with columns "id", "pocket_residues", "sequence", and "cluster"
        pocket: List of residues in the pocket to compare against
    Returns:
        CIF file path of the best conformation
    """
    # First open all concerned cif files

    t0 = time.time()

    structures = {}
    for id in df["id"]:
        structures[id] = df[df["id"] == id]["structure"].values[0]

    def get_rmsd_list(
        structA: Any, structBs: List[Any], pocket: List[Any]
    ) -> List[float] | None:
        sup = Superimposer()

        modelA = structA[0]
        rmsd_vals = []
        for structB in structBs:
            modelB = structB[0]
            atomsA = []
            atomsB = []
            for chain in modelA:
                for residue in chain:
                    if residue in pocket:
                        atomsA.extend(residue.get_atoms())
            for chain in modelB:
                for residue in chain:
                    if residue in pocket:
                        atomsB.extend(residue.get_atoms())
            if not atomsA or not atomsB or len(atomsA) != len(atomsB):
                for chain in modelA:
                    for residue in chain:
                        print(residue)
                print(atomsB, atomsA, structA)
                raise ValueError("Problem with the pocket residues.")
                continue
            sup.set_atoms(atomsA, atomsB)

            rmsd_vals.append(float(sup.rms))
        return rmsd_vals

    rmsd_values = []
    all_ids = list(df["id"])
    for i in range(len(all_ids)):
        idA = all_ids[i]
        structureA = structures[idA]
        structureBs = [structures[all_ids[j]] for j in range(i + 1, len(all_ids))]
        rmsd_values.append(get_rmsd_list(structureA, structureBs, pocket))
    rmsd_matrix = squareform(sum(rmsd_values, []))  # type: ignore
    rmsd_df = pd.DataFrame(rmsd_matrix, index=df["id"], columns=df["id"], dtype=float)
    best_struc = rmsd_df.mean(axis=1).idxmin()
    assert isinstance(best_struc, str)
    t1 = time.time()
    print(
        f"Found best conformation in {t1 - t0:.2f} second (for {len(all_ids)} structures). Best structure: {best_struc}"
    )

    return best_struc


def get_best_conf_pocket_center_width(
    df: pd.DataFrame,
    pocket: List[Any],
    output_dir: str,
    min_pocket_size: int = 8,
    max_pocket_size: int = 30,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Finds the center and width of the pocket residues in the given CIF file.
    Args:
        df: DataFrame with columns "id", "pocket_residues", "sequence", and "cluster"
        sequence: Sequence of the protein
        pocket: List of residues in the pocket
        output_dir: Directory to save the best conformation PDB file
        min_pocket_size: Minimum number of residues in the pocket to consider
        max_pocket_size: Maximum number of residues in the pocket to consider
    Returns:
        Best CIF file path,
        Center of the pocket as a numpy array of shape (3,)
        Width of the pocket as a float (maximum distance from the center to any pocket residue)
    """

    t0 = time.time()

    best_struc = find_best_conf_pocket(df, pocket)

    structure = df[df["id"] == best_struc]["structure"].values[0]
    model = structure[0]
    pocket_coords: List[List[float]] = []

    for chain in model:
        for residue in chain:
            if residue in pocket:
                for atom in residue:
                    pocket_coords.append(atom.coord)
    pocket_coords_array: np.ndarray = np.array(pocket_coords)

    center = (pocket_coords_array.max(axis=0) + pocket_coords_array.min(axis=0)) / 2
    width = pocket_coords_array.max(axis=0) - pocket_coords_array.min(axis=0)
    width = np.clip(width, a_min=min_pocket_size, a_max=max_pocket_size)

    # Save to PDB file in output_dir without the ligand
    file_path = os.path.join(output_dir, "pdb_files", best_struc + ".pdb")
    os.makedirs(os.path.join(output_dir, "pdb_files"), exist_ok=True)

    io = PDBIO()

    # Remove the ligand from the structure
    for chain in model:
        for residue in list(chain):
            if residue.id[0] != " " and residue.id[0] != "W":
                chain.detach_child(residue.id)
    io.set_structure(structure)
    io.save(file_path)

    t1 = time.time()
    logger.info(f"Processed {best_struc} in {t1 - t0:.2f} seconds.")
    return best_struc, center, width


# Steps to generate the dataset.
# 1. Download the .tar.gz files from the SAIR dataset on Hugging Face.
# 2. Open the parquet sair file. Find the top-50% most potent ligands (based on pIC50) for each sequence, and select only the ones with a top 50% confidence score.
# 3. Find the residues in the pocket around the ligand for each structure-ligand pair.
# 4. For each sequence, aggregate the pocket residues from all structures to find the intersection of residues.
# 5. Align the pocket for all conformations, and find the best conformation (the least different from the others).
# 6. Save the best conformation of the pocket residues as a PDB file.
# 7. From the list of all residues in the pocket, aggregate the list of residues to determine the center of the pocket.


@ray.remote(num_cpus=1)
def process_prot(
    cif_files: List[str],
    df_prot: pd.DataFrame,
    topk: int,
    iou_threshold: float,
    output_dir: str,
    bar: Any = None,
) -> pd.DataFrame:
    df = get_seq_to_pocket_residues(cif_files, df_prot, topk)
    assert df.prot_id.nunique() == 1
    df, pocket_dict = process_df_and_aggregate_pockets(df, iou_threshold)

    res = []
    for cluster_id, pocket in pocket_dict.items():
        cluster_df = df[df["cluster"] == cluster_id]
        # Only consider clusters with more than one conformation
        if len(cluster_df) <= 1:
            continue
        res.append(
            get_best_conf_pocket_center_width(cluster_df, pocket, output_dir, 8, 30)
        )
    results = res
    final_df = pd.DataFrame(
        columns=[
            "id",
            "pocket_residues",
            "cluster",
            "center_x",
            "center_y",
            "center_z",
            "width_x",
            "width_y",
            "width_z",
            "n_ligand_poses",
            "sequence",
            "prot_id",
            "avg_pIC50",
            "avg_confidence",
        ]
    )
    idx = 0
    for cluster_id, pocket in pocket_dict.items():
        cluster_df = df[df["cluster"] == cluster_id]
        # Only consider clusters with more than one conformation
        if len(cluster_df) <= 1:
            continue
        best_struc, center, width = results[idx]
        final_df.loc[len(final_df)] = [
            best_struc,
            pocket,
            cluster_id,
            center[0],
            center[1],
            center[2],
            width[0],
            width[1],
            width[2],
            len(cluster_df),
            df.sequence.unique()[0],
            cluster_df["prot_id"].values[0],
            cluster_df["avg_pIC50"].values[0],
            cluster_df["avg_confidence"].values[0],
        ]
        idx += 1

    if bar is not None:
        bar.update.remote(1)

    return final_df


def process_sair_dataset(
    cif_files: List[str],
    df_parquet: pd.DataFrame,
    output_dir: str,
    topk: int = 3,
    iou_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Processes the SAIR dataset to identify and aggregate pocket residues.
    Args:
        cif_files: List of paths to CIF files
        df_parquet: DataFrame containing the SAIR parquet data
        output_dir: Directory to save the best conformation PDB files
        topk: Number of closest residues to the ligand to include in the pocket
        iou_threshold: IoU threshold for clustering pockets
    Returns:
        DataFrame with columns "id", "pocket_residues", "cluster", "center", "width"
    """
    protein_cif_files, protein_dfs = filter_from_parquet(df_parquet, cif_files)
    # Process the CIF files to get pocket residues and sequences
    final_dfs_jobs = []
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(total=len(protein_cif_files), desc="Processing Proteins")

    for prot in protein_cif_files:
        df_prot = protein_dfs[prot]
        cif_file_p = protein_cif_files[prot]
        if len(cif_file_p) == 0:
            pbar.update.remote(1)  # type: ignore
            continue

        final_dfs_jobs.append(
            process_prot.remote(
                cif_file_p, df_prot, topk, iou_threshold, output_dir, pbar
            )
        )
    final_dfs = ray.get(final_dfs_jobs)

    pbar.close.remote()  # type: ignore
    final_df: pd.DataFrame = pd.concat(final_dfs, ignore_index=True)
    return final_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process the SAIR dataset to identify and aggregate pocket residues."
    )
    parser.add_argument(
        "--sair-dir",
        type=str,
        required=True,
        help="Directory containing CIF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sair_pockets",
        help="Directory to save best conformation PDB files.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Padding around the ligand to define the pocket (in Å).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.6,
        help="IoU threshold for clustering pockets.",
    )
    parser.add_argument(
        "--idx-to-download",
        type=int,
        default=-1,
        help="Index of the subset to download. If -1, do not download.",
    )
    args = parser.parse_args()

    df_parquet = pd.read_parquet(os.path.join(args.sair_dir, "sair.parquet"))

    if args.idx_to_download == -1:
        os.makedirs(args.output_dir, exist_ok=True)
        cif_dir = os.path.join(args.sair_dir, "structures")

        cif_files = [
            os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith("0.cif")
        ]
        final_df = process_sair_dataset(
            cif_files,
            df_parquet,
            args.output_dir,
            args.topk,
            args.iou_threshold,
        )
        final_df.to_csv(os.path.join(args.output_dir, "sair_pockets.csv"), index=False)

    else:
        from subprocess import check_call

        structure_folders = os.path.join(args.sair_dir, f"sair_{args.idx_to_download}")
        os.makedirs(structure_folders, exist_ok=True)
        os.makedirs(os.path.join(structure_folders, "structures"), exist_ok=True)
        check_call(
            [
                "python",
                "molrgen/data/SAIR_download.py",
                "--output-dir",
                structure_folders,
                "--start-subset",
                str(args.idx_to_download),
                "--end-subset",
                str(args.idx_to_download + 1),
            ]
        )
        os.makedirs(args.output_dir, exist_ok=True)
        cif_files = [
            os.path.join(structure_folders, "structures", f)
            for f in os.listdir(os.path.join(structure_folders, "structures"))
            if f.endswith("0.cif")
        ]
        final_df = process_sair_dataset(
            cif_files,
            df_parquet,
            args.output_dir,
            args.topk,
            args.iou_threshold,
        )
        final_df.to_csv(
            os.path.join(args.output_dir, f"sair_pockets_{args.idx_to_download}.csv"),
            index=False,
        )
        # Delete the .cif files to save space
        for f in tqdm(cif_files, desc="Deleting CIF files"):
            os.remove(f)
