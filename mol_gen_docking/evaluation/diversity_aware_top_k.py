"""Diversity-aware top-k evaluation metric for molecular generation.

This module implements a diversity-aware variant of the top-k metric that selects
molecules not only based on their scores but also on their chemical diversity.
It ensures selected molecules are sufficiently different from each other, preventing
the selection of similar redundant compounds.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import squareform

from mol_gen_docking.evaluation.fingeprints_utils import get_sim_matrix


def div_aware_top_k_from_dist(
    dist: np.ndarray[float],
    weights: np.ndarray[float],
    k: int,
    t: float,
) -> np.ndarray:
    """Select at most k molecules with highest weights while enforcing minimum distance.

    This function implements a greedy selection algorithm that selects molecules
    with the highest weights while ensuring each selected molecule is at distance
    (dissimilarity) of at least t from all previously selected molecules.

    Args:
        dist: Condensed distance matrix (1D array of upper triangle distances).
            This should be from scipy.spatial.distance.squareform or similar.
            Distance values should be in range [0, 1] where 0 = identical, 1 = completely different.
        weights: 1D array of weights/scores for each molecule. Higher weights are selected first.
            Must have length n where n*(n-1)/2 == len(dist).
        k: Maximum number of molecules to select.
        t: Minimum distance threshold. Selected molecules must be at distance >= t
            from each other (i.e., dissimilarity >= t).

    Returns:
        1D NumPy array of indices of selected molecules, sorted by weight (descending).
        Array may contain fewer than k elements if not enough molecules satisfy
        the distance constraint.

    Raises:
        AssertionError: If the distance matrix size doesn't match the weights array.

    Example:
        ```python
        import numpy as np
        from mol_gen_docking.evaluation.diversity_aware_top_k import div_aware_top_k_from_dist

        # 3 molecules: dist matrix has 3 pairwise distances
        dist = np.array([0.3, 0.8, 0.2])  # condensed distance matrix
        weights = np.array([8.5, 9.2, 7.1])

        selected = div_aware_top_k_from_dist(dist, weights, k=2, t=0.5)
        print(f"Selected indices: {selected}")  # Molecules at distance >= 0.5
        ```

    Notes:
        - Uses a greedy algorithm: sorts by weight and selects molecules in order
        - Once a molecule is selected, it acts as a constraint for future selections
        - No backtracking: if a high-weight molecule can't be selected due to distance
          constraints, it's skipped (lower-weight candidates are checked next)
    """
    n = len(weights)
    assert n * (n - 1) // 2 == len(dist), (
        "Distance matrix size does not match number of weights"
    )
    selected: List[int] = []
    sorted_indices = np.argsort(-weights)  # Sort indices by descending weights
    dist_mat = squareform(dist)
    dist_mat = dist_mat < t

    for idx in sorted_indices:
        if len(selected) >= k:
            break
        is_idx_too_close = dist_mat[idx, selected].any() if selected else False
        if not is_idx_too_close:
            selected.append(idx)
    return np.array(selected)


def diversity_aware_top_k(
    mols: List[Chem.Mol] | List[str] | np.ndarray,
    scores: Sequence[float | int],
    k: int,
    t: float,
    fingerprint_name: Optional[str] = "ecfp4-1024",
    return_idxs: bool = False,
) -> float | Tuple[float, List[int]]:
    """Calculate diversity-aware top-k metric for molecular generation.

        This function computes a diversity-aware top-k metric that selects up to k molecules
        with the highest scores, subject to the constraint that selected molecules must
        have chemical similarity below a threshold (i.e., dissimilarity above 1-t).
        This prevents selecting multiple similar molecules and encourages chemical diversity.

        Args:
            mols: List of molecules in one of three formats:
                - List of SMILES strings (str)
                - List of RDKit Mol objects (Chem.Mol)
                - 2D NumPy array representing a similarity matrix
            scores: Sequence of scores corresponding to each molecule (e.g., docking scores).
                Must have the same length as mols (unless mols is a similarity matrix).
            k: Maximum number of molecules to select.
            t: Similarity threshold (range 0.0 to 1.0). Selected molecules must have
                Tanimoto similarity < t to be considered diverse enough.
                Lower values enforce higher diversity.
            fingerprint_name: Name of the molecular fingerprint to use for similarity calculation.
                Only used when mols are SMILES or Mol objects. Default is "ecfp4-1024".
                See mol_gen_docking.evaluation.fingeprints_utils.fp_name_to_fn for options.
    return_idxs            If False (default), return only the average score.

        Returns:
            If return_idxs is False (default):
                Average score of the selected k diverse molecules. If fewer than k molecules
                are selected due to diversity constraints, unselected slots are padded with 0.0.
            If return_idxs is True:
                Tuple of (average_score, indices) where:
                    - average_score: Average score of selected molecules (with 0.0 padding)
                    - indices: List of indices of selected molecules from the input

        Raises:
            AssertionError: If mols and scores have different lengths, or if input types
                are inconsistent.

        Example:
            ```python
            from mol_gen_docking.evaluation.diversity_aware_top_k import diversity_aware_top_k
            from rdkit import Chem

            # Using SMILES strings
            smiles = [
                "c1ccccc1",                              # benzene
                "CC(C)Cc1ccc(cc1)C(C)C(O)=O",           # ibuprofen
                "c1ccc2ccccc2c1",                       # naphthalene (similar to benzene)
                "CCO"                                    # ethanol
            ]
            scores = [8.5, 9.2, 8.0, 6.5]

            # Select top 2 molecules with similarity threshold 0.8
            metric = diversity_aware_top_k(
                smiles, scores, k=2, t=0.8, fingerprint_name="ecfp4-1024"
            )
            print(f"Diversity-aware top-2 score: {metric}")

            # Get both score and indices
            metric, indices = diversity_aware_top_k(
                smiles, scores, k=2, t=0.8, fingerprint_name="ecfp4-1024", return_idxs=True
            )
            print(f"Score: {metric}, Selected indices: {indices}")

            # Using a pre-computed similarity matrix
            sim_matrix = np.array([
                [1.0, 0.3, 0.9, 0.2],
                [0.3, 1.0, 0.4, 0.6],
                [0.9, 0.4, 1.0, 0.3],
                [0.2, 0.6, 0.3, 1.0]
            ])
            metric, idxs = diversity_aware_top_k(
                sim_matrix, scores, k=2, t=0.8
            )
            ```

        Notes:
            - The function converts similarity matrices to distance matrices (1 - similarity)
            - Higher t values (closer to 1.0) allow selection of more similar molecules
            - Lower t values enforce stricter diversity constraints
            - If a similarity matrix is provided directly, it should be a 2D NumPy array
              with diagonal elements equal to 1.0
            - Padding with 0.0 for unselected slots means diversity constraints can
              result in lower average scores than unconstrained top-k

        References:
            This metric is commonly used in molecular generation benchmarks to evaluate
            both quality and diversity of generated molecules (e.g., De Novo Generation task).
    """
    dist_mat: np.ndarray[float]
    assert len(mols) == len(scores), "Mols and scores must have the same length."

    if isinstance(mols[0], str) or isinstance(mols[0], Chem.Mol):
        assert fingerprint_name is not None, (
            "Fingerprint name must be provided when mols are SMILES or Mol objects."
        )
        mols_list: List[Chem.Mol]
        if isinstance(mols[0], str):
            assert all(isinstance(smi, str) for smi in mols), (
                "All elements must be SMILES strings since the first is a string."
            )
            mols_list = [Chem.MolFromSmiles(smi) for smi in mols]
        else:
            assert all(isinstance(mol, Chem.Mol) for mol in mols), (
                "All elements must be RDKit Mol objects since the first is a Mol."
            )
            mols_list = mols
        dist_mat = 1 - get_sim_matrix(mols_list, fingerprint_name=fingerprint_name)
    else:
        assert isinstance(mols, np.ndarray), "Unknown type for mols."
        assert mols.ndim == 2, (
            "Using distance matrix directly requires a 2D numpy array."
        )
        assert (mols.diagonal() == 1.0).all(), (
            "Similarity matrix diagonal must be all zeros."
        )
        dist_mat = squareform(1 - mols)

    idxs = div_aware_top_k_from_dist(dist_mat, np.array(scores), k, 1 - t)

    scores_arr = np.array(
        [scores[idx] for idx in idxs] + [0.0 for _ in range(len(idxs), k)]
    )
    out_val: float = np.mean(scores_arr)
    if not return_idxs:
        return out_val
    else:
        return out_val, idxs.tolist()
