"""Top-k evaluation metric for molecular generation tasks.

This module implements the standard top-k metric for evaluating molecular generation models.
The metric measures the average quality of the top k unique molecules from a set of generated candidates.
"""

from typing import List

from rdkit import Chem


def top_k(
    mols: List[str] | List[Chem.Mol],
    scores: List[float],
    k: int,
    canonicalize: bool = True,
) -> float:
    """Calculate the top-k metric for molecular generation.

    This function computes the average score of the top k unique molecules from a set
    of candidates. It first deduplicates molecules using canonical SMILES representation,
    then selects the k molecules with the highest scores. This metric is useful for
    evaluating the quality of generated molecules.

    Args:
        mols: List of molecules as SMILES strings or RDKit Mol objects.
        scores: List of scores corresponding to each molecule (e.g., docking scores,
            binding affinity). Must have the same length as mols.
        k: Number of top molecules to consider. If fewer than k unique molecules are
            provided, the remaining slots are filled with 0.0 scores.
        canonicalize: Whether to canonicalize SMILES strings before comparison.
            If True, molecules are converted to canonical SMILES for deduplication.
            Default is True.

    Returns:
        Average score of the top k unique molecules. If fewer than k unique molecules
        are available, the average includes padding with 0.0 scores.

    Raises:
        AssertionError: If mols and scores have different lengths.

    Example:
        ```python
        from molrgen.evaluation.top_k import top_k
        from rdkit import Chem

        # Using SMILES strings
        smiles = ["CC(C)Cc1ccc(cc1)C(C)C(O)=O", "c1ccccc1"]
        scores = [8.5, 7.2]
        metric = top_k(smiles, scores, k=2)
        print(f"Top-2 score: {metric}")

        # Using RDKit Mol objects
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        metric = top_k(mols, scores, k=2)
        ```

    Notes:
        - Duplicate molecules are automatically detected and removed using canonical SMILES
        - If k is larger than the number of unique molecules, remaining slots are filled with 0.0
        - The final metric is the average of the k highest scores
    """
    smi_list: List[str]
    if canonicalize or isinstance(mols[0], Chem.Mol):
        if isinstance(mols[0], str):
            mols_list = [Chem.MolFromSmiles(smi) for smi in mols]
        else:
            mols_list = mols
        smi_list = [Chem.MolToSmiles(mol, canonical=True) for mol in mols_list]
    else:
        smi_list = mols

    # Drop ducplicates and keep idxs
    seen = set()
    unique_idxs = []
    for idx, smi in enumerate(smi_list):
        if smi not in seen:
            seen.add(smi)
            unique_idxs.append(idx)
    unique_scores = [scores[idx] for idx in unique_idxs] + [
        0.0 for _ in range(len(unique_idxs), k)
    ]
    unique_scores = sorted(unique_scores, reverse=True)[:k]
    return sum(unique_scores) / k
