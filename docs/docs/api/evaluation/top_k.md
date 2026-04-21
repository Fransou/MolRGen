# Top-k Metric

## Overview

The **top-k metric** is a standard evaluation metric for molecular generation tasks. It measures the average quality of the top k unique molecules from a set of generated candidates, ensuring that duplicate molecules are not counted multiple times.
!!! note
    ### Uniqueness Constraint
    The top-k metric enforces uniqueness by:

    - Converting all molecules to canonical SMILES representation
    - Removing duplicate molecules
    - Selecting the k molecules with the highest scores

    ### Padding Mechanism
    If fewer than k unique molecules are available (i.e the model cannot generate as many candidates), the remaining slots are padded with 0.0 scores.

## Usage Examples

### Basic Usage with SMILES

```python
from molrgen.evaluation.top_k import top_k

# List of generated molecules as SMILES
smiles = [
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "c1ccccc1",
    "CCO",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O" # Ibuprofen duplicate but different smiles
]

# Docking scores for each molecule
scores = [8.5, 6.2, 6.1, 8.5]

# Calculate top-2 score
metric = top_k(smiles, scores, k=2)
print(f"Top-2 score: {metric}")
# Output:
# >>> 7.35
```

### Using RDKit Mol Objects

```python
from molrgen.evaluation.top_k import top_k
from rdkit import Chem

# Convert to Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in smiles]

# top_k automatically canonicalizes Mol objects
metric = top_k(mols, scores, k=2)
print(metric)

# Output:
# >>> 7.35

```

### Without Canonicalization

```python
# If SMILES strings are already canonical
metric = top_k(smiles, scores, k=2, canonicalize=False)
print(metric)

# Output:
# >>> 8.5 # Since both ibuprofen entries are considered unique without canonicalization
```

## Function Reference

::: molrgen.evaluation.top_k
    options:
        show_root_toc_entry: false
        heading_level: 3

## Benchmark Context

In the MolRGen project, the top-k metric is used to evaluate molecular generation models on raw-quality without diversity constraints.
We assess how well a model can generate multiple high-scoring molecules, possibly from a same chemical serie.

See [Diversity-Aware Top-k](diversity_aware_top_k.md) for a variant that also enforces chemical diversity.
