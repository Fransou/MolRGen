# Diversity-Aware Top-k Metric

## Overview

The **diversity-aware top-k metric** is an advanced evaluation metric for molecular generation tasks that balances both **quality** and **chemical diversity**. Unlike the standard top-k metric which only considers scores, this metric enforces that selected molecules are sufficiently different from each other.


## Key Concepts

### Diversity Constraint

The metric uses a **similarity threshold** (parameter `t`) to enforce diversity:

- Selected molecules must have **Tanimoto similarity < t** with each other
- Lower `t` values enforce stricter diversity (e.g., t=0.1 requires very different molecules), and will be increasingly challenging as `t` decreases.
- Higher `t` values allow more similar molecules (e.g., t=0.9 is more lenient), and the results of the diversity metric should converge to the top-k score as `t` approaches 1 (see [Top-k Metric](top_k.md)).

### Greedy Selection Algorithm

The selection process uses a greedy approach:

1. Sort molecules by score (highest first)
2. Select the highest-scoring molecule
3. For each remaining molecule (in descending score order):

      - If it's sufficiently different from all selected molecules, select it
     - Otherwise, skip it and check the next candidate

4. Stop when k molecules are selected or all candidates are evaluated

### Padding Mechanism

If fewer than k diverse molecules are found, the remaining slots are padded with 0.0 scores, penalizing models unable to generate k molecules meeting the diversity-constraint.

## Molecular Fingerprints

The metric uses **molecular fingerprints** to compute similarity:

- **Default**: ECFP4-1024 (Extended Connectivity Fingerprint, diameter 4, 1024 bits)
- **Supported**: ECFP, MACCS, RDKit, Gobbi2D, Avalon
- See [Fingerprint Utilities](fingeprints_utils.md) for details

## Usage Examples

### Basic Usage with SMILES

```python
from molrgen.evaluation.diversity_aware_top_k import diversity_aware_top_k

# Generated molecules as SMILES
smiles = [
    "c1ccccc1",                           # (score: 8.5)
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",        # (score: 9.2)
    "c1ccc2ccccc2c1",                    # (score: 8.0) - similar to benzene
    "CCO"                                 # (score: 6.5)
]

scores = [8.5, 9.2, 8.0, 6.5]

# Select top 2 molecules with similarity threshold 0.7
# (molecules must have similarity < 0.7, i.e., distance > 0.3)
metric = diversity_aware_top_k(
    smiles,
    scores,
    k=2,
    t=0.9,
    fingerprint_name="ecfp4-1024"
)
print(f"Diversity-aware top-2 score: {metric}")
# Output:
# >>> Diversity-aware top-2 score: 8.85
metric = diversity_aware_top_k(
    smiles,
    scores,
    k=2,
    t=0.05,
    fingerprint_name="ecfp4-1024"
)
print(f"Diversity-aware top-2 score: {metric}")

# Output:
# >>> Diversity-aware top-2 score: 4.6
```

### Using RDKit Mol Objects

```python
from molrgen.evaluation.diversity_aware_top_k import diversity_aware_top_k
from rdkit import Chem

# Convert to Mol objects
mols = [Chem.MolFromSmiles(smi) for smi in smiles]

# Same interface works with Mol objects
metric = diversity_aware_top_k(mols, scores, k=3, t=0.7)
print(metric)

# Output:
# >>> 8.566666666666666
```

### Using Pre-computed Similarity Matrix

```python
import numpy as np
from molrgen.evaluation.diversity_aware_top_k import diversity_aware_top_k

# Pre-computed similarity matrix (diagonal = 1.0)
sim_matrix = np.array([
    [1.0, 0.3, 0.9, 0.2],   # benzene
    [0.3, 1.0, 0.4, 0.6],   # ibuprofen
    [0.9, 0.4, 1.0, 0.3],   # naphthalene
    [0.2, 0.6, 0.3, 1.0]    # ethanol
])

scores = [8.5, 9.2, 8.0, 6.5]

# Use similarity matrix directly
metric = diversity_aware_top_k(
    sim_matrix,
    scores,
    k=2,
    t=0.7
)
print(metric)

# Output:
# >>> 8.85
```


## Function Reference


::: molrgen.evaluation.diversity_aware_top_k
    options:
        show_root_toc_entry: false
        heading_level: 3



## Common Pitfalls

1. **Confusing similarity and distance**: Remember t is similarity threshold, distance = 1 - similarity
2. **Invalid SMILES**: Always validate SMILES strings before passing to function
3. **Threshold interpretation**: Lower t = stricter diversity, not lenient
