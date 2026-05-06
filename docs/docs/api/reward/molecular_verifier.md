# Molecular Verifier

The `MolecularVerifier` is the main orchestrator for reward computation across all molecular tasks. It automatically routes completions to the appropriate task-specific verifier based on metadata.

## Overview

The MolecularVerifier provides:

- **Unified Interface**: Single entry point for all reward computations
- **Automatic Routing**: Routes completions to the correct verifier based on metadata
- **Parallel Processing**: Uses Ray for efficient batch processing
- **GPU Support**: Supports GPU-accelerated molecular docking

## Architecture

```
MolecularVerifier
├── GenerationVerifier  → De novo molecular generation
├── MolPropVerifier     → Property prediction (regression/classification)
└── ReactionVerifier    → Chemical reactions & retro-synthesis
```

## Usage

### Basic Example

```python
from molrgen.reward import (
    MolecularVerifier,
    MolecularVerifierConfigModel,
)
from molrgen.reward.verifiers import (
    GenerationVerifierConfigModel,
    ReactionVerifierConfigModel,
    MolPropVerifierConfigModel
)

# Configure the verifier
config = MolecularVerifierConfigModel(
    generation_verifier_config=GenerationVerifierConfigModel(
        path_to_mappings="data/molgendata",
        reward="property"
    ),
    mol_prop_verifier_config=MolPropVerifierConfigModel(),
    reaction_verifier_config=ReactionVerifierConfigModel()
)

# Create verifier
verifier = MolecularVerifier(config)

# Compute rewards for generation task
completions = ["<answer>CCO</answer>"]
metadata = [{"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]}]

results = verifier(completions, metadata)
print(f"Rewards: {results.rewards}")
# Output:
# >>> Rewards: [0.1127273579103326]
```

### Mixed Batch Processing

The verifier can handle batches with different task types:

```python
completions = [
    "<answer>CCO</answer>",
    "<answer>0.75</answer>",
    "<answer>CCN(CC)C(N)=S + NNC(=O)Cc1ccc(F)cc1 -> C[CH]N(CC)c1nnc(Cc2ccc(F)cc2)[nH]1</answer>"
]

metadata = [
    {"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]},
    {"objectives": ["regression"], "target": [0.8], "norm_var": 0.1},
    {"objectives": ["full_path"], "target": ["C[CH]N(CC)c1nnc(Cc2ccc(F)cc2)[nH]1"]}
]

results = verifier(completions, metadata)

print(results.rewards)
# Output:
# >>> Rewards: [0.1127273579103326, 0.7499999999999996, 1.0]

```

::: molrgen.reward.molecular_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - MolecularVerifierConfigModel
            - BatchMolecularVerifierOutputModel



::: molrgen.reward.molecular_verifier
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - MolecularVerifier


## Task Routing

The verifier automatically determines the task type based on metadata structure:

| Metadata Fields | Task Type | Verifier |
|-----------------|-----------|----------|
| `properties`, `objectives` (maximize/minimize/above/below), `target` | Generation | `GenerationVerifier` |
| `objectives` (regression/classification), `target`, `norm_var` | Property Prediction | `MolPropVerifier` |
| `objectives` (full_path/smarts/...), `target`, `reactants` | Reaction | `ReactionVerifier` |

## Performance Considerations

- **Lazy Loading**: Verifiers are instantiated on first use
- **Ray Parallelization**: Long-running computations (docking) are parallelized
- **GPU Allocation**: Docking uses fractional GPU allocation for concurrency
- **Caching**: Oracle results are cached to avoid redundant computations

## Related

- [Generation Verifier](generation_verifier.md)
- [Property Verifier](mol_prop_verifier.md)
- [Reaction Verifier](reaction_verifier.md)
