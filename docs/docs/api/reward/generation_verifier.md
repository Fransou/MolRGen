# Generation Verifier

The `GenerationVerifier` computes rewards for de novo molecular generation tasks, evaluating generated molecules against property optimization criteria such as docking scores, QED, synthetic accessibility, and other molecular descriptors.

## Overview

The Generation Verifier supports:

- **Multi-property Optimization**: Optimize multiple properties simultaneously
- **Docking Score Computation**: GPU-accelerated molecular docking with AutoDock
- **RDKit Descriptors**: QED, SA score, LogP, molecular weight, etc.
- **SMILES Extraction**: Robust parsing of SMILES from model completions

### Supported Properties

| Property                    | Type | Description                          |
|-----------------------------|------|--------------------------------------|
| Docking Targets             | Slow (GPU) | Binding affinity to protein pockets  |
| Physico-Chemical Properties | Fast | QED, SA score, Molecular Weight, ... |


### SMILES Extraction

The verifier extracts SMILES from completions using:

1. **Answer Tags**: Content between `<answer>` and `</answer>` tags
2. **Pattern Matching**: Identifies possible valid SMILES patterns (extracted word with no characters outside SMILES charset, and that contains at least one `C` character, or multiple `c`)
3. **Validation**: Verifies molecules with RDKit

### Extraction Failures

| Failure Reason | Description |
|----------------|-------------|
| `no_answer` | No answer tags found |
| `no_smiles` | No SMILES-like strings in answer |
| `no_valid_smiles` | SMILES strings are invalid |
| `multiple_smiles` | Multiple valid SMILES found (ambiguous) |



::: molrgen.reward.verifiers.generation_reward.generation_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - DockingConfigModel
            - DockingGPUConfigModel
            - PyscreenerConfigModel

::: molrgen.reward.verifiers.generation_reward.input_metadata
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - GenerationVerifierInputMetadataModel

::: molrgen.reward.verifiers.generation_reward.generation_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - GenerationVerifierOutputModel
            - GenerationVerifierMetadataModel

::: molrgen.reward.verifiers.generation_reward.generation_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - GenerationVerifierConfigModel

::: molrgen.reward.verifiers.generation_reward.generation_verifier
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - GenerationVerifier

## Related

- [Molecular Verifier](molecular_verifier.md) - Main orchestrator
- [Property Verifier](mol_prop_verifier.md) - Molecular property prediction tasks
- [Reaction Verifier](reaction_verifier.md) - Reaction prediction and retro-synthesis tasks
