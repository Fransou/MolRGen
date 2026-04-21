# Property Verifier

The `MolPropVerifier` computes rewards for molecular property prediction tasks, supporting both regression and classification objectives.

## Overview

The Property Verifier evaluates model predictions of molecular properties such as:

- **ADMET Properties**: Absorption, Distribution, Metabolism, Excretion, Toxicity
- **Physicochemical Properties**: Solubility, LogP, pKa
- **Biological Activity**: IC50, EC50, binding affinity

### Answer Extraction

The verifier extracts answers from completions using:

1. **Answer Tags**: Content between `<answer>` and `</answer>` tags
2. **Value Parsing**: Extracts numeric values or boolean keywords

### Supported Answer Formats

| Format | Parsed Value |
|--------|--------------|
| `0.75` | 0.75 (float) |
| `42` | 42 (int) |
| `True`, `Yes` | 1 |
| `False`, `No` | 0 |

### Invalid Answers

Answers are considered invalid (reward = 0.0) if:
- No answer tags found
- Multiple values in answer
- No numeric/boolean value found




::: molrgen.reward.verifiers.mol_prop_reward.mol_prop_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - MolPropVerifierConfigModel


::: molrgen.reward.verifiers.mol_prop_reward.input_metadata
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - MolPropVerifierInputMetadataModel

::: molrgen.reward.verifiers.mol_prop_reward.mol_prop_verifier_pydantic_model
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - MolPropVerifierOutputModel
            - MolPropVerifierMetadataModel

::: molrgen.reward.verifiers.mol_prop_reward.mol_prop_verifier
    handler: python
    heading_level: 3
    options:
        show_root_toc_entry: false
        members:
            - MolPropVerifier

## Related

- [Molecular Verifier](molecular_verifier.md) - Main orchestrator
- [Generation Verifier](generation_verifier.md) - De novo generation tasks
- [Reaction Verifier](reaction_verifier.md) - Reaction prediction and retro-synthesis tasks
