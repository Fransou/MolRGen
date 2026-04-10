"""Reward module for molecular verification and scoring.

This module provides the main interface for computing rewards based on molecular
generation, property prediction, and chemical reaction tasks. It includes verifiers
for different task types and configuration models for customization.

The module is organized into:
- **MolecularVerifier**: Main orchestrator for reward computation across task types
- **GenerationVerifier**: Verifier for de novo molecular generation tasks
- **MolPropVerifier**: Verifier for molecular property prediction tasks
- **ReactionVerifier**: Verifier for chemical reaction and retro-synthesis tasks

Example:
    ```python
    from mol_gen_docking.reward import MolecularVerifier, MolecularVerifierConfigModel

    config = MolecularVerifierConfigModel(
        generation_verifier_config={"path_to_mappings": "data/molgendata"}
    )
    verifier = MolecularVerifier(config)
    results = verifier(completions, metadata)
    ```
"""

from .molecular_verifier import MolecularVerifier
from .molecular_verifier_pydantic_model import (
    MolecularVerifierConfigModel,
)
from .verifiers import (
    BatchVerifiersInputModel,
    DockingGPUConfigModel,
    GenerationVerifier,
    GenerationVerifierConfigModel,
    MolPropVerifier,
    MolPropVerifierConfigModel,
    PyscreenerConfigModel,
    ReactionVerifier,
    ReactionVerifierConfigModel,
)

__all__ = [
    "MolecularVerifier",
    "MolecularVerifierConfigModel",
    "BatchVerifiersInputModel",
    ###
    "GenerationVerifier",
    "MolPropVerifier",
    "ReactionVerifier",
    ###
    "GenerationVerifierConfigModel",
    "MolPropVerifierConfigModel",
    "ReactionVerifierConfigModel",
    ###
    "PyscreenerConfigModel",
    "DockingGPUConfigModel",
]
