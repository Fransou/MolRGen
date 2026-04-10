"""Verifiers module for molecular reward computation.

This module provides specialized verifiers for different molecular tasks:
- **GenerationVerifier**: De novo molecular generation with property optimization
- **MolPropVerifier**: Molecular property prediction (regression/classification)
- **ReactionVerifier**: Chemical reaction and retro-synthesis tasks

Each verifier has associated configuration models, input metadata models,
and output models for type-safe operation.
"""

from .abstract_verifier import Verifier
from .abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
    VerifierOutputModel,
)
from .generation_reward.generation_verifier import GenerationVerifier
from .generation_reward.generation_verifier_pydantic_model import (
    DockingGPUConfigModel,
    GenerationVerifierConfigModel,
    GenerationVerifierMetadataModel,
    GenerationVerifierOutputModel,
    PyscreenerConfigModel,
)
from .generation_reward.input_metadata import GenerationVerifierInputMetadataModel
from .mol_prop_reward.input_metadata import MolPropVerifierInputMetadataModel
from .mol_prop_reward.mol_prop_verifier import MolPropVerifier
from .mol_prop_reward.mol_prop_verifier_pydantic_model import (
    MolPropVerifierConfigModel,
    MolPropVerifierMetadataModel,
    MolPropVerifierOutputModel,
)
from .reaction_reward.input_metadata import ReactionVerifierInputMetadataModel
from .reaction_reward.reaction_verifier import ReactionVerifier
from .reaction_reward.reaction_verifier_pydantic_model import (
    ReactionVerifierConfigModel,
    ReactionVerifierMetadataModel,
    ReactionVerifierOutputModel,
)


def assign_to_inputs(completion: str, metadata: dict) -> str:
    """Determine which verifier should handle a completion based on its metadata.

    This function attempts to validate the metadata against each verifier's input
    model to determine the appropriate verifier type.

    Args:
        completion: The model completion string.
        metadata: Dictionary of metadata associated with the completion.

    Returns:
        String identifier for the verifier type: "generation", "mol_prop", or "reaction".

    Raises:
        NotImplementedError: If the metadata doesn't match any verifier input model.

    Example:
        ```python
        metadata = {"properties": ["QED"], "objectives": ["maximize"], "target": [0.0]}
        verifier_type = assign_to_inputs("<answer>CCO</answer>", metadata)
        print(verifier_type)  # "generation"
        ```
    """
    for b_model_cls, label in [
        (GenerationVerifierInputMetadataModel, "generation"),
        (MolPropVerifierInputMetadataModel, "mol_prop"),
        (ReactionVerifierInputMetadataModel, "reaction"),
    ]:
        try:
            b_model_cls.model_validate({"completion": completion, **metadata})  # type: ignore
            return label
        except Exception:
            continue
    raise NotImplementedError(
        f"Input metadata does not match any verifier input model for completion: {completion} with metadata: {metadata}"
    )


__all__ = [
    "Verifier",
    "GenerationVerifier",
    "MolPropVerifier",
    "ReactionVerifier",
    ###
    "GenerationVerifierConfigModel",
    "MolPropVerifierConfigModel",
    "ReactionVerifierConfigModel",
    ###,
    "VerifierOutputModel",
    "GenerationVerifierOutputModel",
    "MolPropVerifierOutputModel",
    "ReactionVerifierOutputModel",
    ###
    "GenerationVerifierMetadataModel",
    "MolPropVerifierMetadataModel",
    "ReactionVerifierMetadataModel",
    ###
    "GenerationVerifierInputMetadataModel",
    "MolPropVerifierInputMetadataModel",
    "ReactionVerifierInputMetadataModel",
    ###
    "assign_to_inputs",
    "BatchVerifiersInputModel",
    ###
    "DockingGPUConfigModel",
    "PyscreenerConfigModel",
]
