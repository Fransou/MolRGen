"""Abstract base classes for molecular verifiers.

This module provides the base classes and input/output models used by all
verifiers in the reward system. It defines the common interface that all
verifiers must implement.
"""

from typing import Any, List

from pydantic import BaseModel, Field, model_validator

from molrgen.reward.verifiers.generation_reward.input_metadata import (
    GenerationVerifierInputMetadataModel,
)
from molrgen.reward.verifiers.mol_prop_reward.input_metadata import (
    MolPropVerifierInputMetadataModel,
)
from molrgen.reward.verifiers.reaction_reward.input_metadata import (
    ReactionVerifierInputMetadataModel,
)


class BatchVerifiersInputModel(BaseModel):
    """Input model for batch verification requests.

    This model encapsulates a batch of completions and their corresponding
    metadata for efficient batch processing by verifiers.

    Attributes:
        completions: List of model completion strings to verify.
        metadatas: List of metadata corresponding to each completion.
            The metadata type determines which verifier processes it.

    Example:
        ```python
        inputs = BatchVerifiersInputModel(
            completions=["<answer>CCO</answer>", "<answer>c1ccccc1</answer>"],
            metadatas=[
                GenerationVerifierInputMetadataModel(
                    properties=["QED"], objectives=["maximize"], target=[0.0]
                ),
                GenerationVerifierInputMetadataModel(
                    properties=["SA"], objectives=["minimize"], target=[0.0]
                )
            ]
        )
        ```
    """

    completions: List[str] = Field(
        ..., description="List of model completions to be verified."
    )
    metadatas: (
        List[GenerationVerifierInputMetadataModel]
        | List[MolPropVerifierInputMetadataModel]
        | List[ReactionVerifierInputMetadataModel]
    ) = Field(
        ...,
        description="List of metadata corresponding to each completion.",
    )

    @model_validator(mode="after")
    def check_lengths(self) -> "BatchVerifiersInputModel":
        """Validate that completions and metadatas have the same length."""
        if len(self.completions) != len(self.metadatas):
            raise ValueError("`completions` and `metadatas` must have the same length")
        return self


class VerifierOutputModel(BaseModel):
    """Base output model for all verifier results.

    This model defines the common structure for outputs from any verifier,
    containing the computed reward and associated metadata.

    Attributes:
        reward: The computed reward score (typically 0.0 to 1.0).
        parsed_answer: The parsed answer extracted from the model completion.
        verifier_metadata: Additional metadata from the verification process.
    """

    reward: float = Field(..., description="Reward score assigned by the verifier.")
    parsed_answer: str = Field(
        ..., description="The parsed answer extracted from the model completion."
    )
    verifier_metadata: Any = Field(
        ..., description="Additional metadata from the verification process."
    )
