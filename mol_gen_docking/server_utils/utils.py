import math
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from mol_gen_docking.reward.verifiers import (
    GenerationVerifierMetadataModel,
    MolPropVerifierMetadataModel,
    ReactionVerifierMetadataModel,
)


class MolecularVerifierServerQuery(BaseModel):
    """Input query model for the Molecular Verifier server.

    Represents a complete request to the molecular verifier service,
    containing metadata for scoring and completions to evaluate.

    Attributes:
        metadata: List of metadata dictionaries, one per query item.
            Each dictionary will be converted to a MolecularVerifierMetadata
            object for scoring.

        query: List of completion strings from the language model.

            The parser extracts content between these tags based on
            'parsing_method'

        prompts: Optional. Original prompts used to generate the completions.
            Useful for tracking and debugging. If provided, should have
            same length as query list.

    Notes:
    This model also supports single-item requests not in list form.
    For example:
    ```json
    {
      "query": "Here is a molecules: <answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
      "prompt": "Generate a molecule that binds to my target protein with high affinity and has more than 3 rotatable bonds.",
      "metadata": {
        "properties": ["CalcNumRotatableBonds", "sample_228234_model_0"],
        "objectives": ["above", "minimize"],
        "target": [3.0, 0.0]
      }
    }
    ```
    is also accepted and will be internally converted to:
    ```json
    {      "query": ["Here is a molecules: <answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>"],
      "prompt": ["Generate a molecule that binds to my target protein with high affinity and has more than 3 rotatable bonds."],
      "metadata": [
        {
          "properties": ["CalcNumRotatableBonds", "sample_228234_model_0"],
          "objectives": ["above", "minimize"],
          "target": [3.0, 0.0]
        }
      ]
    }
    ```

    Example:
    ```json
    {
      "query": "Here is a molecules: <answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
      "prompt": "Generate a molecule that binds to my target protein with high affinity and has more than 3 rotatable bonds.",
      "metadata": {
          "properties": ["CalcNumRotatableBonds", "sample_228234_model_0"],
          "objectives": ["above", "minimize"],
          "target": [3.0, 0.0]
        }
    }
    ```
    """

    metadata: List[dict[str, Any]] = Field(
        ..., description="List of metadata dictionaries, one per query item."
    )
    query: List[str] = Field(
        ..., description="List of completion strings from the language model."
    )
    prompts: Optional[List[str]] = Field(
        None, description="Original prompts used to generate the completions."
    )

    @field_validator("metadata", "query", "prompts", mode="before")
    @classmethod
    def convert_to_lists(cls, v: Any) -> Any:
        """Automatically convert fields to lists if they are not already.

        Ensures that metadata, query, and prompts fields are lists.
        If a field is not a list, it will be wrapped in a list.
        None values are left as-is for optional fields.

        Args:
            v: The field value to potentially convert

        Returns:
            The value as a list, or None if the input is None
        """
        if v is None:
            return None
        if not isinstance(v, list):
            return [v]
        return v

    @model_validator(mode="after")
    def validate_equal_lengths(self) -> "MolecularVerifierServerQuery":
        """Validate that all fields have equal length.

        Ensures that metadata, query, and prompts lists all have the same length.
        This validator runs after all fields have been set and converted.

        Returns:
            The validated instance

        Raises:
            ValueError: If field lengths are not equal
        """
        # Extract lengths of non-None fields
        lengths = {}
        lengths["metadata"] = len(self.metadata)
        lengths["query"] = len(self.query)
        if self.prompts is not None:
            lengths["prompts"] = len(self.prompts)
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"All query fields must have equal length. Got: {lengths}")
        return self


class MolecularVerifierServerMetadata(BaseModel):
    """Metadata returned with each scored molecule.

    Aggregates detailed scoring information from all verifier types (Generation,
    Molecular Property, and Reaction). Each field may be populated or empty
    depending on which verifier was used.

    Attributes:
        parsed_answer: The parsed answer extracted from the model completion.
        generation_verifier_metadata: Metadata from the generation verifier containing:

            - properties: List of evaluated property names
            - individual_rewards: Individual reward for each property
            - all_smi_rewards: Rewards for all SMILES found in completion
            - all_smi: List of all extracted SMILES strings
            - smiles_extraction_failure: Error message if SMILES extraction failed

        mol_prop_verifier_metadata: Metadata from the molecular property verifier containing:

            - extracted_answer: The numerical answer extracted from the completion
            - extraction_success: Whether the answer extraction was successful

        reaction_verifier_metadata: Metadata from the reaction verifier containing:

            - valid: Proportion of valid reaction steps (0.0 to 1.0)
            - correct_product: Product correctness or similarity to target molecule
            - correct_reactant: Whether all building blocks used are valid
    """

    parsed_answer: str = Field(
        "",
        description="The parsed answer extracted from the model completion.",
    )
    generation_verifier_metadata: Optional[GenerationVerifierMetadataModel] = Field(
        None,
        description="Metadata from the generation verifier, if applicable.",
    )
    mol_prop_verifier_metadata: Optional[MolPropVerifierMetadataModel] = Field(
        None,
        description="Metadata from the molecular property verifier, if applicable.",
    )
    reaction_verifier_metadata: Optional[ReactionVerifierMetadataModel] = Field(
        None,
        description="Metadata from the reaction verifier, if applicable.",
    )


class MolecularVerifierServerResponse(BaseModel):
    """Response from the Molecular Verifier server when the sever is in single-item mode (default).

    Contains the computed reward score and detailed metadata for a single
    evaluated completion.

    Attributes:
        reward: Overall reward score combining all evaluated properties.
            Typically normalized to [0.0, 1.0] range when rescaling is enabled.

        error: Error message if scoring failed. None if successful.

        meta: Metadata dictionary with detailed scoring information.
            Contains extraction failures, property rewards, and verification
            results for the completion.

        next_turn_feedback: Optional feedback for multi-turn conversations.
            Can be used to guide subsequent model generations.

    Example:
        ```json
        {
          "reward": 0.75,
          "error": null,
          "meta": {
              "smiles_extraction_failure": "",
              "all_smi": ["CC(C)Cc1ccc(cc1)"],
              "all_smi_rewards": [0.75],
              "properties": ["GSK3B", "CalcLogP"],
              "individual_rewards": [1.0, 0.5]
            },
          "next_turn_feedback": null
        }
        ```
    """

    reward: float = Field(
        ..., description="Overall reward score combining all evaluated properties."
    )
    error: Optional[str] = Field(None, description="Error message if scoring failed.")
    meta: MolecularVerifierServerMetadata = Field(
        default_factory=MolecularVerifierServerMetadata,
        description="Metadata dictionary with detailed scoring information.",
    )
    next_turn_feedback: Optional[str] = Field(
        None, description="Optional feedback for multi-turn conversations."
    )

    @field_validator("reward")
    @classmethod
    def validate_reward(cls, v: Any) -> float:
        """Validate and normalize reward value.

        If reward is None or NaN, sets it to 0.0.
        Ensures reward is a valid float.

        Args:
            v: The reward value to validate

        Returns:
            The validated reward value (float), or 0.0 if None/NaN

        Raises:
            ValueError: If reward is not a valid numeric type
        """
        # Handle None case
        if v is None:
            return 0.0
        # Handle NaN case
        if math.isnan(float(v)):
            return 0.0
        # Check if it's a numeric type
        if not isinstance(v, (int, float)):
            raise ValueError(f"reward must be a float, got {type(v).__name__}")
        return float(v)


class BatchMolecularVerifierServerResponse(BaseModel):
    """Response from the Molecular Verifier server when in batch mode.

    Contains the computed reward scores and detailed metadata for each
    evaluated completion.

    Attributes:
        rewards: List of individual reward scores, one per evaluated
            completion in the request.

        error: Error message if scoring failed. None if successful.

        metas: List of metadata dictionaries with detailed scoring information.
            Contains extraction failures, property rewards, and verification
            results for each completion. One metadata object per query item.

        next_turn_feedback: Optional feedback for multi-turn conversations.
            Can be used to guide subsequent model generations.

    Example:
        ```json
        {
          "rewards": [0.75],
          "error": null,
          "metas": [
             "generation_verifier_metadata": {
                    "smiles_extraction_failure": "",
                    "all_smi": ["CC(C)Cc1ccc(cc1)"],
                    "all_smi_rewards": [0.75],
                    "properties": ["GSK3B", "CalcLogP"],
                    "individual_rewards": [1.0, 0.5]
                }
            }
          ],
          "next_turn_feedback": null
        }
        ```
    """

    rewards: List[float] = Field(
        ...,
        description="List of individual reward scores, one per evaluated completion.",
    )
    error: Optional[str] = Field(None, description="Error message if scoring failed.")
    metas: List[MolecularVerifierServerMetadata] = Field(
        default_factory=list,
        description="List of metadata with detailed scoring information.",
    )
    next_turn_feedback: Optional[str] = Field(
        None, description="Optional feedback for multi-turn conversations."
    )

    @field_validator("rewards")
    @classmethod
    def validate_rewards(cls, v: Any) -> List[float]:
        """Validate and normalize reward value.

        If any reward is None or NaN, sets it to 0.0.
        Ensures reward is a valid float.

        Args:
            v: The reward value to validate

        Returns:
            The validated reward value (float), or 0.0 if None/NaN

        Raises:
            ValueError: If reward is not a valid numeric type
        """
        assert isinstance(v, list)
        # Handle None case
        for i, val in enumerate(v):
            if val is None:
                val = 0.0
            # Handle NaN case
            if math.isnan(float(val)):
                val = 0.0
            # Check if it's a numeric type
            if not isinstance(val, (int, float)):
                raise ValueError(f"reward must be a float, got {type(val).__name__}")
            v[i] = float(val)
        return v
