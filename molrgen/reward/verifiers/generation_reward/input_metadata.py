from typing import List, Literal

from pydantic import BaseModel, Field, model_validator

GenerationObjT = Literal["maximize", "minimize", "above", "below"]


class GenerationVerifierInputMetadataModel(BaseModel):
    """Input metadata model for generation verifier.

    Defines the verification criteria for molecular generation tasks, including
    properties to optimize, objectives for each property, and target values.

    Attributes:
        properties: List of property names to verify (e.g., "QED", "SA", "docking_target_name").
            Each property should be a valid molecular descriptor or a docking target name.
            Must have the same length as objectives and target.

        objectives: List of objectives for each property.
            Must have the same length as properties and target.
            Valid values:
            - "maximize": Reward increases with property value
            - "minimize": Reward increases as property value decreases
            - "above": Reward is 1.0 if property >= target, 0.0 otherwise
            - "below": Reward is 1.0 if property <= target, 0.0 otherwise

        target: List of target values for each property.
            Must have the same length as properties and objectives.
            For "maximize"/"minimize": Used as reference point for rescaling (when enabled)
            For "above"/"below": Used as threshold for binary reward computation
    """

    properties: List[str] = Field(
        ...,
        description="List of property names to verify.",
    )
    objectives: List[GenerationObjT] = Field(
        ...,
        description="List of objectives for each property: maximize, minimize, above, or below.",
    )
    target: List[float] = Field(
        ...,
        description="List of target values for each property.",
    )

    @model_validator(mode="after")
    def validate_properties(self) -> "GenerationVerifierInputMetadataModel":
        """Validate that properties, objectives, and target have the same length."""
        if not (len(self.properties) == len(self.objectives) == len(self.target)):
            raise ValueError(
                "Length of properties, objectives, and target must be the same."
            )
        return self
