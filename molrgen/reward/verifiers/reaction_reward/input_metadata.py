from typing import List, Literal

from pydantic import BaseModel, Field

ReactionObjT = Literal[
    "final_product",
    "reactant",
    "all_reactants",
    "all_reactants_bb_ref",
    "smarts",
    "full_path",
    "full_path_bb_ref",
    "full_path_smarts_ref",
    "full_path_smarts_bb_ref",
    "full_path_intermediates",
    "full_path_intermediates_gt_reactants",
]


class ReactionVerifierInputMetadataModel(BaseModel):
    """Input metadata model for reaction verifier.

    Defines the verification criteria for chemical reaction and retro-synthesis tasks,
    including objectives, target molecules, reactants, products, and validation constraints.

    Attributes:
        objectives: List of objective types for the reaction verification.
            Valid values:

            - "final_product": Verify only the final product matches the target
            - "reactant": Verify a single reactant is valid
            - "all_reactants": Verify all reactants are chemically valid
            - "all_reactants_bb_ref": Verify all reactants are in the building blocks list
            - "smarts": Verify reaction follows the given SMARTS pattern
            - "full_path": Verify complete synthesis path to target molecule
            - "full_path_bb_ref": Verify synthesis path with building block constraints
            - "full_path_smarts_ref": Verify synthesis path matches SMARTS patterns
            - "full_path_smarts_bb_ref": Verify synthesis path with both SMARTS and building block constraints
            - "analog_gen": Verify analog generation task

        target: List of target molecules (SMILES strings) for verification.
            For synthesis tasks: The desired final product molecule
            For SMARTS tasks: The expected product after reaction
            Empty list if not applicable to the objective type.

        reactants: List of reactant lists for each reaction step.
            Each inner list contains SMILES strings for reactants in one reaction.
            For ground truth verification in SMARTS prediction tasks.
            Empty list if not applicable to the objective type.

        products: List of product molecules (SMILES strings) for each reaction step.
            For ground truth verification in multi-step synthesis.
            Empty list if not applicable to the objective type.

        building_blocks: List of valid building block SMILES strings.
            Optional constraint for synthesis tasks requiring specific building blocks.
            None if no building block constraints apply.

        smarts: Reference SMARTS strings for the reaction steps.
            Used to verify that reactions follow specific reaction templates.
            Empty list if not applicable to the objective type.

        or_smarts: Original reference SMARTS strings for the reaction steps.
            Alternative SMARTS patterns that can also be valid.
            Empty list if not applicable to the objective type.

        n_steps_max: Maximum number of reaction steps allowed in the synthesis route.
            Default is 5. Only applies to full_path objectives.

        idx_chosen: Index of the chosen reaction for multi-reaction tasks.
            Default is 0. Used for tracking in batch processing.
    """

    objectives: List[ReactionObjT] = Field(
        ...,
        description="The type of objective for the reaction verification.",
    )
    target: List[str] = Field(
        default_factory=list,
        description="The target molecule or SMARTS string for verification.",
    )
    reactants: List[List[str]] = Field(
        default_factory=list,
        description="List of reactants in a reaction.",
    )
    intermediate_products: List[str] = Field(
        default_factory=list,
        description="The intermediate product molecules of the reaction steps.",
    )
    products: List[str] = Field(
        default_factory=list,
        description="The product molecule of the reaction.",
    )
    building_blocks: List[str] | None = Field(
        None,
        description="List of valid building blocks for the reaction.",
    )
    smarts: List[str] = Field(
        default_factory=list,
        description="Reference SMARTS strings for the reaction steps.",
    )
    or_smarts: List[str] = Field(
        default_factory=list,
        description="Original Reference SMARTS strings for the reaction steps.",
    )
    n_steps_max: int = Field(
        default=5,
        gt=0,
        description="Maximum number of reaction steps allowed in the synthesis route.",
    )
    idx_chosen: int = Field(
        0,
        description="Index of the chosen reaction.",
    )
