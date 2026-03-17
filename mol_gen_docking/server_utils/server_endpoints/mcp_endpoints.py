"""MCP endpoint tools for the Molecular Verifier server.

This module defines tools that can be exposed through the MCP (Model Context Protocol)
interface for external clients to interact with the molecular verifier functionality.
"""

from typing import Any, Dict, List, Literal

from fastapi import FastAPI
from pydantic import ValidationError

from mol_gen_docking.reward.verifiers import (
    GenerationVerifierInputMetadataModel,
    MolPropVerifierInputMetadataModel,
    ReactionVerifierInputMetadataModel,
)
from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
from mol_gen_docking.server_utils.utils import (
    MolecularVerifierServerQuery,
    MolecularVerifierServerResponse,
)
from mol_gen_docking.utils.property_utils import CLASSICAL_PROPERTIES_NAMES

TaskType = Literal["molecular_generation", "property_prediction", "molecular_reaction"]


def _validate_metadata(
    query: MolecularVerifierServerQuery, model_class: type, task: str
) -> List[str]:
    """
    Validate metadata against a specific Pydantic model for a given task.

    Attempts to cast each metadata dictionary to the provided model class.

    Args:
        query: The MolecularVerifierServerQuery to validate
        model_class: The Pydantic model class to validate metadata against
        task: The task type name (for error messages)

    Returns:
        List of error messages. Empty list if all metadata is valid.
    """
    errors = []
    for idx, metadata in enumerate(query.metadata):
        try:
            model_class(**metadata)
        except ValidationError as e:
            errors.append(f"Metadata[{idx}] invalid for {task}: {str(e)}")
        except Exception as e:
            errors.append(f"Metadata[{idx}] error during {task} validation: {str(e)}")
    return errors


def register_mcp_tools(
    app: FastAPI, server_settings: MolecularVerifierServerSettings
) -> None:
    @app.post("/validate_query/{task}", operation_id="validate_query")  # type: ignore
    def validate_query(
        query: MolecularVerifierServerQuery | Dict[str, Any],
        task: TaskType,
    ) -> Dict[str, Any]:
        """
        Validate a MolecularVerifierServerQuery for a specific task type.

        This tool validates that a query is properly formatted and contains the
        necessary information for the specified task type.

        Args:
            query: A MolecularVerifierServerQuery object or dictionary representation
                of the query to validate.
            task: The type of task to validate for. Options are:
                - "molecular_generation": Validates for molecule generation tasks
                - "property_prediction": Validates for property prediction tasks
                - "molecular_reaction": Validates for reaction planning tasks

        Returns:
            Dict[str, Any]: A validation result dictionary containing:
                - "is_valid": bool indicating whether the query is valid
                - "task": The task type that was validated
                - "errors": List of error messages if validation failed (empty if valid)
                - "validated_query": The validated query object if valid, None otherwise

        Raises:
            ValueError: If the task type is not recognized
            TypeError: If the query cannot be converted to MolecularVerifierServerQuery

        Example:
            ```python
            query = {
                "query": "Generate a molecule with SMILES: <answer>CC(C)Cc1ccc(cc1)C(C)C(=O)O</answer>",
                "metadata": [
                    {
                        "properties": ["logP"],
                        "objectives": ["minimize"],
                        "target": [0.0]
                    }
                ]
            }
            result = validate_query(query, "molecular_generation")
            ```
        """
        errors = []
        validated_query = None

        # Validate task type
        valid_tasks = [
            "molecular_generation",
            "property_prediction",
            "molecular_reaction",
        ]
        if task not in valid_tasks:
            raise ValueError(
                f"Invalid task type '{task}'. Must be one of: {', '.join(valid_tasks)}"
            )

        try:
            # Convert dict to MolecularVerifierServerQuery if necessary
            if isinstance(query, dict):
                query_obj = MolecularVerifierServerQuery(**query)
            elif isinstance(query, MolecularVerifierServerQuery):
                query_obj = query
            else:
                raise TypeError(
                    f"Query must be a dict or MolecularVerifierServerQuery, got {type(query).__name__}"
                )
        except ValueError as e:
            errors.append(f"Query validation failed: {str(e)}")
            return {
                "is_valid": False,
                "task": task,
                "errors": errors,
                "validated_query": None,
            }

        # Validate task-specific requirements
        task_model_map = {
            "molecular_generation": GenerationVerifierInputMetadataModel,
            "property_prediction": MolPropVerifierInputMetadataModel,
            "molecular_reaction": ReactionVerifierInputMetadataModel,
        }
        model_class = task_model_map[task]
        errors.extend(_validate_metadata(query_obj, model_class, task))

        is_valid = len(errors) == 0
        if is_valid:
            validated_query = query_obj

        return {
            "is_valid": is_valid,
            "task": task,
            "errors": errors,
            "validated_query": validated_query.model_dump()
            if validated_query
            else None,
        }

    @app.get(
        "/get_available_rdkit_properties", operation_id="get_available_rdkit_properties"
    )  # type: ignore
    def get_available_properties() -> List[str]:
        """
        Get a list of all available RDKit properties for molecular generation.

        Returns the RDKit property function names (values from CLASSICAL_PROPERTIES_NAMES)
        that can be evaluated by the molecular verifier.

        Returns:
            List[str]: A sorted list of available RDKit property names.
                Examples include: "SA", "QED", "CalcExactMolWt", "CalcNumRotatableBonds", etc.

        Example:
            ```python
            properties = get_available_properties()
            print(properties)
            # Output: ['CalcExactMolWt', 'CalcFractionCSP3', 'CalcHallKierAlpha',
            #          'CalcNumAromaticRings', 'CalcNumHBA', 'CalcNumHBD',
            #          'CalcNumRotatableBonds', 'CalcPhi', 'CalcTPSA', 'QED', 'SA', 'logP']
            ```
        """
        return sorted(CLASSICAL_PROPERTIES_NAMES.values())

    @app.get(
        "/get_reward_mcp",
        operation_id="compute_reward",
        response_model=MolecularVerifierServerResponse,
    )  # type: ignore
    def get_reward_mcp(
        query: MolecularVerifierServerQuery,
    ) -> MolecularVerifierServerResponse:
        """
        Compute a reward value for a completion associated to a given prompt/metadata.
        Handles molecular generation, property predicion and retrosynthesis planning.

        Args:
            query (MolecularVerifierServerQuery): The query containing a textual generation,
             and the metadata used to compute the reward.

        Returns:
            MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse:
                Reward scores and metadata for the query. Type depends on server_mode.
                Returns singleton response in singleton mode or batch response in batch mode.
        """
        if not server_settings.server_mode == "singleton":
            raise ValueError(
                "The 'get_reward_mcp' endpoint is only available in singleton mode"
            )
        response = app.post(
            "/get_reward",
            json=query.model_dump(),
        )
        return response  # type: ignore
