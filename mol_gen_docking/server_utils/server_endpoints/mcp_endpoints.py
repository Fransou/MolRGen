"""MCP endpoint tools for the Molecular Verifier server.

This module defines tools that can be exposed through the MCP (Model Context Protocol)
interface for external clients to interact with the molecular verifier functionality.
"""

import json
from typing import Any, Dict, List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError

from mol_gen_docking.reward.verifiers import (
    GenerationVerifierInputMetadataModel,
    MolPropVerifierInputMetadataModel,
    ReactionVerifierInputMetadataModel,
)
from mol_gen_docking.server_utils.server_endpoints.get_reward import (
    get_reward_endpoint,
)
from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
from mol_gen_docking.server_utils.utils import (
    BatchMolecularVerifierServerResponse,
    MolecularVerifierServerQuery,
)
from mol_gen_docking.utils.property_utils import (
    CLASSICAL_PROPERTIES_NAMES,
    inverse_rescale_property_values,
)

TaskType = Literal["molecular_generation", "property_prediction", "molecular_reaction"]


class CalcPropInput(BaseModel):
    smiles: List[str] = Field(
        default_factory=list, description="Molecules to compute the properties of"
    )
    properties: List[str] = Field(
        default_factory=list,
        description="Properties to compute (for example: QED, logP, SA, etc.)",
    )


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
                "query": "Generate a molecule with SMILES: <answer> ... </answer>",
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
        except (ValidationError, TypeError, ValueError) as e:
            if isinstance(e, ValidationError):
                # Include detailed Pydantic validation errors
                errors.append(f"Query validation failed: {e.errors()}")
            else:
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
        "/get_available_docking_targets", operation_id="get_available_docking_targets"
    )  # type: ignore
    def get_docking_targets() -> Dict[str, str]:
        """
        Get a list of available docking targets (receptors) for molecular generation.
        Returns a dictionary of:  textual description of the target: id of the target.

        The id of the target can then be furhter used to as a property name in compute_reward and get_properties.
        """
        with open(server_settings.data_path + "/names_mapping.json") as f:
            pockets_info: Dict[str, str] = json.load(f)
        return pockets_info

    @app.post(
        "/get_reward_mcp",
        operation_id="compute_reward",
        response_model=BatchMolecularVerifierServerResponse,
        include_in_schema=False,
    )  # type: ignore
    async def get_reward_mcp(
        query: MolecularVerifierServerQuery,
    ) -> BatchMolecularVerifierServerResponse:
        """
        Compute a reward value for a completion associated to a given prompt/metadata.
        Handles molecular generation, property prediction and retrosynthesis planning.
        Examples:
            ```python
            get_reward_mcp(
                query=MolecularVerifierServerQuery(
                    query=[Molecular SMILES],
                    metadata=[
                        {
                            "properties": [<property to minimize>],
                            "objectives": ["minimize"],
                            "target": [0.0]
                        }
                    ]
                )
            ```

        Args:
            query (MolecularVerifierServerQuery): The query containing a textual generation,
             and the metadata used to compute the reward.

        Returns:
            BatchMolecularVerifierServerResponse:
                Reward scores and metadata for the query.
        """
        response = await get_reward_endpoint(
            query=query,
            app=app,
            server_settings=server_settings,
        )
        return response  # type: ignore

    @app.post(
        "/get_properties",
        operation_id="get_properties",
    )  # type: ignore
    async def get_properties(
        input: CalcPropInput,
    ) -> Dict[str, Dict[str, float]]:
        """Compute properties for one or more SMILES strings.

        Args:
            smiles: List of SMILES strings to evaluate.
            properties: List of property names to compute (for example
                ``["QED", "logP", "SA"]``).

        Returns:
            Dict[str, Dict[str, float]]:

        Example:
            ```python
            result = get_properties(
                smiles=["CCO", "c1ccccc1"],
                properties=["QED", "logP"],
            )
            ```
        """
        smiles = input.smiles
        properties = input.properties
        query = MolecularVerifierServerQuery(
            query=smiles,
            metadata=[
                {
                    "properties": properties,
                    "objectives": ["maximize"]
                    * len(properties),  # Trick to get the value of the properties
                    "target": [0.0] * len(properties),
                }
                for s in smiles
            ],
        )
        response = await get_reward_endpoint(
            query=query,
            server_settings=server_settings,
            app=app,
        )

        # Get the values in: response.metas[].generation_verifier_metadata.individual_rewards
        output = {}
        for smile, meta in zip(smiles, response.metas):  # type: ignore
            prop_values = meta.generation_verifier_metadata.individual_rewards
            output[smile] = {
                prop: inverse_rescale_property_values(
                    value=float(v), prop_name=prop, docking=prop in app.state.receptors
                )
                for prop, v in zip(properties, prop_values)
            }

        return output

    @app.delete("/mcp", include_in_schema=False)  # type: ignore
    async def delete_mcp() -> Dict[str, str]:
        """Handle DELETE requests to /mcp endpoint.
        This endpoint is typically called when a client disconnects or wants to clean up resources.
        """
        return {"status": "ok", "message": "MCP endpoint DELETE request handled"}
