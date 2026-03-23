"""MCP endpoint tools for the Molecular Verifier server.

This module defines tools that can be exposed through the MCP (Model Context Protocol)
interface for external clients to interact with the molecular verifier functionality.
"""

import json
import logging
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
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


class ReinventTrainingParams(BaseModel):
    """Parameters for REINVENT model training."""

    model_name: str = Field(
        default="Franso/reinvent_42M", description="Name of the model to fine-tune"
    )
    output_dir: str = Field(
        default="./results", description="Output directory for model checkpoints"
    )
    num_train_epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=128, description="Batch size for training")
    sigma: float = Field(default=0.1, description="Sigma parameter for REINVENT")
    n_output_molecules: int = Field(
        default=10,
        description="Number of molecules to generate and print after training",
    )
    metadata: GenerationVerifierInputMetadataModel = Field(
        ..., description="Metadata defining the reward function and objectives"
    )


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


class ReinventTrainingService:
    """Service for managing REINVENT training jobs."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.server_settings: MolecularVerifierServerSettings | None = None

    def set_server_settings(
        self, server_settings: MolecularVerifierServerSettings
    ) -> None:
        """Set server settings for the training service."""
        self.server_settings = server_settings

    def start_training(self, params: ReinventTrainingParams) -> str:
        """Start a new REINVENT training job."""
        job_id = str(uuid.uuid4())

        # Create a temporary file for custom metadata
        metadata_file = Path(f"/tmp/reinvent_metadata_{job_id}.json")
        with open(metadata_file, "w") as f:
            json.dump(params.metadata.model_dump(), f)

        # Build command line arguments with sensible defaults for removed parameters
        cmd = [
            "python",
            "-m",
            "mol_gen_docking.baselines.reinvent.rl_opt_rpo",
            "--custom_metadata_file",
            str(metadata_file),
            "--model_name",
            params.model_name,
            "--output_dir",
            params.output_dir,
            "--num_train_epochs",
            str(params.num_train_epochs),
            "--batch_size",
            str(params.batch_size),
            "--sigma",
            str(params.sigma),
            "--remote_rm_url",
            "http://localhost:8000",  # Default value - use localhost to avoid network issues
            "--n_output_molecules",
            str(params.n_output_molecules),
        ]
        print(" ".join(cmd))
        # Start the training process
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                bufsize=1,  # Line buffered
                cwd="/home/philippe/MolGenDocking",
            )

            # Store job information
            self.jobs[job_id] = {
                "status": "running",
                "process": process,
                "params": params.model_dump(),
                "start_time": datetime.now().isoformat(),
                "metadata_file": str(metadata_file),
                "logs": [],
            }

            return job_id

        except Exception as e:
            # Clean up metadata file if process creation failed
            if metadata_file.exists():
                metadata_file.unlink()
            raise HTTPException(
                status_code=500, detail=f"Failed to start REINVENT training: {str(e)}"
            )

    def _capture_logs(self, job_id: str) -> None:
        """Capture available logs from the subprocess and add them to the job's log buffer."""
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]
        process = job["process"]

        # Read available output without blocking
        if process.stdout:
            while True:
                line = process.stdout.readline()
                if line:
                    stripped_line = line.rstrip()
                    job["logs"].append(stripped_line)
                    # Also log to console so it appears in uvicorn logs
                    logging.info(f"[REINVENT {job_id}] {stripped_line}")
                else:
                    break

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a training job."""
        if job_id not in self.jobs:
            return None

        job = self.jobs[job_id]
        process = job["process"]

        # Capture any available logs
        self._capture_logs(job_id)

        # Check if process is still running
        if process.poll() is not None:
            # Process has finished, get remaining output
            stdout, _ = process.communicate()
            if stdout:
                remaining_lines = stdout.splitlines()
                job["logs"].extend(remaining_lines)
                # Log remaining output to console
                for line in remaining_lines:
                    logging.info(f"[REINVENT {job_id}] {line}")

            if process.returncode == 0:
                job["status"] = "completed"
                job["end_time"] = datetime.now().isoformat()
                logging.info(f"[REINVENT {job_id}] Training completed successfully")
            else:
                job["status"] = "failed"
                job["end_time"] = datetime.now().isoformat()
                error_msg = "Process failed with return code: " + str(
                    process.returncode
                )
                job["error"] = error_msg
                logging.error(f"[REINVENT {job_id}] {error_msg}")

            # Clean up metadata file
            metadata_file = Path(job["metadata_file"])
            if metadata_file.exists():
                metadata_file.unlink()

        return job.copy()

    def cleanup_job(self, job_id: str) -> bool:
        """Clean up a completed or failed job."""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        process = job["process"]

        # Terminate process if still running
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        # Clean up metadata file
        metadata_file = Path(job["metadata_file"])
        if metadata_file.exists():
            metadata_file.unlink()

        # Remove job from tracking
        del self.jobs[job_id]
        return True


def register_mcp_tools(
    app: FastAPI, server_settings: MolecularVerifierServerSettings
) -> None:
    # Initialize REINVENT training service
    if not hasattr(app.state, "reinvent_training_service"):
        app.state.reinvent_training_service = ReinventTrainingService()
        app.state.reinvent_training_service.set_server_settings(server_settings)

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

    @app.post(
        "/train_reinvent_generator",
        operation_id="train_reinvent_generator",
        response_model=Dict[str, Any],
    )  # type: ignore
    async def train_reinvent_generator(
        params: ReinventTrainingParams,
    ) -> Dict[str, Any]:
        """
        Train a REINVENT model with custom metadata for reward definition.

        This endpoint launches a REINVENT training job with the specified parameters
        and custom metadata that defines the reward function objectives.

        Args:
            params: Training parameters including model configuration and metadata

        Returns:
            Dict[str, Any]: Training job information including:
                - job_id: Unique identifier for the training job
                - status: Current status of the job (queued, running, completed, failed)
                - message: Status message
                - timestamp: ISO 8601 timestamp

        Example:
            ```python
            from mol_gen_docking.reward.verifiers.generation_reward.input_metadata import GenerationVerifierInputMetadataModel

            metadata = GenerationVerifierInputMetadataModel(
                properties=["QED", "SA"],
                objectives=["maximize", "minimize"],
                target=[0.8, 0.5]
            )

            params = ReinventTrainingParams(
                model_name="Franso/reinvent_42M",
                metadata=metadata,
                num_train_epochs=50,
                batch_size=64,
                n_output_molecules=15  # Generate 15 molecules after training
            )
            result = await train_reinvent_generator(params)
            ```
        """
        try:
            # Start the training job
            job_id = app.state.reinvent_training_service.start_training(params)

            return {
                "job_id": job_id,
                "status": "running",
                "message": "REINVENT training job started successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to start REINVENT training: {str(e)}"
            )

    @app.get(
        "/train_reinvent_status/{job_id}",
        operation_id="get_train_reinvent_status",
        response_model=Optional[Dict[str, Any]],
    )  # type: ignore
    async def get_train_reinvent_status(job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a REINVENT training job.

        Args:
            job_id: The job identifier returned by train_reinvent_generator

        Returns:
            Optional[Dict[str, Any]]: Job status information including:
                - status: Current status (running, completed, failed)
                - start_time: ISO 8601 timestamp when job started
                - end_time: ISO 8601 timestamp when job ended (if completed/failed)
                - params: Training parameters used
                - error: Error message if job failed
        """
        job_status = app.state.reinvent_training_service.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=404, detail=f"Training job {job_id} not found"
            )
        return job_status  # type: ignore

    @app.delete("/mcp", include_in_schema=False)  # type: ignore
    async def delete_mcp() -> Dict[str, str]:
        """Handle DELETE requests to /mcp endpoint.
        This endpoint is typically called when a client disconnects or wants to clean up resources.
        """
        return {"status": "ok", "message": "MCP endpoint DELETE request handled"}
