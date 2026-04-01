"""MCP endpoint tools for the Molecular Verifier server.

This module defines tools that can be exposed through the MCP (Model Context Protocol)
interface for external clients to interact with the molecular verifier functionality.
"""

import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

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
    inverse_rescale_property_values,
)

try:
    from openbabel import pybel
except ImportError:
    pybel = None

TaskType = Literal["molecular_generation", "property_prediction", "molecular_reaction"]


class ReinventTrainingParams(BaseModel):
    """Parameters for REINVENT model training."""

    output_dir: str = Field(
        default="./results", description="Output directory for model checkpoints"
    )
    num_train_epochs: int = Field(default=20, description="Number of training epochs")
    eval_batch_size: int = Field(default=64, description="Batch size for evaluation")
    batch_size: int = Field(default=256, description="Batch size for training")
    sigma: float = Field(default=0.4, description="Sigma parameter for REINVENT")
    learning_rate: float = Field(default=1e-5, description="Learning rate for REINVENT")
    smiles_start: List[str] = Field(
        default_factory=list,
        description="Beginning of sequence to start with",
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
        self.server_settings: MolecularVerifierServerSettings | None = None
        self.job_status: Dict[str, Dict[str, Any]] = {}

    def set_server_settings(
        self, server_settings: MolecularVerifierServerSettings
    ) -> None:
        """Set server settings for the training service."""
        self.server_settings = server_settings

    async def train(
        self, params: ReinventTrainingParams, job_id: str
    ) -> Dict[str, Any]:
        """Start REINVENT training as a background task."""
        # Create a temporary file for custom metadata
        metadata_file = Path(f"/tmp/reinvent_metadata_{job_id}.json")
        with open(metadata_file, "w") as f:
            json.dump(params.metadata.model_dump(), f)

        if params.smiles_start == []:
            params.smiles_start = ["<s>"]

        # Build command line arguments
        cmd = [
            "python",
            "-m",
            "mol_gen_docking.baselines.reinvent.rl_opt_rpo",
            "--custom_metadata_file",
            str(metadata_file),
            "--output_dir",
            params.output_dir,
            "--num_train_epochs",
            str(params.num_train_epochs),
            "--batch_size",
            str(params.batch_size),
            "--eval_batch_size",
            str(params.eval_batch_size),
            "--sigma",
            str(params.sigma),
            "--learning_rate",
            str(params.learning_rate),
            "--smiles_start",
            *params.smiles_start,
            "--remote_rm_url",
            "http://localhost:8000",  # Default value - use localhost to avoid network issues
        ]
        print(" ".join(cmd))

        # Update job status
        if job_id:
            self.job_status[job_id] = {
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "command": " ".join(cmd),
            }
        # Run the training process asynchronously
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output_lines = []
            if process.stdout:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode().strip()
                    print(line_str)
                    output_lines.append(line_str)
                    if job_id:
                        self.job_status[job_id]["output"] = "".join(output_lines)
            await process.wait()
            returncode = process.returncode
            assert returncode is not None, "returncode must not be None"
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd, output=b"")
            result_stdout = "".join(output_lines)
            # Clean up metadata file
            if metadata_file.exists():
                metadata_file.unlink()
            # Update job status
            if job_id:
                self.job_status[job_id] = {
                    "status": "completed",
                    "message": "REINVENT training completed successfully",
                    "output": result_stdout,
                    "end_time": datetime.now().isoformat(),
                }
            return {
                "status": "completed",
                "message": "REINVENT training completed successfully",
                "output": result_stdout,
                "timestamp": datetime.now().isoformat(),
                "job_id": job_id,
            }
        except subprocess.CalledProcessError as e:
            # Clean up metadata file
            if metadata_file.exists():
                metadata_file.unlink()
            # Update job status
            if job_id:
                self.job_status[job_id] = {
                    "status": "failed",
                    "message": f"REINVENT training failed for job {job_id}",
                    "output": e.output.decode() if e.output else str(e),
                    "end_time": datetime.now().isoformat(),
                }
            print(
                f"REINVENT training failed for job {job_id}: {e.output.decode() if e.output else str(e)}"
            )
            return {
                "status": "failed",
                "message": f"REINVENT training failed for job {job_id}",
                "output": e.output.decode() if e.output else str(e),
                "timestamp": datetime.now().isoformat(),
                "job_id": job_id,
            }
        except Exception as e:
            # Clean up metadata file
            if metadata_file.exists():
                metadata_file.unlink()
            print(f"Failed to start REINVENT training for job {job_id}: {str(e)}")
            return {
                "status": "failed",
                "message": f"Failed to start REINVENT training for job {job_id}",
                "output": str(e),
                "timestamp": datetime.now().isoformat(),
                "job_id": job_id,
            }


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

        This endpoint starts the training asynchronously and returns immediately with a job_id.
        Use the /get_training_status endpoint to check the job status.

        Args:
            params: Training parameters including model configuration and metadata

        Returns:
            Dict[str, Any]: Response including:
                - status: 'started'
                - job_id: Unique identifier for the training job
                - message: Status message
                - timestamp: Start timestamp

        Example:
            ```python
            from mol_gen_docking.reward.verifiers.generation_reward.input_metadata import GenerationVerifierInputMetadataModel

            metadata = GenerationVerifierInputMetadataModel(
                properties=["QED", "SA"],
                objectives=["maximize", "minimize"],
                target=[0.8, 0.5]
            )

            params = ReinventTrainingParams(
                metadata=metadata,
                num_train_epochs=50,
                batch_size=64
            )
            result = await train_reinvent_generator(params)
            job_id = result["job_id"]

            # Check status later
            status = await get_training_status(job_id)
            ```
        """
        job_id = str(uuid.uuid4())
        # Start training asynchronously without waiting for completion
        asyncio.create_task(app.state.reinvent_training_service.train(params, job_id))

        # Initialize job status
        app.state.reinvent_training_service.job_status[job_id] = {
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "job_id": job_id,
            "message": "REINVENT training started",
        }

        return {
            "status": "started",
            "job_id": job_id,
            "message": "REINVENT training started",
            "timestamp": datetime.now().isoformat(),
        }

    @app.post(
        "/get_training_status/{job_id}",
        operation_id="get_training_status",
        response_model=Dict[str, Any],
    )  # type: ignore
    async def get_training_status(
        job_id: str,
    ) -> Dict[str, Any]:
        """
        Get the status of a REINVENT training job.

        Args:
            job_id: The job identifier returned by train_reinvent_generator

        Returns:
            Dict[str, Any]: Job status information including:
                - status: 'started', 'running', 'completed', or 'failed'
                - job_id: The job identifier
                - message: Status message
                - start_time: When the job was started
                - end_time: When the job completed (if finished)
                - output: Training output (if available)
                - command: Command being executed (if running)

        Example:
            ```python
            status = await get_training_status("abc123-def456")
            print(status["status"])  # "running", "completed", "failed", etc.
            ```
        """
        await asyncio.sleep(2)  # to avoid overloading the server
        job_status: None | Dict[str, Any] = (
            app.state.reinvent_training_service.job_status.get(job_id)
        )

        if job_status is None:
            return {
                "status": "not_found",
                "job_id": job_id,
                "message": f"Job {job_id} not found",
                "timestamp": datetime.now().isoformat(),
            }

        return job_status

    @app.post("/display_molecule/{smiles}", operation_id="display_molecule")  # type: ignore
    async def display_molecule(smiles: str) -> str:
        """
        Uses Open Babel to generate a 2D ASCII art depiction of a molecule
        from a SMILES string.
        Args:
            smiles: SMILES string to generate a 2D ASCII art

        Returns:
            A string containing the ASCII art representation of the molecule.
        """
        try:
            mol = pybel.readstring("smi", smiles)
            mol.make2D()
            ascii_drawing = mol.write("ascii")
            return f"Visual structure for SMILES: {smiles}\n\n```text\n{ascii_drawing}\n```"

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process SMILES '{smiles}'. Error: {str(e)}",
            )

    @app.delete("/mcp", include_in_schema=False)  # type: ignore
    async def delete_mcp() -> Dict[str, str]:
        """Handle DELETE requests to /mcp endpoint.
        This endpoint is typically called when a client disconnects or wants to clean up resources.
        """
        return {"status": "ok", "message": "MCP endpoint DELETE request handled"}
