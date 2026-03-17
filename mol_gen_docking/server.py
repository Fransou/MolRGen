import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import ray
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward import (
    MolecularVerifier,
)
from mol_gen_docking.server_utils.buffer import RewardBuffer
from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
from mol_gen_docking.server_utils.utils import (
    BatchMolecularVerifierServerResponse,
    MolecularVerifierServerQuery,
    MolecularVerifierServerResponse,
)

# Set logging level to info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molecular_verifier_server")
logger.setLevel(logging.INFO)

server_settings = MolecularVerifierServerSettings()
server_settings.ray_init()

RemoteRewardScorer: Any = ray.remote(MolecularVerifier)

server_settings_log = "Server settings:\n"
for field_name, field_value in server_settings.model_dump().items():
    server_settings_log += f"  {field_name}: {field_value}\n"
logger.info(server_settings_log)

_reward_model = None
_valid_reward_model = None


def get_or_create_reward_actor() -> Any:
    """
    Get or create a Ray remote actor for molecular reward scoring.

    Creates a new remote MolecularVerifier actor if one doesn't exist or if the
    existing actor has been terminated. Uses Ray remote execution to enable
    distributed computation of reward scores.

    Returns:
        Any: A Ray remote actor reference for the MolecularVerifier.
    """
    global _reward_model
    global server_settings
    if _reward_model is None or _reward_model.__ray_terminated__:
        _reward_model = RemoteRewardScorer.remote(
            server_settings.to_molecular_verifier_config()
        )
    return _reward_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage the application lifecycle - startup and shutdown.

    Initializes server resources on startup:
    - Reinitializes server settings
    - Creates reward buffer for batching queries
    - Creates or retrieves the reward model actor
    - Initializes receptor processor if using autodock_gpu
    - Loads receptor/pocket information from data files

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control to the application during runtime.
    """
    global server_settings
    server_settings = MolecularVerifierServerSettings()
    logger.info(
        f"Initialized molecular docking verifier lifespan with {server_settings}"
    )
    logger.info("Initializing socket")
    app.state.reward_buffer = RewardBuffer(
        app,
        buffer_time=server_settings.buffer_time,
        max_batch_size=2**16,
        server_mode=server_settings.server_mode,
    )

    app.state.reward_model = get_or_create_reward_actor()
    app.state.receptor_processor = (
        ReceptorProcess(data_path=server_settings.data_path)
        if server_settings.docking_oracle == "autodock_gpu"
        else None
    )

    with open(server_settings.data_path + "/pockets_info.json") as f:
        pockets_info = json.load(f)
    app.state.receptors = list(pockets_info.keys())

    yield


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Initializes the FastAPI application with:
    - Application lifecycle management (startup/shutdown)
    - Static file mounting for documentation and assets
    - Multiple API endpoints for molecular verification
    - Favicon serving

    Returns:
        FastAPI: A configured FastAPI application instance.
    """
    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="docs/docs/assets"), name="static")

    @app.get("/favicon.ico", include_in_schema=False)  # type: ignore
    async def favicon() -> FileResponse:
        """
        Serve the application favicon.

        Returns:
            FileResponse: The favicon image file (logo.png).
        """
        return FileResponse("docs/docs/assets/logo.png")

    @app.get("/liveness", include_in_schema=False)  # type: ignore
    async def live_check() -> dict[str, str]:
        """
        Health check endpoint for server liveness.

        Returns:
            dict[str, str]: A dictionary containing the status "ok" if the server is alive.
        """
        return {"status": "ok"}

    @app.post("/get_reward", include_in_schema=False)  # type: ignore
    async def get_reward(
        query: MolecularVerifierServerQuery,
    ) -> MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse:
        """
        Compute a reward value for a completion associated to a given prompt/metadata.
        Handles molecular generation, property predicion and retrosynthesis planning.

        Args:
            query (MolecularVerifierServerQuery): The query containing SMILES strings,
                docking targets, and metadata.

        Returns:
            MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse:
                Reward scores and metadata for the query. Type depends on server_mode.
                Returns singleton response in singleton mode or batch response in batch mode.
        """
        if server_settings.server_mode == "singleton":
            # Ensures the query does not contain multiple items
            if len(query.metadata) != 1:
                return MolecularVerifierServerResponse(
                    reward=0.0, error="Singleton mode only supports single query items."
                )

        t0 = time.time()
        prepare_res = await prepare_receptor(query)
        status = prepare_res.get("status", "")
        if status == "Error":
            if server_settings.server_mode == "singleton":
                return MolecularVerifierServerResponse(
                    reward=0.0, error="Error in preprocessing"
                )
            elif server_settings.server_mode == "batch":
                return BatchMolecularVerifierServerResponse(
                    rewards=[0.0], error="Error in preprocessing"
                )

        result: (
            MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse
        ) = await app.state.reward_buffer.add_query(query)
        t1 = time.time()
        logger.info(f"Processed batch in {t1 - t0:.2f} seconds")
        if server_settings.debug_logging:
            logger.info(f"=== Query: {query}\n=== Result: {result}")

        # Possibly add next turn feedback
        if server_settings.server_mode == "singleton":
            assert isinstance(result, MolecularVerifierServerResponse), (
                "Expected singleton response"
            )
            if (
                result.meta.generation_verifier_metadata is not None
            ):  # We only consider multi-turn for generation verifier
                if (
                    len(result.meta.generation_verifier_metadata.all_smi) > 0
                    and len(result.meta.generation_verifier_metadata.all_smi_rewards)
                    > 0
                ):
                    result.next_turn_feedback = (
                        "The score of the provided molecules are:\n"
                        + "\n".join(
                            [
                                f"{smi}: {score:.3f}"
                                for smi, score in zip(
                                    result.meta.generation_verifier_metadata.all_smi,
                                    result.meta.generation_verifier_metadata.all_smi_rewards,
                                )
                            ]
                        )
                    )
        return result

    @app.post(
        "/prepare_receptor",
        include_in_schema=False,
        operation_id="""
        prepare_receptor
        """,
    )  # type: ignore
    async def prepare_receptor(query: MolecularVerifierServerQuery) -> Dict[str, str]:
        """
        Prepare docking targets/receptors for molecular docking.

        Extracts target receptors from query metadata and processes them if necessary.
        This function handles receptor preparation steps required for autodock_gpu
        operations. If no receptor processor is configured, returns success immediately.

        Args:
            query (MolecularVerifierServerQuery): The query containing metadata with
                target receptor information.

        Returns:
            Dict[str, str]: A dictionary with:
                - "status": "Success" or "Error" indicating operation status
                - "info": Optional error details if status is "Error"
        """
        metadata = query.metadata

        if app.state.receptor_processor is None:
            # No need to prepare receptors
            return {
                "status": "No need to prepare receptors for the selected docking oracle."
            }
        assert metadata is not None
        assert all(
            [
                "properties" in m and "objectives" in m and "target" in m
                for m in metadata
            ]
        )

        targets = [
            m["properties"][i]
            for m in metadata
            for i in range(len(m["properties"]))
            if m["properties"][i] in app.state.receptors
        ]
        targets = list(set(targets))
        if targets == []:
            return {"status": "Success"}

        missed_receptors_1, missed_receptors_2 = (
            app.state.receptor_processor.process_receptors(
                receptors=targets, allow_bad_res=True
            )
        )
        if len(missed_receptors_2) > 0:
            # Return error if some receptors could not be processed
            return {
                "status": "Error",
                "info": f"Receptors {missed_receptors_2} could not be processed.",
            }
        else:
            return {"status": "Success"}

    return app


app = create_app()
mcp = FastApiMCP(
    app,
    name="Molecular Verifier Server",
    description="""
    An MCP server that can be used to perform molecular generation / property prediction and retrosythesis planning tasks.

    This server can notably be used to evaluate the quality of an answer (generated molecule, predicted property value, ...) based on the query.
    """,
    describe_all_responses=True,
    describe_full_response_schema=True,
    include_operations=["get_reward"],
)
mcp.mount_http()
mcp.setup_server()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
