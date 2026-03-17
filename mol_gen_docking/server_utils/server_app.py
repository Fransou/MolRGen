import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import ray
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

from mol_gen_docking.data.meeko_process import ReceptorProcess
from mol_gen_docking.reward import (
    MolecularVerifier,
)
from mol_gen_docking.server_utils.buffer import RewardBuffer
from mol_gen_docking.server_utils.server_endpoints import (
    register_endpoints,
    register_mcp_tools,
)
from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings

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

    # Register all endpoints
    register_endpoints(app, server_settings)

    return app


def create_mcp(app: FastAPI) -> FastApiMCP:
    """
    Create and configure the FastAPI MCP (Model Context Protocol) server.

    Creates the FastAPI app and wraps it with FastApiMCP for MCP integration.

    Args:
        app (FastAPI): FastAPI app instance.

    Returns:
        FastApiMCP: A configured MCP server instance.
    """
    register_mcp_tools(app, server_settings)
    mcp = FastApiMCP(
        app,
        name="Molecular Verifier Server",
        description="""
    An MCP server that can be used to perform molecular generation / property prediction and retrosythesis planning tasks.

    This server can notably be used to evaluate the quality of an answer (generated molecule, predicted property value, ...) based on the query.
    """,
        describe_all_responses=True,
        describe_full_response_schema=True,
    )
    mcp.mount_http()
    return mcp
