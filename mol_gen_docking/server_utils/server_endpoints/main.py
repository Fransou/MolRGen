from typing import Dict

from fastapi import FastAPI

from mol_gen_docking.server_utils.server_endpoints.get_reward import (
    get_reward_endpoint,
    prepare_receptor_endpoint,
)
from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
from mol_gen_docking.server_utils.utils import (
    BatchMolecularVerifierServerResponse,
    MolecularVerifierServerQuery,
    MolecularVerifierServerResponse,
)


def register_endpoints(
    app: FastAPI, server_settings: MolecularVerifierServerSettings
) -> None:
    """
    Register all API endpoints on the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
        server_settings (Any): The server settings instance containing configuration.
    """

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
        output = await get_reward_endpoint(
            query=query, app=app, server_settings=server_settings
        )
        return output

    @app.post(
        "/prepare_receptor",
        include_in_schema=False,
        operation_id="prepare_receptor",
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
        return await prepare_receptor_endpoint(
            query=query, app=app, server_settings=server_settings
        )
