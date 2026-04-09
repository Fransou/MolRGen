import asyncio
import logging
import time
from typing import Dict

import ray
from fastapi import FastAPI

from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
from mol_gen_docking.server_utils.utils import (
    BatchMolecularVerifierServerResponse,
    MolecularVerifierServerQuery,
    MolecularVerifierServerResponse,
)

logger = logging.getLogger(__name__)


async def prepare_receptor_endpoint(
    query: MolecularVerifierServerQuery,
    server_settings: MolecularVerifierServerSettings,
    app: FastAPI,
) -> Dict[str, str]:
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
        ["properties" in m and "objectives" in m and "target" in m for m in metadata]
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

    missed_receptors_1, missed_receptors_2 = await asyncio.to_thread(
        ray.get,
        app.state.receptor_processor.process_receptors.remote(
            receptors=targets, allow_bad_res=True
        ),
    )
    if len(missed_receptors_2) > 0:
        # Return error if some receptors could not be processed
        return {
            "status": "Error",
            "info": f"Receptors {missed_receptors_2} could not be processed.",
        }
    else:
        return {"status": "Success"}


async def get_reward_endpoint(
    app: FastAPI,
    server_settings: MolecularVerifierServerSettings,
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
    prepare_res = await prepare_receptor_endpoint(
        query=query, server_settings=server_settings, app=app
    )
    status = prepare_res.get("status", "")
    if status == "Error":
        if server_settings.server_mode == "singleton":
            return MolecularVerifierServerResponse(
                reward=0.0, error="Error in preprocessing"
            )
        elif server_settings.server_mode == "batch":
            num_items = len(query.metadata)
            return BatchMolecularVerifierServerResponse(
                rewards=[0.0] * num_items, error="Error in preprocessing"
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
                and len(result.meta.generation_verifier_metadata.all_smi_rewards) > 0
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
