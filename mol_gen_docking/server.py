import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import ray
from fastapi import FastAPI

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
# Only initialize Ray if not already initialized (e.g., by server_launcher)
if server_settings.ray_namespace is not None and not ray.is_initialized():
    ray.init(address="auto", namespace=server_settings.ray_namespace)
elif server_settings.ray_namespace is not None and ray.is_initialized():
    logger.info("Ray already initialized, skipping Ray initialization")

RemoteRewardScorer: Any = ray.remote(MolecularVerifier)

server_settings_log = "Server settings:\n"
for field_name, field_value in server_settings.model_dump().items():
    server_settings_log += f"  {field_name}: {field_value}\n"
logger.info(server_settings_log)

_reward_model = None
_valid_reward_model = None


def get_or_create_reward_actor() -> Any:
    global _reward_model
    global server_settings
    if _reward_model is None or _reward_model.__ray_terminated__:
        _reward_model = RemoteRewardScorer.remote(
            server_settings.to_molecular_verifier_config()
        )
    return _reward_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
    app = FastAPI(lifespan=lifespan)

    @app.get("/liveness")  # type: ignore
    async def live_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/get_reward")  # type: ignore
    async def get_reward(
        query: MolecularVerifierServerQuery,
    ) -> MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse:
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

    @app.post("/prepare_receptor")  # type: ignore
    async def prepare_receptor(query: MolecularVerifierServerQuery) -> Dict[str, str]:
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
