from pathlib import Path
from typing import Literal, Optional

import ray
from pydantic_settings import BaseSettings

from mol_gen_docking.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifierConfigModel,
    MolPropVerifierConfigModel,
    ReactionVerifierConfigModel,
)


class MolecularVerifierServerSettings(BaseSettings):
    """Configuration settings for the Molecular Verifier server.

    This class manages all configuration parameters for running the Molecular Verifier
    server, including docking settings, oracle configuration, and verifier behavior.
    Settings can be loaded from environment variables, .env files, or passed directly.

    The settings are used to initialize verifier configurations for molecular property
    prediction, reaction verification, and molecular generation tasks. It provides
    validation of settings and conversion to verifier config models.

    Attributes:
        server_mode (Literal["singleton", "batch"]): Server operation mode.

            - "singleton": Handles one request at a time.
            - "batch": Accepts multiple concurrent requests for batch processing.
            Default: "singleton"
        scorer_exhaustiveness (int): Exhaustiveness parameter for docking scoring.
            Controls the thoroughness of the docking search. Higher values increase
            accuracy but require more computation time.
            Default: 8

        scorer_ncpus (int): Number of CPU cores to allocate for scoring operations.
            Must be compatible with exhaustiveness.
            Default: 8

        docking_concurrency_per_gpu (int): Number of concurrent docking jobs per GPU.
            Controls GPU utilization.
            Default: 2

        reaction_matrix_path (str): Path to the pickled reaction matrix file.
            Must exist and be accessible for reaction verification tasks.
            Default: "data/rxn_matrix.pkl"

        docking_oracle (Literal["pyscreener", "autodock_gpu"]): The docking software
            to use for molecular docking. "pyscreener" for PyScreener docking or
            "autodock_gpu" for GPU-accelerated AutoDock.
            Default: "autodock_gpu"

        vina_mode (str): Command mode for AutoDock GPU execution.
            Example: "autodock_gpu_256wi" for 256 work items mode.
            Only used when docking_oracle is "autodock_gpu".
            Default: "autodock_gpu_256wi"

        data_path (str): Path to the molecular data directory containing:
            - names_mapping.json: Property name mappings
            - docking_targets.json: Target definitions for docking
            Used by generation tasks, and must be accessible.
            Default: "data/molgendata"

        buffer_time (int): Time in seconds to buffer requests before processing.
            Allows batch processing of concurrent requests to improve efficiency.
            Default: 20

        parsing_method: Method to parse model completions:

            - "none": No parsing, use full completion (not recommended, risks of ambiguity in the answer extraction)
            - "answer_tags": Extract content within special tags
            - "boxed": Extract content within answer tags and boxed in the '\\boxed{...}' LaTeX command

        debug_logging (bool): Enable detailed debug logging for server operations.
            Useful for troubleshooting and development.
            Default: False

        ray_namespace (Optional[str]): Optional namespace for the Ray cluster.
            Used to connect to a specific Ray namespace when running distributed tasks.
            If None, uses the default namespace.
            Default: None

        pg_name (Optional[str]): Optional name of the Ray placement group to use.
            Specifies which placement group to use for scheduling Ray tasks,
            allowing for better resource management and task isolation.
            If None, no specific placement group is used.
            Default: None

        ray_ip (Optional[str]): Optional IP address of the Ray cluster head node.
            Used to connect to an existing Ray cluster. If None and ray_port is also None,
            Ray will be initialized locally with address="auto".
            Default: None

        ray_port (Optional[int]): Optional port number for the Ray cluster head node.
            Used in conjunction with ray_ip to connect to an existing Ray cluster.
            Ignored if ray_ip is None.
            Default: None


    Example:
        ```python
        from mol_gen_docking.server_utils.server_setting import MolecularVerifierServerSettings
        from mol_gen_docking.reward import MolecularVerifier

        # Load settings from environment variables or .env file
        settings = MolecularVerifierServerSettings()

        # Convert to verifier configuration
        config = settings.to_molecular_verifier_config(reward="property")

        # Create verifier instance
        verifier = MolecularVerifier(verifier_config=config)
        ```

    Environment Variables:
        Settings can be configured via environment variables (non-case sensitive) names:

        - SERVER_MODE
        - SCORER_EXHAUSTIVENESS
        - SCORER_NCPUS
        - DOCKING_CONCURRENCY_PER_GPU
        - REACTION_MATRIX_PATH
        - DOCKING_ORACLE
        - VINA_MODE
        - DATA_PATH
        - BUFFER_TIME
        - PARSING_METHOD
        - DEBUG_LOGGING
        - RAY_NAMESPACE
        - PG_NAME
        - RAY_IP
        - RAY_PORT
    """

    server_mode: Literal["singleton", "batch"] = "singleton"
    scorer_exhaustiveness: int = 8
    scorer_ncpus: int = 8
    docking_concurrency_per_gpu: int = 2
    reaction_matrix_path: str = "data/rxn_matrix.pkl"
    docking_oracle: Literal["pyscreener", "autodock_gpu"] = "autodock_gpu"
    vina_mode: str = "autodock_gpu_256wi"
    data_path: str = "data/molgendata"
    buffer_time: int = 20
    parsing_method: Literal["none", "answer_tags", "boxed"] = "answer_tags"
    debug_logging: bool = False
    ray_namespace: Optional[str] = None
    pg_name: Optional[str] = None
    ray_ip: Optional[str] = None
    ray_port: Optional[int] = None
    ray_tmp_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate all settings after initialization.

        Performs critical validation checks to ensure settings are consistent
        and valid before the server starts. This method is called automatically
        after the pydantic BaseSettings initializes the instance.

        Raises:
            AssertionError: If any of the following conditions are not met:

                - scorer_exhaustiveness > 0
                - scorer_ncpus > 0
                - docking_concurrency_per_gpu > 0
                - reaction_matrix_path file exists
        """
        assert self.scorer_exhaustiveness > 0, "Exhaustiveness must be greater than 0"
        assert self.scorer_ncpus > 0, "Number of CPUs must be greater than 0"

        assert Path(self.reaction_matrix_path).exists(), (
            f"Reaction matrix file {self.reaction_matrix_path} does not exist"
        )

    def to_molecular_verifier_config(
        self, reward: Literal["property", "valid_smiles"] = "property"
    ) -> MolecularVerifierConfigModel:
        """Convert server settings to a complete verifier configuration model.

        Creates a MolecularVerifierConfigModel with all sub-verifier configurations
        (generation, reaction, and property) initialized from the server settings.
        This method bridges the gap between server-level settings and verifier-level
        configurations, ensuring consistent parameter propagation.

        Args:
            reward (Literal["property", "valid_smiles"]): Type of reward to compute.

                - "property": Use property-based rewards for molecular optimization.
                  Evaluates predicted property values against target objectives.
                - "valid_smiles": Use validity-based rewards. Returns 1.0 for valid
                  molecules and 0.0 for invalid ones.
                Default: "property"

        Returns:
            MolecularVerifierConfigModel: A fully configured verifier model containing:

                - GenerationVerifierConfigModel: For molecular generation tasks with:
                  - path_to_mappings set to data_path
                  - reward type synchronized
                  - rescale enabled
                  - oracle_kwargs with docking settings
                  - docking_concurrency_per_gpu configured

                - ReactionVerifierConfigModel: For reaction verification with:
                  - reaction_matrix_path configured
                  - reward type synchronized
                  - reaction_reward_type set to "tanimoto" (default)

                - MolPropVerifierConfigModel: For property prediction with:
                  - reward type synchronized

                All sub-configs are automatically synced to the specified reward type.

        Raises:
            ValueError: If any of the referenced configuration paths don't exist.
                This can happen if data_path or reaction_matrix_path are invalid.
        """
        # Create oracle kwargs from server settings
        oracle_kwargs = {
            "exhaustiveness": self.scorer_exhaustiveness,
            "n_cpu": self.scorer_ncpus,
            "docking_oracle": self.docking_oracle,
            "vina_mode": self.vina_mode,
        }

        # Create GenerationVerifierConfigModel
        generation_config = GenerationVerifierConfigModel(
            path_to_mappings=self.data_path,
            reward=reward,
            rescale=True,
            oracle_kwargs=oracle_kwargs,
            docking_concurrency_per_gpu=self.docking_concurrency_per_gpu,
            pg_name=self.pg_name,
        )

        # Create ReactionVerifierConfigModel
        reaction_config = ReactionVerifierConfigModel(
            reaction_matrix_path=self.reaction_matrix_path,
            reward=reward,
            pg_name=self.pg_name,
        )

        # Create MolPropVerifierConfigModel
        molprop_config = MolPropVerifierConfigModel(reward=reward, pg_name=self.pg_name)

        # Create and return MolecularVerifierConfigModel
        return MolecularVerifierConfigModel(
            parsing_method=self.parsing_method,
            pg_name=self.pg_name,
            reward=reward,
            generation_verifier_config=generation_config,
            reaction_verifier_config=reaction_config,
            mol_prop_verifier_config=molprop_config,
        )

    def ray_init(self) -> None:
        if ray.is_initialized():
            return
        init_kwargs = {}
        if self.ray_namespace is not None:
            init_kwargs["namespace"] = self.ray_namespace
        if self.ray_tmp_dir is not None:
            init_kwargs["_temp_dir"] = self.ray_tmp_dir
        if self.ray_ip is not None:
            init_kwargs["address"] = self.ray_ip
            if self.ray_port is not None:
                init_kwargs["address"] += f":{self.ray_port}"
        else:
            init_kwargs["address"] = "auto"

        ray.init(**init_kwargs)
