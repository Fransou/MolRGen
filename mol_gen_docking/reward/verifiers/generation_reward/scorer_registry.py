"""Reward functions for molecular optimization."""

import logging
from typing import Any, Callable, Dict, List, Literal

import ray
from ray.actor import ActorProxy
from tdc.oracles import Oracle

from mol_gen_docking.utils.property_utils import (
    rescale_property_values,
)


@ray.remote(concurrency_groups={"serial_init": 1})  # type: ignore
class ScorerRegistery:
    """
    Class to register oracles returning the rescaled property of molecules
    """

    def __init__(
        self,
        debug: bool = False,
        property_name_mapping: Dict[str, str] = {},
        docking_target_list: List[str] = [],
        docking_concurrency_actor: int = 16,
    ):
        """Initialize the ScorerRegistry.

        Args:
            debug: If True, enables debug mode with additional logging.
            property_name_mapping: Dictionary mapping property names to oracle names.
            docking_target_list: List of valid docking target names.
            docking_concurrency_actor: Number of concurrent docking actors.
        """
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )
        self._evaluator_registry: Dict[str, Callable[[Any], Any]] = {}
        self.property_name_mapping = property_name_mapping
        self.docking_target_list = docking_target_list

        self._docking_gpu_module: ActorProxy[Any] | None = None
        self._preparator: Any | None = None
        self.docking_concurrency_actor = docking_concurrency_actor

    @ray.method(concurrency_group="serial_init")
    def set_docking_gpu_module(
        self,
        docking_executable: str,
        preparator_class: Any = None,
        aggregation_type: Literal["mean", "min", "cluster_min"] = "mean",
        print_msgs: bool = True,
        docking_attempts: int = 10,
        exhaustiveness: int = 16,
        docking_concurrency_per_gpu: int = 8,
        n_conformers: int = 1,
        conformer_attempts: int = 1,
    ) -> Any:
        """Initialize and configure the GPU docking module.

        Args:
            docking_executable: Command mode for AutoDock GPU.
            preparator_class: Class to use for ligand preparation.
            aggregation_type: How to aggregate docking scores ('mean', 'min', 'cluster_min').
            print_msgs: Whether to print messages during docking.
            docking_attempts: Number of docking attempts to make on each GPU.
            exhaustiveness: Docking exhaustiveness parameter.
            docking_concurrency_per_gpu: Number of concurrent docking runs per GPU.
            n_conformers: Number of conformers to generate per molecule.
            conformer_attempts: Number of attempts to generate conformers.

        Returns:
            The initialized docking GPU module actor.
        """
        if self._docking_gpu_module is None:
            from mol_gen_docking.reward.verifiers.generation_reward.oracles.docking_utils.docking_soft import (
                AutoDockGPUDocking,
            )
            from mol_gen_docking.reward.verifiers.generation_reward.oracles.docking_utils.preparators import (
                MeekoLigandPreparator,
            )

            if preparator_class is None:
                preparator_class = MeekoLigandPreparator
            preparator = preparator_class(
                conformer_attempts=conformer_attempts,
                n_conformers=n_conformers,
            )

            self._docking_gpu_module = AutoDockGPUDocking.options(  # type: ignore
                name="docking_actor",
                get_if_exists=True,
                max_concurrency=self.docking_concurrency_actor,
            ).remote(
                str(docking_executable),
                n_conformers=n_conformers,
                agg_type=aggregation_type,
                get_pose_str=False,
                preparator=preparator,
                timeout_duration=None,
                debug=True,
                print_msgs=print_msgs,
                print_output=False,
                docking_attempts=docking_attempts,
                additional_args={
                    "--nrun": exhaustiveness,
                },
                docking_concurrency_per_gpu=docking_concurrency_per_gpu,
            )
            # Wait of at most 5m for the actor to be available, otherwise return an error
            try:
                ray.get(self._docking_gpu_module.ping.remote(), timeout=300)
            except ray.exceptions.GetTimeoutError:
                err_msg = """Docking GPU module is not responding after 5 minutes.\n This could be due to ressources not being available to initialize the actor, consider scaling your ray cluster, or lowering the number of cpus used by the scorer_registry."""

                self.logger.error(err_msg)
                raise TimeoutError(err_msg)

    @ray.method(concurrency_group="serial_init")  # type: ignore
    def assign_evaluator(
        self,
        oracle_name: str,
        path_to_data: str = "",
        docking_oracle: str = "pyscreener",
        **kwargs: Any,
    ) -> None:
        """
        Get the evaluator function for the specified name.
        :param name: Name of the evaluator

        :return: Evaluator function
        """
        if oracle_name in self._evaluator_registry:
            if (
                oracle_name in self.docking_target_list
                and docking_oracle == "autodock_gpu"
            ):
                self.set_docking_gpu_module(**kwargs)  # To ensure the Actor has started
            return

        eval_fn: Callable[[Any], Any]
        if oracle_name in self.docking_target_list:
            if docking_oracle == "pyscreener":
                from mol_gen_docking.reward.verifiers.generation_reward.oracles.pyscreener_oracle import (
                    PyscreenerOracle,
                )

                eval_fn = PyscreenerOracle(
                    oracle_name, path_to_data=path_to_data, **kwargs
                )

            elif docking_oracle == "autodock_gpu":
                from mol_gen_docking.reward.verifiers.generation_reward.oracles.gpu_docking_oracle import (
                    DockingMoleculeGpuOracle,
                )

                self.set_docking_gpu_module(**kwargs)
                eval_fn = DockingMoleculeGpuOracle(
                    path_to_data=path_to_data, receptor_name=oracle_name, **kwargs
                )
        elif oracle_name.lower() in [
            "drd2",
            "gsk3b",
            "jnk3",
            "fpscores",
            "cyp3a4_veith",
        ]:
            eval_fn = Oracle(name=oracle_name, **kwargs)
        else:
            from mol_gen_docking.reward.verifiers.generation_reward.oracles.rdkit_oracle import (
                RDKITOracle,
            )

            eval_fn = RDKITOracle(oracle_name)

        self._evaluator_registry[oracle_name] = eval_fn

    def get_score(
        self,
        smis: List[str],
        oracle_name: str,
        rescale: bool = False,
    ) -> List[float]:
        """
        Score
        """
        oracle_name = oracle_name.replace(".", "")
        oracle_name = self.property_name_mapping.get(oracle_name, oracle_name)
        assert oracle_name in self._evaluator_registry, (
            f"Oracle {oracle_name} not found in registry. Available oracles: {list(self._evaluator_registry.keys())}"
        )
        evaluator = self._evaluator_registry[oracle_name]
        score_list = evaluator(smis)
        if len(score_list) != len(smis):
            raise ValueError("Length of score_list should be equal to length of smis")
        if rescale:
            score_list = [
                rescale_property_values(
                    oracle_name, score, docking=oracle_name in self.docking_target_list
                )
                for score in score_list
            ]
        else:
            score_list = [float(s) for s in score_list]
        return score_list
