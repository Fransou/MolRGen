import json
import os
from pathlib import Path
from typing import Generator, List, Literal, Optional, Tuple, Type, Union

from tdc.metadata import docking_target_info
from tdc.utils import receptor_load

# from rgfn.gfns.reaction_gfn.proxies.docking_proxy.gnina_wrapper import GninaRescorer
from mol_gen_docking.reward.verifiers.generation_reward.oracles.docking_utils.docking_soft import (
    AutoDockGPUDocking,
    BaseDocking,
)
from mol_gen_docking.reward.verifiers.generation_reward.oracles.docking_utils.preparators import (
    BasePreparator,
    MeekoLigandPreparator,
)


def _chunks(lst: List[str], n: int) -> Generator[List[str], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class DockingMoleculeGpuOracle:
    def __init__(
        self,
        path_to_data: str,
        docking_executable: Union[Path, str],
        preparator_class: Type[BasePreparator] = MeekoLigandPreparator,
        aggreggation_type: Literal["mean", "min", "cluster_min"] = "mean",
        gnina: bool = False,
        receptor_path: Optional[Union[Path, str]] = None,
        receptor_name: Optional[str] = None,
        center: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        print_msgs: bool = True,
        failed_score: float = 0.0,
        conformer_attempts: int = 1,
        n_conformers: int = 1,
        docking_attempts: int = 10,
        docking_batch_size: int = 16,
        exhaustiveness: int = 16,
        n_gpu: int = 1,
        gpu_ids: Optional[List[str]] = None,
        n_cpu: Optional[int] = None,
        docking_concurrency_per_gpu: int = 8,
    ):
        """
        Parameters:
            docking_executable (Union[Path, str]): Docking executable.
            preparator_class (Type[BasePreparator]): Class for preparing ligands. Default is MeekoLigandPreparator.
            gnina (bool): Whether to use GNINA for rescoring. Default is False.
            receptor_path (Optional[Union[Path, str]]): Path to the receptor file. Required if receptor_name is not provided.
            receptor_name (Optional[str]): Name of the receptor. Required if receptor_path is not provided.
            center (Optional[List[float]]): Center coordinates of the docking box.
            size (Optional[List[float]]): Size of the docking box. If not provided, uses predefined sizes for known receptors.
            print_msgs (bool): Whether to print messages during docking. Default is False.
            failed_score (float): Score to assign when docking fails. Default is 0.0.
            conformer_attempts (int): Number of attempts for conformer generation. Default is 20.
            n_conformers (int): Number of conformers to generate per molecule. Default is 1.
            docking_attempts (int): Number of docking attempts per molecule. Default is 10.
            docking_batch_size (int): Batch size for docking. Default is 25.
            exhaustiveness (int): Exhaustiveness parameter for docking. Default is 8000. Note: Minimum is 1000.
            n_gpu (int): Number of GPUs to use. Default is 1.
            n_cpu (Optional[int]): Number of CPUs to use. If None, uses all available CPUs.
        """

        super().__init__()
        self.aggregation_type = aggreggation_type
        if receptor_path is None and receptor_name is None:
            raise ValueError(
                "Expected either receptor_path or receptor_name to be specified."
            )
        if receptor_path is not None and receptor_name is not None:
            raise ValueError(
                "Expected only one of receptor_path and receptor_name to be specified."
            )

        if receptor_name is not None:
            if receptor_name.endswith("docking"):
                pdbid = receptor_name.split("_")[0]
                receptor_load(pdbid)
                self.receptor_path = "./oracle/" + pdbid + ".pdbqt"
                center = docking_target_info[pdbid]["center"]
                size = [s for s in docking_target_info[pdbid]["size"]]
                assert center is not None
            else:
                self.receptor_path = os.path.join(
                    path_to_data, "pdb_files", f"{receptor_name}.pdb"
                )
                if center is None:
                    with open(os.path.join(path_to_data, "pockets_info.json")) as f:
                        pockets_info = json.load(f)
                    center = pockets_info[receptor_name]["center"]
                    size = pockets_info[receptor_name]["size"]
            self.center = center
        elif type(receptor_path) is str and center is not None:
            self.receptor_path = receptor_path
            self.center = center
        else:
            raise ValueError(
                "Expected either receptor_path or receptor_name to be specified."
            )

        assert size is not None
        self.name = "[DOCKING]-" + receptor_name if receptor_name else receptor_path

        self.gnina = gnina
        self.size = size
        self.print_msgs = print_msgs
        self.preparator_class = preparator_class
        self.failed_score = failed_score
        self.conformer_attempts = conformer_attempts
        self.n_conformers = n_conformers
        self.docking_attempts = docking_attempts
        self.batch_size = docking_batch_size
        self.exhaustiveness = exhaustiveness
        self.n_gpu = n_gpu
        self.n_cpu = n_cpu

        self.preparator = self.preparator_class(
            conformer_attempts=self.conformer_attempts,
            n_conformers=self.n_conformers,
            num_cpus=self.n_cpu,
        )
        self.docking_module_gpu: BaseDocking
        self.docking_module_gpu = AutoDockGPUDocking(
            str(docking_executable),
            n_conformers=self.n_conformers,
            agg_type=self.aggregation_type,
            get_pose_str=False,
            preparator=self.preparator,
            timeout_duration=None,
            debug=True,
            print_msgs=self.print_msgs,
            print_output=False,
            gpu_ids=gpu_ids,
            docking_attempts=self.docking_attempts,
            additional_args={
                "--nrun": self.exhaustiveness,
            },
            docking_concurrency_per_gpu=docking_concurrency_per_gpu,
        )

        # if self.gnina:
        #     self.gnina_rescorer = GninaRescorer(
        #         receptor_pdbqt_file=self.receptor_path,
        #         pose_format="pdbqt",
        #         center_pos=self.center,
        #         size=self.size,
        #     )

    def dock_batch_qv2gpu(
        self, smiles: List[str]
    ) -> Tuple[List[float | None], List[Optional[str]]]:
        """
        Uses GPU docking implementation to
        calculate docking score against target of choice.

        Note: Failure at any point in the pipeline (reading molecule, pdbqt conversion,
            score calculation) returns self.failed_score for that molecule.
        """

        output = self.docking_module_gpu(smiles, self.receptor_path)
        if output is None:
            raise ValueError("Failed to compute docking score")
        scores, docked_pdbqts = output
        assert len(scores) == len(smiles)

        if self.n_conformers == 1 or (not scores or not docked_pdbqts):
            return scores, docked_pdbqts

        else:
            all_best_scores: List[None | float] = []
            all_best_poses: List[None | str] = []
            # if self.gnina:
            #     if self.print_msgs:
            #         print("Rescoring with gnina...")
            #     pose_batches = _chunks(docked_pdbqts, self.n_conformers)
            #     for batch in pose_batches:
            #         score, pose = self.gnina_rescorer(list(batch))
            #         all_best_scores.append(score * -1)  # gnina returns a positive score
            #         all_best_poses.append(pose)
            #
            #     if self.print_msgs:
            #         print("Rescoring complete.")
            #
            # else:  # we're just ranking the poses by qv2gpu score
            score_pose_batches = _chunks(
                list(zip(scores, docked_pdbqts)),  # type: ignore
                self.n_conformers,
            )
            for batch in score_pose_batches:
                successful_docks = [tup for tup in batch if tup[0] is not None]

                # Total conformer generation / docking failure for a single SMILES
                if len(successful_docks) == 0:
                    all_best_scores.append(self.failed_score)
                    all_best_poses.append(None)

                else:
                    best_pair = sorted(successful_docks, key=lambda x: x[0])[0]
                    all_best_scores.append(best_pair[0])  # type: ignore
                    all_best_poses.append(best_pair[1])

            return all_best_scores, all_best_poses

    def __call__(self, smiles: List[str]) -> List[float]:
        scores: List[float] = []

        for chunk in _chunks(smiles, self.batch_size):
            scores_chunk, docked_pdbqts = self.dock_batch_qv2gpu(chunk)
            if scores_chunk is not None:
                scores_chunk_float = [
                    self.failed_score if s is None else s for s in scores_chunk
                ]
                scores += scores_chunk_float
            else:
                scores += [self.failed_score] * len(chunk)
        return scores
