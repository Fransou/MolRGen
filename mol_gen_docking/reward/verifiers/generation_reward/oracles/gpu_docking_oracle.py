import json
import os
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import ray
from tdc.metadata import docking_target_info
from tdc.utils import receptor_load


def _chunks(lst: List[str], n: int) -> Generator[List[str], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class DockingMoleculeGpuOracle:
    def __init__(
        self,
        path_to_data: str,
        gnina: bool = False,
        receptor_path: Optional[Union[Path, str]] = None,
        receptor_name: Optional[str] = None,
        center: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        print_msgs: bool = True,
        failed_score: float = 0.0,
        n_conformers: int = 1,
        docking_batch_size: int = 16,
        **kwargs: Any,
    ):
        """
        Parameters:
            gnina (bool): Whether to use GNINA for rescoring. Default is False.
            receptor_path (Optional[Union[Path, str]]): Path to the receptor file. Required if receptor_name is not provided.
            receptor_name (Optional[str]): Name of the receptor. Required if receptor_path is not provided.
            center (Optional[List[float]]): Center coordinates of the docking box.
            size (Optional[List[float]]): Size of the docking box. If not provided, uses predefined sizes for known receptors.
            print_msgs (bool): Whether to print messages during docking. Default is False.
            failed_score (float): Score to assign when docking fails. Default is 0.0.
            n_conformers (int): Number of conformers to generate per molecule. Default is 1.
            docking_batch_size (int): Batch size for docking. Default is 25.
        """

        super().__init__()
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
        self.failed_score = failed_score
        self.n_conformers = n_conformers
        self.batch_size = docking_batch_size

        # if self.gnina:
        #     self.gnina_rescorer = GninaRescorer(
        #         receptor_pdbqt_file=self.receptor_path,
        #         pose_format="pdbqt",
        #         center_pos=self.center,
        #         size=self.size,
        #     )

    def dock_batch_qv2gpu(
        self, smiles: List[str], output: Any
    ) -> Tuple[List[float | None], List[Optional[str]]]:
        """
        Uses GPU docking implementation to
        calculate docking score against target of choice.

        Note: Failure at any point in the pipeline (reading molecule, pdbqt conversion,
            score calculation) returns self.failed_score for that molecule.
        """
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
        """Compute docking scores for a list of SMILES strings.

        Args:
            smiles: List of SMILES strings to dock.

        Returns:
            List of docking scores corresponding to each SMILES.
        """
        scores: List[float] = []
        docking_module_gpu = ray.get_actor("docking_actor")
        outputs_jobs = []
        for chunk in _chunks(smiles, self.batch_size):
            outputs_jobs.append(
                docking_module_gpu.dock_smis.remote(
                    smi=smiles, receptor_file=self.receptor_path
                )
            )

        outputs = ray.get(outputs_jobs)
        for chunk, output in zip(_chunks(smiles, self.batch_size), outputs):
            scores_chunk, docked_pdbqts = self.dock_batch_qv2gpu(chunk, output)
            if scores_chunk is not None:
                scores_chunk_float = [
                    self.failed_score if s is None else s for s in scores_chunk
                ]
                scores += scores_chunk_float
            else:
                scores += [self.failed_score] * len(chunk)
        return scores
