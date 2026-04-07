# Adapted from code in PyVina https://github.com/Pixelatory/PyVina
# Copyright (c) 2024 Nicholas Aksamit
# Licensed under the MIT License

import hashlib
import logging
import os
import random
import re
import shutil
import subprocess
import time
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import ray
from rdkit import Chem
from ringtail.parsers import parse_single_dlg

VINA_DOCKING_OUTPUT = None | Tuple[List[float | None], List[str | None]]


def make_dir(rel_path: str, *args: Any, **kwargs: Any) -> None:
    os.makedirs(os.path.abspath(rel_path), *args, **kwargs)


def sanitize_smi_name_for_file(smi: str) -> str:
    """
    Sanitization for file names. Replacement values cannot be part of valid SMILES.
    Add a random value at the end.
    """
    full_hash = hashlib.sha224(smi.encode()).hexdigest()[::3]
    return full_hash + str(int(10**6 * random.random())).zfill(6)


def move_files_from_dir(
    source_dir_path: str, dest_dir_path: str, include_only: List[str] = []
) -> None:
    files = os.listdir(source_dir_path)
    if len(include_only) > 0:
        files = [f for f in files if f in include_only]
    for file in files:
        source_file = os.path.join(source_dir_path, file)
        destination_file = os.path.join(dest_dir_path, file)
        shutil.move(source_file, destination_file)


def split_list(lst: List[Any], n: int) -> List[List[Any]]:
    """Split a list into n equal parts, with any remainder added to the last split."""
    if n <= 0:
        raise ValueError("Number of splits (n) must be a positive integer.")

    quotient, remainder = divmod(len(lst), n)
    splits = [
        lst[
            i * quotient + min(i, remainder) : (i + 1) * quotient
            + min(i + 1, remainder)
        ]
        for i in range(n)
    ]
    return splits


def get_ligand_hashed_fn(
    path: str, n_conformers: int, suffix: str = ""
) -> Callable[[str], List[str]]:
    def ligand_hashed_fn(smi: str) -> List[str]:
        return [
            os.path.abspath(
                f"{path}/{sanitize_smi_name_for_file(smi)}_{i}_{suffix}.pdbqt"
            )
            for i in range(n_conformers)
        ]

    return ligand_hashed_fn


class TimedProfiler:
    def __init__(self) -> None:
        self._count = 0.0
        self._total = 0.0
        self._average = 0.0

    def _add_value(self, value: float) -> None:
        self._total += value
        self._count += 1
        self._average = self._total / self._count

    def time_it(self, fn: Callable[..., Any] | None, *args: Any, **kwargs: Any) -> Any:
        if fn is None:
            return 0
        start_time = time.time()
        res = fn(*args, **kwargs)
        end_time = time.time()
        self._add_value(end_time - start_time)
        return res

    def get_average(self) -> float:
        return self._average


class BaseDocking:
    def __init__(
        self,
        cmd: str,
        receptor_file: str,
        n_conformers: int = 1,
        get_pose_str: bool = False,
        timeout_duration: Optional[int] = None,
        additional_args: Optional[Dict[str, Any]] = None,
        preparator: Optional[Callable[[List[str], List[List[str]]], List[bool]]] = None,
        cwd: Optional[str] = None,
        gpu_ids: Union[None, str, List[str]] = None,
        docking_attempts: int = 10,
        print_msgs: bool = True,
        print_output: bool = False,
        debug: bool = True,
        docking_concurrency_per_gpu: int = 8,
    ) -> None:
        """
            Parameters:
            - cmd: Command line prefix to execute command (e.g. "/path/to/qvina2.1")
            - receptor_file: Cleaned receptor PDBQT file to use for docking
            - n_conformers: how many times are we docking each SMILES string?
            - get_pose_str: Return output pose as string (True) or not (False)
            - timeout_duration: Timeout in seconds before new process automatically stops
            - additional_args: Dictionary of additional Vina command arguments (e.g. {"cpu": "5"})
            - preparator: Function/Class callable to prepare molecule for docking. Should take the \
                argument format (smiles strings, ligand paths)
            - cwd: Change current working directory of Vina shell (sometimes needed for GPU versions \
                and incorrect openCL pathing)
            - gpu_ids: GPU ids to use for multi-GPU docking (0 is default for single-GPU nodes). If None, \
                use all GPUs.
            - docking_attempts: Number of docking attempts to make on each GPU.
            - print_msgs: Show Python print messages in console (True) or not (False)
            - print_output: Show Vina docking output in console (True) or not (False)
            - debug: Profiling the Vina docking process and ligand preparation.
            - docking_concurrency_per_gpu: Number of concurrent docking processes to run on each GPU.
        """
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )
        if not os.path.isfile(receptor_file):
            raise Exception(rf"Receptor file: {receptor_file} not found")

        self.cmd = cmd
        self.receptor_pdbqt_file = os.path.abspath(receptor_file)
        self.n_conformers = n_conformers
        self.get_pose_str = get_pose_str
        self.timeout_duration = timeout_duration
        self.additional_args = additional_args
        self.preparator = preparator
        self.cwd = cwd

        self.docking_attempts = docking_attempts
        self.print_msgs = print_msgs
        self.print_output = print_output
        self.debug = debug
        self.docking_concurrency_per_gpu = docking_concurrency_per_gpu

        if debug:
            self.preparation_profiler = TimedProfiler()
            self.docking_profiler = TimedProfiler()

    def _prepare_ligands(
        self, ligand_paths_by_smiles: Dict[str, List[str]]
    ) -> List[bool]:
        # Perform ligand preparation and save to proper path (tmp/non-tmp ligand dir)
        smis = list(ligand_paths_by_smiles.keys())
        ligand_paths = list(ligand_paths_by_smiles.values())
        assert self.preparator is not None
        if self.debug:
            return self.preparation_profiler.time_it(  # type:ignore
                self.preparator, smis, ligand_paths
            )
        else:
            return self.preparator(smis, ligand_paths)

    def _batched_prepare_ligands(self, ligand_dir_path: str, smis: List[str]) -> Any:
        # create hashed filename for each unique smiles
        ligand_path_fn = get_ligand_hashed_fn(
            ligand_dir_path, n_conformers=self.n_conformers
        )
        # not the fastest implementation, but safe if multiple experiments running at same time (with different tmp file paths)
        ligand_paths_by_smiles: Dict[str, List[str]] = {}
        for i in range(len(smis)):
            current_ligand_paths = ligand_path_fn(smis[i])
            ligand_paths_by_smiles[smis[i]] = current_ligand_paths
        # Prepare ligands that don't have an existing output file (they aren't overlapping)
        if self.print_msgs:
            self.logger.info("Preparing ligands...")
        success_list = self._prepare_ligands(ligand_paths_by_smiles)
        if not all(success_list):
            for smi, success in zip(smis, success_list):
                if not success:
                    self.logger.warning(
                        f"Error preparing: {smi} | {ligand_paths_by_smiles[smi]}"
                    )

                    # Remove unsuccessful Ligands
                    ligand_paths_by_smiles[smi] = []

        return ligand_paths_by_smiles

    def _sub_module_batch_docking(
        self,
        smis: List[str],
        ligand_dir_path: str,
        ligand_paths_by_smiles: Dict[str, List[str]],
        output_dir_path: str,
    ) -> VINA_DOCKING_OUTPUT:
        gpu_ids = ray.get_gpu_ids()

        # Create GPU-specific subdirectories and distribute ligands
        make_dir(ligand_dir_path, exist_ok=True)
        for gpu_id in gpu_ids:
            make_dir(f"{ligand_dir_path}/{gpu_id}/", exist_ok=True)

        # Gather all ligand paths and split them across GPUs
        ligand_paths = sum(
            [lig_paths for lig_paths in ligand_paths_by_smiles.values() if lig_paths],
            [],
        )
        split_ligand_paths = split_list(ligand_paths, len(gpu_ids))

        # Copy ligands to GPU-specific directories
        ligand_dir = []
        for gpu_i in range(len(gpu_ids)):
            gpu_id = gpu_ids[gpu_i]
            ligand_dir.append(os.path.abspath(f"{ligand_dir_path}/{gpu_id}/"))
            for ligand_file in split_ligand_paths[gpu_i]:
                try:
                    shutil.copy(
                        ligand_file,
                        os.path.abspath(f"{ligand_dir_path}/{gpu_id}/"),
                    )
                except FileNotFoundError:
                    if self.print_msgs:
                        self.logger.warning(f"Ligand file not found: {ligand_file}")

        return self._batched_docking_run(
            smis,
            ligand_dir=ligand_dir,
            ligand_dir_path=ligand_dir_path,
            ligand_paths_by_smiles=ligand_paths_by_smiles,
            output_dir_path=output_dir_path,
            gpu_ids=gpu_ids,
        )

    def _batched_docking(self, smis: List[str]) -> VINA_DOCKING_OUTPUT:
        # make temp pdbqt directories.
        ligand_tempdir = TemporaryDirectory(suffix="_lig")
        output_tempdir = TemporaryDirectory(suffix="_out")

        ligand_dir_path = ligand_tempdir.name
        output_dir_path = output_tempdir.name

        make_dir(ligand_dir_path, exist_ok=True)
        make_dir(output_dir_path, exist_ok=True)

        @ray.remote(num_cpus=2)
        def batch_prepare_ligands(ligand_dir_path: str, smis: List[str]) -> Any:
            return self._batched_prepare_ligands(ligand_dir_path, smis)

        ligand_paths_by_smiles = ray.get(
            batch_prepare_ligands.remote(ligand_dir_path=ligand_dir_path, smis=smis)  # type: ignore
        )

        # Run docking attempts multiple times on each GPU in case of failure.
        @ray.remote(num_gpus=1 / self.docking_concurrency_per_gpu, num_cpus=1)
        def sub_module_batch_docking(
            smis: List[str],
            ligand_dir_path: str,
            ligand_paths_by_smiles: Dict[str, List[str]],
            output_dir_path: str,
        ) -> VINA_DOCKING_OUTPUT:
            return self._sub_module_batch_docking(
                smis=smis,
                ligand_dir_path=ligand_dir_path,
                ligand_paths_by_smiles=ligand_paths_by_smiles,
                output_dir_path=output_dir_path,
            )

        output = ray.get(
            sub_module_batch_docking.remote(  # type: ignore
                smis,
                ligand_dir_path=ligand_dir_path,
                ligand_paths_by_smiles=ligand_paths_by_smiles,
                output_dir_path=output_dir_path,
            )
        )

        # clean up temp dirs
        ligand_tempdir.cleanup()
        output_tempdir.cleanup()
        if self.print_msgs:
            self.logger.info("Docking complete.")

        return output

    def _batched_docking_run(
        self,
        smis: List[str],
        ligand_dir: List[str],
        ligand_dir_path: str,
        ligand_paths_by_smiles: Dict[str, List[str]],
        output_dir_path: str,
        gpu_ids: List[int] = [0],
    ) -> VINA_DOCKING_OUTPUT:
        raise NotImplementedError

    def __call__(self, smi: Union[str, List[str]]) -> VINA_DOCKING_OUTPUT:
        """
        Parameters:
        - smi: SMILES strings to perform docking. A single string activates single-ligand docking mode, while \
            multiple strings utilizes batched docking (if Vina version allows it).
        """
        smi_list: List[str] = []
        if type(smi) is str:
            smi_list = [smi]
        elif type(smi) is list:
            for i in range(len(smi)):
                mol = Chem.MolFromSmiles(smi[i])
                if mol is not None:
                    smi[i] = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                smi_list = smi
        else:
            raise Exception("smi must be a string or a list of strings")
        if len(smi) > 0:
            return self._batched_docking(smi_list)
        else:
            return None


class VinaDocking(BaseDocking):
    def __init__(
        self,
        cmd: str,
        receptor_file: str,
        center_pos: List[float],
        size: List[float],
        n_conformers: int = 1,
        get_pose_str: bool = False,
        timeout_duration: Optional[int] = None,
        additional_args: Optional[Dict[str, Any]] = None,
        preparator: Optional[Callable[[List[str], List[List[str]]], List[bool]]] = None,
        cwd: Optional[str] = None,
        gpu_ids: Union[None, str, List[str]] = None,
        docking_attempts: int = 10,
        print_msgs: bool = True,
        print_output: bool = True,
        debug: bool = True,
        docking_concurrency_per_gpu: int = 1,
    ) -> None:
        if len(center_pos) != 3:
            raise Exception(
                f"center_pos must contain 3 values: {center_pos} was provided"
            )

        if len(size) != 3:
            raise Exception(f"size must contain 3 values: {size} was provided")

        self.center_pos = center_pos
        self.size = size
        super().__init__(
            cmd,
            receptor_file,
            n_conformers,
            get_pose_str,
            timeout_duration,
            additional_args,
            preparator,
            cwd,
            gpu_ids,
            docking_attempts,
            print_msgs,
            print_output,
            debug,
        )

    def _batched_docking_run(
        self,
        smis: List[str],
        ligand_dir: List[str],
        ligand_dir_path: str,
        ligand_paths_by_smiles: Dict[str, List[str]],
        output_dir_path: str,
        gpu_ids: List[int] = [0],
    ) -> VINA_DOCKING_OUTPUT:
        # make temp pdbqt directories.
        config_tempdir = TemporaryDirectory(suffix="_config")
        config_dir_path = config_tempdir.name
        make_dir(config_dir_path, exist_ok=True)

        # make different directories to support parallelization across multiple GPUs.
        for gpu_id in gpu_ids:
            make_dir(f"{output_dir_path}/{gpu_id}/", exist_ok=True)

        # create hashed output filename for each unique smiles
        output_path_fn = get_ligand_hashed_fn(
            output_dir_path, n_conformers=self.n_conformers, suffix="_out"
        )

        # not the fastest implementation, but safe if multiple experiments running at same time (with different tmp file paths)
        output_paths = []
        for i in range(len(smis)):
            output_paths += output_path_fn(smis[i])

        tmp_config_file_paths = []
        for i in range(len(gpu_ids)):
            gpu_id = gpu_ids[i]
            tmp_config_file_path = f"{config_dir_path}/config_{gpu_id}"
            tmp_config_file_paths.append(tmp_config_file_path)
            self._write_conf_file(
                tmp_config_file_path,
                {
                    "ligand_directory": f"{ligand_dir_path}/{gpu_id}/",
                    "output_directory": f"{output_dir_path}/{gpu_id}/",
                },
            )

        # Perform docking procedure(s)
        if self.print_msgs:
            self.logger.info("Ligands prepared. Docking...")

        cmd_prefixes = [f"CUDA_VISIBLE_DEVICES={gpu_id} " for gpu_id in gpu_ids]
        log_paths = [f"log_{i}" for i in range(len(tmp_config_file_paths))]
        # Run docking attempts multiple times on each GPU in case of failure.
        for attempt in range(self.docking_attempts):
            if self.debug:
                self.docking_profiler.time_it(
                    self._run_docking,
                    tmp_config_file_paths,
                    cmd_prefixes=cmd_prefixes,
                    blocking=False,
                    log_paths=log_paths,
                )
            else:
                self._run_docking(
                    tmp_config_file_paths, log_paths=log_paths, blocking=False
                )
            if all(
                len(os.listdir(f"{output_dir_path}/{gpu_id}/")) > 0
                for gpu_id in gpu_ids
            ):
                break

            self.logger.warning(
                f"Docking attempt #{attempt + 1} failed on GPU {gpu_ids[i]}."
            )

        # Move files from temporary to proper directory (or delete if redoing calculation)
        for gpu_id in gpu_ids:
            move_files_from_dir(f"{ligand_dir_path}/{gpu_id}/", ligand_dir_path)

        for gpu_id in gpu_ids:
            move_files_from_dir(f"{output_dir_path}/{gpu_id}/", output_dir_path)

        # Something went horribly wrong
        if all(not os.path.exists(path) for path in output_paths):
            raise ValueError
            binding_scores = None

        else:
            # Gather binding scores
            binding_scores_list = []
            for i in range(len(output_paths)):
                binding_scores_list.append(self._get_output_score(output_paths[i]))

            if self.get_pose_str:
                binding_poses = []
                for i in range(len(output_paths)):
                    binding_poses.append(self._get_output_pose(output_paths[i]))
            else:
                binding_poses = [None] * len(output_paths)
            binding_scores = (binding_scores_list, binding_poses)

        config_tempdir.cleanup()
        return binding_scores

    @staticmethod
    def _get_output_score(output_path: str) -> Union[float, None]:
        try:
            score = float("inf")
            with open(output_path) as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", line)[0]
                        score = min(score, float(new_score))
            return score
        except FileNotFoundError:
            return None

    @staticmethod
    def _get_output_pose(output_path: str) -> Union[str, None]:
        try:
            with open(output_path) as f:
                docked_pdbqt = f.read()
            return docked_pdbqt
        except FileNotFoundError:
            return None

    def _write_conf_file(self, config_file_path: str, args: Dict[str, str] = {}) -> str:
        conf = (
            f"receptor = {self.receptor_pdbqt_file}\n"
            + f"center_x = {self.center_pos[0]}\n"
            + f"center_y = {self.center_pos[1]}\n"
            + f"center_z = {self.center_pos[2]}\n"
            + f"size_x = {self.size[0]}\n"
            + f"size_y = {self.size[1]}\n"
            + f"size_z = {self.size[2]}\n"
        )

        if self.additional_args:
            for k, v in self.additional_args.items():
                conf += f"{str(k)} = {str(v)}\n"

        for k, v in args.items():
            if self.cwd:  # vina_cwd is not none, meaning we have to use global paths for config ligand and output dirs
                conf += f"{str(k)} = {os.path.join(os.getcwd(), str(v))}\n"
            else:
                conf += f"{str(k)} = {str(v)}\n"

        with open(config_file_path, "w") as f:
            f.write(conf)
        return conf

    def _run_docking(
        self,
        config_paths: List[str],
        log_paths: Optional[List[str]] = None,
        cmd_prefixes: Optional[List[str]] = None,
        blocking: bool = True,
    ) -> None:
        """
        Runs Vina docking in separate shell process(es).
        """

        if log_paths is not None:
            assert len(config_paths) == len(log_paths)
        if cmd_prefixes is not None:
            assert len(config_paths) == len(cmd_prefixes)
        procs = []

        for i in range(len(config_paths)):
            if cmd_prefixes is not None and cmd_prefixes[i] is not None:
                cmd_str = cmd_prefixes[i]
            else:
                cmd_str = ""

            cmd_str += f"{self.cmd} --config {config_paths[i]}"

            if not self.print_output:
                cmd_str += " > /dev/null 2>&1"
            elif log_paths is not None:
                cmd_str += f" > {log_paths[i]} 2>&1"
            proc = subprocess.Popen(
                cmd_str, shell=True, start_new_session=False, cwd=self.cwd
            )

            if blocking:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                procs.append(proc)

        if not blocking:
            for proc in procs:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()


class AutoDockGPUDocking(BaseDocking):
    def __init__(
        self,
        cmd: str,
        receptor_file: str,
        n_conformers: int = 1,
        get_pose_str: bool = False,
        timeout_duration: Optional[int] = None,
        additional_args: Optional[Dict[str, Any]] = None,
        preparator: Optional[Callable[[List[str], List[List[str]]], List[bool]]] = None,
        cwd: Optional[str] = None,
        gpu_ids: Union[None, str, List[str]] = None,
        docking_attempts: int = 10,
        print_msgs: bool = True,
        print_output: bool = False,
        debug: bool = True,
        agg_type: Literal["mean", "min", "cluster_min"] = "min",
        docking_concurrency_per_gpu: int = 8,
    ) -> None:
        pdbqt_file = receptor_file.replace(".pdb", "_ag.pdbqt")
        grid_map_file = receptor_file.replace(".pdb", "_ag.maps.fld")
        assert os.path.exists(pdbqt_file), "Receptor file not found"
        assert os.path.exists(grid_map_file), "Grid map file not found"
        super().__init__(
            cmd,
            pdbqt_file,
            n_conformers,
            get_pose_str,
            timeout_duration,
            additional_args,
            preparator,
            cwd,
            gpu_ids,
            docking_attempts,
            print_msgs,
            print_output,
            debug,
        )
        self.cmd = cmd + " --filelist {}" + f" --ffile {grid_map_file}"
        if additional_args is not None:
            for k, v in additional_args.items():
                self.cmd += f" {k} {v} "
        self.receptor_pdbqt_file = os.path.abspath(pdbqt_file)
        self.receptor_map_file = os.path.abspath(grid_map_file)
        self.agg_type = agg_type

    def _batched_docking_run(
        self,
        smis: List[str],
        ligand_dir: List[str],
        ligand_dir_path: str,
        ligand_paths_by_smiles: Dict[str, List[str]],
        output_dir_path: str,
        gpu_ids: List[int] = [0],
    ) -> VINA_DOCKING_OUTPUT:
        # Perform docking procedure(s)
        if self.print_msgs:
            self.logger.info("Ligands prepared. Docking...")
        cmd_prefixes = [f"CUDA_VISIBLE_DEVICES={gpu_id} " for gpu_id in gpu_ids]
        # Run docking attempts multiple times on each GPU in case of failure.
        for attempt in range(self.docking_attempts):
            if self.debug:
                self.docking_profiler.time_it(
                    self._run_docking,
                    ligand_dir,
                    cmd_prefixes=cmd_prefixes,
                    blocking=False,
                )
            else:
                self._run_docking(
                    ligand_dir,
                    cmd_prefixes=cmd_prefixes,
                    blocking=False,
                )
            if self._is_docking_complete(ligand_dir):
                break

        # Move files from temporary to proper directory (or delete if redoing calculation)
        for gpu_id in gpu_ids:
            move_files_from_dir(f"{ligand_dir_path}/{gpu_id}/", ligand_dir_path)

        output_paths = [
            file for file in os.listdir(ligand_dir_path) if file.endswith(".dlg")
        ]

        # Something went horribly wrong
        if output_paths == []:
            error_msg = f"Critical error encountered, no output path given for smi list:\n {smis} \n"
            error_msg += f"### ### number of preprocessed smiles: {len(os.listdir(ligand_dir_path))}"

            self.logger.error(error_msg)
        # Gather binding scores
        binding_scores_dict: Dict[str, float | None] = {}
        # Move all output files to a TempDir
        move_files_from_dir(ligand_dir_path, output_dir_path, include_only=output_paths)
        output_paths = [
            os.path.abspath(f"{output_dir_path}/" + os.path.basename(file))
            for file in output_paths
        ]
        for file in output_paths:
            try:
                ligand_dict = parse_single_dlg(file)
                identifier = ligand_dict["ligname"]
                binding_score_val: float | None
                if self.agg_type == "min":
                    binding_score_val = min(ligand_dict["scores"])
                elif self.agg_type == "mean":
                    binding_score_val = sum(ligand_dict["scores"]) / len(
                        ligand_dict["scores"]
                    )
                elif self.agg_type == "cluster_min":
                    cluster_sizes: Dict[str, int] = ligand_dict["cluster_sizes"]
                    largest_cluster = max(cluster_sizes, key=cluster_sizes.get)  # type: ignore
                    binding_score_val = min(
                        [
                            score
                            for c, score in zip(
                                ligand_dict["cluster_list"], ligand_dict["scores"]
                            )
                            if c == largest_cluster
                        ]
                    )
            except ZeroDivisionError:
                with open(file, "rb") as fp:
                    for line in fp.readlines():
                        line_str: str = line.decode("utf-8")
                        # store ligand file name
                        if line_str[0:11] == "Ligand file":
                            identifier = (
                                line_str.split(":", 1)[1]
                                .split("/")[-1]
                                .split(".")[0]
                                .strip()
                            )  # remove path and file extension
                            break
                binding_score_val = 0.0
            # We cap the binding score to a maximum value  of 0.0 kcal/mol
            binding_score_val = (
                min(binding_score_val, 0.0) if binding_score_val is not None else None
            )
            binding_scores_dict[identifier] = binding_score_val
        if self.get_pose_str:
            binding_poses = []
            for i in range(len(output_paths)):
                binding_poses.append(self._get_output_pose(output_paths[i]))
        else:
            binding_poses = [None] * len(output_paths)
        binding_scores_per_smi = {
            smi: [
                binding_scores_dict.get(os.path.basename(p).split(".")[0], 0)
                for p in paths
            ]
            for smi, paths in ligand_paths_by_smiles.items()
        }
        binding_scores_: Dict[str, float | None] = {}
        for smi, scores in binding_scores_per_smi.items():
            scores = [s for s in scores if s is not None]
            if len(scores) > 0:
                binding_scores_[smi] = min(scores)  # type: ignore
            else:
                binding_scores_[smi] = None
        binding_scores = (
            [binding_scores_.get(smi, 0) for smi in smis],
            binding_poses,
        )
        return binding_scores

    @staticmethod
    def _get_output_pose(output_path: str) -> Union[str, None]:
        try:
            with open(output_path) as f:
                docked_pdbqt = f.read()
            return docked_pdbqt
        except FileNotFoundError:
            return None

    def _run_docking(
        self,
        ligand_dir: List[str],
        cmd_prefixes: Optional[List[str]] = None,
        blocking: bool = True,
    ) -> None:
        """
        Runs Vina docking in separate shell process(es).
        """
        if cmd_prefixes is not None:
            assert len(cmd_prefixes) == len(ligand_dir)

        procs = []

        for i in range(len(ligand_dir)):
            if cmd_prefixes is not None and cmd_prefixes[i] is not None:
                cmd_str = f"{cmd_prefixes[i]} {self.cmd.format(ligand_dir[i])}"
            else:
                cmd_str = self.cmd.format(ligand_dir[i])

            if not self.print_output:
                cmd_str += " > /dev/null 2>&1"
            proc = subprocess.Popen(
                cmd_str, shell=True, start_new_session=False, cwd=ligand_dir[i]
            )

            if blocking:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                procs.append(proc)

        if not blocking:
            for proc in procs:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _is_docking_complete(self, ligand_dir: List[str]) -> bool:
        return all(
            len([f for f in os.listdir(lg_dir) if f.endswith(".dlg")]) > 0
            for lg_dir in ligand_dir
        )
