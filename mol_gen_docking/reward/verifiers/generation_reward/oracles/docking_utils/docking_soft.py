# Adapted from code in PyVina https://github.com/Pixelatory/PyVina
# Copyright (c) 2024 Nicholas Aksamit
# Licensed under the MIT License
import copy
import hashlib
import logging
import os
import random
import shutil
import subprocess
import time
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import ray
from pydantic import BaseModel, Field
from rdkit import Chem
from ringtail.parsers import parse_single_dlg


def make_dir(rel_path: str, *args: Any, **kwargs: Any) -> None:
    """Create a directory at the specified path.

    Args:
        rel_path: Relative path to create.
        *args: Additional positional arguments passed to os.makedirs.
        **kwargs: Additional keyword arguments passed to os.makedirs.
    """
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
    """Move files from source directory to destination directory.

    Args:
        source_dir_path: Path to source directory.
        dest_dir_path: Path to destination directory.
        include_only: Optional list of filenames to include (move all if empty).
    """
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
    """Create a function that generates hashed filenames for ligand conformers.

    Args:
        path: Base directory path for ligand files.
        n_conformers: Number of conformers to generate per SMILES.
        suffix: Optional suffix to append to filenames.

    Returns:
        A function that takes a SMILES string and returns a list of file paths.
    """

    def ligand_hashed_fn(smi: str) -> List[str]:
        """Generate a list of hashed file paths for a SMILES string.

        Args:
            smi: SMILES string to hash.

        Returns:
            List of absolute file paths for each conformer.
        """
        return [
            os.path.abspath(
                f"{path}/{sanitize_smi_name_for_file(smi)}_{i}_{suffix}.pdbqt"
            )
            for i in range(n_conformers)
        ]

    return ligand_hashed_fn


class TimedProfiler:
    """Profiler for timing function executions and computing averages."""

    def __init__(self) -> None:
        """Initialize the profiler with zero values."""
        self._count = 0.0
        self._total = 0.0
        self._average = 0.0

    def _add_value(self, value: float) -> None:
        """Add a timing value to the profiler.

        Args:
            value: Time duration to add.
        """
        self._total += value
        self._count += 1
        self._average = self._total / self._count

    def time_it(self, fn: Callable[..., Any] | None, *args: Any, **kwargs: Any) -> Any:
        """Time a function execution and record the duration.

        Args:
            fn: Function to time.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function, or 0 if fn is None.
        """
        if fn is None:
            return 0
        start_time = time.time()
        res = fn(*args, **kwargs)
        end_time = time.time()
        self._add_value(end_time - start_time)
        return res

    def get_average(self) -> float:
        """Get the average time from all recorded values.

        Returns:
            The average time duration.
        """
        return self._average


class DockingOutput(BaseModel):
    """Represents docking output."""

    score: float = Field(default=0.0, description="Docking score.")
    pose: Optional[str] = Field(
        default=None,
        description="Optional string representation of the docked pose (e.g., PDBQT format).",
    )
    success: bool = Field(..., description="Whether the docking was successful.")
    smi: str = Field(
        ..., description="The SMILES string corresponding to this docking output."
    )


class BaseDocking:  # Keep base class fo BC and future softwares
    """Base class for docking implementations."""

    def __init__(
        self,
        cmd: str,
        n_conformers: int = 1,
        get_pose_str: bool = False,
        timeout_duration: Optional[int] = None,
        additional_args: Optional[Dict[str, Any]] = None,
        preparator: Optional[Callable[[List[str], List[List[str]]], List[bool]]] = None,
        cwd: Optional[str] = None,
        docking_attempts: int = 10,
        print_msgs: bool = True,
        print_output: bool = False,
        debug: bool = True,
        docking_concurrency_per_gpu: int = 8,
    ) -> None:
        """Initialize the base docking class.

        Args:
            cmd: Command line prefix to execute command.
            n_conformers: How many times to dock each SMILES string.
            get_pose_str: Return output pose as string (True) or not (False).
            timeout_duration: Timeout in seconds before process automatically stops.
            additional_args: Dictionary of additional command arguments.
            preparator: Function/Class callable to prepare molecule for docking.
            cwd: Change current working directory (sometimes needed for GPU versions).
            docking_attempts: Number of docking attempts to make on each GPU.
            print_msgs: Show Python print messages in console (True) or not (False).
            print_output: Show docking output in console (True) or not (False).
            debug: Profiling the docking process and ligand preparation.
            docking_concurrency_per_gpu: Number of concurrent docking processes to run on each GPU.
        """
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )

        self.cmd = cmd
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

    def ping(self) -> bool:
        """Health check for the docking service.

        Returns:
            True if the service is healthy.
        """
        return True

    def _prepare_ligands(
        self, ligand_paths_by_smiles: Dict[str, List[str]]
    ) -> List[bool]:
        """Prepare ligands for docking.

        Args:
            ligand_paths_by_smiles: Dictionary mapping SMILES to their file paths.

        Returns:
            List of booleans indicating whether each ligand was prepared successfully.
        """
        # Perform ligand preparation and save to proper path (tmp/non-tmp ligand dir)
        smis = copy.deepcopy(list(ligand_paths_by_smiles.keys()))
        ligand_paths = list(ligand_paths_by_smiles.values())
        assert self.preparator is not None
        if self.debug:
            return self.preparation_profiler.time_it(  # type:ignore
                self.preparator, smis, ligand_paths
            )
        else:
            return self.preparator(smis, ligand_paths)

    def _batched_prepare_ligands(self, ligand_dir_path: str, smis: List[str]) -> Any:
        """Prepare a batch of ligands for docking.

        Args:
            ligand_dir_path: Path to directory for ligand files.
            smis: List of SMILES strings to prepare.

        Returns:
            Dictionary mapping SMILES to their prepared file paths.
        """
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
        receptor_file: str,
    ) -> List[DockingOutput]:
        """Perform docking on a sub-module (GPU-specific) batch.

        Args:
            smis: List of SMILES strings to dock.
            ligand_dir_path: Path to directory containing ligand files.
            ligand_paths_by_smiles: Dictionary mapping SMILES to their file paths.
            output_dir_path: Path to directory for docking output.
            receptor_file: Path to receptor file.

        Returns:
            Tuple of (scores, docked_poses) or None.
        """
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
            receptor_file=receptor_file,
        )

    def _batched_docking(
        self, smis: List[str], receptor_file: str
    ) -> List[DockingOutput]:
        """Perform batched docking on a list of SMILES.

        Args:
            smis: List of SMILES strings to dock.
            receptor_file: Path to receptor file.

        Returns:
            Tuple of (scores, docked_poses) or None.
        """
        # make temp pdbqt directories.
        ligand_tempdir = TemporaryDirectory(suffix="_lig")
        output_tempdir = TemporaryDirectory(suffix="_out")

        ligand_dir_path = ligand_tempdir.name
        output_dir_path = output_tempdir.name

        make_dir(ligand_dir_path, exist_ok=True)
        make_dir(output_dir_path, exist_ok=True)
        ligand_paths_by_smiles = self._batched_prepare_ligands(ligand_dir_path, smis)

        output = self._sub_module_batch_docking(
            smis=smis,
            ligand_dir_path=ligand_dir_path,
            ligand_paths_by_smiles=ligand_paths_by_smiles,
            output_dir_path=output_dir_path,
            receptor_file=receptor_file,
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
        gpu_ids: List[int],
        receptor_file: str,
    ) -> List[DockingOutput]:
        """Run batched docking on multiple GPUs.

        Args:
            smis: List of SMILES strings to dock.
            ligand_dir: List of paths to ligand directories for each GPU.
            ligand_dir_path: Base path to ligand directory.
            ligand_paths_by_smiles: Dictionary mapping SMILES to their file paths.
            output_dir_path: Path to directory for docking output.
            gpu_ids: List of GPU IDs to use.
            receptor_file: Path to receptor file.

        Returns:
            Tuple of (scores, docked_poses) or None.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def dock_smis(
        self, smi: List[str], receptor_file: str
    ) -> List[DockingOutput] | None:
        """
        Parameters:
        - smi: SMILES strings to perform docking. A single string activates single-ligand docking mode, while \
            multiple strings utilizes batched docking.
        """
        smi_list: List[str] = []
        for i in range(len(smi)):
            mol = Chem.MolFromSmiles(smi[i])
            if mol is not None:
                smi_list.append(
                    Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                )
            else:
                smi_list.append(smi[i])
        if len(smi_list) > 0:
            out = self._batched_docking(smis=smi_list, receptor_file=receptor_file)
            for s, d in zip(smi, out):
                d.smi = s
            return out
        else:
            return None


@ray.remote
class AutoDockGPUDocking(BaseDocking):
    """AutoDock GPU implementation of docking."""

    def __init__(
        self,
        cmd: str,
        n_conformers: int = 1,
        get_pose_str: bool = False,
        timeout_duration: Optional[int] = None,
        additional_args: Optional[Dict[str, Any]] = None,
        preparator: Optional[Callable[[List[str], List[List[str]]], List[bool]]] = None,
        cwd: Optional[str] = None,
        docking_attempts: int = 10,
        print_msgs: bool = True,
        print_output: bool = False,
        debug: bool = True,
        agg_type: Literal["mean", "min", "cluster_min"] = "min",
        docking_concurrency_per_gpu: int = 8,
    ) -> None:
        """Initialize AutoDock GPU docking.

        Args:
            cmd: Command line prefix for AutoDock GPU.
            n_conformers: Number of conformers to generate per molecule.
            get_pose_str: Return output pose as string (True) or not (False).
            timeout_duration: Timeout in seconds before process automatically stops.
            additional_args: Dictionary of additional command arguments.
            preparator: Function/Class callable to prepare molecule for docking.
            cwd: Change current working directory.
            docking_attempts: Number of docking attempts to make on each GPU.
            print_msgs: Show Python print messages in console (True) or not (False).
            print_output: Show docking output in console (True) or not (False).
            debug: Profiling the docking process and ligand preparation.
            agg_type: Aggregation type for docking scores ('mean', 'min', 'cluster_min').
            docking_concurrency_per_gpu: Number of concurrent docking processes per GPU.
        """
        super().__init__(
            cmd=cmd,
            n_conformers=n_conformers,
            get_pose_str=get_pose_str,
            timeout_duration=timeout_duration,
            additional_args=additional_args,
            preparator=preparator,
            cwd=cwd,
            docking_attempts=docking_attempts,
            print_msgs=print_msgs,
            print_output=print_output,
            debug=debug,
            docking_concurrency_per_gpu=docking_concurrency_per_gpu,
        )
        self.cmd = cmd + " --filelist {filelist}" + " --ffile {ffile}"  # grid_map_file
        if additional_args is not None:
            for k, v in additional_args.items():
                self.cmd += f" {k} {v} "
        self.agg_type = agg_type

    def _batched_docking_run(
        self,
        smis: List[str],
        ligand_dir: List[str],
        ligand_dir_path: str,
        ligand_paths_by_smiles: Dict[str, List[str]],
        output_dir_path: str,
        gpu_ids: List[int],
        receptor_file: str,
    ) -> List[DockingOutput]:
        """Run batched docking on multiple GPUs for AutoDock GPU.

        Args:
            smis: List of SMILES strings to dock.
            ligand_dir: List of paths to ligand directories for each GPU.
            ligand_dir_path: Base path to ligand directory.
            ligand_paths_by_smiles: Dictionary mapping SMILES to their file paths.
            output_dir_path: Path to directory for docking output.
            gpu_ids: List of GPU IDs to use.
            receptor_file: Path to receptor file.

        Returns:
            Tuple of (scores, docked_poses) or None.
        """
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
                    receptor_file=receptor_file,
                )
            else:
                self._run_docking(
                    ligand_dir,
                    cmd_prefixes=cmd_prefixes,
                    blocking=False,
                    receptor_file=receptor_file,
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

        binding_scores_dict: Dict[str, float | None] = {}
        # Move all output files to a TempDir
        move_files_from_dir(ligand_dir_path, output_dir_path, include_only=output_paths)
        output_paths = [
            os.path.abspath(f"{output_dir_path}/" + os.path.basename(file))
            for file in output_paths
        ]

        binding_results_dict: Dict[str, DockingOutput] = {}
        ligand_identifier_to_smi = {
            v[0].split("/")[-1].split(".")[0]: smi
            for smi, v in ligand_paths_by_smiles.items()
            if len(v) > 0
        }
        for file in output_paths:
            binding_score_val: float | None
            try:
                ligand_dict = parse_single_dlg(file)
                identifier = ligand_dict["ligname"]
            except ZeroDivisionError:  # We catch the identifier
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
                            )
                            break
                binding_score_val = 0.0
            smi = ligand_identifier_to_smi[identifier]
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
            # We cap the binding score to a maximum value  of 0.0 kcal/mol
            binding_score_val = (
                min(binding_score_val, 0.0) if binding_score_val is not None else None
            )
            assert smi not in binding_scores_dict
            binding_results_dict[smi] = DockingOutput(
                score=binding_score_val if binding_score_val is not None else 0.0,
                smi=smi,
                pose=self._get_output_pose(file) if self.get_pose_str else None,
                success=binding_score_val is not None,
            )
        for smi in smis:
            if smi not in binding_results_dict:
                binding_results_dict[smi] = DockingOutput(
                    score=0.0,
                    smi=smi,
                    pose=None,
                    success=False,
                )
        return [binding_results_dict[smi] for smi in smis]

    @staticmethod
    def _get_output_pose(output_path: str) -> Union[str, None]:
        """Get the output pose from a docking output file.

        Args:
            output_path: Path to the docking output file.

        Returns:
            The docked PDBQT string, or None if file not found.
        """
        try:
            with open(output_path) as f:
                docked_pdbqt = f.read()
            return docked_pdbqt
        except FileNotFoundError:
            return None

    def _run_docking(
        self,
        ligand_dir: List[str],
        blocking: bool,
        receptor_file: str,
        cmd_prefixes: Optional[List[str]] = None,
    ) -> None:
        """Run docking in separate shell process(es).

        Args:
            ligand_dir: List of paths to ligand directories.
            blocking: Whether to wait for docking to complete.
            receptor_file: Path to receptor file.
            cmd_prefixes: Optional list of command prefixes for each GPU.
        """
        grid_map_file = os.path.abspath(receptor_file.replace(".pdb", "_ag.maps.fld"))
        assert os.path.exists(grid_map_file), "Grid map file not found"

        if cmd_prefixes is not None:
            assert len(cmd_prefixes) == len(ligand_dir)

        procs = []

        for i in range(len(ligand_dir)):
            if cmd_prefixes is not None and cmd_prefixes[i] is not None:
                cmd_str = cmd_prefixes[i] + self.cmd.format(
                    filelist=ligand_dir[i], ffile=grid_map_file
                )
            else:
                cmd_str = self.cmd.format(filelist=ligand_dir[i], ffile=grid_map_file)

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
        """Check if docking is complete for all ligand directories.

        Args:
            ligand_dir: List of paths to ligand directories.

        Returns:
            True if docking is complete for all directories.
        """
        return all(
            len([f for f in os.listdir(lg_dir) if f.endswith(".dlg")]) > 0
            for lg_dir in ligand_dir
        )
