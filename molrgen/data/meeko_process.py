"""A script to process receptors using Meeko in order to prepare them for molecular docking with AutoDock-GPU."""

import argparse
import json
import logging
import os
import re
import subprocess as sp
from typing import Any, Dict, Tuple

import numpy as np
import ray
from Bio.PDB import PDBParser
from ray.experimental import tqdm_ray


def residue_in_box(
    residue: Any, box_center: list[float], box_size: list[float]
) -> bool:
    """
    Check if any atom of the residue is inside the box.
    box_center: (x, y, z)
    box_size: (sx, sy, sz)
    """
    box_center_arr = np.array(box_center)
    box_min = box_center_arr - np.array(box_size) / 2
    box_max = box_center_arr + np.array(box_size) / 2

    for atom in residue.get_atoms():
        coord = atom.get_coord()
        if np.all(coord >= box_min) and np.all(coord <= box_max):
            return True
    return False


def check_failed_residues_in_box(
    pdb_file: str,
    failed_residues: set[str],
    box_center: list[float],
    box_size: list[float],
) -> list[str]:
    """
    Return the list of failed residues that are inside the docking box.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    residues_in_box = []

    for chain in structure.get_chains():
        for res in chain:
            res_id = f"{chain.id}:{res.get_id()[1]}"
            if res_id in failed_residues and residue_in_box(res, box_center, box_size):
                residues_in_box.append(res_id)

    return residues_in_box


class ReceptorProcess:
    """Receptor processing pipeline for preparing proteins for AutoDock-GPU molecular docking.

    This class provides a complete workflow for processing receptor PDB files through
    Meeko's `mk_prepare_receptor.py` tool and AutoGrid4. It handles common issues like
    unrecognized residues and heteroatoms, and supports parallel processing via Ray.

    The processing pipeline consists of three main steps:

    1. **Preprocessing (optional)**: Remove heteroatoms (ligands, waters, ions) from the
       receptor using PDBFixer/OpenMM.

    2. **Meeko processing**: Convert PDB to PDBQT format and generate grid parameter
       files (.gpf) with specified docking box dimensions.

    3. **AutoGrid computation**: Precompute affinity maps for faster docking.

    The class handles edge cases gracefully:

    - Skips already-processed receptors (checks for `_ag.pdbqt` and `_ag.maps.fld`)
    - Retries with `--allow_bad_res` flag if unrecognized residues are outside the
      docking box (non-critical)
    - Reports critical errors when unrecognized residues are inside the docking box

    Attributes:
        data_path (str): Root directory containing receptor data.
        pre_process_receptors (bool): Whether to remove heteroatoms before Meeko.
        logger (logging.Logger): Logger instance for this class.
        receptor_path (str): Path to directory containing PDB files.
        pockets (Dict[str, Dict[str, Any]]): Docking box specifications loaded from
            `pockets_info.json`. Each entry contains 'center' and 'size' keys.
        cmd (str): Template command for mk_prepare_receptor.py.

    Example:
        ```python
        import ray
        ray.init()

        processor = ReceptorProcess(
            data_path="/path/to/data",
            pre_process_receptors=True
        )

        # Process all receptors defined in pockets_info.json
        warnings, errors = processor.process_receptors(use_pbar=True)

        # Process specific receptors
        warnings, errors = processor.process_receptors(
            receptors=["1abc", "2def"],
            allow_bad_res=True
        )
        ```

    Note:
        Requires Ray to be initialized before calling `process_receptors()`.
        External dependencies: Meeko, AutoGrid4, OpenMM (optional), PDBFixer (optional).
    """

    def __init__(self, data_path: str, pre_process_receptors: bool = False) -> None:
        """Initialize the receptor processor with data paths and configuration.

        Validates that required files and directories exist, and loads the docking
        box specifications from `pockets_info.json`.

        Args:
            data_path (str): Path to the data directory. Must contain:

                - `pdb_files/`: Directory with receptor PDB files
                - `pockets_info.json`: JSON file mapping receptor names to docking
                  box specifications with 'center' [x, y, z] and 'size' [sx, sy, sz]

            pre_process_receptors (bool): If True, remove heteroatoms (ligands,
                waters, ions) from receptors before Meeko processing. Uses PDBFixer.
                Default: False.

        Raises:
            AssertionError: If `data_path` does not exist.
            AssertionError: If `pdb_files/` subdirectory does not exist.
            AssertionError: If `pockets_info.json` does not exist.
        """
        self.data_path: str = data_path
        self.pre_process_receptors: bool = pre_process_receptors
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )
        self.receptor_path = os.path.join(self.data_path, "pdb_files")

        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        assert os.path.exists(self.receptor_path), (
            f"Receptor path {self.receptor_path} does not exist."
        )
        assert os.path.exists(os.path.join(data_path, "pockets_info.json")), (
            f"Pockets info file does not exist in {data_path}."
        )

        with open(os.path.join(data_path, "pockets_info.json")) as f:
            self.pockets: Dict[str, Dict[str, Any]] = json.load(f)
        self.cmd = "mk_prepare_receptor.py -i {INPUT} -o {OUTPUT} -p -g --box_size {BOX_SIZE} --box_center {BOX_CENTER}"

    def _run_meeko(
        self, input_path: str, receptor: str, bad_res: bool = False
    ) -> Tuple[str, int, str]:
        """Execute Meeko's mk_prepare_receptor.py on a single receptor.

        Runs the Meeko command to convert a PDB file to PDBQT format and generate
        grid parameter files for AutoDock-GPU. The docking box dimensions are
        retrieved from the loaded pockets configuration.

        Args:
            input_path (str): Full path to the input PDB file.
            receptor (str): Receptor identifier (key in pockets dict) used to
                look up docking box center and size.
            bad_res (bool): If True, add `--allow_bad_res` flag to permit
                unrecognized residues. Default: False.

        Returns:
            Tuple[str, int, str]: A tuple containing:

                - stderr_text: Standard error output from Meeko (useful for
                  parsing error messages about unrecognized residues)
                - returncode: Process return code (0 = success)
                - output_path: Path to output files (without extension)

        Raises:
            subprocess.TimeoutExpired: If Meeko takes longer than 300 seconds.
        """
        output_path = input_path.replace(".pdb", "_ag")

        box_center = " ".join([str(x) for x in self.pockets[receptor]["center"]])
        box_size = " ".join([str(x) for x in self.pockets[receptor]["size"]])

        command = self.cmd.format(
            INPUT=input_path,
            OUTPUT=output_path,
            BOX_SIZE=box_size,
            BOX_CENTER=box_center,
        )
        if bad_res:
            command += " --allow_bad_res"

        self.logger.info(f"Running command: {command}")
        process = sp.Popen(
            command,
            shell=True,
            stdout=sp.DEVNULL,
            stderr=sp.PIPE,
            preexec_fn=os.setpgrp,
        )
        _, stderr = process.communicate(
            timeout=300,
        )

        stderr_text = stderr.decode("utf-8")

        return stderr_text, process.returncode, output_path

    def meeko_process(
        self, receptor: str, allow_bad_res: bool = False
    ) -> Tuple[int, str]:
        """Process a receptor with Meeko, handling unrecognized residue errors.

        This method implements smart error handling for Meeko failures:

        1. First attempts processing without `--allow_bad_res`
        2. If unrecognized residues are found, checks if they're inside the docking box
        3. If outside the box: retries with `--allow_bad_res` (returns 0)
        4. If inside the box: retries with `--allow_bad_res` but flags as warning (returns 1)

        Args:
            receptor (str): Receptor identifier matching a key in `pockets_info.json`
                and a file `{receptor}.pdb` in the pdb_files directory.
            allow_bad_res (bool): If True, always use `--allow_bad_res` flag.
                Default: False.

        Returns:
            Tuple[int, str]: A tuple containing:

                - status code:
                    - 0: Success (no issues or issues outside docking box)
                    - 1: Warning (unrecognized residues inside docking box)
                - output_path: Path to the processed output files (without extension)

        Raises:
            ValueError: If Meeko fails for reasons other than unrecognized residues.
        """
        input_path = os.path.join(self.receptor_path, f"{receptor}.pdb")
        stderr_text, returncode, output_path = self._run_meeko(
            input_path, receptor, allow_bad_res
        )
        if returncode != 0:
            # Check if failing resiudes are inside the docking box
            failed_residues = set()
            for line in stderr_text.splitlines():
                match = re.search(
                    r"No template matched for residue_key='(\w+:\d+)'", line
                )
                if match:
                    failed_residues.add(match.group(1))

            if failed_residues:
                residues_in_box = check_failed_residues_in_box(
                    input_path,
                    failed_residues,
                    self.pockets[receptor]["center"],
                    self.pockets[receptor]["size"],
                )
                if residues_in_box == []:
                    _, returncode, output_path = self._run_meeko(
                        input_path, receptor, bad_res=True
                    )
                    return returncode, output_path
                else:
                    _, returncode, output_path = self._run_meeko(
                        input_path, receptor, bad_res=True
                    )
                    return 1, output_path
            else:
                raise ValueError(
                    f"Error in Meeko processing {receptor}:\n{stderr_text}"
                )
        return 0, output_path

    def _run_autogrid(self, path: str) -> None:
        """Run AutoGrid4 to precompute affinity maps for a processed receptor.

        Executes `autogrid4` on the grid parameter file (.gpf) generated by Meeko.
        The resulting map files are required for AutoDock-GPU docking.

        Args:
            path (str): Path to the receptor files (with or without extension).
                The basename is extracted and used to locate the .gpf file
                in the receptor_path directory.

        Raises:
            RuntimeError: If AutoGrid4 returns a non-zero exit code.

        Note:
            AutoGrid4 must be installed and available in PATH.
        """
        path = os.path.basename(path)
        grid_command = f"autogrid4 -p {path}.gpf -l {path}.glg -d"
        self.logger.info(f"Running command: {grid_command}")
        process = sp.Popen(
            grid_command,
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=self.receptor_path,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            self.logger.error(
                f"Error in running AutoGrid on {path}:\n{stderr.decode()}"
            )
            raise RuntimeError(f"AutoGrid failed for {path}")
        self.logger.info(f"Successfully ran AutoGrid on {path}")

    def remove_heterogenous(self, receptor: str) -> None:
        """Remove heteroatoms from a receptor PDB file using PDBFixer.

        Removes all heteroatoms including ligands, water molecules, and ions
        from the receptor structure. The cleaned structure overwrites the
        original PDB file.

        Args:
            receptor (str): Receptor identifier. The file
                `{receptor_path}/{receptor}.pdb` will be modified in-place.

        Note:
            Requires OpenMM and PDBFixer packages.

        Warning:
            This method modifies the original PDB file. Make a backup if needed.
        """
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer

        pdb_file = os.path.join(self.receptor_path, f"{receptor}.pdb")

        fixer = PDBFixer(filename=pdb_file)
        fixer.removeHeterogens(True)

        PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_file, "w"))

    def process_receptors(
        self,
        receptors: list[str] = [],
        allow_bad_res: bool = False,
        use_pbar: bool = False,
    ) -> Tuple[list[str], list[str]]:
        """Process multiple receptors in parallel using Ray.

        Orchestrates the full processing pipeline for a batch of receptors:

        1. Filters out already-processed receptors (with existing `_ag.pdbqt`
           and `_ag.maps.fld` files)
        2. Optionally removes heteroatoms (if `pre_process_receptors=True`)
        3. Runs Meeko processing with error recovery
        4. Runs AutoGrid4 for successful conversions
        5. Collects and categorizes failures

        Processing is parallelized using Ray remote functions, with each receptor
        allocated 4 CPU cores.

        Args:
            receptors (list[str]): List of receptor identifiers to process.
                If empty, processes all receptors in `pockets_info.json`.
                Default: [].
            allow_bad_res (bool): If True, always use `--allow_bad_res` flag
                in Meeko. Default: False.
            use_pbar (bool): If True, display a tqdm progress bar via Ray.
                Default: False.

        Returns:
            Tuple[list[str], list[str]]: Two lists of receptor identifiers:

                - warnings: Receptors with non-critical issues (unrecognized
                  residues inside docking box, but processing completed)
                - errors: Receptors with critical failures (exceptions during
                  processing)

        Raises:
            AssertionError: If any receptor in the list is not found in
                `pockets_info.json`.

        Note:
            Ray must be initialized before calling this method.
        """

        @ray.remote(num_cpus=4)
        def process_receptor(
            receptor: str,
            pbar: Any,
            allow_bad_res: bool = False,
        ) -> int:
            """
            Outputs the level of the error:
            0: Success
            1: Failed residues inside the box
            2: Other errors (critical)
            """
            try:
                if self.pre_process_receptors:
                    self.remove_heterogenous(receptor)
                result, processed_path = self.meeko_process(receptor, allow_bad_res)
                if result <= 1:
                    self._run_autogrid(processed_path)
                if pbar is not None:
                    pbar.update.remote(1)
            except Exception as e:
                self.logger.error(f"Error processing {receptor}: {e}")
                if pbar is not None:
                    pbar.update.remote(1)
                return 2
            return result

        if receptors == []:
            receptors = list(self.pockets.keys())
        else:
            assert all(r in self.pockets for r in receptors), (
                "Some receptors are not in pockets_info.json: "
                + ",".join([r for r in receptors if r not in self.pockets])
            )

        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        if use_pbar:
            pbar = remote_tqdm.remote(total=len(receptors), desc="Processing receptors")
        else:
            pbar = None
        # Find receptors that already have a _ag.pdbqt and_ag.maps.fld file
        receptors_to_process = []
        for receptor in receptors:
            ag_pdbqt_path = os.path.join(self.receptor_path, f"{receptor}_ag.pdbqt")
            ag_maps_fld_path = os.path.join(
                self.receptor_path, f"{receptor}_ag.maps.fld"
            )
            if not (os.path.exists(ag_pdbqt_path) and os.path.exists(ag_maps_fld_path)):
                receptors_to_process.append(receptor)
            else:
                self.logger.info(f"Receptor {receptor} already processed. Skipping.")
                if use_pbar:
                    pbar.update.remote(1)  # type: ignore
        if len(receptors_to_process) == 0:
            self.logger.info("All receptors already processed.")
            if use_pbar:
                pbar.close.remote()  # type: ignore
            return [], []

        self.logger.info(f"Processing {len(receptors_to_process)} receptors.")
        futures = [
            process_receptor.remote(receptor, pbar, allow_bad_res)
            for receptor in receptors_to_process
        ]
        results = ray.get(futures)

        missed_receptors_1 = [
            receptor
            for receptor, success in zip(receptors_to_process, results)
            if success == 1
        ]
        missed_receptors_2 = [
            receptor
            for receptor, success in zip(receptors_to_process, results)
            if success == 2
        ]
        return missed_receptors_1, missed_receptors_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process receptors using Meeko for AutoDock-GPU."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory containing pdb_files and pockets_info.json",
    )
    args = parser.parse_args()

    processor = ReceptorProcess(data_path=args.data_path, pre_process_receptors=True)
    missed_receptors_1, missed_receptors_2 = processor.process_receptors(use_pbar=True)
    print(
        "###########\n Missed receptors with critical errors: \n",
        len(missed_receptors_2),
    )
    print(missed_receptors_2)

    # Remove receptors with critical errors from pockets_info.json and docking_targets.json
    if len(missed_receptors_2) > 0:
        with open(os.path.join(args.data_path, "pockets_info.json")) as f:
            pockets_info = json.load(f)
        for receptor in missed_receptors_2:
            if receptor in pockets_info:
                del pockets_info[receptor]
        with open(os.path.join(args.data_path, "pockets_info.json"), "w") as f:
            json.dump(pockets_info, f, indent=4)

        if os.path.exists(os.path.join(args.data_path, "docking_targets.json")):
            with open(os.path.join(args.data_path, "docking_targets.json")) as f:
                docking_targets = json.load(f)
            docking_targets = [
                target for target in docking_targets if target not in missed_receptors_2
            ]
            with open(os.path.join(args.data_path, "docking_targets.json"), "w") as f:
                json.dump(docking_targets, f, indent=4)
