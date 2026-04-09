import json
import os
import warnings
from itertools import chain
from typing import Any, List

import pyscreener as ps
import ray
from pyscreener.docking import DockingVirtualScreen, get_runner
from tdc.metadata import docking_target_info
from tdc.utils import receptor_load
from tqdm import tqdm


class DockingVirtualScreenWithTimeout(DockingVirtualScreen):
    def __init__(self, timeout: int = 60, *args: Any, **kwargs: Any) -> None:
        """Initialize docking virtual screen with timeout.

        Args:
            timeout: Timeout in seconds for docking simulations.
            *args: Additional positional arguments passed to parent class.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def run(self, simulationss: List[List[Any]]) -> List[List[Any]]:
        """Run docking simulations with timeout handling.

        Args:
            simulationss: List of lists of simulation parameters.

        Returns:
            List of lists of docking results.
        """
        refss = [
            [self.prepare_and_run.remote(s) for s in sims] for sims in simulationss
        ]
        resultss = []
        for refs in tqdm(refss, desc="Docking", unit="ligand", smoothing=0.0):
            # Wait for all docking simulations to complete with a timeout
            try:
                resultss.append(ray.get(refs, timeout=self.timeout))
            except ray.exceptions.GetTimeoutError:
                warnings.warn(
                    "Docking simulations timed out. Returning None for results."
                )
                resultss.append([None] * len(refs))
                continue

        self.run_simulationss.extend(simulationss)
        self.resultss.extend(resultss)
        self.num_ligands += len(resultss)
        self.num_simulations += len(list(chain(*resultss)))

        return resultss


class PyscreenerOracle:
    def __init__(
        self,
        target_name: str,
        path_to_data: str,
        software_class: str = "vina",
        n_cpu: int = 16,
        exhaustiveness: int = 8,
        **kwargs: Any,
    ):
        """Initialize PyScreener docking oracle.

        Args:
            target_name: Name of the docking target.
            path_to_data: Path to directory containing receptor files and pocket info.
            software_class: Docking software to use (vina, qvina, smina, etc.).
            n_cpu: Number of CPUs to use for docking.
            exhaustiveness: Docking exhaustiveness parameter.
            **kwargs: Additional keyword arguments for docking configuration.

        Raises:
            ValueError: If software_class is not implemented.
        """
        if software_class not in [
            "vina",
            "qvina",
            "smina",
            "psovina",
            "dock",
            "dock6",
            "ucsfdock",
        ]:
            raise ValueError(
                'The value of software_class is not implemented. Currently available:["vina", "qvina", "smina", "psovina", "dock", "dock6", "ucsfdock"]'
            )

        self.name = "[DOCKING]-" + target_name
        if (
            not os.path.isfile(target_name)
            and target_name.endswith("docking")
            or target_name.endswith("docking_vina")
        ):
            pdbid = target_name.split("_")[0]
            receptor_load(pdbid)
            receptor_pdb_file = "./oracle/" + pdbid + ".pdbqt"
            box_center = docking_target_info[pdbid]["center"]
            box_size = tuple([s for s in docking_target_info[pdbid]["size"]])
        else:
            pdb_id = target_name
            receptor_pdb_file = os.path.join(
                path_to_data, "pdb_files", f"{target_name}.pdb"
            )
            with open(os.path.join(path_to_data, "pockets_info.json")) as f:
                pockets_info = json.load(f)
            box_center = tuple(pockets_info[pdb_id]["center"])
            box_size = tuple(pockets_info[pdb_id]["size"])

        if not ray.is_initialized():
            ray.init()

        if hasattr(ps, "build_metadata"):
            metadata = ps.build_metadata(
                software_class, metadata={"exhaustiveness": exhaustiveness}
            )
        else:
            raise OSError(
                "Pyscreener version is not compatible. Please update to the latest version."
            )

        if hasattr(ps, "virtual_screen"):
            self.scorer = DockingVirtualScreenWithTimeout(
                120,  # Set a timeout of 2m for each docking simulation
                get_runner(software_class),
                [receptor_pdb_file],
                box_center,
                box_size,
                metadata,
                ncpu=n_cpu,
            )
        else:
            raise OSError(
                "Pyscreener version is not compatible. Please update to the latest version."
            )

    def __call__(
        self, test_smiles: str | List[str], error_value: float | None = None
    ) -> Any:
        """Compute docking scores for SMILES string(s).

        Args:
            test_smiles: SMILES string or list of SMILES strings to dock.
            error_value: Value to return when docking fails.

        Returns:
            Docking score or list of docking scores.
        """
        final_score = self.scorer(test_smiles)

        if isinstance(test_smiles, str):
            return list(final_score)[0]
        else:
            score_lst = []
            for i, smiles in enumerate(test_smiles):
                score = final_score[i]
                if score is None:
                    warnings.warn(f"Docking failed for {smiles}.")
                    score = error_value
                score_lst.append(score)
            return score_lst
