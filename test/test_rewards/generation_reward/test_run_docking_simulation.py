"""Tests for the run_docking_simulation MCP tool.

These tests cover:
- The new ``dock_and_save`` method on ``BaseDocking`` / ``AutoDockGPUDocking``
- The ``RunDockingSimulationInput`` Pydantic model
- The ``/run_docking_simulation`` server endpoint (requires a running server)
"""

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from mol_gen_docking.reward.verifiers.generation_reward.oracles.docking_utils.docking_soft import (
    BaseDocking,
    AutoDockGPUDocking,
)
from mol_gen_docking.server_utils.server_endpoints.mcp_endpoints import (
    RunDockingSimulationInput,
)


# =============================================================================
# Unit tests: BaseDocking.dock_and_save signature & basic behaviour
# =============================================================================


class TestDockAndSaveMethod:
    """Tests that BaseDocking exposes the dock_and_save method correctly."""

    def test_dock_and_save_exists(self) -> None:
        """BaseDocking must expose a dock_and_save method."""
        assert hasattr(BaseDocking, "dock_and_save"), (
            "BaseDocking is missing the dock_and_save method"
        )

    def test_dock_and_save_returns_none_for_empty_smiles(self, tmp_path: Any) -> None:
        """dock_and_save must return None when given an empty SMILES list."""
        # Create a minimal concrete subclass that is never actually run.
        class _NeverRun(BaseDocking):
            def _batched_docking_run(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("Should not be called for empty input")

        # We need a mock receptor file so __init__ doesn't raise.
        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("ATOM 1 ...\n")
        instance = _NeverRun(cmd="echo", receptor_file=str(receptor))
        result = instance.dock_and_save([], output_dlg_dir=str(tmp_path))
        assert result is None

    def test_batched_docking_copies_files_to_output_dir(self, tmp_path: Any) -> None:
        """_batched_docking must copy output files to output_dlg_dir before cleanup."""

        class _FakeDocking(BaseDocking):
            """Simulates docking by writing a dummy .dlg file to output_dir_path."""

            def _batched_docking_run(
                self,
                smis: Any,
                ligand_dir: Any,
                ligand_dir_path: Any,
                ligand_paths_by_smiles: Any,
                output_dir_path: Any,
                gpu_ids: Any = None,
            ) -> Any:
                # Write a fake .dlg file to output_dir_path.
                fake_dlg = os.path.join(output_dir_path, "result.dlg")
                with open(fake_dlg, "w") as fh:
                    fh.write("DOCKED: fake result\n")
                return ([None], [None])

        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("ATOM 1 ...\n")
        persistent_dir = tmp_path / "persistent"
        persistent_dir.mkdir()

        with patch.object(_FakeDocking, "_prepare_ligands", return_value=[True]):
            with patch.object(
                _FakeDocking,
                "_batched_prepare_ligands",
                return_value=({"CCO": ["dummy.pdbqt"]}, [str(tmp_path)]),
            ):
                instance = _FakeDocking(cmd="echo", receptor_file=str(receptor))
                instance._batched_docking(
                    ["CCO"],
                    gpu_ids=[0],
                    output_dlg_dir=str(persistent_dir),
                )

        dlg_files = list(persistent_dir.glob("*.dlg"))
        assert len(dlg_files) == 1, (
            "Expected exactly one .dlg file to be copied to the persistent dir"
        )


# =============================================================================
# Unit tests: RunDockingSimulationInput Pydantic model
# =============================================================================


class TestRunDockingSimulationInput:
    """Tests for the RunDockingSimulationInput Pydantic model."""

    def test_minimal_valid_input(self) -> None:
        """Mandatory fields must be accepted without optional fields."""
        inp = RunDockingSimulationInput(
            smiles=["CCO"],
            receptor_name="1abc",
        )
        assert inp.smiles == ["CCO"]
        assert inp.receptor_name == "1abc"
        assert inp.output_dir is None
        assert inp.gpu_ids == [0]

    def test_full_valid_input(self) -> None:
        """All fields must be accepted together."""
        inp = RunDockingSimulationInput(
            smiles=["CCO", "c1ccccc1"],
            receptor_name="2xyz",
            output_dir="/tmp/docking_run",
            gpu_ids=[0, 1],
        )
        assert inp.output_dir == "/tmp/docking_run"
        assert inp.gpu_ids == [0, 1]

    def test_missing_required_fields_raises(self) -> None:
        """Missing ``smiles`` or ``receptor_name`` must raise a validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunDockingSimulationInput(receptor_name="1abc")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            RunDockingSimulationInput(smiles=["CCO"])  # type: ignore[call-arg]


# =============================================================================
# Server tests: /run_docking_simulation endpoint
# =============================================================================


@pytest.fixture(scope="module")  # type: ignore
def ensure_server(uvicorn_server: Any, server_url: str) -> str:
    """Ensure the reward server is reachable before running server-based tests."""
    try:
        response = requests.get(f"{server_url}/liveness", timeout=5)
        if response.status_code == 200:
            return server_url
    except requests.exceptions.RequestException:
        pass

    pytest.skip("Reward server is not available")
    return ""


class TestRunDockingSimulationEndpoint:
    """Server-level tests for /run_docking_simulation."""

    def test_returns_400_when_not_autodock_gpu(self, ensure_server: str) -> None:
        """The endpoint must return HTTP 400 when docking_oracle != 'autodock_gpu'."""
        # Check the server settings exposed by the liveness endpoint first.
        # If the server *is* using autodock_gpu we simply skip this test.
        import os

        if os.environ.get("DOCKING_ORACLE", "autodock_gpu") == "autodock_gpu":
            pytest.skip(
                "Server is configured with autodock_gpu; "
                "cannot test the 400 path here."
            )

        response = requests.post(
            f"{ensure_server}/run_docking_simulation",
            json={"smiles": ["CCO"], "receptor_name": "1abc"},
        )
        assert response.status_code == 400
        assert "autodock_gpu" in response.json()["detail"].lower()

    def test_returns_404_for_unknown_receptor(self, ensure_server: str) -> None:
        """The endpoint must return HTTP 404 for a non-existent receptor name."""
        import os

        if os.environ.get("DOCKING_ORACLE", "autodock_gpu") != "autodock_gpu":
            pytest.skip("Server is not using autodock_gpu; skipping receptor test.")

        response = requests.post(
            f"{ensure_server}/run_docking_simulation",
            json={
                "smiles": ["CCO"],
                "receptor_name": "this_receptor_does_not_exist_12345",
            },
        )
        assert response.status_code == 404
        assert "not a known docking target" in response.json()["detail"]
