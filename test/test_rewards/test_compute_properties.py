"""Tests for MolecularVerifier.compute_properties and server compute_properties endpoint."""

from typing import Optional

import pytest
import requests

from mol_gen_docking.reward import MolecularVerifier, MolecularVerifierConfigModel
from mol_gen_docking.utils.property_utils import CLASSICAL_PROPERTIES_NAMES


# =============================================================================
# Unit Tests for MolecularVerifier.compute_properties
# =============================================================================


@pytest.fixture(scope="module")  # type: ignore
def verifier() -> MolecularVerifier:
    """Create a minimal MolecularVerifier for property computation tests."""
    return MolecularVerifier(MolecularVerifierConfigModel())


class TestComputePropertiesMethod:
    """Tests for MolecularVerifier.compute_properties."""

    def test_valid_smiles_all_properties(self, verifier: MolecularVerifier) -> None:
        """Valid SMILES with all available properties should return all float values."""
        smiles = "CCO"  # Ethanol
        properties = list(CLASSICAL_PROPERTIES_NAMES.values())
        result = verifier.compute_properties(smiles, properties)

        assert set(result.keys()) == set(properties)
        for prop, value in result.items():
            assert value is not None, f"Property {prop} should not be None for valid SMILES"
            assert isinstance(value, float), f"Property {prop} should be a float"

    def test_valid_smiles_subset_of_properties(self, verifier: MolecularVerifier) -> None:
        """Valid SMILES with a subset of properties returns correct values."""
        smiles = "c1ccccc1"  # Benzene
        properties = ["QED", "SA", "logP"]
        result = verifier.compute_properties(smiles, properties)

        assert set(result.keys()) == {"QED", "SA", "logP"}
        for value in result.values():
            assert value is not None
            assert isinstance(value, float)

    def test_invalid_smiles_returns_none(self, verifier: MolecularVerifier) -> None:
        """Invalid SMILES string should return None for all requested properties."""
        smiles = "NOT_A_VALID_SMILES"
        properties = ["QED", "SA", "logP"]
        result = verifier.compute_properties(smiles, properties)

        assert set(result.keys()) == {"QED", "SA", "logP"}
        for value in result.values():
            assert value is None

    def test_unknown_property_returns_none(self, verifier: MolecularVerifier) -> None:
        """Unknown property name should return None for that property."""
        smiles = "CCO"
        properties = ["QED", "NOT_A_PROPERTY"]
        result = verifier.compute_properties(smiles, properties)

        assert result["QED"] is not None
        assert result["NOT_A_PROPERTY"] is None

    def test_empty_properties_list(self, verifier: MolecularVerifier) -> None:
        """Empty properties list should return empty dict."""
        smiles = "CCO"
        result = verifier.compute_properties(smiles, [])
        assert result == {}

    def test_qed_known_range(self, verifier: MolecularVerifier) -> None:
        """QED value should be between 0 and 1."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        result = verifier.compute_properties(smiles, ["QED"])
        assert result["QED"] is not None
        assert 0.0 <= result["QED"] <= 1.0

    def test_sa_known_range(self, verifier: MolecularVerifier) -> None:
        """SA score should be between 1 and 10."""
        smiles = "CCO"  # Ethanol - simple molecule
        result = verifier.compute_properties(smiles, ["SA"])
        assert result["SA"] is not None
        assert 1.0 <= result["SA"] <= 10.0


# =============================================================================
# Server Tests for /compute_properties endpoint
# =============================================================================


class TestComputePropertiesEndpoint:
    """Tests for the /compute_properties MCP endpoint."""

    def test_valid_smiles(self, ensure_server: str) -> None:
        """Valid SMILES should return float values for all requested properties."""
        response = requests.post(
            f"{ensure_server}/compute_properties",
            json={"smiles": "CCO", "properties": ["QED", "SA", "logP"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert set(result.keys()) == {"QED", "SA", "logP"}
        for value in result.values():
            assert value is not None
            assert isinstance(value, float)

    def test_invalid_smiles(self, ensure_server: str) -> None:
        """Invalid SMILES should return null values for all properties."""
        response = requests.post(
            f"{ensure_server}/compute_properties",
            json={"smiles": "NOT_VALID", "properties": ["QED", "SA"]},
        )
        assert response.status_code == 200
        result = response.json()
        for value in result.values():
            assert value is None

    def test_empty_properties(self, ensure_server: str) -> None:
        """Empty properties list should return empty dict."""
        response = requests.post(
            f"{ensure_server}/compute_properties",
            json={"smiles": "CCO", "properties": []},
        )
        assert response.status_code == 200
        result = response.json()
        assert result == {}

    def test_known_property_values(self, ensure_server: str) -> None:
        """Computed values should match expected ranges."""
        response = requests.post(
            f"{ensure_server}/compute_properties",
            json={"smiles": "CC(=O)Oc1ccccc1C(=O)O", "properties": ["QED"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["QED"] is not None
        assert 0.0 <= result["QED"] <= 1.0
