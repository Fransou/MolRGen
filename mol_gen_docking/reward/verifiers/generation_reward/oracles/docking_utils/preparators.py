# Adapted from code in PyVina https://github.com/Pixelatory/PyVina
# Copyright (c) 2024 Nicholas Aksamit
# Licensed under the MIT License

import abc
import logging
import os
from typing import Any, List, Optional

from meeko import AtomTyper, MoleculePreparation, PDBQTWriterLegacy, RDKitMoleculeSetup
from meeko.reactive import assign_reactive_types
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem


class RDKitMoleculeSetupNoSym(RDKitMoleculeSetup):
    """RDKit molecule setup without symmetry computation for performance."""

    @staticmethod
    def get_symmetries_for_rmsd(mol: Any, max_matches: int = 17) -> tuple:
        """We remove symmetry computation to reduce the overhead.

        Args:
            mol: RDKit molecule.
            max_matches: Maximum number of symmetry matches.

        Returns:
            Empty tuple (no symmetry computation).
        """
        return tuple()


class MoleculePreparator(MoleculePreparation):
    """Molecule preparator using RDKitMoleculeSetupNoSym for performance."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize molecule preparator with no-symmetry setup.

        Args:
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self._classes_setup = {Chem.rdchem.Mol: RDKitMoleculeSetupNoSym}

    def prepare(
        self,
        mol: Any,
        root_atom_index: Optional[int] = None,
        not_terminal_atoms: Optional[List[int]] = None,
        delete_ring_bonds: Optional[List[tuple[int, int]]] = None,
        glue_pseudo_atoms: Optional[dict] = None,
        conformer_id: int = -1,
    ) -> RDKitMoleculeSetup:
        """
        Create an RDKitMoleculeSetup from an RDKit Mol object.

        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
            An RDKit Mol with explicit hydrogens and 3D coordinates.
        root_atom_index: int
            Used to set ROOT of torsion tree instead of searching.
        not_terminal_atoms: list
            Makes bonds with terminal atoms rotatable (e.g. C-Alpha carbon in flexres).
        delete_ring_bonds: list[tuple[int, int]]
            Bonds deleted for macrocycle flexibility. Each bond is a tuple of two ints (atom 0-indices).
        glue_pseudo_atoms: dict
            Mapping from parent atom indices to coordinates.
        conformer_id: int

        Returns
        -------
        setups: list[RDKitMoleculeSetup]
            Returns a list of generated RDKitMoleculeSetups
        """
        if not_terminal_atoms is None:
            not_terminal_atoms = []
        if delete_ring_bonds is None:
            delete_ring_bonds = []
        if glue_pseudo_atoms is None:
            glue_pseudo_atoms = {}
        mol_type = type(mol)
        if mol_type not in self._classes_setup:
            raise TypeError(
                "Molecule is not an instance of supported types: %s" % type(mol)
            )
        setup_class = self._classes_setup[mol_type]
        setup = setup_class.from_mol(
            mol,
            keep_chorded_rings=self.keep_chorded_rings,
            keep_equivalent_rings=self.keep_equivalent_rings,
            assign_charges=self.charge_model == "gasteiger",
            conformer_id=conformer_id,
        )

        self.check_external_ring_break(setup, delete_ring_bonds, glue_pseudo_atoms)

        # 1.  assign atom params
        AtomTyper.type_everything(
            setup,
            self.atom_params,
            self.charge_model,
            self.offatom_params,
            self.dihedral_params,
        )

        # Convert molecule to graph and apply trained Espaloma model
        if self.dihedral_model == "espaloma" or self.charge_model == "espaloma":
            molgraph = self.espaloma_model.get_espaloma_graph(setup)

        # Grab dihedrals from graph node and set them to the molsetup
        if self.dihedral_model == "espaloma":
            self.espaloma_model.set_espaloma_dihedrals(setup, molgraph)

        # Grab charges from graph node and set them to the molsetup
        if self.charge_model == "espaloma":
            self.espaloma_model.set_espaloma_charges(setup, molgraph)

        # merge hydrogens (or any terminal atoms)
        indices = set()
        for atype_to_merge in self.merge_these_atom_types:
            for atom in setup.atoms:
                if atom.atom_type == atype_to_merge:
                    indices.add(atom.index)
        setup.merge_terminal_atoms(indices)

        # 3.  assign bond types
        #     - all single bonds rotatable except some amides and SMARTS rigidification
        #     - macrocycle code breaks rings only at rotatable bonds
        #     - bonds in rigid rings are set as non-rotatable after the flex_model is built
        self._bond_typer(
            setup,
            self.flexible_amides,
            self.rigidify_bonds_smarts,
            self.rigidify_bonds_indices,
        )

        # 4 . hydrate molecule
        if self.hydrate:
            self._water_builder.hydrate(setup)

        self.calc_flex(
            setup,
            root_atom_index,
            not_terminal_atoms,
            delete_ring_bonds,
            glue_pseudo_atoms,
        )

        if self.reactive_smarts is None:
            setups = [setup]
        else:
            reactive_types_dicts = assign_reactive_types(
                setup,
                self.reactive_smarts,
                self.reactive_smarts_idx,
            )

            if len(reactive_types_dicts) == 0:
                raise RuntimeError("reactive SMARTS didn't match")

            setups = []
            for r in reactive_types_dicts:
                new_setup = setup.copy()
                # There is no guarantee that the addition order in the dictionary will be the correct order to
                # create the list in, so first sorts the keys from the dictionary then extracts the values in order
                # to construct the new atom type list.
                for idx, atom_type in r.items():
                    new_setup.atoms[idx].atom_type = atom_type
                setups.append(new_setup)

        # for a gentle introduction of the new API
        self.deprecated_setup_access = setups

        return setups


class BasePreparator(abc.ABC):
    """
    Base Preparator class.

    Args:
        conformer_attempts: Should an embedding attempt fail, how many times to retry conformer generation.
        n_conformers: How many conformers to generate per SMILES string.
    """

    def __init__(self, conformer_attempts: int, n_conformers: int, **kwargs: Any):
        """Initialize base preparator.

        Args:
            conformer_attempts: Number of times to retry conformer generation on failure.
            n_conformers: Number of conformers to generate per SMILES string.
            **kwargs: Additional keyword arguments.
        """
        self.conformer_attempts = conformer_attempts
        self.n_conformers = n_conformers
        self._check_install()

    @abc.abstractmethod
    def _check_install(self) -> None:
        """Check if necessary dependencies are installed.

        Raises:
            Exception: If required dependencies are not installed.
        """

    def __call__(
        self, smis: List[str], ligand_paths_by_smiles: List[List[str]]
    ) -> List[bool]:
        """Prepare a list of SMILES strings.

        Args:
            smis: List of SMILES strings to prepare.
            ligand_paths_by_smiles: List of paths where ligand files should be saved.

        Returns:
            A list of booleans indicating whether preparation was successful.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_ligand(self, smi: str, ligand_path: List[str], **kwargs: Any) -> bool:
        """Prepare a single SMILES string.

        Args:
            smi: SMILES string to prepare.
            ligand_path: List of paths where ligand conformer files should be saved.
            **kwargs: Additional keyword arguments.

        Returns:
            A boolean indicating whether preparation was successful.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class MeekoLigandPreparator(BasePreparator):
    """
    Using Meeko to prepare ligand.
    """

    def __init__(
        self,
        conformer_attempts: int = 3,
        n_conformers: int = 1,
        pH: float = 7.4,
        remove_stereo_chemistry_on_fail: bool = True,  # TODO: Check Bredt's rule for bridged bonds
    ):
        """Initialize Meeko ligand preparator.

        Args:
            conformer_attempts: Number of times to retry conformer generation on failure.
            n_conformers: Number of conformers to generate per SMILES string.
            pH: pH value for ligand preparation.
            remove_stereo_chemistry_on_fail: Whether to remove stereochemistry on failure.
        """
        super().__init__(conformer_attempts, n_conformers)
        self.ligand_critic_errors: List[str] = []
        self.logger = logging.getLogger(
            __name__ + "/" + self.__class__.__name__,
        )
        self.pH = pH
        self.remove_stereo_chemistry_on_fail = remove_stereo_chemistry_on_fail

    def _check_install(self) -> None:
        """Check if Meeko package is installed.

        Raises:
            Exception: If Meeko package is not installed.
        """
        try:
            pass
        except ImportError:
            raise Exception("Meeko package isn't installed.")

    def __call__(
        self, smis: List[str], ligand_paths_by_smiles: List[List[str]]
    ) -> List[bool]:
        """Prepare a list of SMILES strings for docking.

        Args:
            smis: List of SMILES strings to prepare.
            ligand_paths_by_smiles: List of paths where ligand files should be saved.

        Returns:
            A list of booleans indicating whether new ligand file was created (True) or already exists (False).
        """
        results: List[bool] = []
        for i in range(len(smis)):
            results.append(self._prepare_ligand(smis[i], ligand_paths_by_smiles[i]))

        return results

    def _prepare_ligand(self, smi: str, ligand_path: List[str], **kwargs: Any) -> bool:
        """Prepare a single SMILES string for docking.

        Args:
            smi: SMILES string to prepare.
            ligand_path: List of paths where ligand conformer files should be saved.
            **kwargs: Additional keyword arguments.

        Returns:
            True if preparation was successful, False otherwise.
        """
        for j in range(0, len(ligand_path), self.n_conformers):
            if any(
                os.path.exists(path) for path in ligand_path[j : j + self.n_conformers]
            ):
                return False

        obc = ob.OBConversion()
        obc.SetInAndOutFormats("smi", "smi")
        obmol = ob.OBMol()
        obc.ReadString(obmol, smi)
        obmol.CorrectForPH(self.pH)
        new_smi = obc.WriteString(obmol).strip()
        mol = Chem.MolFromSmiles(new_smi)

        if mol is None:
            mol = Chem.MolFromSmiles(smi)
        else:
            smi = new_smi
        attempt = 0
        while attempt < self.conformer_attempts:
            attempt += 1
            try:
                assert mol is not None, (
                    f"Error with smiles not caught by scorer : {new_smi, smi}"
                )
                mol = Chem.AddHs(mol)
                if self.n_conformers == 1:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=10_000)
                    preparator = MoleculePreparator()
                    mol_setups = preparator.prepare(mol)
                    setup = mol_setups[0]
                    pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)
                    with open(ligand_path[0], "w") as file:
                        file.write(pdbqt_string)

                else:
                    AllChem.EmbedMultipleConfs(
                        mol, self.n_conformers, AllChem.ETKDGv3()
                    )

                    if mol.GetNumConformers() == 0:
                        raise Exception(
                            "AllChem.EmbedMultipleConfs() failed to yield any valid conformers."
                        )

                    preparator = MoleculePreparation()
                    for i in range(mol.GetNumConformers()):
                        AllChem.UFFOptimizeMolecule(mol, confId=i)
                        mol_setups = preparator.prepare(mol, conformer_id=i)
                        setup = mol_setups[0]
                        pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)
                        with open(ligand_path[i], "w") as file:
                            file.write(pdbqt_string)
            except Exception:
                if attempt == 1:
                    try:
                        self.logger.warning(f"Error when preparing ligand: {smi}")
                        if self.remove_stereo_chemistry_on_fail:
                            old_smi = smi
                            mol = Chem.RemoveHs(mol)
                            Chem.RemoveStereochemistry(mol)
                            smi = Chem.MolToSmiles(mol, canonical=True)
                            self.logger.warning(
                                f"Removing stereo chemistry in: {old_smi} | results: {smi}"
                            )
                    except Exception:
                        self.ligand_critic_errors.append(old_smi)
                        self.logger.error(
                            f"List of problematic ligands: {self.ligand_critic_errors}"
                        )
                else:
                    self.logger.error(
                        f"[attempt: {attempt}/{self.conformer_attempts}] Error when preparing ligand: {smi}"
                    )
                continue
            return True

            # except Exception as e:
            #     print(f"Failed embedding attempt #{attempt} with error: '{e}'.")

        return False
