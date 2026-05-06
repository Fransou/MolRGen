"""Reaction verifier for chemical reaction and retro-synthesis tasks.

This module provides the ReactionVerifier class which computes rewards for
chemical reaction tasks including retro-synthesis planning, SMARTS prediction,
and reaction product verification.
"""

import itertools
import json
import logging
import pickle
import re
from typing import Any, Dict, List, Literal, Tuple

from rdkit import RDLogger

from molrgen.data.reactions.mol import Molecule
from molrgen.data.reactions.reaction import Reaction, ReactionContainer
from molrgen.data.reactions.reaction_matrix import ReactantReactionMatrix
from molrgen.reward.verifiers.abstract_verifier import (
    Verifier,
)
from molrgen.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
)
from molrgen.reward.verifiers.reaction_reward.input_metadata import (
    ReactionVerifierInputMetadataModel,
)
from molrgen.reward.verifiers.reaction_reward.reaction_verifier_pydantic_model import (
    ReactionVerifierConfigModel,
    ReactionVerifierMetadataModel,
    ReactionVerifierOutputModel,
)

RDLogger.DisableLog("rdApp.*")


class ReactionVerifier(Verifier):
    """Verifier for chemical reaction and retro-synthesis tasks.

    This verifier computes rewards for various reaction-related tasks including:
    - Final product prediction
    - Reactant identification
    - SMARTS pattern prediction
    - Full retro-synthesis path validation

    The verifier uses a reaction matrix to validate synthesis steps and supports
    both binary and Tanimoto-based reward computation.

    Attributes:
        verifier_config: Configuration for the reaction verifier.
        rxn_matrix: Pre-loaded reaction matrix for validation.
        check_ground_truth_tasks: List of task types requiring ground truth comparison.
        run_validation_tasks: List of task types requiring path validation.
        logger: Logger instance for the verifier.
    """

    def __init__(
        self,
        verifier_config: ReactionVerifierConfigModel,
    ):
        """Initialize the ReactionVerifier.

        Args:
            verifier_config: Configuration containing reaction matrix path
                and reward type settings.
        """
        super().__init__(verifier_config)
        self.verifier_config: ReactionVerifierConfigModel = verifier_config

        self.rxn_matrix: ReactantReactionMatrix
        with open(verifier_config.reaction_matrix_path, "rb") as f:
            self.rxn_matrix = pickle.load(f)

        self.reactants_csi: List[str] = [m.csmiles for m in self.rxn_matrix.reactants]
        self.check_ground_truth_tasks = [
            "final_product",
            "reactant",
            "all_reactants",
            "all_reactants_bb_ref",
        ]
        self.run_validation_tasks = [
            "full_path",
            "full_path_bb_ref",
            "full_path_smarts_ref",
            "full_path_smarts_bb_ref",
            "full_path_intermediates_gt_reactants",
            "full_path_intermediates",  # We treat this task as a normal path validation
        ]
        self.logger = logging.getLogger("ReactionVerifier")

    def r_ground_truth_mols(
        self,
        mol_y: List[Molecule],
        mol_label: List[Molecule],
        reactants: List[str],
        product: str,
        smarts: str,
        objective: str,
    ) -> float:
        """Compute reward for molecule prediction against ground truth."""
        self.logger.info(
            f"Computed molecules: {[mol.smiles for mol in mol_y]} vs labels: {[mol.smiles for mol in mol_label]}"
        )
        smi_y = {mol.csmiles for mol in mol_y}
        smi_y_true = {mol.csmiles for mol in mol_label}
        if smi_y == smi_y_true:
            return 1.0

        # Check if the reaction still works
        rxn = Reaction(smarts)
        if objective == "final_product":
            if len(mol_y) != 1:
                return 0.0
            react_mols = [Molecule(smi) for smi in reactants]
            possible_products = []
            for r_ in itertools.permutations(react_mols):
                possible_products.extend(rxn(list(r_)))
            # Change to csmiles to avoid unexpected behavior with python in
            possible_products_csi = {m.csmiles for m in possible_products}
            if mol_y[0].csmiles in possible_products_csi:  # Can be only one product
                return 1.0
            else:
                return 0.0
        elif objective in ["reactant", "all_reactants", "all_reactants_bb_ref"]:
            # Start from all explicitly specified reactants, excludeing placeholders
            # representing the label
            react_mols = [Molecule(smi) for smi in reactants if not smi == "???"]
            react_mols += mol_y
        # Check that the number of reactants matches
        if len(react_mols) != rxn.num_reactants:
            return 0.0
        # Check if all reactants are in the building blocks
        if any(m.csmiles not in self.reactants_csi for m in react_mols):
            return 0.0
        possible_products = []
        for r_ in itertools.permutations(react_mols):
            possible_products.extend(rxn(list(r_)))
        # Change to csmiles to avoid unexpected behavior with python in
        possible_products_csi = {m.csmiles for m in possible_products}
        prod_mol = Molecule(product)
        if prod_mol.csmiles in possible_products_csi:
            return 1.0
        return 0.0

    def ground_truth_reward_mol(
        self,
        answer: Dict[str, Any],
        labels: List[str],
        reactants: List[str],
        product: str,
        smarts: str,
        objective: str,
    ) -> float:
        """Compute reward for molecule prediction tasks.

        Notes
        The answer must be contained in a JSON object with an "answer" key, which can be either a single SMILES string or a list of SMILES strings.

        Args:
            answer: Model completion containing the answer.
            labels: List of ground truth SMILES strings.
            reactants: List of reactant SMILES strings.
            product: Expected product SMILES string.
            objective: The type of objective for the reaction verification.

        Returns:
            Reward value between 0.0 and 1.0.
        """
        if answer == {}:
            return 0.0
        mol_label = [Molecule(smi) for smi in labels]
        if not all([m.is_valid for m in mol_label]):
            self.logger.error("Invalid ground truth molecule")
            return 0.0
        smiles = answer["answer"]
        if isinstance(smiles, str):
            smiles_list = [smiles]
        elif isinstance(smiles, list):
            smiles_list = smiles
        else:
            return 0.0
        mols = [Molecule(smi) for smi in smiles_list]

        if any([not m.is_valid for m in mols]):
            self.logger.info("Invalid molecule found in prediction")
            return 0.0

        return self.r_ground_truth_mols(
            mols,
            mol_label,
            reactants=reactants,
            product=product,
            smarts=smarts,
            objective=objective,
        )

    def reward_smarts(
        self,
        answer: Dict[str, Any],
        labels: List[str],
        reactants: List[str],
        product: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for SMARTS prediction tasks.

        Notes
        The answer must be contained in a JSON object with an "answer" key,
        which should be a SMARTS string representing the reaction. The reward is computed based
        on whether the proposed SMARTS can produce the expected product from the given reactants.
        A reward of 1.0 is given for an exact match with the ground truth SMARTS, 0.1 if the SMARTS
        is valid and produces the correct product, and 0.0 otherwise.

        Args:
            answer: Model completion containing the SMARTS answer.
            labels: List containing the ground truth SMARTS string.
            reactants: List of reactant SMILES strings.
            product: Expected product SMILES string.

        Returns:
            Tuple of (reward, metadata_dict) where metadata contains
            'Reactants_contained' and 'Products_contained' flags.
        """
        if answer == {}:
            return 0.0, {"Reactants_contained": False, "Products_contained": False}
        gt_smarts = labels[0]
        smarts_pred = answer["answer"]
        if not isinstance(smarts_pred, str):
            return 0.0, {"Reactants_contained": False, "Products_contained": False}

        if smarts_pred.strip() == gt_smarts:
            return 1.0, {"Reactants_contained": True, "Products_contained": True}
        self.logger.info(
            f"Proposed SMARTS: {smarts_pred.strip()} | GT SMARTS: {gt_smarts}, checking reaction..."
        )
        try:
            rxnB = Reaction(smarts_pred.strip())
            if rxnB.num_reactants != len(reactants):
                return 0.0, {"Reactants_contained": False, "Products_contained": False}
            p = rxnB([Molecule(r) for r in reactants])
            reward = 0.0
            if Molecule(product).csmiles in {prod.csmiles for prod in p}:
                reward = 0.1
            return reward, {
                "Reactants_contained": True,
                "Products_contained": reward == 0.1,
            }
        except Exception as e:
            self.logger.info(
                f"Error in reaction SMARTS parsing: {e} (proposed: {smarts_pred} | gt: {gt_smarts})"
            )
            return 0.0, {"Reactants_contained": False, "Products_contained": False}

    def _find_reaction_smarts(
        self,
        reactants_step: List[Molecule],
        products_step: List[Molecule],
        allowed_smarts: ReactionContainer,
    ) -> List[Reaction]:
        """Find valid reaction SMARTS that can produce products from reactants.

        Args:
            reactants_step: List of reactant molecules for this step.
            products_step: List of expected product molecules.
            allowed_smarts: Container of allowed reaction SMARTS patterns.

        Returns:
            List of Reaction objects that successfully produce the expected products.
        """
        found_reactions: List[Reaction] = []
        id_poss_smarts: List[Dict[int, tuple[int, ...]]] = []
        for r in reactants_step:
            id_poss_smarts.append(allowed_smarts.match_reactions(r))

        for id_reaction in id_poss_smarts[0]:
            # Check if reaction can take the correct number of reactants
            if allowed_smarts[id_reaction].num_reactants != len(reactants_step):
                continue
            if any(
                id_reaction not in id_poss_smarts[i]
                for i in range(1, len(reactants_step))
            ):
                continue

            # Generate all permutations of reactants and test the reaction
            possible_products = []
            for reactants_ord in itertools.permutations(reactants_step):
                possible_products.extend(
                    allowed_smarts[id_reaction](list(reactants_ord))
                )
            possible_products_csi = {m.csmiles for m in possible_products}
            all_found: bool = all(
                p.csmiles in possible_products_csi for p in products_step
            )
            if all_found:
                found_reactions.append(allowed_smarts[id_reaction])
        return found_reactions

    def _check_valid_step(
        self,
        reactants_step: List[Molecule],
        products_step: List[Molecule],
        possible_reactants: List[str],
        allowed_smarts: ReactionContainer,
    ) -> Tuple[bool, str]:
        """Check if a synthesis step is valid.
        Used in the reward_run_path method to validate each step of the proposed synthesis path.

        Validates that:
        1. All reactants and products are valid molecules
        2. All reactants are in building blocks or previous products
        3. At least one reaction can produce the products from the reactants

        Args:
            reactants_step: List of reactant molecules for this step.
            products_step: List of expected product molecules.
            possible_reactants: List of valid starting materials (building blocks + previous products).
            allowed_smarts: Container of allowed reaction SMARTS patterns.

        Returns:
            Tuple of (is_valid, fail_reason) where fail_reason is empty string if valid,
            or one of "reactants", "products", "reaction" indicating what failed.
        """
        # 1. Check that all reactants and products are valid molecules
        if not all([r.is_valid for r in reactants_step]):
            self.logger.info(
                "Reactants not valid in {}".format([r.smiles for r in reactants_step])
            )
            return False, "reactants"
        if not all([p.is_valid for p in products_step]):
            self.logger.info(
                "Products not valid in {}".format([p.smiles for p in products_step])
            )
            return False, "products"
        # 2. Check that all reactants are in building blocks or previous products
        for r in reactants_step:
            if r.csmiles not in possible_reactants:
                self.logger.info(
                    "Reactant {} not in building blocks or previous products".format(
                        r.smiles
                    )
                )
                return False, "reactants"
        if products_step == []:
            self.logger.info("No products in step")
            return False, "products"

        # 3. Check that there is at least one reaction that can produce the products from the reactants
        found_reactions = self._find_reaction_smarts(
            reactants_step, products_step, allowed_smarts
        )
        if len(found_reactions) == 0:
            self.logger.info(
                "No reaction found for step: {} -> {}".format(
                    [r.smiles for r in reactants_step],
                    [p.smiles for p in products_step],
                )
            )
            return False, "reaction"
        # Log success
        self.logger.info(
            "Found valid reaction for step: {} -> {}".format(
                [r.smiles for r in reactants_step],
                [p.smiles for p in products_step],
            )
        )
        return True, ""

    def reward_run_path(
        self,
        answer: Dict[str, Any],
        label: str,
        building_blocks: List[str],
        smarts: List[str],
        n_steps_max: int,
        reward_type: Literal["binary", "tanimoto"] = "binary",
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for retro-synthesis path validation.

        Validates a multi-step synthesis path by checking:
        1. All reactants are valid building blocks or previous products
        2. Each reaction step has a valid SMARTS pattern
        3. The final product matches the target (exactly or by Tanimoto similarity)

        Notes
        The answer must be contained in a JSON object with an "answer" key,
        which should contain a list of steps: Dictionaries containing the keys:

            - reactants: List of reactant SMILES strings for this step
            - product: List of product SMILES strings for this step

        Args:
            answer: Model completion containing the synthesis path.
            label: Target product SMILES string.
            building_blocks: List of valid starting building block SMILES.
            smarts: List of allowed SMARTS patterns (empty = use reaction matrix).
            n_steps_max: Maximum allowed number of synthesis steps.
            reward_type: "binary" for exact match or "tanimoto" for similarity-based.

        Returns:
            Tuple of (reward, metadata_dict) containing validation results.
        """
        if answer == {}:
            self.logger.info("No synthesis path found in completion")
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
        steps: List[Dict[str, Any]] = answer["answer"]
        if not isinstance(steps, list) or len(steps) == 0 or len(steps) > n_steps_max:
            self.logger.info("Synthesis path answer is not a list or is too long/short")
            return 0.0, {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
        # Validate each step structure
        for step in steps:
            if "reactants" not in step or "product" not in step:
                self.logger.info("Synthesis step missing reactants or product")
                return 0.0, {
                    "valid": 0.0,
                    "correct_product": 0.0,
                    "correct_reactant": False,
                }
            if step["reactants"] == [] or step["product"] == []:
                self.logger.info("Synthesis step has empty reactants or product")
                return 0.0, {
                    "valid": 0.0,
                    "correct_product": 0.0,
                    "correct_reactant": False,
                }
            if not isinstance(step["reactants"], list) or not all(
                isinstance(r, str) for r in step["reactants"]
            ):
                self.logger.info("One or more reactants in step are not strings")
                return 0.0, {
                    "valid": 0.0,
                    "correct_product": 0.0,
                    "correct_reactant": False,
                }
            if not isinstance(step["product"], list) or not all(
                isinstance(p, str) for p in step["product"]
            ):
                self.logger.info("One or more products in step are not strings")
                return 0.0, {
                    "valid": 0.0,
                    "correct_product": 0.0,
                    "correct_reactant": False,
                }

        reactants = [
            [Molecule(smi.strip()) for smi in step["reactants"]] for step in steps
        ]
        products = [
            [Molecule(smi.strip()) for smi in step["product"]] for step in steps
        ]
        n_steps = len(reactants)
        label_mol = Molecule(label)
        reward_mult: List[float] = [1.0 for _ in products]
        if reward_type == "binary" and not any(
            [label_mol == last_p for last_p in products[-1]]
        ):
            self.logger.info("Product not found")
            return 0.0, {
                "valid": 0.0,
                "correct_last_product": 0.0,
                "correct_reactant": False,
            }
        elif reward_type == "tanimoto":
            # Compute the tanimoto similarity between the label and products at each step
            for i, product in enumerate(products):
                all_sims = label_mol.tanimoto_similarity(product)
                reward_mult[i] = max(all_sims) ** 3

        reactions: ReactionContainer
        if smarts == []:
            reactions = self.rxn_matrix.reactions
        else:
            reactions = ReactionContainer([Reaction(sma) for sma in smarts])

        building_blocks_csi = (
            self.reactants_csi
        )  # For the moment, the building blocks passed in the metadata
        # are only used to  help the model but not proper constraints.

        n_valid = 0
        fail_reason = ""
        for i_reac, (reactant, product) in enumerate(zip(reactants, products)):
            is_valid, fail_reason = self._check_valid_step(
                reactant,
                product,
                building_blocks_csi
                + [p.csmiles for step in products[:i_reac] for p in step],
                reactions,
            )
            if not is_valid:
                self.logger.info(f"Invalid step at index {i_reac} due to {fail_reason}")
                break
            else:
                n_valid += 1
        if n_valid < n_steps:
            return reward_mult[n_valid - 1] * (n_valid / n_steps) ** 2, {
                "valid": n_valid / n_steps,
                "correct_product": reward_mult[n_valid - 1],
                "correct_reactant": fail_reason != "reactants",
            }

        return reward_mult[n_valid - 1], {
            "valid": 1.0,
            "correct_product": reward_mult[n_valid - 1],
            "correct_reactant": True,
        }

    def parse_json_content(self, content: str) -> Dict[str, Any]:
        """Parse JSON content from the model completion.

        Args:
            content: The extracted answer content from the model completion.
        Returns:
            Parsed JSON as a dictionary.
        """
        # Find the first and last curly braces to extract JSON
        # as a regex with { "answer": ... }
        possible_json = re.search(r"\{.*\}", content, re.DOTALL)
        parsed: Dict[str, Any]
        if possible_json is None:
            return {}
        try:
            parsed = json.loads(possible_json.group(0))
        except json.JSONDecodeError as e:
            self.logger.info(f"JSON decode error: {e}")
            return {}
        if "answer" not in parsed:
            return {}
        return parsed

    def get_score(
        self, inputs: BatchVerifiersInputModel
    ) -> List[ReactionVerifierOutputModel]:
        """Compute reaction rewards for a batch of completions.

        This method routes each completion to the appropriate reward function
        based on the objective type specified in the metadata.

        Args:
            inputs: Batch of completions and metadata for verification.

        Returns:
            List of ReactionVerifierOutputModel containing rewards and metadata.

        Notes:
            - Ground truth tasks: final_product, reactant, all_reactants
            - SMARTS tasks: smarts prediction with reaction validation
            - Path tasks: full_path with step-by-step validation
        """
        completions = inputs.completions
        assert all(
            isinstance(meta, ReactionVerifierInputMetadataModel)
            for meta in inputs.metadatas
        )
        metadatas: List[ReactionVerifierInputMetadataModel] = inputs.metadatas  # type: ignore

        output_models = []
        for answer, meta in zip(completions, metadatas):
            objective = meta.objectives[0]
            reward = 0.0
            reward_metadata = {
                "valid": 0.0,
                "correct_product": 0.0,
                "correct_reactant": False,
            }
            extracted_answer = self.parse_answer(answer)
            json_answer = self.parse_json_content(extracted_answer)

            if objective in self.check_ground_truth_tasks:
                reward = self.ground_truth_reward_mol(
                    json_answer,
                    meta.target,
                    reactants=meta.reactants[meta.idx_chosen],
                    product=meta.products[meta.idx_chosen],
                    smarts=meta.or_smarts[meta.idx_chosen],
                    objective=objective,
                )
                reward_metadata = {
                    "valid": float(reward > 0.0),
                    "correct_product": reward > 0.0,
                    "correct_reactant": reward > 0.0,
                }
            elif objective == "smarts":
                assert len(meta.reactants) > 0, (
                    "Reactants must be provided for SMARTS objective"
                )
                assert len(meta.products) > 0, (
                    "Product must be provided for SMARTS objective"
                )
                reward, raw_metadata = self.reward_smarts(
                    json_answer,
                    meta.target,
                    meta.reactants[0],
                    meta.products[0],
                )
                reward_metadata = {
                    "valid": reward,
                    "correct_product": raw_metadata.get("Products_contained", False),
                    "correct_reactant": raw_metadata.get("Reactants_contained", False),
                }
            elif objective in self.run_validation_tasks:
                assert len(meta.target) > 0, (
                    "Target must be provided for run validation tasks"
                )
                # If smarts constraint, add it
                if "smarts" in objective:
                    assert len(meta.smarts) > 0, (
                        "SMARTS must be provided for smarts reference path validation"
                    )
                    smarts = meta.smarts
                else:
                    smarts = []
                reward, raw_metadata = self.reward_run_path(
                    json_answer,
                    meta.target[0],
                    meta.building_blocks if meta.building_blocks else [],
                    smarts=smarts,
                    n_steps_max=meta.n_steps_max,
                    reward_type=self.verifier_config.reaction_reward_type,
                )
                reward_metadata = raw_metadata
            else:
                raise ValueError(
                    "Unknown objective {} type for reaction verifier".format(objective)
                )

            if self.verifier_config.reward == "valid_smiles":
                reward = float(reward > 0.0)

            # Create the output model
            output_model = ReactionVerifierOutputModel(
                reward=reward,
                parsed_answer=f"{json_answer}",
                verifier_metadata=ReactionVerifierMetadataModel(
                    valid=reward_metadata["valid"],
                    correct_product=reward_metadata["correct_product"],
                    correct_reactant=reward_metadata["correct_reactant"],
                ),
            )
            output_models.append(output_model)

        return output_models
