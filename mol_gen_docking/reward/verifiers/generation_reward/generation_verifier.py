"""Generation verifier for de novo molecular generation tasks.

This module provides the GenerationVerifier class which computes rewards for
molecular generation based on property optimization objectives such as docking
scores, QED, synthetic accessibility, and other molecular descriptors.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from rdkit import Chem, RDLogger

from mol_gen_docking.reward.verifiers.abstract_verifier import (
    Verifier,
)
from mol_gen_docking.reward.verifiers.abstract_verifier_pydantic_model import (
    BatchVerifiersInputModel,
)
from mol_gen_docking.reward.verifiers.generation_reward.generation_verifier_pydantic_model import (
    GenerationVerifierConfigModel,
    GenerationVerifierMetadataModel,
    GenerationVerifierOutputModel,
)
from mol_gen_docking.reward.verifiers.generation_reward.input_metadata import (
    GenerationObjT,
    GenerationVerifierInputMetadataModel,
)
from mol_gen_docking.reward.verifiers.generation_reward.oracle_wrapper import (
    OracleWrapper,
    get_oracle,
)
from mol_gen_docking.utils.property_utils import (
    has_bridged_bond,
    rescale_property_values,
)


class GenerationVerifier(Verifier):
    """Verifier for de novo molecular generation tasks.

    This verifier computes rewards for generated molecules based on how well
    they meet specified property optimization criteria. It supports multiple
    property types including docking scores, QED, SA score, and RDKit descriptors.

    The verifier uses Ray for parallel computation and supports GPU-accelerated
    docking calculations when configured with AutoDock GPU.

    Attributes:
        verifier_config: Configuration for the generation verifier.
        property_name_mapping: Mapping of property names to oracle names.
        docking_target_list: List of valid docking target names.
        oracles: Cache of oracle instances for property computation.
        debug: If True, enables debug mode with additional logging.

    Example:
        ```python
        from mol_gen_docking.reward.verifiers import (
            GenerationVerifier,
            GenerationVerifierConfigModel,
            BatchVerifiersInputModel,
            GenerationVerifierInputMetadataModel
        )

        config = GenerationVerifierConfigModel(
            path_to_mappings="data/molgendata",
            reward="property"
        )
        verifier = GenerationVerifier(config)

        inputs = BatchVerifiersInputModel(
            completions=["<answer>CCO</answer>"],
            metadatas=[GenerationVerifierInputMetadataModel(
                properties=["QED"], objectives=["maximize"], target=[0.0]
            )]
        )
        results = verifier.get_score(inputs)
        ```
    """

    def __init__(
        self,
        verifier_config: GenerationVerifierConfigModel,
    ):
        """Initialize the GenerationVerifier.

        Args:
            verifier_config: Configuration containing paths to mappings,
                reward type, and docking oracle settings.
        """
        super().__init__(verifier_config)
        self.verifier_config: GenerationVerifierConfigModel = verifier_config
        self.logger = logging.getLogger("GenerationVerifier")

        with open(
            os.path.join(verifier_config.path_to_mappings, "names_mapping.json")
        ) as f:
            property_name_mapping = json.load(f)
        with open(
            os.path.join(verifier_config.path_to_mappings, "docking_targets.json")
        ) as f:
            docking_target_list = json.load(f)

        self.property_name_mapping = property_name_mapping
        self.docking_target_list = docking_target_list
        self.slow_props = docking_target_list  # + ["GSK3B", "JNK3", "DRD2"]

        self.oracles: Dict[str, OracleWrapper] = {}
        self.debug = False  # Only for tests

    def get_smiles_from_completion(self, comp: str) -> Tuple[List[str], str]:
        """Extract SMILES strings from a model completion.

        This method parses a model completion to extract valid SMILES strings.
        It handles various formats including answer tags and markdown formatting.

        Args:
            comp: The model completion string to parse.

        Returns:
            Tuple containing:
                - List of valid SMILES strings found in the completion
                - Failure reason string (empty if successful, otherwise one of:
                  "no_answer", "no_smiles", "no_valid_smiles", "multiple_smiles")

        Example:
            ```python
            smiles, failure = verifier.get_smiles_from_completion("<answer>CCO</answer>")
            # smiles = ["CCO"], failure = ""
            ```
        """
        comp = comp.strip()
        reason: str = ""
        comp = self.parse_answer(comp)

        # Now we identify which elements are possibly SMILES
        # First we split the completion by newlines and spaces
        # Then we filter by removing any string that does not contain "C"
        valid_smiles_pattern = re.compile(r"^[A-Za-z0-9=#:\+\-\[\]\(\)/\\@.%]+$")
        mkd_pattern = re.compile(r"^(\*\*|[-*'])(.+)\1$")

        def filter_smiles(x: str) -> str:
            x = x.replace("<|im_end|>", "")
            if len(x) < 3:
                return ""
            # Check if the string is encapsulated in some kind of markdown
            m = mkd_pattern.match(x)
            x = m.group(2) if m else x
            if len(x) < 3:
                return ""
            if (
                "C" in x
                or x.count("c") > 2
                and valid_smiles_pattern.fullmatch(x) is not None
            ):
                return x
            return ""

        # Finally we remove any string that is not a valid SMILES
        def test_is_valid_batch(smis: list[str]) -> list[bool]:
            RDLogger.DisableLog("rdApp.*")
            results = []
            for smi in smis:
                if len(smi) >= 130:
                    results.append(False)
                    continue
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        results.append(False)
                        continue
                    if has_bridged_bond(mol):  ### WE REMOVE BRIDGED MOLS
                        results.append(False)
                        continue
                    Chem.MolToMolBlock(mol)
                    results.append(True)
                except Exception:
                    results.append(False)
            return results

        s_poss = [
            filter_smiles(x) for x in re.split("\n| |\\.|\t|:|`|'|,|\\{|\\}", comp)
        ]
        s_poss = [x for x in s_poss if x != ""]
        s_poss = list(set(s_poss))

        if len(s_poss) == 0:
            if reason == "":
                reason = "no_smiles"
            return [], reason

        is_valid: List[bool] = test_is_valid_batch(s_poss)

        s_spl = [x for (x, val) in zip(s_poss, is_valid) if val]
        if s_spl == [] and reason == "":
            reason = "no_valid_smiles"
        elif len(s_spl) > 1:
            reason = "multiple_smiles"
        elif reason == "":
            reason = ""
        return s_spl, reason

    def get_all_completions_smiles(
        self, completions: List[str]
    ) -> Tuple[List[List[str]], List[str]]:
        """Extract SMILES from multiple completions.

        Args:
            completions: List of model completion strings.

        Returns:
            Tuple containing:
                - List of SMILES lists (one per completion)
                - List of failure reasons (one per completion)
        """
        smiles = []
        failures = []
        for completion in completions:
            if isinstance(completion, list):
                assert len(completion) == 1
                completion = completion[0]
            if isinstance(completion, dict):
                assert "content" in completion
                completion = completion["content"]
            smi, failure = self.get_smiles_from_completion(completion)
            smiles.append(smi)
            failures.append(failure)
        return smiles, failures

    def fill_df_properties(self, df_properties: pd.DataFrame) -> None:
        """Compute property values for all molecules in a DataFrame.

        This method fills in the 'value' column of the DataFrame with computed
        property values using the appropriate oracles. It uses Ray for parallel
        computation, with GPU resources allocated for docking calculations.

        Args:
            df_properties: DataFrame with columns ['smiles', 'property', 'value',
                'obj', 'target_value', 'id_completion']. The 'value' column will
                be filled with computed property values.
        """

        def _get_property(
            smiles: List[str],
            prop: str,
            rescale: bool = True,
            kwargs: Dict[str, Any] = {},
        ) -> List[float]:
            """
            Get property reward
            """
            oracle_fn = self.oracles.get(
                prop,
                get_oracle(
                    prop,
                    path_to_data=self.verifier_config.path_to_mappings
                    if self.verifier_config.path_to_mappings
                    else "",
                    docking_target_list=self.docking_target_list,
                    property_name_mapping=self.property_name_mapping,
                    **kwargs,
                ),
            )
            if prop not in self.oracles:
                self.oracles[prop] = oracle_fn
            property_reward: np.ndarray | float = oracle_fn(smiles, rescale=rescale)
            assert isinstance(property_reward, np.ndarray)

            return [float(p) for p in property_reward]

        _get_property_fast = ray.remote(num_cpus=0)(_get_property)
        _get_property_long = ray.remote(
            num_cpus=1,
            num_gpus=float("gpu" in self.verifier_config.oracle_kwargs.docking_oracle)
            / self.verifier_config.docking_concurrency_per_gpu,
        )(_get_property)

        all_properties = df_properties["property"].unique().tolist()
        prop_smiles = {
            p: df_properties[df_properties["property"] == p]["smiles"].unique().tolist()
            for p in all_properties
        }

        values_job = []
        for p in all_properties:
            # If the reward is long to compute, use ray
            smiles = prop_smiles[p]
            if p in self.slow_props:
                _get_property_remote = _get_property_long
            else:
                _get_property_remote = _get_property_fast

            values_job.append(
                _get_property_remote.remote(
                    smiles,
                    p,
                    rescale=self.verifier_config.rescale,
                    kwargs=self.verifier_config.oracle_kwargs.model_dump(),
                )
            )
        all_values = ray.get(values_job)
        for idx_p, p in enumerate(all_properties):
            values = all_values[idx_p]
            smiles = prop_smiles[p]
            for s, v in zip(smiles, values):
                df_properties.loc[
                    (df_properties["smiles"] == s) & (df_properties["property"] == p),
                    "value",
                ] = v

    def get_reward(self, row: pd.Series) -> float:
        """Compute reward for a single property-molecule pair.

        This method computes the reward based on the objective type:
        - "below": 1.0 if property <= target, else 0.0
        - "above": 1.0 if property >= target, else 0.0
        - "maximize": Returns the property value directly
        - "minimize": Returns 1 - property value
        - "equal": Returns clipped value based on squared error

        Args:
            row: DataFrame row containing 'obj', 'value', 'target_value', 'property'.

        Returns:
            Computed reward value (typically 0.0 to 1.0).
        """
        reward: float = 0
        obj = row["obj"]
        mol_prop = row["value"]
        target_value = row["target_value"]
        prop = row["property"]
        is_docking = prop in self.docking_target_list
        # Replace 0 docking score by the worst outcome
        if is_docking and prop == 0.0:
            return 0.0
        if self.verifier_config.rescale:
            target_value = rescale_property_values(
                prop, target_value, docking=is_docking
            )
        if obj == "below":
            reward += float(mol_prop <= target_value)
        elif obj == "above":
            reward += float(mol_prop >= target_value)
        elif obj == "maximize":
            reward += mol_prop
        elif obj == "minimize":
            reward += 1 - mol_prop
        elif obj == "equal":
            reward += np.clip(1 - 100 * (mol_prop - target_value) ** 2, 0, 1)
        return float(reward)

    def _get_prop_to_smiles_dataframe(
        self,
        smiles_list_per_completion: List[List[str]],
        objectives: List[dict[str, Tuple[GenerationObjT, float]]],
    ) -> pd.DataFrame:
        """Create a DataFrame mapping properties to SMILES for batch processing.

        Args:
            smiles_list_per_completion: List of SMILES lists, one per completion.
            objectives: List of objective dictionaries mapping property names
                to (objective_type, target_value) tuples.

        Returns:
            DataFrame with columns: smiles, property, value, obj, target_value, id_completion.
        """
        df_properties = pd.DataFrame(
            [
                (s, p, None, obj, target_value, i)
                for i, (props, smiles_list) in enumerate(
                    zip(objectives, smiles_list_per_completion)
                )
                for s in smiles_list
                for p, (obj, target_value) in props.items()
            ],
            columns=[
                "smiles",
                "property",
                "value",
                "obj",
                "target_value",
                "id_completion",
            ],
        )
        return df_properties

    def get_score(
        self, inputs: BatchVerifiersInputModel
    ) -> List[GenerationVerifierOutputModel]:
        """Compute generation rewards for a batch of completions.

        This method extracts SMILES from completions, computes property values,
        and calculates rewards based on the specified objectives. The final reward
        is the geometric mean of per-property rewards.

        Args:
            inputs: Batch of completions and metadata for verification.

        Returns:
            List of GenerationVerifierOutputModel containing rewards and metadata
            for each completion.

        Notes:
            - If reward type is "valid_smiles", returns 1.0 for valid single SMILES
            - Multiple SMILES in a completion result in 0.0 reward
            - Uses geometric mean to aggregate multi-property rewards
        """
        smiles_per_completion, extraction_failures = self.get_all_completions_smiles(
            inputs.completions
        )
        if self.verifier_config.reward == "valid_smiles":
            return [
                GenerationVerifierOutputModel(
                    reward=float(len(smis) == 1),
                    parsed_answer=self.parse_answer("; ".join(smis)),
                    verifier_metadata=GenerationVerifierMetadataModel(
                        smiles_extraction_failure=fail
                    ),
                )
                for smis, fail in zip(smiles_per_completion, extraction_failures)
            ]
        assert all(
            isinstance(meta, GenerationVerifierInputMetadataModel)
            for meta in inputs.metadatas
        )
        metadatas: List[GenerationVerifierInputMetadataModel] = inputs.metadatas  # type: ignore

        objectives = []
        for m in metadatas:
            props = {}
            for p, obj, target in zip(m.properties, m.objectives, m.target):
                props[p] = (obj, float(target))
            objectives.append(props)

        df_properties = self._get_prop_to_smiles_dataframe(
            smiles_per_completion, objectives
        )
        self.fill_df_properties(df_properties)
        df_properties["reward"] = df_properties.apply(
            lambda x: self.get_reward(x), axis=1
        )

        output_models = []
        for id_completion, smiles in enumerate(smiles_per_completion):
            properties: List[str] = []
            individual_rewards: List[float] = []
            compl_reward: List[float] = []
            if len(smiles) > 0:
                for idx_s, s in enumerate(smiles):
                    rows_completion = df_properties[
                        (df_properties["id_completion"] == id_completion)
                        & (df_properties["smiles"] == s)
                    ]
                    rewards_l = rows_completion["reward"].to_numpy().clip(0, 1)
                    reward = np.power(
                        rewards_l.prod(), (1 / len(rewards_l))
                    )  # Geometric mean
                    if idx_s == 0:
                        for i in range(len(rows_completion["smiles"])):
                            properties.append(rows_completion["property"].iloc[i])
                            individual_rewards.append(rows_completion["reward"].iloc[i])

                    if self.verifier_config.rescale and not self.debug:
                        reward = np.clip(reward, 0, 1)
                    compl_reward.append(float(reward))
            else:
                reward = 0
                compl_reward = [0.0]

            if np.isnan(reward) or reward is None:
                self.logger.warning(
                    f"Warning: Reward is None or NaN for completion id {id_completion} with smiles {smiles}\n"
                )
                reward = 0.0
            if len(smiles) > 1:
                reward = 0.0

            # Create the output model
            output_model = GenerationVerifierOutputModel(
                reward=float(reward),
                parsed_answer=self.parse_answer("; ".join(smiles)),
                verifier_metadata=GenerationVerifierMetadataModel(
                    properties=properties,
                    individual_rewards=individual_rewards,
                    all_smi_rewards=compl_reward,
                    all_smi=smiles,
                    smiles_extraction_failure=extraction_failures[id_completion],
                ),
            )
            output_models.append(output_model)

        return output_models
