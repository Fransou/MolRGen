"""Dataset for generating prompts for molecule generation"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from numpy import random
from tqdm import tqdm

from molrgen.data.pydantic_dataset import Conversation, Message, Sample
from molrgen.utils.property_utils import (
    CLASSICAL_PROPERTIES_NAMES,
    inverse_rescale_property_values,
)

from .utils import (
    DOCKING_OBJECTIVES,
    DOCKING_SOLO_OBJECTIVES,
    DOCKING_SUFFIX,
    OBJECTIVES,
    OBJECTIVES_TEMPLATES,
    PROMPT_TEMPLATE,
    PROPERTY_ALLOWED_OBJECTIVES,
    TARGET_VALUE_OBJECTIVES,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleSet:
    """A simple class to keep track of the rules for generating prompts"""

    prompt_ids: Dict[int, List[str]] = field(
        default_factory=dict
    )  # n_props -> list of prompt_ids
    n_occ_prop: Dict[int, Dict[str, int]] = field(
        default_factory=dict
    )  # n_props -> {prop: n_occurrences}
    probs_docking_targets: float = field(
        default=0.5
    )  # Probability to draw a docking property
    max_occ: int = field(
        default=10
    )  # Maximum number of occurrences of a property per n_props
    max_occ_exceptions: List[str] = field(
        default_factory=lambda: [
            "SA",
            "QED",
            "CalcExactMolWt",
            "CalcNumAromaticRings",
            "CalcNumHBA",
            "CalcNumHBD",
            "CalcNumRotatableBonds",
            "CalcFractionCSP3",
            "CalcTPSA",
            "CalcHallKierAlpha",
            "CalcPhi",
            "logP",
        ]
    )
    max_docking_per_prompt: int = field(default=2)
    prohibited_props_at_n: Dict[int, List[str]] = field(default_factory=dict)

    def verify_and_update(self, metadata: Dict[str, Any]) -> bool:
        """
        Verify if the metadata meets the rules and update the occurrences.
        Returns True if the metadata is valid, False otherwise.
        """
        n_props = metadata["n_props"]
        properties = metadata["properties"]
        max_occ = self.max_occ if n_props > 0 else 2

        if n_props not in self.n_occ_prop:
            self.n_occ_prop[n_props] = {}
            self.prompt_ids[n_props] = []

        if (
            metadata["prompt_id"] in self.prompt_ids[n_props]
        ):  # Already generated this prompt
            return False

        for prop in properties:
            if prop not in self.n_occ_prop[n_props]:
                self.n_occ_prop[n_props][prop] = 0
            if (
                self.n_occ_prop[n_props][prop] >= max_occ
            ):  # Too many occurrences of this property
                logger.info("Property %s has too many occurrences: %d", prop, max_occ)
                return False

        self.update_occurence(n_props, metadata, max_occ, properties)

        return True

    def update_occurence(
        self,
        n_props: int,
        metadata: Dict[str, Any],
        max_occ: int,
        properties: List[str],
    ) -> None:
        # Update occurrences
        self.prompt_ids[n_props].append(metadata["prompt_id"])
        for prop in properties:
            # Only account for docking targets
            if prop in self.max_occ_exceptions and n_props > 1:
                continue
            self.n_occ_prop[n_props][prop] += 1
            if self.n_occ_prop[n_props][prop] >= max_occ:
                if n_props not in self.prohibited_props_at_n:
                    self.prohibited_props_at_n[n_props] = []
                self.prohibited_props_at_n[n_props].append(prop)
                logger.info(
                    "Prohibiting %s for n_props=%d",
                    prop,
                    n_props,
                )

    def partial_reset(self) -> None:
        """Reinitialize the rule set, keeping the prompt_ids"""
        self.n_occ_prop = {}
        self.prohibited_props_at_n = {}


@dataclass
class DatasetConfig:
    """Configuration for the MolGenerationInstructionsDatasetGenerator"""

    data_path: str
    max_n_props: int = 3
    props_prob: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    vina: bool = True
    split_docking: List[float] = field(default_factory=lambda: [1])
    probs_docking_targets: float = 0.2
    max_occ: int = 4
    max_docking_per_prompt: int = 2
    min_n_pocket_infos: int = -1
    chat_template: Dict[str, str] = field(
        default_factory=lambda: {"user": "role", "content": "content"}
    )


class MolGenerationInstructionsDatasetGenerator:
    """A simple Dataset generating rule-based prompts for molecule generation"""

    def __init__(
        self,
        config: DatasetConfig,
    ):
        """
        :param max_n_props: Maximal number of properties to optimize
        """
        self.config = config
        self.system_prompt = (
            "A conversation between User and Assistant. "
            "The User asks a question, and the Assistant solves it.\n"
            "The reasoning process is enclosed within <think> </think>"
            " and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> "
            "<answer> SMILES here </answer>."
        )
        properties_csv = pd.read_csv(
            os.path.join(os.path.dirname(config.data_path), "properties.csv")
        )
        self.possible_smiles = properties_csv[
            properties_csv.smiles.apply(len) < 32
        ].smiles
        self.chat_temp = config.chat_template

        self.docking_targets: List[str] = []
        self.prop_name_mapping: Dict[str, str] = {}
        self.pockets_info: Dict[str, Any] = {}

        self.add_pocket_info = config.min_n_pocket_infos > 0
        self.min_n_pocket_infos = config.min_n_pocket_infos
        self._load_props(config.data_path)
        self.max_n_props = config.max_n_props

        self.std_properties: List[str] = [
            k
            for k in self.prop_name_mapping
            if self.prop_name_mapping[k] not in self.docking_targets
            and k in CLASSICAL_PROPERTIES_NAMES
        ]
        self.docking_properties: List[str] = []
        self.docking_properties_split: List[List[str]] = [[]] * len(
            config.split_docking
        )
        if config.vina:
            # shuffle the docking properties
            self.docking_properties = [
                k
                for k in self.prop_name_mapping
                if self.prop_name_mapping[k] in self.docking_targets
            ]
            self._extract_splits(config.split_docking)  # Train, Val, Test (no leakage)

        self.obj_templates: Dict[str, List[str]] = OBJECTIVES_TEMPLATES
        self.templates: List[str] = PROMPT_TEMPLATE
        self.prop_key_list = list(self.prop_name_mapping.keys())

        self.rule_set = RuleSet(
            probs_docking_targets=self.config.probs_docking_targets,
            max_occ=self.config.max_occ,
            max_docking_per_prompt=self.config.max_docking_per_prompt,
        )
        self.needs_dock = False  # Only for ood

    @staticmethod
    def _get_allowed_props(
        original_prop_list: List[str], rule_set: RuleSet, n_props: int
    ) -> List[str]:
        return [
            p
            for p in original_prop_list
            if p not in rule_set.prohibited_props_at_n.get(n_props, [])
        ]

    def _load_props(self, path: str) -> None:
        assert os.path.exists(path)
        docking_target_list_path = os.path.join(path, "docking_targets.json")
        prop_name_mapping_path = os.path.join(path, "names_mapping.json")
        pocket_info_path = os.path.join(path, "pockets_info.json")

        assert os.path.exists(docking_target_list_path), (
            f"File {docking_target_list_path} does not exist. Please check the data path."
        )
        assert os.path.exists(prop_name_mapping_path), (
            f"File {prop_name_mapping_path} does not exist. Please check the data path."
        )
        assert os.path.exists(pocket_info_path), (
            f"File {pocket_info_path} does not exist. Please check the data path."
        )

        with open(docking_target_list_path) as f:
            self.docking_targets = json.load(f)
        with open(prop_name_mapping_path) as f:
            self.prop_name_mapping = json.load(f)
        with open(pocket_info_path) as f:
            self.pockets_info = json.load(f)

    def _extract_splits(self, split_docking: List[float]) -> None:
        rng = np.random.default_rng(0)
        rng.shuffle(self.docking_properties)
        i0 = 0
        for idx, p in enumerate(split_docking):
            i1 = i0 + int(len(self.docking_properties) * p)
            i1 = min(i1, len(self.docking_properties))
            self.docking_properties_split[idx] = self.docking_properties[i0:i1]
            i0 = i1
        # Verify that there is no leakage
        assert (
            np.intersect1d(
                self.docking_properties_split[0], self.docking_properties_split[1]
            ).shape[0]
            == 0
        )
        assert (
            np.intersect1d(
                self.docking_properties_split[0], self.docking_properties_split[2]
            ).shape[0]
            == 0
        )
        assert (
            np.intersect1d(
                self.docking_properties_split[1], self.docking_properties_split[2]
            ).shape[0]
            == 0
        )

    def fill_prompt(
        self, props: List[str], objs: List[str], has_docking: bool
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Takes a list of properties and corresponding objectives
        and returns a diverse natural language prompt for multi-objective optimization.
        """
        if len(props) != len(objs):
            raise ValueError("props and objs must have the same length.")

        # Phrase templates for each type of objective
        phrases = [""]  # Empty string for the join method later
        phrases_mm = [""]  # to add the prefix in front of the 1st obj
        path_to_mm_object = []
        for prop, obj in zip(props, objs):
            obj_l = obj.lower()
            phrase: str = random.choice(self.obj_templates[obj_l.split()[0]])
            if obj_l.startswith("maximize") or obj_l.startswith("minimize"):
                phrase = phrase.format(prop=prop)
            elif obj_l.startswith("above"):
                val = obj_l.split()[-1]
                phrase = phrase.format(prop=prop, val=val)
            elif obj_l.startswith("below"):
                val = obj_l.split()[-1]
                phrase = phrase.format(prop=prop, val=val)
            elif obj_l.startswith("equal"):
                val = obj_l.split()[-1]
                phrase = phrase.format(prop=prop, val=val)
            else:
                raise ValueError("Unknown objective.")

            phrases.append(phrase)
            # Check if it is a docking target
            if self.prop_name_mapping.get(prop, prop) in self.docking_targets:
                phrases_mm.append("<|image_pad|> " + phrase)
                path_to_mm_object.append(
                    {
                        "type": "image",  # HACK
                        "path": os.path.join(
                            "pockets_embeddings",
                            self.prop_name_mapping.get(prop, prop) + "_embeddings.pt",
                        ),
                    }
                )
            else:
                phrases_mm.append(phrase)

        # Top-level prompt templates
        prompt: str = random.choice(self.templates).split("|")[int(len(props) > 1)]
        full_prompt = prompt.format(objectives="\n    - ".join(phrases))
        prompt_mm = prompt.format(objectives="\n    - ".join(phrases_mm))

        if has_docking:
            full_prompt += DOCKING_SUFFIX
            prompt_mm += DOCKING_SUFFIX

        full_prompt_mm: List[Dict[str, str]] = [
            {"type": "text", "text": prompt_mm}
        ] + path_to_mm_object

        return full_prompt, full_prompt_mm

    def _get_prompt_metadata(
        self,
        properties: List[str],
        objectives: List[str],
        identifier: str,
        n_props: int,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        metadata["properties"] = [self.prop_name_mapping[p] for p in properties]
        metadata["objectives"] = [obj.split(" ")[0] for obj in objectives]
        metadata["target"] = [
            0.0 if len(obj.split(" ")) == 1 else float(obj.split(" ")[1])
            for obj in objectives
        ]
        metadata["prompt_id"] = identifier
        metadata["n_props"] = n_props
        metadata["docking_metadata"] = []
        for p in properties:
            if self.prop_name_mapping[p] in self.docking_targets:
                pdb_id = self.prop_name_mapping[p]
                if pdb_id in self.pockets_info:
                    infos = {}

                    pocket_data = self.pockets_info[pdb_id]
                    if not isinstance(pocket_data, dict):
                        pocket_data = dict(pocket_data)
                    infos = pocket_data
                    if "pdb_id" not in infos:
                        infos["pdb_id"] = pdb_id
                else:
                    infos = {"pdb_id": pdb_id.split("_")[0]}
                metadata["docking_metadata"].append(infos)
        return metadata

    def _sample_properties(
        self, n_props: int, docking_properties_list: List[str]
    ) -> List[str]:
        allowed_docking_props = self._get_allowed_props(
            docking_properties_list, self.rule_set, n_props=n_props
        )
        allowed_std_props = self._get_allowed_props(
            self.std_properties, self.rule_set, n_props=n_props
        )

        if len(allowed_docking_props) == 0:
            probas = None
        else:
            allowed_docking_props = np.random.choice(
                allowed_docking_props,
                min(self.rule_set.max_docking_per_prompt, len(allowed_docking_props)),
                replace=False,
            ).tolist()

            probas_docking = np.ones(len(allowed_docking_props)) / len(
                allowed_docking_props
            )
            probas_docking = (
                probas_docking
                * self.rule_set.probs_docking_targets
                / (1 - self.rule_set.probs_docking_targets)
            )
            if n_props == 1:  # We consider all props equally
                probas_std = np.ones(len(allowed_std_props)) / len(allowed_std_props)
            else:
                probas_std = np.array(
                    [
                        PROPERTY_ALLOWED_OBJECTIVES[self.prop_name_mapping[p]][
                            "frequency"
                        ]
                        for p in allowed_std_props
                    ]
                )
                probas_std = probas_std / probas_std.sum()
            probas = np.concatenate([probas_docking, probas_std])
            probas = probas / probas.sum()

        property_list = allowed_docking_props + allowed_std_props

        properties = list(
            random.choice(property_list, n_props, replace=False, p=probas)
        )
        # If n_props>=2 or needs dock, we ensure that we have at least one docking property
        if (n_props >= 2 or self.needs_dock) and len(
            np.intersect1d(allowed_docking_props, properties)
        ) == 0:
            properties[0] = allowed_docking_props[0]
        return properties

    def add_pocket_info_to_prompt(
        self, full_prompt: str, pocket_data: Dict[str, Any]
    ) -> str:
        if not pocket_data == {}:
            new_sentence = (
                " Here are some descriptors of the pocket"
                + "s" * int(len(pocket_data) > 1)
                + " we want the compound to bind to: \n"
            )
            for p in pocket_data:
                new_sentence += p + ":\n"
                for k in pocket_data[p]:
                    new_sentence += "    " + k + ":" + str(pocket_data[p][k]) + "\n"

            full_prompt += new_sentence
        return full_prompt

    def get_obj_from_prop(self, properties: List[str]) -> List[str]:
        objectives: List[str] = []

        solo_docking = True  # First docking property sees its values taken from DOCKING_SOLO_OBJECTIVES
        for prop in properties:
            short_prop = self.prop_name_mapping[prop]
            if solo_docking and short_prop in self.docking_targets:
                obj = random.choice(
                    DOCKING_SOLO_OBJECTIVES[0], p=DOCKING_SOLO_OBJECTIVES[1]
                )
                solo_docking = False
            elif short_prop in self.docking_targets:
                obj = random.choice(DOCKING_OBJECTIVES[0], p=DOCKING_OBJECTIVES[1])
            else:
                obj = random.choice(
                    PROPERTY_ALLOWED_OBJECTIVES[short_prop]["objectives"],
                    p=PROPERTY_ALLOWED_OBJECTIVES[short_prop]["p_objectives"],
                )

            if obj in TARGET_VALUE_OBJECTIVES:
                # Find the value to target
                if short_prop in self.docking_targets:
                    v = random.random() * 0.4 + 0.1
                    # Only docking scores between -10 and -7
                else:
                    v = random.random() * 0.6 + 0.2

                obj += f" {inverse_rescale_property_values(short_prop, v / 10, short_prop in self.docking_targets):.2f}"
            objectives.append(obj)
        return objectives

    def generate_text_prompts(
        self,
        properties: List[str],
        objectives: List[str],
        has_docking: bool,
    ) -> Dict[str, List[Dict[str, Any]]]:
        prompt_text, prompt_multimodal = self.fill_prompt(
            properties, objectives, has_docking=has_docking
        )

        sys_prompt = self.system_prompt

        prompt: Dict[str, List[Dict[str, Any]]] = {
            "standard": [
                {
                    self.chat_temp["user"]: "system",
                    self.chat_temp["content"]: sys_prompt,
                },
                {
                    self.chat_temp["user"]: "user",
                    self.chat_temp["content"]: prompt_text,
                },
            ],
            "multimodal": [
                {
                    self.chat_temp["user"]: "system",
                    self.chat_temp["content"]: [{"type": "text", "text": sys_prompt}],
                },
                {
                    self.chat_temp["user"]: "user",
                    self.chat_temp["content"]: prompt_multimodal,
                },
            ],
        }
        return prompt

    def generate(
        self,
        n: int,
        docking_properties_list: List[str],
    ) -> Iterator[Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]]:
        """
        Generates n prompts randomly to generate molecules
        :param n: number of prompts to generate
        :param return_n_props: if True, returns the number of properties to optimize
        :return:
        """

        for i_sampled in range(n):
            n_props_probs = np.array(self.config.props_prob)
            for i in range(len(n_props_probs)):
                if len(self.rule_set.prohibited_props_at_n.get(i + 1, [])) > 0:
                    if i == 0:
                        if len(self.rule_set.prohibited_props_at_n[1]) >= len(
                            docking_properties_list
                        ) + len(self.std_properties):
                            n_props_probs[i] = 0
                    else:
                        if len(self.rule_set.prohibited_props_at_n[i + 1]) >= len(
                            docking_properties_list
                        ):
                            n_props_probs[i] = 0
            n_props_probs = n_props_probs / n_props_probs.sum()
            possible_n = np.arange(1, self.max_n_props + 1)
            n_props: int = int(np.random.choice(possible_n, p=n_props_probs))
            properties = self._sample_properties(n_props, docking_properties_list)
            objectives = self.get_obj_from_prop(properties)

            identifier = "".join(
                sorted(
                    [
                        str(self.prop_key_list.index(prop))
                        + str(OBJECTIVES.index(obj.split(" ")[0]))
                        + str(
                            int(float(obj.split(" ")[-1]) * 10)
                            if len(obj.split(" ")) > 1
                            else 0
                        )
                        for prop, obj in zip(properties, objectives)
                    ]
                )
            )
            has_docking = any(
                self.prop_name_mapping[p] in self.docking_targets for p in properties
            )

            prompt = self.generate_text_prompts(
                properties, objectives, has_docking=has_docking
            )

            metadata = self._get_prompt_metadata(
                properties, objectives, identifier, n_props
            )
            yield prompt, metadata

    def generate_with_rule(
        self, n: int, docking_split: int = 0
    ) -> Iterator[
        Tuple[
            Dict[str, List[Dict[str, Any]]],
            Dict[str, Any],
        ]
    ]:
        """
        Generates prompts, with at most n tries to obtain a prompt that meets the rule.
        """
        docking_prop_list: List[str] = self.docking_properties_split[docking_split]
        for _ in range(n):
            found = False
            for prompt, metadata in self.generate(
                4 * n, docking_properties_list=docking_prop_list
            ):
                allowed = self.rule_set.verify_and_update(metadata)
                if allowed:
                    found = True
                    break
            if not found:
                break
            yield prompt, metadata

    def generate_dataset(self, n: int, docking_split: int) -> Dataset:
        assert docking_split < len(self.docking_properties_split), (
            f"docking_split must be less than the number of docking splits, here:{len(self.docking_properties_split)}"
        )
        data_dict: Dict[str, Any] = {
            "prompt": [],
            "prompt_multimodal": [],
            "properties": [],
            "objectives": [],
            "target": [],
            "prompt_id": [],
            "n_props": [],
            "docking_metadata": [],
        }
        p_bar = tqdm(total=n)
        for prompt, metadata in self.generate_with_rule(n, docking_split=docking_split):
            jsonize_dict(metadata)
            data_dict["prompt"].append(prompt["standard"])
            data_dict["prompt_multimodal"].append(prompt["multimodal"])
            for k in metadata:
                if k in data_dict:
                    data_dict[k].append(metadata[k])
            tqdm.update(p_bar)
        # Reinitialize rule set
        self.rule_set = RuleSet(
            probs_docking_targets=self.config.probs_docking_targets,
            max_occ=self.config.max_occ,
            max_docking_per_prompt=self.config.max_docking_per_prompt,
        )
        return data_dict


def data_dict_to_hf_dataset(data_dict: dict) -> Dataset:
    hf_dataset = Dataset.from_dict(data_dict)
    return hf_dataset


def data_dict_to_pydantic(
    data_dict: dict, key: str = "prompt", origin: str = "train"
) -> List[Sample]:
    sample_list: List[Sample] = []

    for i in range(len(data_dict[key])):
        messages: List[Message] = [Message(**msg) for msg in data_dict[key][i]]
        conv = Conversation(
            messages=messages,
            system_prompt=None,
            available_tools=None,
            truncate_at_max_tokens=None,
            truncate_at_max_image_tokens=None,
            output_modalities=None,
            identifier=data_dict["prompt_id"][i],
            references=[],
            rating=None,
            source=None,
            training_masks_strategy="none",
            custom_training_masks=None,
            meta={
                "properties": data_dict["properties"][i],
                "objectives": data_dict["objectives"][i],
                "target": data_dict["target"][i],
                "prompt_id": data_dict["prompt_id"][i],
                "n_props": data_dict["n_props"][i],
                "n_docking_props": len(data_dict["docking_metadata"][i]),
                "docking_metadata": data_dict["docking_metadata"][i],
                "source": origin,
            },
        )
        sample_list.append(
            Sample(
                identifier=data_dict["prompt_id"][i],
                conversations=[conv],
                trajectories=[],
                meta={},
                source=None,
            )
        )
    return sample_list


def jsonize_dict(d: Dict[Any, Any]) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            jsonize_dict(v)
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
