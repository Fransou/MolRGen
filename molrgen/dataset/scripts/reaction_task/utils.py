import argparse
from typing import List, Tuple

import numpy as np

from molrgen.data.reactions.mol import Molecule
from molrgen.data.reactions.reaction_matrix import ReactantReactionMatrix
from molrgen.data.reactions.utils import PROMPT_TASKS


class ReactionTaskSampler:
    def __init__(
        self,
        args: argparse.Namespace,
        reaction_matrix: ReactantReactionMatrix,
    ) -> None:
        self.args = args
        self.reaction_matrix = reaction_matrix
        self.all_reactants = reaction_matrix.reactants
        self.all_reactants_csmi = [r.csmiles for r in self.all_reactants]
        self.all_reactions = reaction_matrix.reactions

        self.choose_idx_reaction = [
            "reactant",
            "all_reactants",
            "all_reactants_bb_ref",
            "smarts",
        ]

    def get_eval_obj_and_label(self, product: str) -> Tuple[str, int, List[str]]:
        prop = "full_path_bb_ref" if self.args.n_bb_max > 0 else "full_path"
        label = [product]
        idx_chosen = 0
        return prop, idx_chosen, label

    def get_training_obj_and_label(
        self, reactants: List[List[str]], products: List[str], or_smarts: List[str]
    ) -> Tuple[str, int, List[str], List[List[str]], List[str], List[str]]:
        prop: str
        if len(reactants) == 1:
            prop = np.random.choice(
                PROMPT_TASKS[:-2],
                p=np.array(self.args.proba_obj[:-2]) / sum(self.args.proba_obj[:-2]),
            )
        else:
            prop = np.random.choice(PROMPT_TASKS, p=self.args.proba_obj)

        idx_chosen: int = 0
        label: List[str] = ["n/a"]
        if prop in self.choose_idx_reaction:
            idx_chosen = int(np.random.choice(list(range(len(reactants)))))
            if prop == "smarts":
                # If objective is SMARTS we select a random step and we only
                # keep the corresponding reaction template and reactants/products
                reactants = [reactants[idx_chosen]]
                products = [products[idx_chosen]]
                or_smarts = [or_smarts[idx_chosen]]
                label = [or_smarts[0]]
            else:
                # We keep all steps up until the idx_chosen step, and we remove
                # the labels from the reaction, replacing them with ???
                if idx_chosen == 0:
                    last_product = None
                else:
                    last_product = products[idx_chosen - 1]
                poss_label = [r for r in reactants[idx_chosen] if r != last_product]
                #  If there is no possible label, we set idx_chosen to 0
                if len(poss_label) == 0:
                    idx_chosen = 0
                    last_product = None
                    poss_label = [r for r in reactants[idx_chosen] if r != last_product]

                reactants = reactants[: idx_chosen + 1]
                products = products[: idx_chosen + 1]
                or_smarts = or_smarts[: idx_chosen + 1]
                if prop in ["all_reactants", "all_reactants_bb_ref"]:
                    label = poss_label
                else:
                    label = [np.random.choice(poss_label)]
                reactants[idx_chosen] = [
                    "???" if r in label else r for r in reactants[idx_chosen]
                ]
        elif prop == "final_product" or prop.startswith("full_path"):
            label = [products[-1]]
        else:
            raise NotImplementedError(f"Unknown property {prop}")
        return prop, idx_chosen, label, reactants, products, or_smarts

    def get_building_blocks_ref(
        self, prop: str, reactants: List[List[str]], target_smi: str
    ) -> Tuple[List[str], List[str]]:
        if prop not in [
            "all_reactants_bb_ref",
            "full_path_bb_ref",
            "full_path_smarts_bb_ref",
            "full_path_intermediates_gt_reactants",
        ]:
            return [], []
        original_building_blocks = []
        for l_reactants in reactants:
            for smi in l_reactants:
                mol = Molecule(smi)
                if mol.csmiles in self.all_reactants_csmi:
                    original_building_blocks.append(smi)
        assert len(original_building_blocks) > 0 or reactants == [] or prop == "smarts"

        if prop == "full_path_reordering":
            np.random.shuffle(original_building_blocks)
            return original_building_blocks, original_building_blocks

        if prop == "all_reactants_bb_ref":
            n_bb_max = np.random.choice(
                [
                    self.args.n_bb_max,
                    self.args.n_bb_max // 2,
                    self.args.n_bb_max // 4,
                ],
                p=[0.25, 0.25, 0.5],
            )
            bb = (
                original_building_blocks
                + np.random.choice(
                    self.all_reactants_csmi, size=n_bb_max, replace=False
                ).tolist()
            )
            bb = bb[:n_bb_max]
            bb = list(set(bb))
        elif prop == "full_path_intermediates_gt_reactants":
            # Add original building blocks plus some random ones
            n_bb_max = np.random.choice(
                [
                    self.args.n_bb_max,
                    self.args.n_bb_max // 2,
                    self.args.n_bb_max // 4,
                ],
                p=[0.4, 0.3, 0.3],
            )
            bb = list(
                set(
                    original_building_blocks
                    + np.random.choice(
                        self.all_reactants_csmi, size=n_bb_max, replace=False
                    ).tolist()
                )
            )
        else:
            n_bb_max = np.random.choice(
                [
                    self.args.n_bb_max,
                    self.args.n_bb_max // 2,
                    self.args.n_bb_max // 4,
                ],
                p=[0.4, 0.3, 0.3],
            )
            bb = self.get_building_blocks_tanim_sim(
                target_smi,
                n_bb_max,
                [Molecule(smi) for smi in original_building_blocks],
            )

        np.random.shuffle(bb)
        return bb, original_building_blocks

    def get_building_blocks_tanim_sim(
        self,
        target_smi: str,
        n_bb_max: int,
        or_bb: List[Molecule],
    ) -> List[str]:
        # Find the most similar building blocks
        target_mol = Molecule(target_smi)

        ### We only add similar building blocks that can be part of the last reaction
        # Find all reactions where the target can be obtained
        poss_reactions = self.all_reactions.match_product_reactions(target_mol)
        if len(poss_reactions) == 0:
            idx_poss_last = np.array(list(range(len(self.all_reactants))))
        else:
            # Only keep the reactants that can participate in one of these reactions
            idx_poss_last = np.where(
                (self.reaction_matrix.matrix[:, np.array(poss_reactions)] > 0).any(1)
            )[0]
        reactants_for_tanimoto_sim = [self.all_reactants[i] for i in idx_poss_last]
        # Compute Tanimoto similarity
        tanimoto_sim = target_mol.tanimoto_similarity(reactants_for_tanimoto_sim)
        sorted_idx = np.argsort(tanimoto_sim)[::-1]
        # Get the most similar building blocks for 50% of n_bb_max
        building_blocks = [reactants_for_tanimoto_sim[i] for i in sorted_idx[:n_bb_max]]
        building_blocks = list(set(building_blocks))
        return [bb.csmiles for bb in building_blocks]

    def get_smarts(self, or_smarts: List[str], prop: str) -> List[str]:
        if prop not in ["full_path_smarts_ref", "full_path_smarts_bb_ref"]:
            return or_smarts
        assert self.args.n_smarts_max >= len(or_smarts) * 2, (
            f"n_smarts_max {self.args.n_smarts_max} too small for or_smarts length {len(or_smarts)}"
        )

        n_smarts_max = np.random.randint(
            low=len(or_smarts), high=self.args.n_smarts_max - len(or_smarts) + 1
        )
        idx_random_reactions = np.random.choice(
            list(range(len(self.all_reactions))), n_smarts_max, replace=False
        )
        smarts = [
            self.all_reactions._reactions[i].smarts for i in idx_random_reactions
        ] + or_smarts
        smarts = list(set(smarts))
        np.random.shuffle(smarts)
        return smarts

    def sample(
        self,
        reactants: List[List[str]],
        products: List[str],
        or_smarts: List[str],
    ) -> Tuple[
        str, int, List[str], List[str], List[str], List[str], List[List[str]], List[str]
    ]:
        prop, idx_chosen, label, reactants, products, or_smarts = (
            self.get_training_obj_and_label(reactants, products, or_smarts)
        )
        if prop == "all_reactants_bb_ref":
            # If the objective is to find all reactants with building block reference,
            # We only provide building block reference for the chosen step.
            # This information is only present in the label as we replaced the other reactants with ???
            bb, or_bb = self.get_building_blocks_ref(prop, [label], products[-1])
        else:
            bb, or_bb = self.get_building_blocks_ref(prop, reactants, products[-1])
        smarts = self.get_smarts(or_smarts, prop)
        return prop, idx_chosen, label, bb, smarts, or_bb, reactants, products

    def sample_eval(
        self,
        product: str,
    ) -> Tuple[str, int, List[str], List[str]]:
        prop, idx_chosen, label = self.get_eval_obj_and_label(product)
        bb, or_bb = self.get_building_blocks_ref(prop, [], product)
        return prop, idx_chosen, label, bb


### JINJA TEMPLATES

product_reactant_jinja = """You are given an incomplete synthetic route, with a missing reactant or product (replaced by ???). Find the missing element of the synthetic route.
You are provided the molecular reaction template of the last step in the SMARTS format.

Reaction SMARTS:
    {{ smarts[idx_chosen] }}
Incomplete Reaction:
    {% for r, p in zip(reactants, products) -%}
        {{ " + ".join(r) }} >> {{ p.replace(target[0], "???") }}
    {% endfor %}
Provide your answer in the json format, in the "answer" field:
{% raw %}{
    "answer": "missing element"
}{% endraw %}"""


all_reactants_jinja = """You are given an incomplete synthetic route, with some reactants replaced by ???. Find the missing elements of this synthetic route in order to complete it.
You are provided the molecular reaction template of the last step in the SMARTS format.

Reaction SMARTS:
    {{ smarts[idx_chosen] }}
Incomplete Reaction:
    {% for r, p in zip(reactants, products) -%}
        {{ " + ".join(r) }} >> {{ p.replace(target[0], "???") }}
    {% endfor %}
Provide your answer in the json format, in the "answer" field:
{% raw %}{
    "answer": ["missing element 1", "missing element 2", ...]
}{% endraw %}"""

all_reactants_bbref_jinja = """You are given an incomplete synthetic route, with some reactants replaced by ???. Find the missing elements of this synthetic route in order to complete it.
You are provided the molecular reaction template of the last step in the SMARTS format.

Choose your reactants among the following building blocks:
{% for bb in building_blocks -%}
- {{ bb }}
{% endfor %}

Reaction SMARTS:
    {{ smarts[idx_chosen] }}
Incomplete Reaction:
    {% for r, p in zip(reactants, products) -%}
        {{ " + ".join(r) }} >> {{ p.replace(target[0], "???") }}
    {% endfor %}
Provide your answer in the json format, in the "answer" field:
{% raw %}{
    "answer": ["missing element 1", "missing element 2", ...]
}{% endraw %}"""

smarts_jinja = """You are given a molecular reaction with reactants and products in the SMILES format. Given this list of reactants and products, find which reaction it corresponds to and provide its SMARTS representation.

Reaction:
    {% for r, p in zip(reactants, products) -%}
        {{ " + ".join(r) }} >> {{ p }}
    {% endfor %}
Provide your answer in the json format, in the "answer" field:
{% raw %}{
    "answer": "reaction smarts"
}{% endraw %}"""

full_path_inter_jinja = """Given a target molecule to synthesize, provide a full synthetic route to synthesize it.
To achieve this, you will need to synthesize the intermediate products that can then be used as reactants in subsequent steps.
The intermediate products are given in a shuffled order, and you must determine the correct order to use them in the synthesis.{% if "full_path_intermediates_gt_reactants" in objectives %} You are also provided with a list of commercially available building blocks that you should use to synthesize the target molecule.{% endif %}

{% if "full_path_intermediates_gt_reactants" in objectives -%}
Building Blocks:
    {% for bb in building_blocks -%}
    - {{ bb }}
    {% endfor %}
{%- endif %}
Intermediate Products:
    {% for p in intermediate_products -%}
    - {{ p }}
    {% endfor %}
Target Molecule:
    {{ target[0] }}

Provide your answer in the json format, in the "answer" field:
{% raw %}{
    "answer": [
        {
            "step": 1,
            "reactants": ["reactant1_smiles", "reactant2_smiles"],
            "products": ["product_smiles1"]
        },
        {
            "step": 2,
            "reactants": ["reactant3_smiles", "product_smiles1"],
            "products": ["product_smiles2"]
        },
        ...
    ]
}{% endraw %}
"""

full_path_jinja = """Given a target molecule to synthesize, provide a full synthetic route to synthesize it from commercially available building blocks in at most 5 steps.{% if "full_path_bb_ref" in objectives or "full_path_smarts_bb_ref" in objectives %} You are provided the top-{{ building_blocks.__len__() }} most similar building blocks to the target molecule, that you may or may not use for your synthesis.{% endif %}{% if "full_path_smarts_bb_ref" in objectives or "full_path_smarts_ref" in objectives %} You are provided the available reaction templates (SMARTS) that you can use to perform the synthesis.{% endif %}

{% if "full_path_bb_ref" in objectives or "full_path_smarts_bb_ref" in objectives -%}
Most similar Building Blocks:
    {% for bb in building_blocks -%}
    - {{ bb }}
    {% endfor -%}
{%- endif %}
{%- if "full_path_smarts_bb_ref" in objectives or "full_path_smarts_ref" in objectives %}
Available Reaction SMARTS:
    {% for s in smarts -%}
    - {{ s }}
    {% endfor -%}
{%- endif %}
Target Molecule:
    {{ target[0] }}

Provide your answer in the json format, in the "answer" field:
{% raw %}{
    "answer": [
        {
            "step": 1,
            "reactants": ["reactant1_smiles", "reactant2_smiles"],
            "products": ["product_smiles1"]
        },
        {
            "step": 2,
            "reactants": ["reactant3_smiles", "product_smiles1"],
            "products": ["product_smiles2"]
        },
        ...
    ]
}{% endraw %}"""

full_jinja = (
    """
{%- if "full_path_intermediates" == objectives[0] or "full_path_intermediates_gt_reactants" == objectives[0] -%}
"""
    + full_path_inter_jinja
    + """
{%- elif "full_path" in objectives[0] -%}
"""
    + full_path_jinja
    + """
{%- elif "smarts" == objectives[0] -%}
"""
    + smarts_jinja
    + """
{%- elif "all_reactants" == objectives[0] -%}
"""
    + all_reactants_jinja
    + """
{%- elif "all_reactants_bb_ref" == objectives[0] -%}
"""
    + all_reactants_bbref_jinja
    + """
{%- else -%}
"""
    + product_reactant_jinja
    + """
{%- endif -%}
"""
)
