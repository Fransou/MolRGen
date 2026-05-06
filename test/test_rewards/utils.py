from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from molrgen.utils.property_utils import (
    CLASSICAL_PROPERTIES_NAMES,
    has_bridged_bond,
    inverse_rescale_property_values,
    rescale_property_values,
)

properties_csv = pd.read_csv("data/properties.csv", index_col=0)

# =============================================================================
# Helper Functions
# =============================================================================


def is_reward_valid(
    rewards: List[float],
    smiles: List[str],
    properties: List[str],
) -> None:
    """
    Check if the reward is valid.

    Args:
        rewards: List of reward values.
        smiles: List of SMILES strings.
        properties: List of property names.

    Raises:
        AssertionError: If rewards don't match expected values.
    """
    # Get the bridged_bond mask
    smiles = list(set(smiles))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    bridged_mask = torch.tensor(
        [not has_bridged_bond(mol) if mol is not None else False for mol in mols]
    )

    if len(smiles) > 0:
        props = torch.tensor(
            properties_csv.set_index("smiles").loc[smiles, properties].values
        )
        assert props.shape == (len(smiles), len(properties))
        # Rescale properties to [0, 1]
        for i, prop in enumerate(properties):
            props[:, i] = torch.tensor(
                rescale_property_values(
                    prop,
                    props[:, i].numpy(),
                    prop not in CLASSICAL_PROPERTIES_NAMES.values(),
                )
            )
        props = props.clip(0, 1)
        if bridged_mask.sum() == 0:
            props = torch.tensor(0.0)
        else:
            all_props = props[bridged_mask].float()
            props = all_props.prod(-1).pow(1 / len(properties)).mean()
        rewards = torch.tensor(rewards).mean()
        try:
            assert torch.isclose(rewards, props, atol=1e-3)
        except AssertionError as e:
            print(f"Rewards: {rewards}, Props: {props}")
            raise e


def compare_obj_reward_to_max(
    objective: str,
    target: float,
    rewards_max: torch.Tensor,
    mask: torch.Tensor,
    prop: str,
) -> torch.Tensor:
    """
    Compute expected reward based on objective type.

    Args:
        objective: The optimization objective (maximize, minimize, below, above, equal).
        target: Target value for threshold objectives.
        rewards_max: Raw reward values from maximize objective.
        mask: Mask for valid molecules (no bridged bonds).
        prop: Property name for rescaling.

    Returns:
        Expected reward tensor.
    """
    if objective == "maximize":
        val = rewards_max
    elif objective == "minimize":
        val = 1 - rewards_max
    else:
        target = rescale_property_values(prop, target, False)
        if objective == "below":
            val = (rewards_max <= target).float()
        elif objective == "above":
            val = (rewards_max >= target).float()
        elif objective == "equal":
            val = torch.tensor(np.clip(1 - 100 * (rewards_max - target) ** 2, 0, 1))
        else:
            raise ValueError(f"Unknown objective: {objective}")
    return val * mask


def get_unscaled_obj(obj: str, prop: str) -> tuple[str, float]:
    """
    Get unscaled objective function and target value.

    Args:
        obj: Objective string (e.g., "above 0.5").
        prop: Property name.

    Returns:
        Tuple of (objective function name, unscaled target value).
    """
    if len(obj.split()) == 1:
        return obj, 0
    func = obj.split()[0]
    val = float(obj.split()[1])
    val = inverse_rescale_property_values(
        prop, val, prop not in CLASSICAL_PROPERTIES_NAMES
    )
    return func, val
