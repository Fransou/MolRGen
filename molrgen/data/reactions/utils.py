from typing import Callable

import torch
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

PROMPT_TASKS = [
    "final_product",
    "reactant",
    "all_reactants",
    "all_reactants_bb_ref",
    "smarts",
    "full_path",
    "full_path_bb_ref",
    "full_path_smarts_ref",
    "full_path_smarts_bb_ref",
    "full_path_intermediates",
    "full_path_intermediates_gt_reactants",
]


def get_prop(k: str, mol: Chem.rdchem.Mol) -> float:
    if k.lower() == "qed":
        return float(QED.qed(mol))
    if hasattr(rdMolDescriptors, k):
        return float(getattr(rdMolDescriptors, k)(mol))
    raise NotImplementedError(f"Unknown property {k}")


PROP_RANGE = {
    "qed": (1, 0.3),
    "CalcExactMolWt": (600, 0),
    "CalcTPSA": (160, 0),
    "CalcNumHBA": (10, 0),
    "CalcNumHBD": (10, 0),
    "CalcNumRotatableBonds": (10, 1),
    "CalcNumRings": (7, 0),
    "CalcNumAromaticRings": (6, 0),
}


def logbeta(
    a: float, b: float, min_x: float = 0.0, max_x: float = 1.0
) -> Callable[[float | torch.Tensor], float]:
    def logbeta_fn(x: float | torch.Tensor) -> float | torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x_norm = torch.tensor((x - min_x) / (max_x - min_x))
        else:
            x_norm = (x - min_x) / (max_x - min_x)
        x_norm = torch.clip(x_norm, min=1e-6, max=1 - 1e-6)
        return torch.log(x_norm) * (a - 1) + torch.log(1 - x_norm) * (b - 1)

    # Compute the normalizing constant
    x = torch.linspace(min_x, max_x, 100)
    log_cst = torch.logsumexp(logbeta_fn(x), 0) - torch.log(torch.tensor(100.0))
    return lambda x: float(logbeta_fn(x) - log_cst)


PROP_TARGET_DISTRIB_FN: dict[str, Callable[[float], float]] = {
    "qed": logbeta(5.0, 1.2, 0.0, 1.0),
    "CalcExactMolWt": logbeta(11, 11, 0, 600),
    "CalcTPSA": logbeta(4, 6, 0, 160),
    "CalcNumHBA": logbeta(3.2, 6, -1, 11),
    # "CalcNumHBD": beta(4.5,11,-1,7),
    "CalcNumRotatableBonds": logbeta(4, 5.5, -1, 11),
    "CalcNumAromaticRings": logbeta(4.5, 9, -1, 7),
}
