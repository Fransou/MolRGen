from typing import Dict

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
CLASSICAL_PROPERTIES_NAMES: Dict[str, str] = {
    # "Probability to inhibate glycogen synthase kinase-3 beta": "GSK3B",
    # "Probability to inhibate c-Jun N-terminal kinase-3": "JNK3",
    # "Bioactive probability against dopamine receptor D2": "DRD2",
    "Synthetic accessibility": "SA",
    "Quantitative estimate of drug-likeness": "QED",
    "Molecular Weight": "CalcExactMolWt",
    "Number of Aromatic Rings": "CalcNumAromaticRings",
    "Number of H-bond acceptors": "CalcNumHBA",
    "Number of H-bond donors": "CalcNumHBD",
    "Number of Rotatable Bonds": "CalcNumRotatableBonds",
    "Fraction of C atoms Sp3 hybridised": "CalcFractionCSP3",
    "Topological Polar Surface Area": "CalcTPSA",
    "Hall-Kier alpha": "CalcHallKierAlpha",
    # "Hall-Kier kappa 1": "CalcKappa1",
    # "Hall-Kier kappa 2": "CalcKappa2",
    # "Hall-Kier kappa 3": "CalcKappa3",
    "Kier Phi": "CalcPhi",
    "logP": "logP",
}

RESCALE = {
    "SA": (5.470811952699881, 1.7681737515295974),
    "QED": (0.9328462736405657, 0.3399751728859344),
    "CalcExactMolWt": (481.0863048, 187.08996052009),
    "CalcNumAromaticRings": (8.0, 0.0),
    "CalcNumHBA": (10.0, 0.0),
    "CalcNumHBD": (10.0, 0.0),
    "CalcNumRotatableBonds": (15.0, 1.0),
    "CalcFractionCSP3": (1.0, 0.0),
    "CalcTPSA": (122.08, 16.61),
    "CalcHallKierAlpha": (-0.08, -4.049999999999999),
    "CalcKappa1": (23.032512685470213, 9.2146124327443),
    "CalcKappa2": (9.997827635363464, 3.483474762113614),
    "CalcKappa3": (6.342258100784633, 1.5863338770790505),
    "CalcPhi": (7.3951454949896, 2.3228902519325008),
    "logP": (3.00048897578959, -6.37141324990104),
}

INTEGER_PROPS = [
    "CalcNumAromaticRings",
    "CalcNumHBA",
    "CalcNumHBD",
    "CalcNumRotatableBonds",
]


def has_bridged_bond(mol: Chem.Mol | None) -> bool:
    """
    Returns True if the molecule contains a bridged ring system.
    A bridged system is defined as two rings sharing more than two atoms.
    """
    if mol is None:
        return True  # If None, we consider it invalid / bridged
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Compare all ring pairs
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            shared_atoms = set(atom_rings[i]) & set(atom_rings[j])
            if len(shared_atoms) > 2:  # more than 2 shared atoms → bridged
                return True
    return False


def rescale_property_values(
    prop_name: str, value: float, docking: bool = False
) -> float:
    if docking:
        # Rescale the values by adding 11, dividing by 10
        # a docking score of -10 is therefore a 0.1 and -7 is 0.4
        return (value + 11) / 10
    if prop_name not in RESCALE:
        return value
    max_val, min_val = RESCALE[prop_name]
    return (value - min_val) / (max_val - min_val)


def inverse_rescale_property_values(
    prop_name: str,
    value: float,
    docking: bool,
) -> float:
    if docking:
        return 10 * value - 11
    if prop_name not in RESCALE:
        return value
    max_val, min_val = RESCALE[prop_name]
    value = value * (max_val - min_val) + min_val
    if prop_name in INTEGER_PROPS:
        value = int(value)
        # if value <= min_val + 1:
        #     value = min_val + 2
        # elif value >= max_val - 1:
        #     value = max_val - 2
    return value
