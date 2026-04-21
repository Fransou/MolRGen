from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import overload

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from molrgen.data.reactions.mol import Molecule


class Template:
    def __init__(self, smarts: str) -> None:
        super().__init__()
        self._smarts = smarts.strip()

    def __getstate__(self) -> str:
        return self._smarts

    def __setstate__(self, state: str) -> None:
        self._smarts = state

    @property
    def smarts(self) -> str:
        return self._smarts

    @cached_property
    def _rdmol(self) -> Chem.Mol:
        return AllChem.MolFromSmarts(self._smarts)

    def match(self, mol: Molecule) -> bool:
        has = mol._rdmol.HasSubstructMatch(self._rdmol)
        assert isinstance(has, bool)
        return has

    def __hash__(self) -> int:
        return hash(self._smarts)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Reaction) and self.smarts == __value.smarts


class Reaction:
    def __init__(self, smarts: str) -> None:
        super().__init__()
        self._smarts = smarts.strip()

    def __getstate__(self) -> str:
        return self._smarts

    def __setstate__(self, state: str) -> None:
        self._smarts = state

    @property
    def smarts(self) -> str:
        return self._smarts

    @cached_property
    def _reaction(self) -> AllChem.ChemicalReaction:
        r = AllChem.ReactionFromSmarts(self._smarts)
        rdChemReactions.ChemicalReaction.Initialize(r)
        return r

    @cached_property
    def num_reactants(self) -> int:
        n_reac = self._reaction.GetNumReactantTemplates()
        assert isinstance(n_reac, int)
        return n_reac

    @cached_property
    def num_agents(self) -> int:
        n_agents = self._reaction.GetNumAgentTemplates()
        assert isinstance(n_agents, int)
        return n_agents

    @cached_property
    def num_products(self) -> int:
        n_prod = self._reaction.GetNumProductTemplates()
        assert isinstance(n_prod, int)
        return n_prod

    @cached_property
    def reactant_templates(self) -> tuple[Template, ...]:
        # reactant_smarts = self.smarts.split(">")[0].split(".")
        reactant_smarts = [
            Chem.MolToSmarts(self._reaction.GetReactantTemplate(i))
            for i in range(self.num_reactants)
        ]
        return tuple(Template(s) for s in reactant_smarts)

    def match_reactant_templates(self, mol: Molecule) -> tuple[int, ...]:
        matched: list[int] = []
        for i, template in enumerate(self.reactant_templates):
            if template.match(mol):
                matched.append(i)
        return tuple(matched)

    @cached_property
    def product_templates(self) -> tuple[Template, ...]:
        product_smarts = self.smarts.split(">")[2].split(".")

        # verify parenthesis
        def update_parenthesis(s: str) -> str:
            if not s[0] == "(" and not s[-1] == ")":
                return s
            open_p = s.count("(")
            close_p = s.count(")")
            if s[0] == "(" and open_p > close_p:
                s = s[1:]
                open_p -= 1
            if s[-1] == ")" and close_p > open_p:
                s = s[:-1]
                close_p -= 1
            return s

        return tuple(Template(update_parenthesis(s)) for s in product_smarts)

    def match_product_templates(self, mol: Molecule) -> bool:
        for i, template in enumerate(self.product_templates):
            if template.match(mol):
                return True
        return False

    def is_reactant(self, mol: Molecule) -> bool:
        isreac = self._reaction.IsMoleculeReactant(mol._rdmol)
        assert isinstance(isreac, bool)
        return isreac

    def is_agent(self, mol: Molecule) -> bool:
        isag = self._reaction.IsMoleculeAgent(mol._rdmol)
        assert isinstance(isag, bool)
        return isag

    def is_product(self, mol: Molecule) -> bool:
        isprod = self._reaction.IsMoleculeProduct(mol._rdmol)
        assert isinstance(isprod, bool)
        return isprod

    def __call__(self, reactants: Sequence[Molecule]) -> list[Molecule]:
        products = [
            Molecule.from_rdmol(p[0])
            for p in self._reaction.RunReactants([m._rdmol for m in reactants])
        ]
        products = [p for p in products if p.is_valid]
        return products

    def __hash__(self) -> int:
        return hash(self._smarts)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Reaction) and self.smarts == __value.smarts


class ReactionContainer(Sequence[Reaction]):
    def __init__(self, reactions: Iterable[Reaction]) -> None:
        super().__init__()
        self._reactions = tuple(reactions)

    @overload
    def __getitem__(self, index: int) -> Reaction: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Reaction, ...]: ...

    def __getitem__(self, index: int | slice) -> Reaction | tuple[Reaction, ...]:
        return self._reactions[index]

    def __len__(self) -> int:
        return len(self._reactions)

    def match_reactions(self, mol: Molecule) -> dict[int, tuple[int, ...]]:
        matched: dict[int, tuple[int, ...]] = {}
        for i, rxn in enumerate(self._reactions):
            m = rxn.match_reactant_templates(mol)
            if len(m) > 0:
                matched[i] = m
        return matched

    def match_product_reactions(self, mol: Molecule) -> list[int]:
        matched = []
        for i, rxn in enumerate(self._reactions):
            if rxn.match_product_templates(mol):
                matched.append(i)

        return matched
