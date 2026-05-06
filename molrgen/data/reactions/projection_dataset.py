from itertools import permutations, product
from typing import Iterator

import ray
from ray.experimental import tqdm_ray
from torch.utils.data import IterableDataset

from molrgen.data.reactions.mol import Molecule
from molrgen.data.reactions.reaction import Reaction
from molrgen.data.reactions.reaction_matrix import ReactantReactionMatrix
from molrgen.data.reactions.stack import (
    Stack,
    create_stack_ray,
    pass_filters_p,
)


class TextualProjectionDataset(IterableDataset[ReactantReactionMatrix]):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        max_num_atoms: int = 80,
        max_smiles_len: int = 192,
        max_num_reactions: int = 5,
        init_stack_weighted_ratio: float = 0.0,
        n_retry: int = 1,
        n_attempts_per_reaction: int = 1,
        virtual_length: int = 65536,
        ray_batch_size: int = 64,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_smiles_len = max_smiles_len
        self._max_num_reactions = max_num_reactions
        self._init_stack_weighted_ratio = init_stack_weighted_ratio
        self._virtual_length = virtual_length
        self._n_retry = n_retry
        self._n_attempts_per_reaction = n_attempts_per_reaction
        self._ray_batch_size = ray_batch_size

    def __len__(self) -> int:
        return self._virtual_length

    def iter_ray(
        self,
    ) -> Iterator[tuple[list[list[str]], list[str], list[str], list[bool]]]:
        if not ray.is_initialized():
            ray.init()
        reaction_matrix_ref = ray.put(self._reaction_matrix)
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        pbar = remote_tqdm.remote(total=self._virtual_length)
        running_tasks: list[ray.ObjectRef] = []
        for i in range(self._virtual_length):
            if len(running_tasks) >= self._ray_batch_size:
                done_ids, running_tasks = ray.wait(running_tasks)
                stacks = ray.get(done_ids)
                for stack in stacks:
                    if stack is None:
                        continue
                    out = self.post_process_stack(stack)
                    if out is not None:
                        yield out
            task = create_stack_ray.remote(
                reaction_matrix_ref,
                max_num_reactions=self._max_num_reactions,
                max_num_atoms=self._max_num_atoms,
                init_stack_weighted_ratio=self._init_stack_weighted_ratio,
                n_retry=self._n_retry,
                n_attempts_per_reaction=self._n_attempts_per_reaction,
                pbar=pbar,
            )
            running_tasks.append(task)

        last_tasks = ray.get(running_tasks)
        pbar.close.remote()  # type: ignore
        for stack in last_tasks:
            if stack is None:
                continue
            out = self.post_process_stack(stack)
            if out is not None:
                yield out

    def post_process_stack(
        self, stack: Stack
    ) -> tuple[list[list[str]], list[str], list[str], list[bool]] | None:
        rxn_smarts = [
            rxn.smarts for rxn in stack.rxns if rxn is not None
        ]  # TODO find how to handle this better
        is_product = [rxn is not None for rxn in stack.rxns]
        reactants: list[list[Molecule]] = [[]]
        products: list[Molecule] = []
        for mol, is_prod in zip(stack.mols, is_product):
            if not is_prod:
                reactants[-1].append(mol)
            else:
                products.append(mol)
                reactants.append([mol])
        reactants = reactants[:-1]
        try:
            reactants_smiles = [
                self.find_mol_order(r, p, smarts)
                for r, p, smarts in zip(reactants, products, rxn_smarts)
            ]
            product_smiles = [p.smiles for p in products]
        except ValueError as e:
            print(f"Error in post_process_stack: {e}")
            return None
        filter_logps = [pass_filters_p(p.smiles) for p in products]
        filters = [f for f, _, _ in filter_logps]

        return reactants_smiles, product_smiles, rxn_smarts, filters

    @staticmethod
    def find_mol_order(
        reactants: list[Molecule], prod: Molecule, smarts: str
    ) -> list[str]:
        reaction = Reaction(smarts)
        for reactants_order in permutations(reactants):
            prods = reaction(reactants_order)
            if prod in prods:
                return [r.smiles for r in reactants_order]
        raise ValueError(f"{product} not in {reactants}")

    def __iter__(
        self,
    ) -> Iterator[tuple[list[list[str]], list[str], list[str], list[bool]]]:
        return self.iter_ray()
