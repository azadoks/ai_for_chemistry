# -*- coding: utf-8 -*-
"""Various utility functions."""

import random
import selfies
from rdkit import Chem

__all__ = ('smiles_to_inchi', 'smiles_to_selfies', 'randomize_smiles',)

def canonicalize_smiles(smiles: str, fail_val=None) -> str:
    """Return the canonical SMILES string of a given SMILES string."""
    try:
        return Chem.CanonSmiles(smiles)
    except:
        return fail_val


def smiles_to_inchi(smiles: str, fail_val=None) -> str:
    """Convert a SMILES string to an InChI string."""
    try:
        res = Chem.rdinchi.MolToInchi(Chem.MolFromSmiles(smiles))
        return res[0].split('=')[1]
    except:
        return fail_val


def smiles_to_selfies(smiles: str, strict: bool=False, fail_val=None) -> str:
    """Convert a SMILES string to a SELFIES string."""
    try:
        return selfies.encoder(smiles, strict=strict)
    except:
        return fail_val


def randomize_smiles(mol, random_type: str="restricted") -> str:
    """
    Returns a random SMILES given a SMILES of a molecule.

    Taken from:
    https://github.com/undeadpixel/reinvent-randomized/blob/df63cab67df2a331afaedb4d0cea93428ef8a9f7/utils/chem.py

    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None
    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError("Type '{}' is not valid".format(random_type))
