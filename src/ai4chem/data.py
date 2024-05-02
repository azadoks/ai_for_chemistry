# -*- coding: utf-8 -*-
"""Data loaders."""
import os
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem

__all__ = ('PhotoEmissionDataset', 'ChemFluorDataset', 'Deep4ChemDataset')


class PhotoEmissionDataset(Dataset):
    """Dataset containing molecules and their photemission properties."""

    def __init__(
        self,
        data_file: os.PathLike,
        canonicalize_smiles: bool = True,
        chromophore_smiles: str = 'SMILES',
        solvent_smiles: str = 'solvent_smiles',
        absorption_max: str = 'absorption_max',
        emission_max: str = 'emission_max'
    ):
        self._chromophore_smiles = chromophore_smiles
        self._solvent_smiles_column = solvent_smiles
        self._absorption_max_column = absorption_max
        self._emission_max_column = emission_max

        self.raw_data = pd.read_csv(data_file)
        self.clean_data = self.raw_data.dropna(axis='index')

        if canonicalize_smiles:
            self.clean_data.loc[:,self._chromophore_smiles
                            ] = self.clean_data[self._chromophore_smiles].apply(Chem.CanonSmiles)
            self.clean_data.loc[:,self._solvent_smiles_column
                            ] = self.clean_data[self._solvent_smiles_column].apply(Chem.CanonSmiles)

    def __len__(self) -> int:
        return self.raw_data.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """Get a data sample.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: ((chromophore_smiles, solvent_smiles,), (absorption_max, emission_max,)); abs./emi. max. in nm.
        """

        row = self.clean_data.iloc[idx]
        chromophore_smiles = row[self._chromophore_smiles]
        solvent_smiles = row[self._solvent_smiles_column]
        absorption_max = row[self._absorption_max_column]
        emission_max = row[self._emission_max_column]

        return ((chromophore_smiles, solvent_smiles), (
            absorption_max,
            emission_max,
        ))


class ChemFluorDataset(PhotoEmissionDataset):
    """ChemFluor dataset.

    10.1021/acs.jcim.0c01203
    """

    def __init__(self, data_file: os.PathLike, canonicalize_smiles: bool = True):
        self._chromophore_smiles = 'SMILES'
        self._solvent_smiles_column = 'solvent_smiles'
        self._absorption_max_column = 'Absorption/nm'
        self._emission_max_column = 'Emission/nm'

        super().__init__(
            data_file,
            canonicalize_smiles=canonicalize_smiles,
            chromophore_smiles=self._chromophore_smiles,
            solvent_smiles=self._solvent_smiles_column,
            absorption_max=self._absorption_max_column,
            emission_max=self._emission_max_column
        )


class Deep4ChemDataset(PhotoEmissionDataset):
    """Deep4Chem dataset.

    10.1038/s41597-020-00634-8
    """

    def __init__(self, data_file: os.PathLike, canonicalize_smiles: bool = True):
        self._chromophore_smiles = 'Chromophore'
        self._solvent_smiles_column = 'Solvent'
        self._absorption_max_column = 'Absorption max (nm)'
        self._emission_max_column = 'Emission max (nm)'

        super().__init__(
            data_file,
            canonicalize_smiles=canonicalize_smiles,
            chromophore_smiles=self._chromophore_smiles,
            solvent_smiles=self._solvent_smiles_column,
            absorption_max=self._absorption_max_column,
            emission_max=self._emission_max_column
        )
