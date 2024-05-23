# -*- coding: utf-8 -*-
"""Tokenizers."""

import re
from typing import List, Callable

__all__ = ('schwaller_smiles_regex', 'Tokenizer')


def schwaller_smiles_regex(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string into a list of string tokens.

    Args:
        smiles (str): SMILES string

    Returns:
        List[str]: List of string tokens
    """
    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

    return re.findall(SMI_REGEX_PATTERN, smiles)


class Tokenizer:
    """Class which handles tokenization and detokenization using a vocabulary.

    Taken from Kevin Jablonka's llm-from-scratch notebook.
    """
    def __init__(
        self,
        vocabulary: List[str],
        eos: str = '[EOS]',
        bos: str = '[BOS]',
        pad: str = '[PAD]',
        unk: str = '[UNK]',
    ):
        self.vocabulary = [pad, bos, eos, unk] + vocabulary
        self._token_to_index = {token: index for index, token in enumerate(self.vocabulary)}
        self.index_to_token = {index: token for index, token in enumerate(self.vocabulary)}

    def token_to_index(self, token: str) -> int:
        try:
            return self._token_to_index[token]
        except KeyError:
            return self._token_to_index['[UNK]']

    def __len__(self):
        return len(self.vocabulary)

    def __getitem__(self, item):
        return self.token_to_index(item)

    def __contains__(self, item):
        return item in self.vocabulary

    def encode(self, smiles: str, tokenizer: Callable=schwaller_smiles_regex, add_sos: bool=False, add_eos: bool=False) -> List[int]:
        """
        Encode a SMILES into a list of indices

        Args:
            smiles (str): SMILES string
            add_sos (bool): Add start of sentence token
            add_eos (bool): Add end of sentence token

        Returns:
            List[int]: List of indices
        """
        tokens = []
        if add_sos:
            tokens.append(self.token_to_index('[BOS]'))
        tokens += [self.token_to_index(token) for token in tokenizer(smiles)]
        if add_eos:
            tokens.append(self.token_to_index('[EOS]'))
        return tokens

    def decode(self, indices: List[int], strip_special_tokens: bool = True) -> str:
        """
        Decode a list of indices into a SMILES

        Args:
            indices (List[int]): List of indices

        Returns:
            str: SMILES string
        """
        decoded = ''.join([self.index_to_token[index] for index in indices])
        if '[UNK]' in decoded:
            raise ValueError('Unknown vocabulary token in decoded SMILES')
        if strip_special_tokens:
            return decoded.replace('[PAD]', '').replace('[BOS]', '').replace('[EOS]', '')
        return decoded
