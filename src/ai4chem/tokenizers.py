# -*- coding: utf-8 -*-
"""Tokenizers."""

import re
from typing import List, Callable

import tiktoken

__all__ = ('schwaller_smiles_regex', 'get_vocabulary', 'Tokenizer', 'gpt_num_tokens_from_messages')


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


def get_vocabulary(smiles: List[str], tokenizer: Callable=schwaller_smiles_regex) -> List[str]:
    """
    Get the vocabulary of a list of SMILES strings.

    Args:
        smiles (List[str]): List of SMILES strings

    Returns:
        List[str]: List of unique tokens
    """
    return list(set([token for smiles in smiles for token in tokenizer(smiles)]))


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

    def encode(self, smiles: str, tokenizer: Callable=schwaller_smiles_regex, add_bos: bool=False, add_eos: bool=False, pad_to_length: int=0, pad_justify: str='left') -> List[int]:
        """
        Encode a SMILES into a list of indices

        Args:
            smiles (str): SMILES string
            add_bos (bool): Add beginning of sentence token
            add_eos (bool): Add end of sentence token
            pad_to_length (int): Pad to a certain length

        Returns:
            List[int]: List of indices
        """
        smiles_tokens = tokenizer(smiles)
        tokens = []
        if add_bos:
            tokens.append(self.token_to_index('[BOS]'))
        if pad_to_length > 0 and pad_justify == 'right':
            tokens += [self.token_to_index('[PAD]')] * (pad_to_length - len(smiles_tokens))
        tokens += [self.token_to_index(token) for token in smiles_tokens]
        if pad_to_length > 0 and pad_justify == 'left':
            tokens += [self.token_to_index('[PAD]')] * (pad_to_length - len(smiles_tokens))
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


def gpt_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return gpt_num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return gpt_num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
