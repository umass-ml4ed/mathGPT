from typing import Dict, Tuple
import json

from constants import TYPE_STR_TO_INT, TokenType

VOCAB_FILE_PATH = "ref_data/vocab.json"

BaseVocab = Dict[str, Dict[str, int]]

Vocab = Dict[TokenType, Dict[str, int]]

VocabInv = Dict[TokenType, Dict[int, str]]

class Vocabulary:
    _base_vocab: BaseVocab = {}
    _vocab: Vocab = {}
    _vocab_inv: VocabInv = {}
    _loaded: bool = False

    @classmethod
    def add(cls, str_type: str, symbol: str):
        """
        Add the given type to the vocab if not present, and add the given symbol to the type if not present
        """
        # Create an entry for the type (if nonexistent) with type token and dict for associated symbols
        symbol_dict = cls._base_vocab.setdefault(str_type, {})
        # Create an entry for the symbol (if nonexistent) with incrementing token value
        symbol_dict.setdefault(symbol, len(symbol_dict))

    @classmethod
    def get_token(cls, str_type: str, symbol: str) -> Tuple[TokenType, int]:
        """
        Get the type and token id for the associated type and symbol strings
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        # Get token from type mapping
        token_type = TYPE_STR_TO_INT[str_type]
        type_dict = cls._vocab[token_type]
        symbol_token = type_dict.get(symbol)
        if symbol_token is None:
            raise Exception(f"Symbol {symbol} not found for type {str_type}")
        return token_type, symbol_token

    @classmethod
    def get_symbol(cls, token_type: TokenType, token_id: int) -> str:
        """
        Get the symbol string associated with the given type and token id
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        # Get symbol from inverted mapping
        inv_type_dict = cls._vocab_inv[token_type]
        symbol = inv_type_dict.get(token_id)
        if symbol is None:
            raise Exception(f"Token ID {token_id} not found for type {token_type}!")
        return symbol

    @classmethod
    def num_tokens_in_type(cls, token_type: TokenType) -> int:
        """
        Get the number of tokens associated with the given type enum
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        return len(cls._vocab[token_type])

    @classmethod
    def dump(cls):
        """
        Dump vocab to file
        """
        with open(VOCAB_FILE_PATH, "w") as out_file:
            json.dump(cls._base_vocab, out_file, indent=2)

    @classmethod
    def load(cls):
        """
        Load vocab from file and mark as loaded
        """

        # Load base vocab from file
        with open(VOCAB_FILE_PATH) as vocab_file:
            base_vocab: BaseVocab = json.load(vocab_file)

        # Compute vocab and inverted vocab by collapsing types and assigning new token IDs
        cls._vocab = {}
        cls._vocab_inv = {}
        for str_type, symbols in base_vocab.items():
            token_type = TYPE_STR_TO_INT[str_type]
            type_dict = cls._vocab.setdefault(token_type, {})
            inv_type_dict = cls._vocab_inv.setdefault(token_type, {})
            for symbol in symbols:
                # Skip if symbol already seen (from other matching base type) to avoid gaps in the token ID list
                if symbol in type_dict:
                    continue
                new_token_id = len(type_dict)
                type_dict[symbol] = new_token_id
                inv_type_dict[new_token_id] = symbol
        cls._loaded = True
