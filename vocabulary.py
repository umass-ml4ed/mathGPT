from typing import Dict, Tuple, TypedDict
import json

VOCAB_FILE_PATH = "ref_data/vocab.json"

class VocabType(TypedDict):
    token: int
    symbols: Dict[str, int]

class Vocabulary:
    _vocab: Dict[str, VocabType] = {}
    _loaded: bool = False

    @classmethod
    def add(cls, sym_type: str, symbol: str):
        """
        Add the given type to the vocab if not present, and add the given symbol to the type if not present
        """
        # TODO: should hard-code token int types, ensure consistency and no overlap with special types
        # Create an entry for the type (if nonexistent) with type token and dict for associated symbols
        symbol_dict = cls._vocab.setdefault(sym_type, {
            "token": len(cls._vocab),
            "symbols": {}
        })["symbols"]
        # Create an entry for the symbol (if nonexistent) with incrementing token value
        symbol_dict.setdefault(symbol, len(symbol_dict))

    @classmethod
    def get_token(cls, sym_type: str, symbol: str) -> Tuple[int, int]:
        """
        Get the type int and token for the associated type and symbol strings
        Loads vocab from file if unloaded
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        # Retrieve tokens for given type and symbol
        type_dict = cls._vocab.get(sym_type)
        if not type_dict:
            raise Exception(f"Type {sym_type} not found!")
        symbol_token = type_dict["symbols"].get(symbol)
        if symbol_token is None:
            raise Exception(f"Symbol {symbol} not found for type {sym_type}!")
        return type_dict["token"], symbol_token

    @classmethod
    def dump(cls):
        """
        Dump vocab to file
        """
        with open(VOCAB_FILE_PATH, "w") as out_file:
            json.dump(cls._vocab, out_file, indent=2)

    @classmethod
    def load(cls):
        """
        Load vocab from file and mark as loaded
        """
        with open(VOCAB_FILE_PATH) as vocab_file:
            cls._vocab = json.load(vocab_file)
        cls._loaded = True
