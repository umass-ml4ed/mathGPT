from typing import Dict, Tuple, Union
import json

from constants import TYPE_STR_TO_INT, TYPE_STR_TO_MAX_NUM_TOKENS, TokenType, SpecialOpToken, SpecialVarToken, SpecialNumToken

VOCAB_FILE_PATH = "ref_data/vocab.json"

BaseVocab = Dict[str, Dict[str, int]]

Vocab = Dict[TokenType, Dict[str, int]]

VocabInv = Dict[TokenType, Dict[int, str]]

UNK_MAP = {
    TokenType.OP: SpecialOpToken.UNK.value,
    TokenType.VAR: SpecialVarToken.UNK.value,
    TokenType.NUM: SpecialNumToken.UNK.value,
}

def get_matrix_symbol(symbol: str):
    """
    Create special symbol for matrix identifiers.
    The first character is the type of matrix (ex: L, V, M, etc.) followed by details added by TangentCFT.
    Discard the details, keep the type, and add a prefix so the type doesn't get confused with other OP symbols during coalescence.
    """
    return "matrix-" + symbol[0]

class Vocabulary:
    # Maps type to symbol to occurrence frequency in dataset
    _base_vocab: BaseVocab = {}
    # Maps TokenType to symbol to token ID
    _vocab: Vocab = {}
    # Maps TokenType to token ID to symbol
    _vocab_inv: VocabInv = {}
    # If the vocab has been loaded for use
    _loaded: bool = False

    @classmethod
    def add(cls, str_type: str, symbol: str):
        """
        Add the given type to the vocab if not present, and add the given symbol to the type if not present
        """
        # Create an entry for the type (if nonexistent) with dict for associated symbols
        symbol_dict = cls._base_vocab.setdefault(str_type, {})
        # Create an entry for the symbol (if nonexistent) and increment value
        symbol_dict.setdefault(symbol, 0)
        symbol_dict[symbol] += 1

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
        symbol_token = type_dict.get(symbol, UNK_MAP[token_type])
        return token_type, symbol_token

    @classmethod
    def get_symbol(cls, token_type: Union[TokenType, int], token_id: int) -> str:
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
        with open(VOCAB_FILE_PATH, "w", encoding="utf-8") as out_file:
            json.dump(cls._base_vocab, out_file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls):
        """
        Load vocab from file and mark as loaded
        """
        # Initially fill vocab with special tokens - even if not looked up directly, need to assign token IDs from base vocab above special token IDs
        cls._vocab = {
            TokenType.OP: {str(token): token.value for token in SpecialOpToken},
            TokenType.VAR: {str(token): token.value for token in SpecialVarToken},
            TokenType.NUM: {str(token): token.value for token in SpecialNumToken},
        }
        cls._vocab_inv = {
            TokenType.OP: {token.value: str(token) for token in SpecialOpToken},
            TokenType.VAR: {token.value: str(token) for token in SpecialVarToken},
            TokenType.NUM: {token.value: str(token) for token in SpecialNumToken},
        }

        # Load base vocab from file
        with open(VOCAB_FILE_PATH, encoding="utf-8") as vocab_file:
            base_vocab: BaseVocab = json.load(vocab_file)

        # Compute vocab and inverted vocab by collapsing types and assigning new token IDs from base vocab
        # Note: we are sorting the types and symbols to guarantee that token IDs are identical across loads, even if the base vocab file is reordered
        for str_type, symbols in sorted(base_vocab.items()):
            # Skip types that are all converted away in post-processing
            if str_type in ("E", "+"):
                continue

            # Create entries for token type
            token_type = TYPE_STR_TO_INT[str_type]
            type_dict = cls._vocab.setdefault(token_type, {})
            inv_type_dict = cls._vocab_inv.setdefault(token_type, {})

            # For specified types, only keep most frequent symbols
            if str_type in TYPE_STR_TO_MAX_NUM_TOKENS:
                most_freq_symbols = sorted(symbols.items(), key=lambda symbol: symbol[1], reverse=True)
                symbols_to_keep = most_freq_symbols[:TYPE_STR_TO_MAX_NUM_TOKENS[str_type]]
            else:
                symbols_to_keep = symbols.items()

            # Generate token ID for each symbol and add to vocab
            for symbol, _ in sorted(symbols_to_keep):
                # Special processing for matrix symbols
                if str_type == "M":
                    symbol = get_matrix_symbol(symbol)
                # Skip if symbol already seen (from other matching base type) to avoid gaps in the token ID list
                if symbol in type_dict:
                    continue
                new_token_id = len(type_dict)
                type_dict[symbol] = new_token_id
                inv_type_dict[new_token_id] = symbol
        cls._loaded = True
