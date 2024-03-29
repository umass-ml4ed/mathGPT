from typing import Dict, Tuple, Union, List
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

MATH_TYPES = [TokenType.OP, TokenType.VAR, TokenType.NUM]

NUM_SYMBOLS = [str(digit) for digit in range(10)] + ["."]

def get_matrix_symbol(symbol: str):
    """
    Create special symbol for matrix identifiers.
    The first character is the type of matrix (ex: L, V, M, etc.) followed by details added by TangentCFT.
    Discard the details, keep the type, and add a prefix so the type doesn't get confused with other OP symbols during coalescence.
    """
    return "matrix-" + symbol[0]

class Vocabulary:
    # Path to base vocab
    _vocab_file_path = VOCAB_FILE_PATH
    # Maps type to symbol to occurrence frequency in dataset
    _base_vocab: BaseVocab = {}
    # Maps TokenType to symbol to token ID
    _vocab: Vocab = {}
    # Maps TokenType to token ID to symbol
    _vocab_inv: VocabInv = {}
    # Maps TokenType to (type str, symbol) to frequency
    _frequency: Dict[TokenType, Dict[Tuple[str, str], int]] = {}
    # Subset of the full vocab that just contains tokens with special meaning
    _special_tokens: VocabInv = {}
    # Maps TokenType to number of tokens in that type (at the time of loading)
    _sizes: Dict[TokenType, int] = {}
    # If number tokens should be expanded into subtrees
    _num_to_tree: bool = False
    # If UNKs will be converted to sub-trees
    _math_text: bool = False
    # If the vocab has been loaded for use
    _loaded: bool = False

    @classmethod
    def override_vocab_file(cls, vocab_file_path: bool):
        cls._vocab_file_path = vocab_file_path

    @classmethod
    def set_num_to_tree(cls, num_to_tree: bool):
        cls._num_to_tree = num_to_tree

    @classmethod
    def set_math_text(cls, math_text: bool):
        cls._math_text = math_text

    @classmethod
    def math_text(cls):
        return cls._math_text

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
    def get_token(cls, str_type: str, symbol: str, assign_new: bool = False) -> Tuple[TokenType, int]:
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
            if assign_new:
                symbol_token = len(cls._vocab[token_type])
                cls._vocab[token_type][symbol] = symbol_token
                cls._vocab_inv[token_type][symbol_token] = symbol
            else:
                symbol_token = UNK_MAP[token_type]
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
    def is_special_token(cls, token_type: TokenType, token_id: int):
        """
        Get if the given token has special meaning, i.e. does not direclty map to a symbol in the data
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        if token_type not in cls._special_tokens:
            return False
        return token_id in cls._special_tokens[token_type]

    @classmethod
    def num_tokens_in_type(cls, token_type: TokenType) -> int:
        """
        Get the number of tokens associated with the given type enum
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        return cls._sizes[token_type]

    @classmethod
    def most_frequent(cls, token_type: TokenType, num: int = 0):
        """
        Get <num> most frequent tokens for given type
        Returns all digits for NUM type if using num_to_tree
        """
        # Load data if not yet loaded
        if not cls._loaded:
            cls.load()

        if token_type == TokenType.NUM and cls._num_to_tree:
            return [("N", symbol) for symbol in NUM_SYMBOLS]

        most_freq: List[Tuple[Tuple[str, str], int]] = sorted(cls._frequency[token_type].items(), key=lambda freq: freq[1], reverse=True)[:num]
        return [symbol_tup for symbol_tup, _ in most_freq]

    @classmethod
    def dump(cls):
        """
        Dump vocab to file
        """
        with open(cls._vocab_file_path, "w", encoding="utf-8") as out_file:
            json.dump(cls._base_vocab, out_file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, infreq_to_unk: bool = True):
        """
        Load vocab from file and mark as loaded
        """
        # Initially fill vocab with special tokens - even if not looked up directly, need to assign token IDs from base vocab above special token IDs
        op_tokens = [token for token in SpecialOpToken
            # Include option-specific special tokens, always include NUM_SUB_TREE_HEAD if math_text is set to not leave gap in token IDs
            if (cls._num_to_tree or cls._math_text or token != SpecialOpToken.NUM_SUB_TREE_HEAD) and (cls._math_text or token != SpecialOpToken.MATH_TEXT_HEAD)]
        cls._vocab = {
            TokenType.OP: {str(token): token.value for token in op_tokens},
            TokenType.VAR: {str(token): token.value for token in SpecialVarToken},
            TokenType.NUM: {str(token): token.value for token in SpecialNumToken},
        }
        cls._vocab_inv = {
            TokenType.OP: {token.value: str(token) for token in op_tokens},
            TokenType.VAR: {token.value: str(token) for token in SpecialVarToken},
            TokenType.NUM: {token.value: str(token) for token in SpecialNumToken},
        }
        cls._special_tokens = {key: value.copy() for key, value in cls._vocab_inv.items()}

        # Load base vocab from file
        with open(cls._vocab_file_path, encoding="utf-8") as vocab_file:
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
            freq_dict = cls._frequency.setdefault(token_type, {})

            # If expanding numbers to trees, no need to save individual numbers in the vocab
            if token_type == TokenType.NUM and cls._num_to_tree:
                continue

            # For specified types, only keep most frequent symbols
            if infreq_to_unk and str_type in TYPE_STR_TO_MAX_NUM_TOKENS:
                most_freq_symbols = sorted(symbols.items(), key=lambda symbol: symbol[1], reverse=True)
                symbols_to_keep = most_freq_symbols[:TYPE_STR_TO_MAX_NUM_TOKENS[str_type]]
            else:
                symbols_to_keep = symbols.items()

            # Generate token ID for each symbol and add to vocab
            for symbol, frequency in sorted(symbols_to_keep):
                # Special processing for matrix symbols
                if str_type == "M":
                    symbol = get_matrix_symbol(symbol)
                # Skip if symbol already seen (from other matching base type) to avoid gaps in the token ID list
                if symbol in type_dict:
                    freq_dict[(str_type, symbol)] = frequency
                    continue
                new_token_id = len(type_dict)
                type_dict[symbol] = new_token_id
                inv_type_dict[new_token_id] = symbol
                freq_dict[(str_type, symbol)] = frequency

        # If expanding numbers to trees, just assign token IDs for each digit and the period character
        if cls._num_to_tree:
            for num_symb in NUM_SYMBOLS:
                token_id = len(cls._vocab[TokenType.NUM])
                cls._vocab[TokenType.NUM][num_symb] = token_id
                cls._vocab_inv[TokenType.NUM][token_id] = num_symb

        # Save size of each type's vocab
        for token_type in MATH_TYPES:
            cls._sizes[token_type] = len(cls._vocab[token_type])

        cls._loaded = True
