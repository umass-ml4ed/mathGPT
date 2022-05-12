from unittest.mock import patch, MagicMock

from vocabulary import Vocabulary
from constants import TokenType, SpecialOpToken, SpecialVarToken, SpecialNumToken

def test_load():
    """
    Test that internal vocab dictionaries are created correctly.
    - Types should be coalesced
    - New token IDs should be assigned with increasing values
    - Duplicate token IDs should be skipped per type
    - Special tokens should be added
    - Order of processing is alphabetical for types and symbols
    - + and E types should be skipped
    - Matrix symbols should be reassigned
    - Low frequency tokens should be skipped
    """
    base_vocab_mock = MagicMock(side_effect=[{
        "+": {
            "O!limit": 100,
        },
        "E": {
            "": 1000,
        },
        "O": {
            "sup": 500,
            "lt": 100,
        },
        "U": {
            "eq": 1000,
        },
        "F": {
            "lt": 23,
        },
        "N": {
            "1": 100000,
            "42": 1232203,
            "0": 132999923,
            "3.14159": 23454,
        },
        "M": {
            "V-29": 1,
        },
        "V": {
            "x": 34564,
            "lt": 2,
            "y": 23493,
        },
    }])
    max_tokens_mock = {
        "N": 3,
    }

    expected_vocab = {
        TokenType.OP: {
            str(SpecialOpToken.UNK): SpecialOpToken.UNK.value,
            str(SpecialOpToken.ANON_OP): SpecialOpToken.ANON_OP.value,
            str(SpecialOpToken.CERR_OP): SpecialOpToken.CERR_OP.value,
            "lt": 3,
            "matrix-V": 4,
            "sup": 5,
            "eq": 6,
        },
        TokenType.VAR: {
            str(SpecialVarToken.UNK): SpecialVarToken.UNK.value,
            str(SpecialVarToken.MAT_PAD): SpecialVarToken.MAT_PAD.value,
            "lt": 2,
            "x": 3,
            "y": 4,
        },
        TokenType.NUM: {
            str(SpecialNumToken.UNK): SpecialNumToken.UNK.value,
            "0": 1,
            "1": 2,
            "42": 3,
            # 3.14159 removed as 3 most frequent already placed
        },
    }
    expected_vocab_inv = {
        TokenType.OP: {val: key for key, val in expected_vocab[TokenType.OP].items()},
        TokenType.VAR: {val: key for key, val in expected_vocab[TokenType.VAR].items()},
        TokenType.NUM: {val: key for key, val in expected_vocab[TokenType.NUM].items()},
    }

    with patch("json.load", base_vocab_mock):
        with patch("vocabulary.TYPE_STR_TO_MAX_NUM_TOKENS", max_tokens_mock):
            Vocabulary.load()
            assert Vocabulary._loaded
            assert Vocabulary._vocab == expected_vocab
            assert Vocabulary._vocab_inv == expected_vocab_inv

def test_get_token():
    """
    Test that for symbols in the vocab, type and token ID are correctly retrieved
    For symbols not in the vocab, type and UNK are returned
    """
    Vocabulary._vocab = {
        TokenType.OP: {
            "eq": 2020,
        },
        TokenType.VAR: {
            "x": 42,
        },
        TokenType.NUM: {
            "2": 12
        }
    }

    assert Vocabulary.get_token("O", "eq") == (TokenType.OP, 2020)
    assert Vocabulary.get_token("V", "x") == (TokenType.VAR, 42)
    assert Vocabulary.get_token("N", "2") == (TokenType.NUM, 12)

    assert Vocabulary.get_token("O", "lt") == (TokenType.OP, SpecialOpToken.UNK.value)
    assert Vocabulary.get_token("V", "y") == (TokenType.VAR, SpecialVarToken.UNK.value)
    assert Vocabulary.get_token("N", "3.14159") == (TokenType.NUM, SpecialNumToken.UNK.value)
