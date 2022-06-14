from math_tokenize import encode_pos, tokenize_formula, POS_ENCODING_REGION_NUM_BITS
from vocabulary import Vocabulary
from constants import MAX_FORMULA_DEPTH, TPE, TokenType, SpecialOpToken, SpecialVarToken

def test_encode_pos_forte():
    pos_vec = [0, 1, 5] + ([0] * (MAX_FORMULA_DEPTH - 3))
    pos_level = 2

    encoding = encode_pos(pos_vec, pos_level, TPE.FORTE.value)

    padding = ([1, 0] * (POS_ENCODING_REGION_NUM_BITS - 3)) # Will use 3 bits to represent numbers below, follow with 0-bit padding
    assert encoding == [1, 0, 1, 0, 1, 0] + padding + [0, 1, 1, 0, 1, 0] + padding + [0, 1, 1, 0, 0, 1] + padding +\
                       [0] * ((MAX_FORMULA_DEPTH - 3) * POS_ENCODING_REGION_NUM_BITS * 2)

def test_tokenize_formula():
    """
    Test tokenization, includes the following cases
    - Coalescing string type to TokenType
    - Inserting END tokens after as the last child of each OP
    - For + type, convert to anonymous operator
    - For cerror E type, remove first child and convert to special operator
    - For matrix E type, convert to padding
    - For M type, convert to special symbol
    """
    formula = [
        "+", "O!limit", [
            [
                "O", "SUB", [
                    [
                        "O", "limit", None
                    ],
                    [
                        "E", "", [
                            [
                                "O", "fragments", None
                            ],
                            [
                                "V", "N", None
                            ],
                            [
                                "F", "normal-\u2192", None
                            ],
                            [
                                "C", "infinity", None
                            ]
                        ]
                    ]
                ]
            ],
            [
                "M", "V-3", [
                    [
                        "N", "1", None
                    ],
                    [
                        "N", "2", None
                    ],
                    [
                        "E", "", None
                    ]
                ]
            ]
        ]
    ]

    expected_tokens = [
        SpecialOpToken.ANON_OP.value,
            Vocabulary.get_token("O", "SUB")[1],
                Vocabulary.get_token("O", "limit")[1],
                    0,
                SpecialOpToken.CERR_OP.value,
                    Vocabulary.get_token("V", "N")[1],
                    Vocabulary.get_token("F", "normal-\u2192")[1],
                        0,
                    Vocabulary.get_token("C", "infinity")[1],
                    0,
                0,
            Vocabulary.get_token("M", "matrix-V")[1],
                Vocabulary.get_token("N", "1")[1],
                Vocabulary.get_token("N", "2")[1],
                SpecialVarToken.MAT_PAD.value,
                0,
            0,
    ]
    expected_types = [
        TokenType.OP,
            TokenType.OP,
                TokenType.OP,
                    TokenType.END,
                TokenType.OP,
                    TokenType.VAR,
                    TokenType.OP,
                        TokenType.END,
                    TokenType.VAR,
                    TokenType.END,
                TokenType.END,
            TokenType.OP,
                TokenType.NUM,
                TokenType.NUM,
                TokenType.VAR,
                TokenType.END,
            TokenType.END,
    ]
    padding = [0] * (MAX_FORMULA_DEPTH - 5)
    expected_pos_vecs = [
        [0, 0, 0, 0, 0] + padding,
            [0, 0, 0, 0, 0] + padding,
                [0, 0, 0, 0, 0] + padding,
                    [0, 0, 0, 0, 0] + padding,
                [0, 0, 1, 0, 0] + padding,
                    [0, 0, 1, 0, 0] + padding,
                    [0, 0, 1, 1, 0] + padding,
                        [0, 0, 1, 1, 0] + padding,
                    [0, 0, 1, 2, 0] + padding,
                    [0, 0, 1, 3, 0] + padding,
                [0, 0, 2, 0, 0] + padding,
            [0, 1, 0, 0, 0] + padding,
                [0, 1, 0, 0, 0] + padding,
                [0, 1, 1, 0, 0] + padding,
                [0, 1, 2, 0, 0] + padding,
                [0, 1, 3, 0, 0] + padding,
            [0, 2, 0, 0, 0] + padding,
    ]
    expected_pos_levels = [0, 1, 2, 3, 2, 3, 3, 4, 3, 3, 2, 1, 2, 2, 2, 2, 1]

    tokenized = tokenize_formula(formula, TPE.FORTE.value)

    assert tokenized.token_ids == expected_tokens
    assert tokenized.token_types == expected_types
    assert tokenized.pos_vecs == expected_pos_vecs
    assert tokenized.pos_levels == expected_pos_levels
    assert tokenized.pos_encodings == [encode_pos(pos_vec, pos_level, TPE.FORTE.value) for pos_vec, pos_level in zip(expected_pos_vecs, expected_pos_levels)]
