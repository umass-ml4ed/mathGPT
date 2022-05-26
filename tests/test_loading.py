from unittest.mock import patch, MagicMock

from loading import PreTrainDataset
from math_tokenize import encode_pos, EMPTY_POS_VECTOR, EMPTY_POS_ENCODING
from vocabulary import Vocabulary
from constants import TokenType, EOS_TOKEN_ID, FORMULA_IDENTIFIER

SAMPLE_ARTICLE_1 = {
    "text": f"The derivative of {FORMULA_IDENTIFIER} is {FORMULA_IDENTIFIER}.",
    "formulas": {
        "0": {
            "opt": [
                "O",
                "SUP",
                [
                    [
                        "V",
                        "x",
                        None
                    ],
                    [
                        "N",
                        "2",
                        None
                    ]
                ]
            ]
        },
        "1": {
            "opt": [
                "U",
                "times",
                [
                    [
                        "N",
                        "2",
                        None
                    ],
                    [
                        "V",
                        "x",
                        None
                    ]
                ]
            ]
        }
    }
}

def test_new_dataset():
    Vocabulary.load() # Pre-load vocab so unaffected by mock
    load_article_mock = MagicMock(side_effect=[SAMPLE_ARTICLE_1])
    with patch("json.load", load_article_mock):
        dataset = PreTrainDataset(["data/GCD.json"], 11)

    assert dataset.data[0].token_ids == [
        464, 27255, 286, 220,
        0, Vocabulary.get_token("O", "SUP")[1], Vocabulary.get_token("V", "x")[1], Vocabulary.get_token("N", "2")[1], 0, 0,
        318
    ]
    assert dataset.data[0].token_types == [
        TokenType.TEXT, TokenType.TEXT, TokenType.TEXT, TokenType.TEXT,
        TokenType.START_FORMULA, TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END, TokenType.END_FORMULA,
        TokenType.TEXT
    ]
    assert dataset.data[0].pos_vecs[5][:1] == [0]
    assert dataset.data[0].pos_levels[5] == 0
    assert dataset.data[0].pos_encodings[5] == encode_pos([0], 0)
    assert dataset.data[0].pos_vecs[6][:2] == [0, 0]
    assert dataset.data[0].pos_levels[6] == 1
    assert dataset.data[0].pos_encodings[6] == encode_pos([0, 0], 1)
    assert dataset.data[0].pos_vecs[7][:2] == [0, 1]
    assert dataset.data[0].pos_levels[7] == 1
    assert dataset.data[0].pos_encodings[7] == encode_pos([0, 1], 1)
    assert dataset.data[0].pos_vecs[8][:2] == [0, 2]
    assert dataset.data[0].pos_levels[8] == 1
    assert dataset.data[0].pos_encodings[8] == encode_pos([0, 2], 1)
    for non_form_idx in list(range(5)) + list(range(9, len(dataset.data[0].token_ids))):
        assert dataset.data[0].pos_vecs[non_form_idx] == EMPTY_POS_VECTOR
        assert dataset.data[0].pos_levels[non_form_idx] == 0
        assert dataset.data[0].pos_encodings[non_form_idx] == EMPTY_POS_ENCODING

    assert dataset.data[1].token_ids == [
        220,
        0, Vocabulary.get_token("U", "times")[1], Vocabulary.get_token("N", "2")[1], Vocabulary.get_token("V", "x")[1], 0, 0,
        13, EOS_TOKEN_ID
    ]
    assert dataset.data[1].token_types == [
        TokenType.TEXT,
        TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END, TokenType.END_FORMULA,
        TokenType.TEXT, TokenType.TEXT
    ]
    assert dataset.data[1].pos_vecs[2][:1] == [0]
    assert dataset.data[1].pos_levels[2] == 0
    assert dataset.data[1].pos_encodings[2] == encode_pos([0], 0)
    assert dataset.data[1].pos_vecs[3][:2] == [0, 0]
    assert dataset.data[1].pos_levels[3] == 1
    assert dataset.data[1].pos_encodings[3] == encode_pos([0, 0], 1)
    assert dataset.data[1].pos_vecs[4][:2] == [0, 1]
    assert dataset.data[1].pos_levels[4] == 1
    assert dataset.data[1].pos_encodings[4] == encode_pos([0, 1], 1)
    assert dataset.data[1].pos_vecs[5][:2] == [0, 2]
    assert dataset.data[1].pos_levels[5] == 1
    assert dataset.data[1].pos_encodings[5] == encode_pos([0, 2], 1)
    for non_form_idx in list(range(2)) + list(range(6, len(dataset.data[1].token_ids))):
        assert dataset.data[1].pos_vecs[non_form_idx] == EMPTY_POS_VECTOR
        assert dataset.data[1].pos_levels[non_form_idx] == 0
        assert dataset.data[1].pos_encodings[non_form_idx] == EMPTY_POS_ENCODING

def test_split_sequence():
    pass # TODO

def test_collator():
    pass # TODO
