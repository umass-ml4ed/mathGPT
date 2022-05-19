from loading import Dataset
from vocabulary import Vocabulary
from constants import TokenType

SAMPLE_ARTICLE_1 = {
    "text": "The derivative of [0] is [1].",
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
    # TODO: mock json.load
    dataset = Dataset([SAMPLE_ARTICLE_1], 11)

    assert dataset.data[0].token_ids == [464, 27255, 286, 220, 0, Vocabulary.get_token("O", "SUP")[1], Vocabulary.get_token("V", "x")[1], Vocabulary.get_token("N", "2")[1], 0, 0, 318]
    assert dataset.data[0].token_types == [0, 0, 0, 0, TokenType.START_FORMULA, TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END, TokenType.END_FORMULA, 0]
    assert dataset.data[0].pos_vecs[5][:1] == [0]
    assert dataset.data[0].pos_levels[5] == 0
    assert dataset.data[0].pos_vecs[6][:2] == [0, 0]
    assert dataset.data[0].pos_levels[6] == 1
    assert dataset.data[0].pos_vecs[7][:2] == [0, 1]
    assert dataset.data[0].pos_levels[7] == 1
    assert dataset.data[0].pos_vecs[8][:2] == [0, 2]
    assert dataset.data[0].pos_levels[8] == 1
    # TODO: pos_encodings

    assert dataset.data[1].token_ids == [220, 0, Vocabulary.get_token("U", "times")[1], Vocabulary.get_token("N", "2")[1], Vocabulary.get_token("V", "x")[1], 0, 0, 13]
    assert dataset.data[1].token_types == [0, TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END, TokenType.END_FORMULA, 0]
    assert dataset.data[1].pos_vecs[2][:1] == [0]
    assert dataset.data[1].pos_levels[2] == 0
    assert dataset.data[1].pos_vecs[3][:2] == [0, 0]
    assert dataset.data[1].pos_levels[3] == 1
    assert dataset.data[1].pos_vecs[4][:2] == [0, 1]
    assert dataset.data[1].pos_levels[4] == 1
    assert dataset.data[1].pos_vecs[5][:2] == [0, 2]
    assert dataset.data[1].pos_levels[5] == 1
    # TODO: pos_encodings

def test_split_sequence():
    pass # TODO

def test_collator():
    pass # TODO
