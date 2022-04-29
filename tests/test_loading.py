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
    # TODO: second sample
    dataset = Dataset([SAMPLE_ARTICLE_1], 10)
    assert dataset.data == [
        {
            "token_ids": [464, 27255, 286, 220, 0, 3, 6, 1, 0, 318],
            "token_types": [0, 0, 0, 0, 0, TokenType.OP, TokenType.VAR, TokenType.NUM, 0, 0],
            "positions": ["", "", "", "", "", "0", "00", "01", "", ""],
        },
        {
            "token_ids": [220, 0, 3, 1, 6, 0, 13],
            "token_types": [0, 0, TokenType.OP, TokenType.NUM, TokenType.VAR, 0, 0],
            "positions": ["", "", "0", "00", "01", "", ""]
        }
    ]

def test_split_sequence():
    pass # TODO

def test_collator():
    pass # TODO
