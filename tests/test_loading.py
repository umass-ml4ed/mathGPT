from loading import Dataset
from vocabulary import Vocabulary
from constants import Token

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
    # TODO: use constants for types
    assert dataset.data == [
        {
            "token_ids": [464, 27255, 286, 220, Token.SWITCH_CONTEXT.value, 3, 6, 1, Token.SWITCH_CONTEXT.value, 318],
            "token_types": [0, 0, 0, 0, 0, Vocabulary._vocab["O"]["token"], Vocabulary._vocab["V"]["token"], Vocabulary._vocab["N"]["token"], 0, 0],
            "positions": ["", "", "", "", "", "0", "00", "01", "", ""],
        },
        {
            "token_ids": [220, Token.SWITCH_CONTEXT.value, 3, 1, 6, Token.SWITCH_CONTEXT.value, 13],
            "token_types": [0, 0, Vocabulary._vocab["U"]["token"], Vocabulary._vocab["N"]["token"], Vocabulary._vocab["V"]["token"], 0, 0],
            "positions": ["", "", "0", "00", "01", "", ""]
        }
    ]

def test_split_sequence():
    pass
