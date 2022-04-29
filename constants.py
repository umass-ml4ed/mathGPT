from typing import Dict, Tuple, List, Optional, TypedDict
from enum import IntEnum
import torch

OPT = Tuple[str, str, Optional[List['OPT']]]

class Formula(TypedDict):
    opt: OPT
    tex: str

class Article(TypedDict):
    text: str
    formulas: Dict[str, Formula]

class TokenType(IntEnum):
    TEXT = 0
    START_FORMULA = 1
    END_FORMULA = 2
    VAR = 3
    NUM = 4
    OP = 5

TYPE_STR_TO_INT: Dict[str, TokenType] = {
    "U": TokenType.OP, # Unordered ops
    "O": TokenType.OP, # Ordered ops
    "F": TokenType.OP, # Functions
    "M": TokenType.OP, # Matrices/vectors
    "+": TokenType.OP, # Nested apply operators
    "N": TokenType.NUM, # Numbers
    "V": TokenType.VAR, # Variables
    "C": TokenType.VAR, # Constants
    "T": TokenType.VAR, # Text
    "E": TokenType.VAR, # Error
    "W": TokenType.VAR, # TODO
    "-": TokenType.VAR, # TODO
}

class CollatedBatch(TypedDict):
    token_ids: torch.LongTensor
    token_types: torch.LongTensor
    positions: torch.LongTensor
    attention_mask: torch.FloatTensor

class Mode(IntEnum):
    PRETRAIN = 1
