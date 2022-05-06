from typing import Dict, Tuple, List, Optional, TypedDict
from enum import IntEnum
import torch

class TokenType(IntEnum):
    TEXT = 0
    START_FORMULA = 1
    END_FORMULA = 2
    VAR = 3
    NUM = 4
    OP = 5
    END = 6

OPT = Tuple[str, str, Optional[List['OPT']]]

class Formula(TypedDict):
    opt: OPT
    tex: str

class Article(TypedDict):
    name: str
    text: str
    formulas: Dict[str, Formula]

class Sequence:
    def __init__(self, src_article: str):
        self.src_article = src_article
        self.token_ids: List[int] = []
        self.token_types: List[TokenType] = []
        self.pos_vecs: List[List[int]] = []
        self.pos_levels: List[int] = []
        self.pos_encodings: List[List[int]] = []

    def split_at(self, split_point):
        pre_split = Sequence(self.src_article)
        pre_split.token_ids = self.token_ids[:split_point]
        pre_split.token_types = self.token_types[:split_point]
        pre_split.pos_vecs = self.pos_vecs[:split_point]
        pre_split.pos_levels = self.pos_levels[:split_point]
        pre_split.pos_encodings = self.pos_encodings[:split_point]
        post_split = Sequence(self.src_article)
        post_split.token_ids = self.token_ids[split_point:]
        post_split.token_types = self.token_types[split_point:]
        post_split.pos_vecs = self.pos_vecs[split_point:]
        post_split.pos_levels = self.pos_levels[split_point:]
        post_split.pos_encodings = self.pos_encodings[split_point:]
        return pre_split, post_split

class CollatedBatch(TypedDict):
    articles: List[str]
    token_ids: torch.Tensor
    token_types: torch.Tensor
    pos_vecs: torch.Tensor
    pos_levels: torch.Tensor
    pos_encodings: torch.Tensor
    attention_mask: torch.Tensor

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
    "E": TokenType.OP, # Error # TODO: remove these?
    "W": TokenType.VAR, # TODO
    "-": TokenType.VAR, # TODO
}

class Mode(IntEnum):
    PRETRAIN = 1

MAX_FORMULA_DEPTH = 64 # TODO: check actual max depth seen
MAX_FORMULA_WIDTH = 32 # TODO: check actual max width seen

PADDING_TOKEN_ID = -100 # This is the default padding token for transformers and the default ignore_index for NLLLoss
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 50256
