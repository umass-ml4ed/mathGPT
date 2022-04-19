from typing import Dict, Tuple, List, Optional, TypedDict
from enum import Enum
import torch

OPT = Tuple[str, str, Optional[List['OPT']]]

class Formula(TypedDict):
    opt: OPT
    tex: str

class Article(TypedDict):
    text: str
    formulas: Dict[str, Formula]

class Token(Enum):
    SWITCH_CONTEXT = 0

class TokenType(Enum):
    TEXT = 0

class CollatedBatch(TypedDict):
    token_ids: torch.LongTensor
    token_types: torch.LongTensor
    positions: torch.LongTensor
    attention_mask: torch.FloatTensor

class Mode(Enum):
    PRETRAIN = 1
