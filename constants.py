from typing import Dict, Optional, TypedDict
from enum import IntEnum, Enum
import torch

class TokenType(IntEnum):
    TEXT = 0
    START_FORMULA = 1
    END_FORMULA = 2
    VAR = 3
    NUM = 4
    OP = 5
    END = 6

class SpecialOpToken(IntEnum):
    UNK = 0
    ANON_OP = 1
    CERR_OP = 2

class SpecialVarToken(IntEnum):
    UNK = 0
    MAT_PAD = 1

class SpecialNumToken(IntEnum):
    UNK = 0

TYPE_STR_TO_INT: Dict[str, TokenType] = {
    "N": TokenType.NUM, # Numbers
    "U": TokenType.OP, # Unordered ops
    "O": TokenType.OP, # Ordered ops
    "F": TokenType.OP, # Functions
    "M": TokenType.OP, # Matrices/groups
    "+": TokenType.OP, # Nested apply operators
    "T": TokenType.OP, # Text, can have children sometimes
    "W": TokenType.OP, # Empty field, can have children or be terminal
    "V": TokenType.VAR, # Variables
    "C": TokenType.VAR, # Constants
    "-": TokenType.VAR, # "unknown", but generally represents some math symbol
    # We're not including the "E" type, as all instances should be removed during post-processing
}

# For given types, keep n most frequent tokens and discard the rest (convert discarded to UNK during loading)
TYPE_STR_TO_MAX_NUM_TOKENS: Dict[str, int] = {
    "N": 2000,
    "T": 1000,
    "V": 1000,
    "F": 5000,
}

FORMULA_IDENTIFIER = "[_mathGPT_formula_]" # Replace formulas in raw text with this to recover after loading

class Checkpoint(TypedDict):
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: dict
    scaler_state_dict: Optional[dict]
    rng_state: torch.Tensor
    epoch: int

class DownstreamTask(Enum):
    HEADLINES = "headlines"
    SOLVING = "solving"
    ANSWER_SCORING = "answer_scoring"
    FEEDBACK_GEN = "feedback"
    KC_PRED = "kc_pred"

DOWNSTREAM_TASK_TO_NUM_CLASSES = {
    DownstreamTask.ANSWER_SCORING: 5,
    DownstreamTask.KC_PRED: 10,
}

class TPE(Enum):
    FORTE = "forte"
    SIN_PART = "sin_part"
    SIN_ADD = "sin_add"

class Gen(Enum):
    GREEDY = "greedy"
    NUCLEUS = "nucleus"
    BEAM = "beam"

MAX_FORMULA_DEPTH = 32
MAX_FORMULA_WIDTH = 64

EMB_SIZE = 768
TEXT_VOCAB_SIZE = 50257

# The following are the defaults for the transformers tokenizer
PADDING_TOKEN_ID = -100
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 50256
SEP_TOKEN = "[SEP]"
CLS_TOKEN = "[CLS]"
DOLLAR_TOK = 720

# File paths
DATA = "data"
WIKI_DATA = "data/wikipedia"
EXEQ_DATA = "data/EXEQ-300k"
OFEQ_DATA = "data/OFEQ-10k"
AS_PROBLEMS = "data/answer_scoring/problems.json"
AS_ANSWERS = "data/answer_scoring/answers.json"
