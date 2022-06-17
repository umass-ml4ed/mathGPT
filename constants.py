from typing import Any, Dict, Tuple, List, Optional, TypedDict
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

OPT = Tuple[str, str, Optional[List['OPT']]] # Type, token, children

class Formula(TypedDict):
    opt: OPT
    tex: str

class Article(TypedDict):
    text: str
    formulas: Dict[str, Formula]

class GenTaskSample(TypedDict):
    prompt: Article
    label: Article

class ClassifyTaskSample(Article):
    label: Any

class Sequence:
    def __init__(self, name: str, label = None):
        self.name = name
        self.token_ids: List[int] = []
        self.token_types: List[TokenType] = []
        self.pos_vecs: List[List[int]] = []
        self.pos_levels: List[int] = []
        self.pos_encodings: List[List[int]] = []
        self.label = label

    def split_at(self, split_point):
        pre_split = Sequence(self.name, self.label)
        pre_split.token_ids = self.token_ids[:split_point]
        pre_split.token_types = self.token_types[:split_point]
        pre_split.pos_vecs = self.pos_vecs[:split_point]
        pre_split.pos_levels = self.pos_levels[:split_point]
        pre_split.pos_encodings = self.pos_encodings[:split_point]
        post_split = Sequence(self.name, self.label)
        post_split.token_ids = self.token_ids[split_point:]
        post_split.token_types = self.token_types[split_point:]
        post_split.pos_vecs = self.pos_vecs[split_point:]
        post_split.pos_levels = self.pos_levels[split_point:]
        post_split.pos_encodings = self.pos_encodings[split_point:]
        return pre_split, post_split

    def __add__(self, seq_2: 'Sequence'):
        new_seq = Sequence(self.name, self.label)
        new_seq.token_ids = self.token_ids + seq_2.token_ids
        new_seq.token_types = self.token_types + seq_2.token_types
        new_seq.pos_vecs = self.pos_vecs + seq_2.pos_vecs
        new_seq.pos_levels = self.pos_levels + seq_2.pos_levels
        new_seq.pos_encodings = self.pos_encodings + seq_2.pos_encodings
        return new_seq

    def __len__(self):
        return len(self.token_ids)

class CollatedBatch(TypedDict):
    sources: List[str]
    token_ids: torch.Tensor
    token_types: torch.Tensor
    pos_vecs: torch.Tensor
    pos_levels: torch.Tensor
    pos_encodings: torch.Tensor
    attention_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    prompt_lengths: Optional[torch.Tensor]
    gen_labels: Optional[torch.Tensor]
    cls_labels: Optional[torch.Tensor]

class DownstreamTask(Enum):
    HEADLINES = "headlines"
    SOLVING = "solving"
    GRADING = "grading"
    FEEDBACK_GEN = "feedback"
    KC_PRED = "kc_pred"

DOWNSTREAM_TASK_TO_NUM_CLASSES = {
    DownstreamTask.GRADING: 5,
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

# File paths
DATA = "data"
WIKI_DATA = "data/wikipedia"
EXEQ_DATA = "data/EXEQ-300k"
OFEQ_DATA = "data/OFEQ-10k"
