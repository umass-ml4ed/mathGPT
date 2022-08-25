from typing import Dict, Tuple, List, Optional, Union, TypedDict
import torch

from constants import TokenType

OPT = Tuple[Union[str, TokenType], Union[str, int], Optional[List['OPT']]] # Type, token, children

class Formula(TypedDict):
    opt: OPT
    tex: str

class Article(TypedDict):
    text: str
    formulas: Dict[str, Formula]

class GenTaskSample(TypedDict):
    prompt: Article
    label: Article

class AnswerScoringSample(TypedDict):
    answer: Article
    problem_id: int
    problem_log_id: int
    grade: int

class FeedbackTaskSample(TypedDict):
    problem_id: str
    answer: Article
    feedback: Article

class ProblemSolvingTaskSample(TypedDict):
    problem: Article
    steps: Article
    answer: Article

class SequenceMetaData(TypedDict):
    # For gen tasks
    prompt_length: Optional[int]
    # For classify tasks
    label: Optional[int]
    # For answer scoring
    problem_id: Optional[int]
    problem_log_id: Optional[int]

class Sequence:
    def __init__(self, name: str):
        self.name = name
        self.token_ids: List[int] = []
        self.token_types: List[TokenType] = []
        self.pos_vecs: List[List[int]] = []
        self.pos_levels: List[int] = []
        self.gpt_tokens: List[List[int]] = []
        self.meta: SequenceMetaData = {}

    def split_at(self, split_point):
        pre_split = Sequence(self.name)
        pre_split.token_ids = self.token_ids[:split_point]
        pre_split.token_types = self.token_types[:split_point]
        pre_split.pos_vecs = self.pos_vecs[:split_point]
        pre_split.pos_levels = self.pos_levels[:split_point]
        pre_split.gpt_tokens = self.gpt_tokens[:split_point]
        post_split = Sequence(self.name)
        post_split.token_ids = self.token_ids[split_point:]
        post_split.token_types = self.token_types[split_point:]
        post_split.pos_vecs = self.pos_vecs[split_point:]
        post_split.pos_levels = self.pos_levels[split_point:]
        post_split.gpt_tokens = self.gpt_tokens[split_point:]
        return pre_split, post_split

    def __add__(self, seq_2: 'Sequence'):
        new_seq = Sequence(self.name)
        new_seq.token_ids = self.token_ids + seq_2.token_ids
        new_seq.token_types = self.token_types + seq_2.token_types
        new_seq.pos_vecs = self.pos_vecs + seq_2.pos_vecs
        new_seq.pos_levels = self.pos_levels + seq_2.pos_levels
        new_seq.gpt_tokens = self.gpt_tokens + seq_2.gpt_tokens
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
    gpt_tokens: Optional[torch.Tensor]
    use_shared_emb: Optional[torch.Tensor]
    attention_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    prompt_lengths: Optional[torch.Tensor]
    gen_labels: Optional[torch.Tensor]
    cls_labels: Optional[torch.Tensor]
