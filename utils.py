import random
from typing import Optional
import numpy as np
import torch

from constants import DownstreamTask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

def enum_choices(enum):
    return [choice.value for choice in enum]

def enum_value_to_member(value, enum):
    return next((member for member in enum if member.value == value), None)

def is_cls_task(task: Optional[DownstreamTask]):
    return task in (DownstreamTask.GRADING, DownstreamTask.KC_PRED)

class TrainOptions:
    def __init__(self, options: dict):
        self.lr: float = options.get("lr", 1e-5)
        self.weight_decay: float = options.get("weight_decay", 1e-2)
        self.epochs: int = options.get("epochs", 20)
        self.patience: Optional[int] = options.get("patience", None)
        self.batch_size: int = options.get("batch_size", 64)
        self.grad_accum_batches: int = options.get("grad_accum_batches", 1)
        self.max_seq_len: int = options.get("max_seq_len", 1024)
        self.amp: bool = options.get("amp", False)
        self.ns_p: float = options.get("ns_p", 0.90)
        self.stride: Optional[int] = options.get("stride", None)
        self.num_classes: Optional[int] = options.get("num_classes", None)
        self.use_type_embs: bool = options.get("use_type_embs", True)

    def update(self, options: dict):
        self.__dict__.update(options)
