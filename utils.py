import random
from typing import Optional
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seedNum):
    random.seed(seedNum)
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seedNum)
        torch.cuda.manual_seed_all(seedNum)

class TrainOptions:
    def __init__(self, options: dict):
        self.lr: float = options.get("lr", 1e-5)
        self.weight_decay: float = options.get("weight_decay", 1e-2)
        self.epochs: int = options.get("epochs", 100)
        self.patience: Optional[int] = options.get("patience", None)
        self.batch_size: int = options.get("batch_size", 64)
        self.grad_accum_batches: int = options.get("grad_accum_batches", 1)
        self.max_seq_len: int = options.get("max_seq_len", 1024)
        self.stride: Optional[int] = options.get("stride", None)

    def update(self, options: dict):
        self.__dict__.update(options)
