from functools import lru_cache
import random
import os
from typing import Optional, Dict
import numpy as np
import torch
import torch.distributed as dist
# import neptune.new as neptune
from transformers import GPT2TokenizerFast

from constants import DownstreamTask, TPE, Gen, Optimizer, ModelSize, DOWNSTREAM_TASK_TO_NUM_CLASSES

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

def new_neptune_run():
    return None

def setup_proc_group(rank: int, world_size: int):
    global device

    # Set the device to the GPU given to the process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize the process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_proc_group():
    dist.destroy_process_group()

def enum_choices(enum):
    return [choice.value for choice in enum]

def enum_value_to_member(value, enum):
    return next((member for member in enum if member.value == value), None)

def is_cls_task(task: Optional[DownstreamTask]):
    return task in DOWNSTREAM_TASK_TO_NUM_CLASSES

@lru_cache
def text_tokenizer():
    """
    Retrieve the GPT-2 text tokenizer
    Note that all model sizes have exactly the same vocabulary
    """
    return GPT2TokenizerFast.from_pretrained("gpt2")

def load_pretrained(model: torch.nn.Module, pretrained_state_dict: Dict[str, torch.Tensor]):
    """
    Given a pre-trained model's state_dict, load its parameters that are relevant to the current model
    """
    state_dict = model.state_dict()
    if hasattr(model, "transform_pretrained_state_dict"):
        _pretrained_state_dict = model.transform_pretrained_state_dict(pretrained_state_dict)
    else:
        _pretrained_state_dict = pretrained_state_dict
    for param_name, param_val in _pretrained_state_dict.items():
        if param_name in state_dict:
            state_dict[param_name] = param_val
    model.load_state_dict(state_dict)

class TrainOptions:
    def __init__(self, options: dict):
        # Training/testing params
        self.dataset: Optional[str] = options.get("dataset", None)
        self.data_dir: Optional[str] = options.get("data_dir", None)
        self.split: float = options.get("split", 0.95)
        self.optim: str = options.get("optim", Optimizer.ADAMW.value)
        self.lr: float = options.get("lr", 1e-5)
        self.weight_decay: float = options.get("weight_decay", 1e-2)
        self.epochs: int = options.get("epochs", 20)
        self.patience: Optional[int] = options.get("patience", None)
        self.batch_size: int = options.get("batch_size", 4)
        self.grad_accum_batches: int = options.get("grad_accum_batches", 4)
        self.max_seq_len: int = options.get("max_seq_len", 1024)
        self.amp: bool = options.get("amp", True)
        self.gen: str = options.get("gen", Gen.BEAM.value)
        self.ns_p: float = options.get("ns_p", 0.90)
        self.beam_width: int = options.get("beam_width", 3)
        self.min_gen_len: int = options.get("min_gen_len", 0)
        self.eval_formulas: bool = options.get("eval_formulas", False)
        self.eval_text: bool = options.get("eval_text", False)
        self.eval_final: bool = options.get("eval_final", False)
        self.stride: Optional[int] = options.get("stride", None)
        self.ddp: bool = options.get("ddp", False)
        # Model/tree structure config
        self.model_size: str = options.get("model_size", ModelSize.SMALL.value)
        self.baseline: bool = options.get("baseline", False)
        self.post_proc: bool = options.get("post_proc", False)
        self.joint: bool = options.get("joint", False)
        self.use_type_embs: bool = options.get("use_type_embs", True)
        self.tpe: str = options.get("tpe", TPE.FORTE.value)
        self.num_classes: Optional[int] = options.get("num_classes", None)
        self.num_to_tree: bool = options.get("num_to_tree", True)
        self.sd_to_tree: bool = options.get("sd_to_tree", False)
        self.math_text: bool = options.get("math_text", True)
        self.shared_emb: bool = options.get("shared_emb", True)
        self.cdt: bool = options.get("cdt", True)
        self.freeze_wte: bool = options.get("freeze_wte", False)
        self.init_math_pred: bool = options.get("init_math_pred", False)
        self.lmhb: bool = options.get("lmhb", True)

    def as_dict(self):
        self_dict = self.__dict__
        return self_dict

    def update(self, options: dict):
        self.__dict__.update(options)
