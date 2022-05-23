import torch

def assert_tensors_equal(t1: torch.Tensor, t2: torch.Tensor):
    assert torch.sum(torch.abs(t1 - t2)) < 0.00001

def assert_tensor_sums_to(t1: torch.Tensor, val: float):
    assert torch.abs(val - torch.sum(t1)) < 0.00001
