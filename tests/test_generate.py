import torch

from generate import infer_math_pos
from constants import TokenType

def test_infer_math_pos():
    # Test each of the 3 cases (var/num share a case)
    prev_pos_vecs = torch.LongTensor([
        [0, 0, 0],
        [0, 1, 0],
        [0, 3, 1],
        [0, 5, 3]
    ])
    prev_pos_levels = torch.LongTensor([
        0,
        1,
        2,
        2,
    ])
    prev_token_types = torch.LongTensor([
        TokenType.OP,
        TokenType.VAR,
        TokenType.NUM,
        TokenType.END,
    ])

    new_pos_vecs, new_pos_levels = infer_math_pos(prev_pos_vecs, prev_pos_levels, prev_token_types)
    assert torch.all(new_pos_vecs == torch.LongTensor([
        [0, 0, 0],
        [0, 2, 0],
        [0, 3, 2],
        [0, 5, 0],
    ]))
    assert torch.all(new_pos_levels == torch.LongTensor([
        1,
        1,
        2,
        1,
    ]))

def test_generate():
    pass # TODO: mock model, only need to generate one new token (until we have new techniques)

def test_decode_batch():
    pass # TODO
