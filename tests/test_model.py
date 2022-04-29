import torch

from model_math_gpt import MathGPT, EMB_SIZE, get_collapsed_predictions
from vocabulary import Vocabulary
from constants import TokenType

def test_get_input_embeddings():
    batch = {
        "token_ids": torch.LongTensor([
            [10, 0, 5, 4, 3, 0]
        ]),
        "token_types": torch.LongTensor([
            [TokenType.TEXT, TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END_FORMULA]
        ]),
    }

    model = MathGPT()
    input_embeddings = model.get_input_embeddings(batch)

    assert input_embeddings.shape == (1, 6, EMB_SIZE)
    # TODO: math positions
    assert all(input_embeddings[0, 0] == (
        model.transformer.wte(torch.LongTensor([10]))[0] + model.type_embeddings(torch.LongTensor([TokenType.TEXT]))[0]))
    assert all(input_embeddings[0, 1] == (
        model.token_embeddings[str(TokenType.START_FORMULA.value)](torch.LongTensor([0]))[0] + model.type_embeddings(torch.LongTensor([TokenType.START_FORMULA]))[0]))
    assert all(input_embeddings[0, 2] == (
        model.token_embeddings[str(TokenType.OP.value)](torch.LongTensor([5]))[0] + model.type_embeddings(torch.LongTensor([TokenType.OP]))[0]))
    assert all(input_embeddings[0, 3] == (
        model.token_embeddings[str(TokenType.NUM.value)](torch.LongTensor([4]))[0] + model.type_embeddings(torch.LongTensor([TokenType.NUM]))[0]))
    assert all(input_embeddings[0, 4] == (
        model.token_embeddings[str(TokenType.VAR.value)](torch.LongTensor([3]))[0] + model.type_embeddings(torch.LongTensor([TokenType.VAR]))[0]))
    assert all(input_embeddings[0, 5] == (
        model.token_embeddings[str(TokenType.END_FORMULA.value)](torch.LongTensor([0]))[0] + model.type_embeddings(torch.LongTensor([TokenType.END_FORMULA]))[0]))

def test_predictions():
    type_to_token_probs = {
        TokenType.TEXT: torch.Tensor([
            [[0.1, 0.2, 0.1], [0.4, 0.2, 0.1]],
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.1]]
        ]),
        TokenType.START_FORMULA: torch.Tensor([
            [[0.1], [0.4]],
            [[0.0], [0.5]]
        ]),
        TokenType.END_FORMULA: torch.Tensor([
            [[0.1], [0.4]],
            [[0.0], [0.5]]
        ]),
        TokenType.OP: torch.Tensor([
            [[0.1, 0.2, 0.1], [0.4, 0.2, 0.1]],
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.1]]
        ]),
        TokenType.NUM: torch.Tensor([
            [[0.1, 0.2, 0.1], [0.4, 0.2, 0.1]],
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.1]]
        ]),
        TokenType.VAR: torch.Tensor([
            [[0.1, 0.2, 0.1], [0.4, 0.2, 0.1]],
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.1]]
        ])
    }
    type_preds, token_preds = get_collapsed_predictions(type_to_token_probs)
    assert all(type_preds == [])
    assert all(token_preds == [])

def test_text_only_decode():
    # TODO: compare our model's decoding to pre-trained gpt2 with LM head and ensure same results.
    # only use text data for this (obviously).
    # test out greedy, tree, and top-p.
    pass
