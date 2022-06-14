import torch

from math_tokenize import encode_pos, get_empty_pos_encoding
from model_math_gpt import MathGPTLM
from vocabulary import Vocabulary
from constants import TokenType, TPE, PADDING_TOKEN_ID, MAX_FORMULA_DEPTH, EMB_SIZE, TEXT_VOCAB_SIZE
from utils import TrainOptions
from test_utils import assert_tensors_equal, assert_tensor_sums_to

class GPTOutputMock:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

def test_get_input_embeddings():
    with torch.no_grad():
        pos_vecs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 0], [0, 0]]
        pos_levels = [0, 0, 0, 1, 1, 1, 0, 0]
        pos_encodings = [
            get_empty_pos_encoding(TPE.FORTE.value), get_empty_pos_encoding(TPE.FORTE.value),
            *[encode_pos(pos_vec, pos_level, TPE.FORTE.value) for pos_vec, pos_level in zip(pos_vecs[2:6], pos_levels[2:6])],
            get_empty_pos_encoding(TPE.FORTE.value), get_empty_pos_encoding(TPE.FORTE.value)
        ]
        batch = {
            "token_ids": torch.LongTensor([
                [10, 0, 5, 4, 3, 0, 0, PADDING_TOKEN_ID]
            ]),
            "token_types": torch.LongTensor([
                [TokenType.TEXT, TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END, TokenType.END_FORMULA, TokenType.TEXT]
            ]),
            "pos_vecs": torch.LongTensor([
                pos_vecs
            ]),
            "pos_levels": torch.LongTensor([
                pos_levels
            ]),
            "pos_encodings": torch.FloatTensor([
                pos_encodings
            ]),
        }

        model = MathGPTLM(TrainOptions({}))
        input_embeddings = model.get_input_embeddings(batch)

        assert input_embeddings.shape == (1, 8, EMB_SIZE)
        assert_tensors_equal(input_embeddings[0, 0], (
            model.transformer.wte(torch.LongTensor([10]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.TEXT]))[0]
        ))
        assert_tensors_equal(input_embeddings[0, 1], (
            model.token_embeddings[str(TokenType.START_FORMULA.value)](torch.LongTensor([0]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.START_FORMULA]))[0]
        ))
        assert_tensors_equal(input_embeddings[0, 2], (
            model.token_embeddings[str(TokenType.OP.value)](torch.LongTensor([5]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.OP]))[0] +\
            model.math_embedding_projection(torch.FloatTensor(pos_encodings[2]))
        ))
        assert_tensors_equal(input_embeddings[0, 3], (
            model.token_embeddings[str(TokenType.NUM.value)](torch.LongTensor([4]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.NUM]))[0] +\
            model.math_embedding_projection(torch.FloatTensor(pos_encodings[3]))
        ))
        assert_tensors_equal(input_embeddings[0, 4], (
            model.token_embeddings[str(TokenType.VAR.value)](torch.LongTensor([3]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.VAR]))[0] +\
            model.math_embedding_projection(torch.FloatTensor(pos_encodings[4]))
        ))
        assert_tensors_equal(input_embeddings[0, 5], (
            model.token_embeddings[str(TokenType.END.value)](torch.LongTensor([0]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.END]))[0] +\
            model.math_embedding_projection(torch.FloatTensor(pos_encodings[5]))
        ))
        assert_tensors_equal(input_embeddings[0, 6], (
            model.token_embeddings[str(TokenType.END_FORMULA.value)](torch.LongTensor([0]))[0] +\
            model.type_embeddings(torch.LongTensor([TokenType.END_FORMULA]))[0]
        ))
        assert_tensors_equal(input_embeddings[0, 7], (
            model.type_embeddings(torch.LongTensor([TokenType.TEXT]))[0]
        ))

def test_masks():
    with torch.no_grad():
        # Test mask for each type, as well as both conditions for final token in a formula
        batch = {
            "token_ids": torch.LongTensor([
                [10, 0, 5, 4, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]),
            "token_types": torch.LongTensor([
                [TokenType.TEXT, TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END, TokenType.END_FORMULA],
                [TokenType.START_FORMULA, TokenType.VAR, TokenType.END_FORMULA, 0, 0, 0, 0],
            ]),
            "pos_vecs": torch.LongTensor([
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ]),
            "pos_levels": torch.LongTensor([
                [0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ])
        }

        model = MathGPTLM(TrainOptions({}))
        type_idxs, final_formula_token_idx = model.get_prediction_masks(batch)

        expected_type_idxs = {
            TokenType.TEXT: torch.tensor([
                [True, False, False, False, False, False, False],
                [False, False, False, True, True, True, True],
            ]),
            TokenType.START_FORMULA: torch.tensor([
                [False, True, False, False, False, False, False],
                [True, False, False, False, False, False, False],
            ]),
            TokenType.END_FORMULA: torch.tensor([
                [False, False, False, False, False, False, True],
                [False, False, True, False, False, False, False],
            ]),
            TokenType.VAR: torch.tensor([
                [False, False, False, False, True, False, False],
                [False, True, False, False, False, False, False],
            ]),
            TokenType.NUM: torch.tensor([
                [False, False, False, True, False, False, False],
                [False, False, False, False, False, False, False],
            ]),
            TokenType.OP: torch.tensor([
                [False, False, True, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]),
            TokenType.END: torch.tensor([
                [False, False, False, False, False, True, False],
                [False, False, False, False, False, False, False],
            ]),
        }
        for token_type in TokenType:
            assert torch.all(type_idxs[token_type] == expected_type_idxs[token_type])
        assert torch.all(final_formula_token_idx == torch.tensor([
            [False, False, False, False, False, True, False],
            [False, True, False, False, False, False, False]
        ]))

def test_type_probs():
    with torch.no_grad():
        # The second sequence in the batch makes no sense realistically, it's just to test the max width/depth protection
        vec_pad = [0] * (MAX_FORMULA_DEPTH - 2)
        batch = {
            "token_ids": torch.LongTensor([
                [10, 0, 5, 4, 3, 0, 0, PADDING_TOKEN_ID],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]),
            "token_types": torch.LongTensor([
                [TokenType.TEXT, TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END, TokenType.END_FORMULA, 0],
                [TokenType.VAR, TokenType.END, TokenType.OP, TokenType.VAR, TokenType.END, TokenType.VAR, TokenType.END, TokenType.OP]
            ]),
            "pos_vecs": torch.LongTensor([
                [[0, 0] + vec_pad, [0, 0] + vec_pad, [0, 0] + vec_pad, [0, 0] + vec_pad, [0, 1] + vec_pad, [0, 2] + vec_pad, [0, 0] + vec_pad, [0, 0] + vec_pad],
                [[0, 0] + vec_pad, [0, 0] + vec_pad, [0, 0] + vec_pad, [0, 0] + vec_pad, [0, 0] + vec_pad,
                 [0, 62] + vec_pad, [0, 62, 2] + [0] * (MAX_FORMULA_DEPTH - 3), [0, 62] + vec_pad]
            ]),
            "pos_levels": torch.LongTensor([
                [0, 0, 0, 1, 1, 1, 0, 0],
                [31, 31, 30, 30, 30, 1, 2, 1]
            ])
        }
        gpt_output_mock = GPTOutputMock(torch.Tensor([[[1] * EMB_SIZE] * 8] * 2))

        model = MathGPTLM(TrainOptions({}))
        type_idxs, final_formula_token_idx = model.get_prediction_masks(batch)
        type_probs = model.get_type_probs(gpt_output_mock, type_idxs, final_formula_token_idx, batch)

        def assert_probs(batch_idx, idx, allowed_types):
            assert_tensor_sums_to(type_probs[batch_idx][idx], 1)
            assert all([
                (type_probs[batch_idx][idx][token_type] > 0) if token_type in allowed_types else (type_probs[batch_idx][idx][token_type] == 0)
                for token_type in TokenType
            ])

        assert type_probs.shape == (2, 8, 7)
        assert_probs(0, 0, (TokenType.TEXT, TokenType.START_FORMULA))
        assert_probs(0, 1, (TokenType.OP, TokenType.VAR, TokenType.NUM))
        assert_probs(0, 2, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(0, 3, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(0, 4, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(0, 5, (TokenType.END_FORMULA,))
        assert_probs(0, 6, (TokenType.TEXT,))
        assert_probs(1, 0, (TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(1, 1, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(1, 2, (TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(1, 3, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(1, 4, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))
        assert_probs(1, 5, (TokenType.END,))
        assert_probs(1, 6, (TokenType.END,))
        assert_probs(1, 7, (TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END))

def test_token_probs():
    with torch.no_grad():
        batch = {
            "token_ids": torch.LongTensor([
                [10, 0, 5, 4, 3, 0, 0]
            ]),
            "token_types": torch.LongTensor([
                [TokenType.TEXT, TokenType.START_FORMULA, TokenType.OP, TokenType.NUM, TokenType.VAR, TokenType.END, TokenType.END_FORMULA]
            ]),
            "pos_vecs": torch.LongTensor([
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 0]]
            ]),
            "pos_levels": torch.LongTensor([
                [0, 0, 0, 1, 1, 1, 0]
            ])
        }
        gpt_output_mock = GPTOutputMock(torch.Tensor([
            [[1] * EMB_SIZE, [1] * EMB_SIZE, [1] * EMB_SIZE, [1] * EMB_SIZE, [1] * EMB_SIZE, [1] * EMB_SIZE, [1] * EMB_SIZE]
        ]))

        model = MathGPTLM(TrainOptions({}))
        type_idxs, final_formula_token_idx = model.get_prediction_masks(batch)
        type_probs = model.get_type_probs(gpt_output_mock, type_idxs, final_formula_token_idx, batch)
        type_to_token_probs = model.get_token_probs(gpt_output_mock, type_probs)

        def assert_probs(idx):
            assert_tensor_sums_to(torch.concat([type_to_token_probs[token_type][0, idx] for token_type in TokenType], dim=-1), 1)
            for token_type in TokenType:
                assert_tensor_sums_to(type_to_token_probs[token_type][0, idx], type_probs[0, idx, token_type])

        assert type_to_token_probs[TokenType.TEXT].shape == (1, 7, TEXT_VOCAB_SIZE)
        assert type_to_token_probs[TokenType.START_FORMULA].shape == (1, 7, 1)
        assert type_to_token_probs[TokenType.END_FORMULA].shape == (1, 7, 1)
        assert type_to_token_probs[TokenType.OP].shape == (1, 7, Vocabulary.num_tokens_in_type(TokenType.OP))
        assert type_to_token_probs[TokenType.VAR].shape == (1, 7, Vocabulary.num_tokens_in_type(TokenType.VAR))
        assert type_to_token_probs[TokenType.NUM].shape == (1, 7, Vocabulary.num_tokens_in_type(TokenType.NUM))
        assert type_to_token_probs[TokenType.END].shape == (1, 7, 1)
        for idx in range(7):
            assert_probs(idx)

def test_loss():
    batch = {
        "token_ids": torch.LongTensor([
            [1, 0, 1, 2, 0, 0, 3],
            [0, 1, 0, PADDING_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID],
        ]),
        "token_types": torch.LongTensor([
            [TokenType.TEXT, TokenType.START_FORMULA, TokenType.OP, TokenType.VAR, TokenType.END, TokenType.END_FORMULA, TokenType.TEXT],
            [TokenType.START_FORMULA, TokenType.VAR, TokenType.END_FORMULA, 0, 0, 0, 0],
        ]),
        "pos_levels": torch.LongTensor([ # Use dummy values since needed by code but not used in loss calculation
            [0] * 7,
            [0] * 7,
        ]),
        "gen_labels": None,
    }
    type_to_token_probs = {
        TokenType.TEXT: torch.Tensor([
            [
                [0.1, 0.1, 0.2, 0.2],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.3, 0.1, 0.3, 0.2],
                [0.2, 0.2, 0.2, 0.3],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.3, 0.1, 0.1],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]),
        TokenType.START_FORMULA: torch.Tensor([
            [[0.4], [0.0], [0.0], [0.0], [0.0], [0.0], [0.1]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        ]),
        TokenType.END_FORMULA: torch.Tensor([
            [[0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        ]),
        TokenType.OP: torch.Tensor([
            [
                [0.0, 0.0],
                [0.1, 0.5],
                [0.1, 0.1],
                [0.2, 0.1],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.5, 0.1],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ]),
        TokenType.VAR: torch.Tensor([
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.1],
                [0.1, 0.35, 0.25],
                [0.1, 0.1, 0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.15, 0.05, 0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]),
        TokenType.END: torch.Tensor([
            [[0.0], [0.0], [0.1], [0.4], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        ]),
    }

    model = MathGPTLM(TrainOptions({}))
    type_idxs, _ = model.get_prediction_masks(batch)
    loss = model.get_prediction_loss(type_to_token_probs, type_idxs, batch, None)

    # Loss is the average negative log prob of the following token at each timestamp
    expected_loss = -torch.log(torch.tensor([
        0.4, 0.5, 0.25, 0.4, 1.0, 0.2,
        0.05, 1.0
    ])).mean()
    assert loss == expected_loss

def test_text_only_decode():
    # TODO: compare our model's decoding to pre-trained gpt2 with LM head and ensure same results.
    # only use text data for this (obviously).
    # test out greedy, tree, and top-p.
    pass
