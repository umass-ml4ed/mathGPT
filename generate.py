from typing import Dict, List
import torch
import torch.nn.functional as F
from tqdm import tqdm
from mathGPT.utils import TrainOptions

from model_math_gpt import MathGPTLM
from math_tokenize import encode_pos
from utils import device
from constants import CollatedBatch, TokenType, EOS_TOKEN_ID

def infer_math_pos(prev_pos_vecs: torch.Tensor, prev_pos_levels: torch.Tensor, prev_token_types: torch.Tensor):
    """
    Calculate math positions for the next token for each sequence in a batch - can always be determined by position and type of preceding token
    Return (next pos vectors for the batch, next pos levels for the batch)
    """
    # Start with copies of previous positions and modify based on case
    new_pos_vecs = torch.clone(prev_pos_vecs)
    new_pos_levels = torch.clone(prev_pos_levels)
    # Mask to select current level in each pos vector
    level_mask = F.one_hot(new_pos_levels, num_classes=new_pos_vecs.shape[1]).type(torch.bool).to(device)

    # An OP is always followed by a child
    op_idx = prev_token_types == TokenType.OP
    # Increment level; vec doesn't change since higher levels filled with 0 by default
    new_pos_levels[op_idx] += 1

    # A non-END terminal token is always followed by a sibling (or formula ends)
    term_idx = ((prev_token_types == TokenType.VAR) | (prev_token_types == TokenType.NUM)) & (new_pos_levels != 0)
    # Increment sibling; level doesn't change
    term_level_mask = level_mask * term_idx.unsqueeze(1)
    new_pos_vecs[term_level_mask] += 1

    # An END token always pops up and is followed by parent's sibling (or formula ends)
    end_idx = prev_token_types == TokenType.END
    # Set current level to 0 since we're popping up
    end_level_mask = level_mask * end_idx.unsqueeze(1)
    new_pos_vecs[end_level_mask] = 0
    # Decrement current level
    new_pos_levels[end_idx] -= 1
    # For tokens that don't end the formula, increase sibling value of new level
    level_mask = F.one_hot(new_pos_levels, num_classes=new_pos_vecs.shape[1]).type(torch.bool).to(device)
    end_level_mask = level_mask * (end_idx & new_pos_levels != 0).unsqueeze(1)
    new_pos_vecs[end_level_mask] += 1

    return new_pos_vecs, new_pos_levels

def get_most_likely_predictions(type_to_token_probs: Dict[TokenType, torch.Tensor]):
    batch_size, max_seq_len = type_to_token_probs[TokenType.TEXT].shape[:2]
    predicted_types = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    predicted_tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    max_token_probs = torch.zeros((batch_size, max_seq_len)).to(device)
    for token_type in TokenType:
        # Get most likely token for this type
        max_values, max_indices = torch.max(type_to_token_probs[token_type], dim=-1)
        # Find indices where most likely token is higher than previously most likely token
        new_highest_prob_idx = max_values > max_token_probs
        # For those indices, set the predicted token and type
        predicted_tokens[new_highest_prob_idx] = max_indices[new_highest_prob_idx]
        predicted_types[new_highest_prob_idx] = token_type
        # Update highest seen probabilities
        max_token_probs = torch.maximum(max_token_probs, max_values)
    return predicted_types, predicted_tokens

def get_nucleus_sample_predictions(type_to_token_probs: Dict[TokenType, torch.Tensor], options: TrainOptions):
    # TODO: apply temperature

    # Convert dictionary to single tensor with concatenated token probabilites across types
    type_to_start_idx = [0] * len(TokenType)
    cur_start_idx = 0
    token_prob_tensors = []
    for token_type in TokenType:
        type_to_start_idx[token_type] = cur_start_idx
        cur_start_idx += type_to_token_probs[token_type].shape[2]
        token_prob_tensors.append(type_to_token_probs[token_type][:, -1])
    token_probs = torch.concat(token_prob_tensors, dim=1)

    predicted_types: List[int] = []
    predicted_tokens: List[torch.Tensor] = []
    for batch_idx in range(token_probs.shape[0]):
        # Use top-p sampling to get the next token for each batch
        sorted_probs, sorted_indices = torch.sort(token_probs[batch_idx], descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        idx_filter_mask = cdf > options.ns_p
        idx_filter_mask[0] = False # If the most likely token is more likely than p, we still want to keep it
        filtered_indices = sorted_indices[idx_filter_mask]
        token_probs[batch_idx, filtered_indices] = 0
        next_token = torch.multinomial(token_probs[batch_idx], num_samples=1)

        # Convert next tokens from concatenated tensor back to token ID and type
        for token_type, start_idx in enumerate(type_to_start_idx):
            if token_type == len(type_to_start_idx) - 1 or next_token < type_to_start_idx[token_type + 1]:
                predicted_types.append(token_type)
                predicted_tokens.append(next_token - start_idx)
                break
    return torch.tensor(predicted_types).to(device), torch.concat(predicted_tokens)

def generate(model: MathGPTLM, gen_batch: CollatedBatch, options: TrainOptions):
    """
    Given a model and a batch, generate tokens up to the given length
    Given batch is modified
    """
    model.eval()
    batch_size, starting_len = gen_batch["token_ids"].shape
    for _ in tqdm(range(starting_len, options.max_seq_len)):
        # TODO: can we make faster by removing sequences in the batch that hit EOS? or set attention mask to 0 after EOS?
        # TODO: extract and pass past_key_values
        _, type_to_token_probs = model(gen_batch)

        scheme = "nucleus"
        if scheme == "greedy":
            type_preds, token_preds = get_most_likely_predictions(type_to_token_probs)
            type_preds = type_preds[:, -1]
            token_preds = token_preds[:, -1]
        if scheme == "nucleus":
            type_preds, token_preds = get_nucleus_sample_predictions(type_to_token_probs, options)

        new_pos_vecs, new_pos_levels = infer_math_pos(gen_batch["pos_vecs"][:, -1], gen_batch["pos_levels"][:, -1], gen_batch["token_types"][:, -1])
        new_pos_encodings = torch.LongTensor([encode_pos(pos_vec, pos_level) for pos_vec, pos_level in zip(new_pos_vecs, new_pos_levels)]).to(device)
        new_attention_mask = torch.ones(batch_size).to(device)

        gen_batch["token_ids"] = torch.concat([gen_batch["token_ids"], token_preds.unsqueeze(1)], dim=1)
        gen_batch["token_types"] = torch.concat([gen_batch["token_types"], type_preds.unsqueeze(1)], dim=1)
        gen_batch["pos_vecs"] = torch.concat([gen_batch["pos_vecs"], new_pos_vecs.unsqueeze(1)], dim=1)
        gen_batch["pos_levels"] = torch.concat([gen_batch["pos_levels"], new_pos_levels.unsqueeze(1)], dim=1)
        gen_batch["pos_encodings"] = torch.concat([gen_batch["pos_encodings"], new_pos_encodings.unsqueeze(1)], dim=1)
        gen_batch["attention_mask"] = torch.concat([gen_batch["attention_mask"], new_attention_mask.unsqueeze(1)], dim=1)

        if batch_size == 1 and gen_batch["token_ids"][0, -1] == EOS_TOKEN_ID:
            break
