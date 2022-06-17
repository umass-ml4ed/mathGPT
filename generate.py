from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loading import trim_batch
from model_math_gpt import MathGPTLM
from math_tokenize import encode_pos
from utils import device, TrainOptions
from constants import CollatedBatch, TokenType, Gen, EOS_TOKEN_ID, PADDING_TOKEN_ID

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

def collapse_token_probs(type_to_token_probs: Dict[TokenType, torch.Tensor]):
    """
    Convert dictionary to single tensor with concatenated token probabilites across types
    """
    type_to_start_idx = [0] * len(TokenType)
    cur_start_idx = 0
    token_prob_tensors = []
    for token_type in TokenType:
        type_to_start_idx[token_type] = cur_start_idx
        cur_start_idx += type_to_token_probs[token_type].shape[2]
        token_prob_tensors.append(type_to_token_probs[token_type][:, -1])
    token_probs = torch.concat(token_prob_tensors, dim=1)
    return token_probs, type_to_start_idx

def uncollapse_token_idx(token_idx: int, type_to_start_idx: List[int]):
    """
    Given a token's index from the collapsed vocabulary, get the original token ID and type
    """
    for token_type, start_idx in enumerate(type_to_start_idx):
        if token_type == len(type_to_start_idx) - 1 or token_idx < type_to_start_idx[token_type + 1]:
            return token_idx - start_idx, token_type

def get_nucleus_sample_predictions(type_to_token_probs: Dict[TokenType, torch.Tensor], options: TrainOptions):
    # TODO: apply temperature

    token_probs, type_to_start_idx = collapse_token_probs(type_to_token_probs)
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
        token_id, token_type = uncollapse_token_idx(next_token, type_to_start_idx)
        predicted_tokens.append(token_id)
        predicted_types.append(token_type)
    return torch.tensor(predicted_types).to(device), torch.concat(predicted_tokens)

def add_to_batch(batch: CollatedBatch, token_preds: torch.Tensor, type_preds: torch.Tensor, options: TrainOptions):
    batch_size = batch["token_ids"].shape[0]
    new_pos_vecs, new_pos_levels = infer_math_pos(batch["pos_vecs"][:, -1], batch["pos_levels"][:, -1], batch["token_types"][:, -1])
    new_pos_encodings = torch.LongTensor([encode_pos(pos_vec, pos_level, options.tpe) for pos_vec, pos_level in zip(new_pos_vecs, new_pos_levels)]).to(device)
    new_attention_mask = torch.ones(batch_size).to(device)
    batch["token_ids"] = torch.concat([batch["token_ids"], token_preds.unsqueeze(1)], dim=1)
    batch["token_types"] = torch.concat([batch["token_types"], type_preds.unsqueeze(1)], dim=1)
    batch["pos_vecs"] = torch.concat([batch["pos_vecs"], new_pos_vecs.unsqueeze(1)], dim=1)
    batch["pos_levels"] = torch.concat([batch["pos_levels"], new_pos_levels.unsqueeze(1)], dim=1)
    batch["pos_encodings"] = torch.concat([batch["pos_encodings"], new_pos_encodings.unsqueeze(1)], dim=1)
    batch["attention_mask"] = torch.concat([batch["attention_mask"], new_attention_mask.unsqueeze(1)], dim=1)
    if batch["gen_labels"] is not None:
        batch["gen_labels"] = torch.concat([batch["gen_labels"], token_preds.unsqueeze(1)], dim=1)

def generate(model: MathGPTLM, start_batch: CollatedBatch, options: TrainOptions):
    """
    Given a model and a batch, generate tokens up to the given length
    """
    if options.gen == Gen.BEAM.value:
        return generate_beam(model, start_batch, options)

    model.eval()
    gen_batch = trim_batch(start_batch, 0, options.max_seq_len)
    batch_size, starting_len = gen_batch["token_ids"].shape
    for _ in range(starting_len, options.max_seq_len):
        # TODO: can we make faster by removing sequences in the batch that hit EOS? or set attention mask to 0 after EOS?
        # TODO: extract and pass past_key_values
        _, type_to_token_probs = model(gen_batch)

        if options.gen == Gen.GREEDY.value:
            type_preds, token_preds = get_most_likely_predictions(type_to_token_probs)
            type_preds = type_preds[:, -1]
            token_preds = token_preds[:, -1]
        elif options.gen == Gen.NUCLEUS.value:
            type_preds, token_preds = get_nucleus_sample_predictions(type_to_token_probs, options)

        add_to_batch(gen_batch, token_preds, type_preds, options)

        if batch_size == 1 and gen_batch["token_ids"][0, -1] == EOS_TOKEN_ID:
            break
    return gen_batch

def generate_beam(model: MathGPTLM, start_batch: CollatedBatch, options: TrainOptions):
    model.eval()
    batch_size, starting_len = start_batch["token_ids"].shape
    assert batch_size == 1 # It's just easier this way...
    candidate_batches = [(0., trim_batch(start_batch, 0, options.max_seq_len))]
    for _ in range(starting_len, options.max_seq_len):
        # For each candidate batch, get the n most likely continuations
        top_candidate_info: List[Tuple[float, int, int, int]] = []
        for batch_idx, (nll, batch) in enumerate(candidate_batches):
            # If an EOS has already been generated, just keep the single path as a candidate
            if batch["token_ids"][0, -1] in (EOS_TOKEN_ID, PADDING_TOKEN_ID):
                top_candidate_info.append((
                    nll,
                    PADDING_TOKEN_ID,
                    TokenType.TEXT.value,
                    batch_idx
                ))
                continue
            loss, type_to_token_probs = model(batch)
            token_probs, type_to_start_idx = collapse_token_probs(type_to_token_probs)
            sorted_probs, sorted_indices = torch.sort(token_probs[0], descending=True)
            labels = batch["token_ids"] if batch["gen_labels"] is None else batch["gen_labels"]
            loss_denom = torch.sum(labels != PADDING_TOKEN_ID)
            for beam in range(options.beam_width):
                new_token_id, new_token_type = uncollapse_token_idx(sorted_indices[beam], type_to_start_idx)
                top_candidate_info.append((
                    (loss * loss_denom - torch.log(sorted_probs[beam])).detach().cpu().numpy(),
                    new_token_id,
                    new_token_type,
                    batch_idx
                ))
        # Create a new candidate batch for the n most likely continuations
        new_candidate_batches: List[Tuple[float, CollatedBatch]] = []
        for nll, new_token_id, new_token_type, batch_idx in sorted(top_candidate_info)[:options.beam_width]:
            new_batch = trim_batch(candidate_batches[batch_idx][1], 0, options.max_seq_len)
            token_preds = torch.tensor([new_token_id]).to(device)
            type_preds = torch.tensor([new_token_type]).to(device)
            add_to_batch(new_batch, token_preds, type_preds, options)
            new_candidate_batches.append((nll, new_batch))
        candidate_batches = new_candidate_batches
        # Stop if all beams hit an EOS
        if all(batch[1]["token_ids"][0, -1] in (EOS_TOKEN_ID, PADDING_TOKEN_ID) for batch in candidate_batches):
            break
    return sorted(candidate_batches, key=lambda candidate: candidate[0])[0][1]

def generate_batch(model: MathGPTLM, gen_batch: CollatedBatch, options: TrainOptions):
    """
    Given a model and a batch, generate tokens up to the given length
    Given batch is modified
    """
    model.eval()
    # Pad batch to max_seq_len
    batch_size, starting_len = gen_batch["token_ids"].shape
    padding_size = options.max_seq_len - starting_len
    gen_batch["token_ids"] = torch.concat([
        gen_batch["token_ids"], torch.full((batch_size, padding_size), PADDING_TOKEN_ID).to(device)], dim=1)
    gen_batch["token_types"] = torch.concat([
        gen_batch["token_types"], torch.zeros((batch_size, padding_size), dtype=torch.long).to(device)], dim=1)
    gen_batch["pos_vecs"] = torch.concat([
        gen_batch["pos_vecs"], torch.zeros((batch_size, padding_size, gen_batch["pos_vecs"].shape[2]), dtype=torch.long).to(device)], dim=1)
    gen_batch["pos_levels"] = torch.concat([
        gen_batch["pos_levels"], torch.zeros((batch_size, padding_size), dtype=torch.long).to(device)], dim=1)
    gen_batch["pos_encodings"] = torch.concat([
        gen_batch["pos_encodings"], torch.zeros((batch_size, padding_size, gen_batch["pos_encodings"].shape[2])).to(device)], dim=1)
    gen_batch["attention_mask"] = torch.concat([
        gen_batch["attention_mask"], torch.zeros((batch_size, padding_size)).to(device)], dim=1)
    # Tells model to not compute loss
    gen_batch["gen_labels"] = torch.full((batch_size, options.max_seq_len), PADDING_TOKEN_ID).to(device)

    starting_idx = torch.min(gen_batch["sequence_lengths"])
    for _ in tqdm(range(starting_idx, options.max_seq_len)):
        # TODO: can we make faster by removing sequences in the batch that hit EOS? or set attention mask to 0 after EOS?
        # TODO: extract and pass past_key_values
        # Get predictions from model
        _, type_to_token_probs = model(gen_batch)
        batch_all_idx = torch.arange(batch_size)
        last_idx = gen_batch["sequence_lengths"] - 1

        # Collapse predictions via decoding scheme
        scheme = "nucleus"
        if scheme == "greedy":
            type_preds, token_preds = get_most_likely_predictions(type_to_token_probs)
            type_preds = type_preds[batch_all_idx, last_idx]
            token_preds = token_preds[batch_all_idx, last_idx]
        if scheme == "nucleus":
            type_preds, token_preds = get_nucleus_sample_predictions(
                {token_type: probs[batch_all_idx, last_idx] for token_type, probs in type_to_token_probs.items()},
                options
            )

        new_pos_vecs, new_pos_levels = infer_math_pos(
            gen_batch["pos_vecs"][batch_all_idx, last_idx],
            gen_batch["pos_levels"][batch_all_idx, last_idx],
            gen_batch["token_types"][batch_all_idx, last_idx]
        )
        new_pos_encodings = torch.FloatTensor([encode_pos(pos_vec, pos_level, options.tpe) for pos_vec, pos_level in zip(new_pos_vecs, new_pos_levels)]).to(device)

        # TODO: next_idx could be equal to max_seq_len, so include that check in not_eos_idx
        not_eos_idx = gen_batch["token_ids"][batch_all_idx, last_idx] != EOS_TOKEN_ID
        batch_cont_idx = batch_all_idx[not_eos_idx]
        next_idx = gen_batch["sequence_lengths"][not_eos_idx]
        gen_batch["token_ids"][batch_cont_idx, next_idx] = token_preds[batch_cont_idx]
        gen_batch["token_types"][batch_cont_idx, next_idx] = type_preds[batch_cont_idx]
        gen_batch["pos_vecs"][batch_cont_idx, next_idx] = new_pos_vecs[batch_cont_idx]
        gen_batch["pos_levels"][batch_cont_idx, next_idx] = new_pos_levels[batch_cont_idx]
        gen_batch["pos_encodings"][batch_cont_idx, next_idx] = new_pos_encodings[batch_cont_idx]
        gen_batch["attention_mask"][batch_cont_idx, next_idx] = 1
        gen_batch["sequence_lengths"][batch_cont_idx] += 1

        if torch.all(gen_batch["token_ids"][batch_all_idx, gen_batch["sequence_lengths"] - 1] == EOS_TOKEN_ID):
            break
