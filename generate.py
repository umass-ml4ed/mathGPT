from typing import List
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from vocabulary import Vocabulary
from math_tokenize import encode_pos
from utils import device
from constants import CollatedBatch, TokenType, EOS_TOKEN_ID, PADDING_TOKEN_ID

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

    # A non-END terminal token is always followed by a sibling
    term_idx = (prev_token_types == TokenType.VAR) | (prev_token_types == TokenType.NUM)
    # Mask to select current level for applicable sequences
    term_level_mask = level_mask * term_idx.unsqueeze(1)
    # Increment sibling; level doesn't change
    new_pos_vecs[term_level_mask] += 1

    # An END token always pops up and is followed by parent's sibling (or formula ends)
    end_idx = prev_token_types == TokenType.END
    # Mask to select current level for applicable sequences
    end_level_mask = level_mask * end_idx.unsqueeze(1)
    # Set current level to 0 since we're popping up
    new_pos_vecs[end_level_mask] = 0
    # Decrement current level
    new_pos_levels[end_idx] -= 1

    return new_pos_vecs, new_pos_levels

def generate(model, gen_batch: CollatedBatch, max_seq_len: int):
    """
    Given a model and a seed batch, generate tokens up to the given length
    Seed batch is modified
    """
    batch_size, starting_len = gen_batch["token_ids"].shape
    for _ in range(starting_len, max_seq_len):
        # TODO: can we make faster by removing sequences in the batch that hit EOS? or set attention mask to 0 after EOS?
        # TODO: extract and pass past_key_values
        _, _, type_preds, token_preds = model(gen_batch)
        # TODO: apply temperature
        # TODO: from transformers import tf_top_k_top_p_filtering
        gen_batch["token_ids"] = torch.concat([gen_batch["token_ids"], token_preds[:, -1].unsqueeze(1)], dim=1)
        gen_batch["token_types"] = torch.concat([gen_batch["token_types"], type_preds[:, -1].unsqueeze(1)], dim=1)
        new_pos_vecs, new_pos_levels = infer_math_pos(gen_batch["pos_vecs"][:, -1], gen_batch["pos_levels"][:, -1], gen_batch["token_types"][:, -2])
        gen_batch["pos_vecs"] = torch.concat([gen_batch["pos_vecs"], new_pos_vecs.unsqueeze(1)], dim=1)
        gen_batch["pos_levels"] = torch.concat([gen_batch["pos_levels"], new_pos_levels.unsqueeze(1)], dim=1)
        new_pos_encodings = torch.LongTensor([encode_pos(pos_vec, pos_level) for pos_vec, pos_level in zip(new_pos_vecs, new_pos_levels)]).to(device)
        gen_batch["pos_encodings"] = torch.concat([gen_batch["pos_encodings"], new_pos_encodings.unsqueeze(1)], dim=1)
        gen_batch["attention_mask"] = torch.concat([gen_batch["attention_mask"], torch.ones(batch_size).unsqueeze(1).to(device)], dim=1)

def decode_batch(batch: CollatedBatch, text_tokenizer: GPT2TokenizerFast) -> List[str]:
    """
    Given a batch, decode it into human-readable text
    Return text translation for each sequence in the batch
    """
    all_decoded_sequences: List[str] = []
    for seq_idx in range(len(batch["token_ids"])):
        result = ""
        is_text = True # Assumption: sequence will always either start with a text token, or start formula token
        sub_seq_start = sub_seq_end = 0
        for tok_idx, (token_type, token_id) in enumerate(zip(batch["token_types"][seq_idx], batch["token_ids"][seq_idx])):
            # At start formula, switch to math context, and decode any prior text tokens
            if token_type == TokenType.START_FORMULA:
                is_text = False
                if sub_seq_start != sub_seq_end:
                    result += text_tokenizer.decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])
                result += "<START_FORMULA>"
                sub_seq_start = sub_seq_end = tok_idx + 1
                continue

            # At end formula, switch to text context
            if token_type == TokenType.END_FORMULA:
                is_text = True
                # TODO: decode the OPT and add to result
                result += "<END_FORMULA>"
                sub_seq_start = sub_seq_end = tok_idx + 1
                continue

            # NOTE: this is temporary until we implement tree decoding
            if not is_text:
                if token_type != TokenType.END:
                    result += Vocabulary.get_symbol(int(token_type), int(token_id)) + " "

            sub_seq_end = tok_idx + 1

            # Stop decoding at EOS token
            if token_type == TokenType.TEXT and token_id == EOS_TOKEN_ID:
                break

        # Decode any trailing text tokens at the end
        if sub_seq_start != sub_seq_end:
            # TODO: probably a more elegant way to handle this
            if not all(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end] == PADDING_TOKEN_ID):
                result += text_tokenizer.decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])

        all_decoded_sequences.append(result)

    return all_decoded_sequences
