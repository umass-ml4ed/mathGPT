import json
import os
import re
from typing import List

import torch
from transformers import GPT2TokenizerFast

from math_tokenize import tokenize_formula
from constants import Article, Token, TokenType, CollatedBatch
from utils import device

def load_articles():
    articles: List[Article] = []
    for article_filename in os.listdir("data"):
        with open(os.path.join("data", article_filename)) as article_file:
            articles.append(json.load(article_file))
    return articles


# TODO: better typing for dict with token data
def split_sequence(token_ids, token_types, positions, max_seq_len: int) -> List[dict]:
    """
    Split the given sequence into sub-sequences which are no longer than the maximum length
    Will always split the sequence at text tokens, since splitting within a formula would deprive model of tree context
    Will thus skip all formulas that are longer than the maximum length
    Conflict strategy: if the split point is within a formula, try to place the split point at the beginning of that formula
    # TODO: alternate strategies: preserve some prior text context, or duplicate some text information to keep some context
    """
    seq_len = len(token_ids)

    # If sequence is within the max length, just return the sequence
    if seq_len <= max_seq_len:
        return [{
            "token_ids": token_ids,
            "token_types": token_types,
            "positions": positions
        }]

    # If the split point (at max length) is a text token, we can just split there and keep applying recursively
    if token_types[max_seq_len] == 0: # TODO: text type
        return [{
            "token_ids": token_ids[:max_seq_len],
            "token_types": token_types[:max_seq_len],
            "positions": positions[:max_seq_len]
        }] + split_sequence(token_ids[max_seq_len:], token_types[max_seq_len:], positions[max_seq_len:], max_seq_len)

    # If the split point was not a text token (was in a formula), split at the start of the formula
    pre_form_text_tok_id = next((tok_idx for tok_idx in range(max_seq_len - 1, -1, -1) if token_types[tok_idx] == 0), None) # TODO: constant
    if not pre_form_text_tok_id:
        # No text tokens before the split point, so skip this formula and keep processing right after it ends
        print("Formula too long! Skipping...")
        # To skip this formula, we need to find the end of it and start the next split there
        end_of_form = next((tok_idx for tok_idx in range(max_seq_len, seq_len) if token_types[tok_idx] == 0), None) # TODO: constant
        if not end_of_form:
            # The sequence ends with this formula, so there is nothing left to salvage
            return []
        return split_sequence(token_ids[end_of_form:], token_types[end_of_form:], positions[end_of_form:], max_seq_len)

    # Current sequence ends with last text token before formula, and next starts with formula
    split_point = pre_form_text_tok_id + 1
    return [{
        "token_ids": token_ids[:split_point],
        "token_types": token_types[:split_point],
        "positions": positions[:split_point]
    }] + split_sequence(token_ids[split_point:], token_types[split_point:], positions[split_point:], max_seq_len)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, articles: List[Article], max_seq_len: int):
        self.data = [] # TODO: assign type
        self.text_tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

        for article in articles:
            token_ids: List[int] = []
            token_types: List[int] = []
            positions: List[str] = []

            # TODO: add end of sequence token? just needed at end of article, or needed at the end of each subsequence?

            # Split article text in between formulas
            text_chunks = re.split(r"\[\d+\]", article["text"]) # TODO: commonize regex
            for text_chunk_idx, text_chunk in enumerate(text_chunks):
                # TODO: see what happens with empty strings
                # Tokenize the text chunk and add it to the sequence
                text_token_ids = self.text_tokenizer(text_chunk)["input_ids"]
                token_ids += text_token_ids
                token_types += [TokenType.TEXT.value] * len(text_token_ids)
                positions += [""] * len(text_token_ids)

                # TODO: just testing with text for now to see if that works
                if True:
                    continue

                # Article will end with a text chunk (even if it's an empty string)
                if text_chunk_idx == len(text_chunks) - 1:
                    continue

                # Skip formula if it wasn't captured
                if str(text_chunk_idx) not in article["formulas"]:
                    continue

                # Add formula start token
                token_ids.append(Token.SWITCH_CONTEXT.value)
                token_types.append(TokenType.TEXT.value)
                positions.append("")

                # Tokenize the formula and add it to the sequence
                math_token_ids, math_token_types, math_positions = tokenize_formula(article["formulas"][str(text_chunk_idx)]["opt"])
                token_ids += math_token_ids
                token_types += math_token_types
                positions += math_positions

                # Add formula end token
                token_ids.append(Token.SWITCH_CONTEXT.value)
                token_types.append(TokenType.TEXT.value)
                positions.append("")

            # Split sequence according to max length and add to final data
            # TODO: use named tuple or typed dict for better semantics
            split_sequences = split_sequence(token_ids, token_types, positions, max_seq_len)
            self.data += split_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def trim_batch(batch: CollatedBatch, trim_point: int) -> CollatedBatch:
    """
    Return a copy of a trimmed collated batch up to a given point
    """
    return {
        "token_ids": batch["token_ids"][:, :trim_point],
        "token_types": batch["token_types"][:, :trim_point],
        # "positions": batch["positions"][:, :trim_point],
        "attention_mask": batch["attention_mask"][:, :trim_point],
    }


class Collator:
    def __init__(self):
        pass

    # TODO: type for batch
    def __call__(self, batch: List[dict]) -> CollatedBatch:
        token_id_batches = []
        token_type_batches = []
        position_batches = []
        attention_mask = []

        for sequence in batch:
            token_id_batches.append(torch.LongTensor(sequence["token_ids"]))
            token_type_batches.append(torch.LongTensor(sequence["token_types"]))
            # position_batches.append(torch.LongTensor(sequence["positions"])) # TODO: gotta encode positions first
            attention_mask.append(torch.ones(len(sequence["token_ids"])))

        # TODO: try padding with pad_token_id from model.config
        return {
            "token_ids": torch.nn.utils.rnn.pad_sequence(token_id_batches, batch_first=True).to(device),
            "token_types": torch.nn.utils.rnn.pad_sequence(token_type_batches, batch_first=True).to(device),
            # "positions": torch.nn.utils.rnn.pad_sequence(position_batches, batch_first=True).to(device),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device),
        }
