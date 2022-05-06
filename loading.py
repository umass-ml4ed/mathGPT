import json
import os
import re
from typing import List

import torch
from transformers import GPT2TokenizerFast

from math_tokenize import tokenize_formula, EMPTY_POS_VECTOR, EMPTY_POS_ENCODING
from constants import Article, TokenType, Sequence, CollatedBatch, PADDING_TOKEN_ID, EOS_TOKEN
from utils import device

def load_articles():
    articles: List[Article] = []
    for article_filename in os.listdir("data"):
        with open(os.path.join("data", article_filename)) as article_file:
            article = json.load(article_file)
            article["name"] = article_filename
            articles.append(article)
    return articles


def split_sequence(sequence: Sequence, max_seq_len: int) -> List[Sequence]:
    """
    Split the given sequence into sub-sequences which are no longer than the maximum length
    Will always split the sequence at text tokens, since splitting within a formula would deprive model of tree context
    Will thus skip all formulas that are longer than the maximum length
    Conflict strategy: if the split point is within a formula, try to place the split point at the beginning of that formula
    # TODO: alternate strategies: preserve some prior text context, or duplicate some text information to keep some context
    """
    seq_len = len(sequence.token_ids)

    # If sequence is within the max length, just return the sequence
    if seq_len <= max_seq_len:
        return [sequence]

    # If the split point (at max length) is a text token, we can just split there and keep applying recursively
    if sequence.token_types[max_seq_len] == TokenType.TEXT:
        pre_split, post_split = sequence.split_at(max_seq_len)
        return [pre_split] + split_sequence(post_split, max_seq_len)

    # If the split point was not a text token (was in a formula), split at the start of the formula
    pre_form_text_tok_id = next((tok_idx for tok_idx in range(max_seq_len - 1, -1, -1) if sequence.token_types[tok_idx] == TokenType.TEXT), None)
    if not pre_form_text_tok_id:
        # No text tokens before the split point, so skip this formula and keep processing right after it ends
        print("Formula too long! Skipping...")
        # To skip this formula, we need to find the end of it and start the next split there
        end_of_form = next((tok_idx for tok_idx in range(max_seq_len, seq_len) if sequence.token_types[tok_idx] == TokenType.TEXT), None)
        if not end_of_form:
            # The sequence ends with this formula, so there is nothing left to salvage
            return []
        _, post_split = sequence.split_at(end_of_form)
        return split_sequence(post_split, max_seq_len)

    # Current sequence ends with last text token before formula, and next starts with formula
    pre_split, post_split = sequence.split_at(pre_form_text_tok_id + 1)
    return [pre_split] + split_sequence(post_split, max_seq_len)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, articles: List[Article], max_seq_len: int):
        self.data: List[Sequence] = []
        self.text_tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

        for article in articles:
            sequence = Sequence(article["name"])

            # Split article text in between formulas
            article_text = article["text"] + EOS_TOKEN
            text_chunks = re.split(r"\[\d+\]", article_text) # TODO: commonize regex
            for text_chunk_idx, text_chunk in enumerate(text_chunks):
                # TODO: see what happens with empty strings
                # Tokenize the text chunk and add it to the sequence
                text_token_ids: List[int] = self.text_tokenizer(text_chunk)["input_ids"]
                sequence.token_ids += text_token_ids
                sequence.token_types += [TokenType.TEXT] * len(text_token_ids)
                sequence.pos_vecs += [EMPTY_POS_VECTOR] * len(text_token_ids)
                sequence.pos_levels += [0] * len(text_token_ids)
                sequence.pos_encodings += [EMPTY_POS_ENCODING] * len(text_token_ids)

                # Article will end with a text chunk (even if it's an empty string)
                if text_chunk_idx == len(text_chunks) - 1:
                    continue

                # Skip formula if it wasn't captured
                if str(text_chunk_idx) not in article["formulas"]:
                    continue

                # Add formula start token
                sequence.token_ids.append(0)
                sequence.token_types.append(TokenType.START_FORMULA)
                sequence.pos_vecs.append(EMPTY_POS_VECTOR)
                sequence.pos_levels.append(0)
                sequence.pos_encodings.append(EMPTY_POS_ENCODING)

                # Tokenize the formula and add it to the sequence
                formula_sequence = tokenize_formula(article["formulas"][str(text_chunk_idx)]["opt"])
                sequence.token_ids += formula_sequence.token_ids
                sequence.token_types += formula_sequence.token_types
                sequence.pos_vecs += formula_sequence.pos_vecs
                sequence.pos_levels += formula_sequence.pos_levels
                sequence.pos_encodings += formula_sequence.pos_encodings

                # Add formula end token
                sequence.token_ids.append(0)
                sequence.token_types.append(TokenType.END_FORMULA)
                sequence.pos_vecs.append(EMPTY_POS_VECTOR)
                sequence.pos_levels.append(0)
                sequence.pos_encodings.append(EMPTY_POS_ENCODING)

            # Split sequence according to max length and add to final data
            split_sequences = split_sequence(sequence, max_seq_len)
            self.data += split_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def trim_batch(batch: CollatedBatch, trim_start: int, trim_end: int) -> CollatedBatch:
    """
    Return a copy of a trimmed collated batch in a given range
    """
    return {
        "articles": batch["articles"],
        "token_ids": batch["token_ids"][:, trim_start : trim_end],
        "token_types": batch["token_types"][:, trim_start : trim_end],
        "pos_vecs": batch["pos_vecs"][:, trim_start : trim_end],
        "pos_levels": batch["pos_levels"][:, trim_start : trim_end],
        "pos_encodings": batch["pos_encodings"][:, trim_start : trim_end],
        "attention_mask": batch["attention_mask"][:, trim_start : trim_end],
    }


class Collator:
    def __init__(self):
        pass

    def __call__(self, batch: List[Sequence]) -> CollatedBatch:
        token_id_batches = []
        token_type_batches = []
        pos_vec_batches = []
        pos_level_batches = []
        pos_encoding_batches = []
        attention_mask = []

        for sequence in batch:
            token_id_batches.append(torch.LongTensor(sequence.token_ids))
            token_type_batches.append(torch.LongTensor(sequence.token_types))
            pos_vec_batches.append(torch.LongTensor(sequence.pos_vecs))
            pos_level_batches.append(torch.LongTensor(sequence.pos_levels))
            pos_encoding_batches.append(torch.FloatTensor(sequence.pos_encodings))
            attention_mask.append(torch.ones(len(sequence.token_ids)))

        return {
            "articles": [sequence.src_article for sequence in batch],
            "token_ids": torch.nn.utils.rnn.pad_sequence(token_id_batches, batch_first=True, padding_value=PADDING_TOKEN_ID).to(device),
            "token_types": torch.nn.utils.rnn.pad_sequence(token_type_batches, batch_first=True).to(device),
            "pos_vecs": torch.nn.utils.rnn.pad_sequence(pos_vec_batches, batch_first=True).to(device),
            "pos_levels": torch.nn.utils.rnn.pad_sequence(pos_level_batches, batch_first=True).to(device),
            "pos_encodings": torch.nn.utils.rnn.pad_sequence(pos_encoding_batches, batch_first=True).to(device),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device),
        }
