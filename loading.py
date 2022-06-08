import json
from typing import List, Optional, Dict
from tqdm import tqdm
import torch
from transformers import GPT2TokenizerFast

from math_tokenize import tokenize_formula, EMPTY_POS_VECTOR, EMPTY_POS_ENCODING
from constants import Article, GenTaskSample, ClassifyTaskSample, TokenType, Formula, Sequence, CollatedBatch, DownstreamTask, PADDING_TOKEN_ID, EOS_TOKEN, SEP_TOKEN, CLS_TOKEN, FORMULA_IDENTIFIER
from utils import device, is_cls_task

def split_sequence(sequence: Sequence, max_seq_len: int) -> List[Sequence]:
    """
    Split the given sequence into sub-sequences which are no longer than the maximum length
    Will always split the sequence at text tokens, since splitting within a formula would deprive model of tree context
    Will thus skip all formulas that are longer than the maximum length
    Conflict strategy: if the split point is within a formula, try to place the split point at the beginning of that formula
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

def tokenize_sequence(name: str, text: str, formulas: Dict[str, Formula], text_tokenizer: GPT2TokenizerFast):
    sequence = Sequence(name)
    text_chunks = text.split(FORMULA_IDENTIFIER)
    num_missing_formulas = 0
    for text_chunk_idx, text_chunk in enumerate(text_chunks):
        # Tokenize the text chunk and add it to the sequence
        text_token_ids: List[int] = text_tokenizer(text_chunk)["input_ids"]
        sequence.token_ids += text_token_ids
        sequence.token_types += [TokenType.TEXT] * len(text_token_ids)
        sequence.pos_vecs += [EMPTY_POS_VECTOR] * len(text_token_ids)
        sequence.pos_levels += [0] * len(text_token_ids)
        sequence.pos_encodings += [EMPTY_POS_ENCODING] * len(text_token_ids)

        # Sequence will end with a text chunk (even if it's an empty string)
        if text_chunk_idx == len(text_chunks) - 1:
            continue

        # Skip formula if it wasn't captured
        if str(text_chunk_idx) not in formulas:
            num_missing_formulas += 1
            continue

        # Add formula start token
        sequence.token_ids.append(0)
        sequence.token_types.append(TokenType.START_FORMULA)
        sequence.pos_vecs.append(EMPTY_POS_VECTOR)
        sequence.pos_levels.append(0)
        sequence.pos_encodings.append(EMPTY_POS_ENCODING)

        # Tokenize the formula and add it to the sequence
        formula_sequence = tokenize_formula(formulas[str(text_chunk_idx)]["opt"])
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

    return sequence, num_missing_formulas

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data: List[Sequence] = []
        self.text_tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        print("Processing data...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class PreTrainDataset(Dataset):
    def __init__(self, article_filenames: List[str], max_seq_len: Optional[int] = None):
        super().__init__()
        num_missing_formulas = 0
        for article_name in tqdm(article_filenames):
            with open(article_name, encoding="utf-8") as article_file:
                article: Article = json.load(article_file)

            article_text = article["text"] + EOS_TOKEN
            sequence, cur_missing_formulas = tokenize_sequence(article_name, article_text, article["formulas"], self.text_tokenizer)
            num_missing_formulas += cur_missing_formulas

            if max_seq_len:
                split_sequences = split_sequence(sequence, max_seq_len)
                self.data += split_sequences
            else:
                self.data.append(sequence)
        print("Missing", num_missing_formulas, "formulas")

class PreTrainDatasetPreloaded(Dataset):
    def __init__(self, articles: List[Article], max_seq_len: Optional[int] = None):
        super().__init__()
        num_missing_formulas = 0
        for article in tqdm(articles):
            article_text = article["text"]
            sequence, cur_missing_formulas = tokenize_sequence("", article_text, article["formulas"], self.text_tokenizer)
            num_missing_formulas += cur_missing_formulas

            if max_seq_len:
                split_sequences = split_sequence(sequence, max_seq_len)
                self.data += split_sequences
            else:
                self.data.append(sequence)
        print("Missing", num_missing_formulas, "formulas")

class GenTaskDataset(Dataset):
    def __init__(self, samples: List[GenTaskSample], max_seq_len: int):
        super().__init__()
        num_missing_formulas = 0
        trimmed_sequences = 0
        for sample in tqdm(samples):
            # Tokenize the prompt and label sequences
            prompt_text = sample["prompt"]["text"] + SEP_TOKEN
            prompt_sequence, cur_missing_formulas = tokenize_sequence("", prompt_text, sample["prompt"]["formulas"], self.text_tokenizer)
            num_missing_formulas += cur_missing_formulas
            label_text = sample["label"]["text"] + EOS_TOKEN
            label_sequence, cur_missing_formulas = tokenize_sequence("", label_text, sample["label"]["formulas"], self.text_tokenizer)
            num_missing_formulas += cur_missing_formulas
            # Trim the prompt if we go over the max length
            overflow = len(prompt_sequence) + len(label_sequence) - max_seq_len
            if overflow > 0:
                import pdb; pdb.set_trace()
                trimmed_sequences += 1
                prompt_sequence = split_sequence(prompt_sequence, len(prompt_sequence) - overflow)[0]
            # Concatenate into single sequence, and save the length of the prompt for creating generative labels
            sequence = prompt_sequence + label_sequence
            sequence.label = len(prompt_sequence)
            self.data.append(sequence)
        print("Missing", num_missing_formulas, "formulas")
        print("Trimmed", trimmed_sequences, "long sequences")

class ClassifyTaskDataset(Dataset):
    def __init__(self, samples: List[ClassifyTaskSample], max_seq_len: int):
        super().__init__()
        num_missing_formulas = 0
        trimmed_sequences = 0
        for sample in tqdm(samples):
            # Tokenize sequence and save the label
            text = sample["text"] + CLS_TOKEN
            sequence, cur_missing_formulas = tokenize_sequence("", text, sample["formulas"], self.text_tokenizer)
            num_missing_formulas += cur_missing_formulas
            sequence.label = sample["label"]
            # Trim the sequence if we go over the max length
            if len(sequence) > max_seq_len:
                trimmed_sequences += 1
                sequence = split_sequence(sequence, max_seq_len)[0]
            self.data.append(sequence)
        print("Missing", num_missing_formulas, "formulas")
        print("Trimmed", trimmed_sequences, "long sequences")

def trim_batch(batch: CollatedBatch, trim_start: int, trim_end: int) -> CollatedBatch:
    """
    Return a copy of a trimmed collated batch in a given range
    """
    return {
        "sources": batch["sources"],
        "token_ids": batch["token_ids"][:, trim_start : trim_end],
        "token_types": batch["token_types"][:, trim_start : trim_end],
        "pos_vecs": batch["pos_vecs"][:, trim_start : trim_end],
        "pos_levels": batch["pos_levels"][:, trim_start : trim_end],
        "pos_encodings": batch["pos_encodings"][:, trim_start : trim_end],
        "attention_mask": batch["attention_mask"][:, trim_start : trim_end],
        "sequence_lengths": torch.tensor([min(trim_end - trim_start, max(seq_len - trim_start, 0)) for seq_len in batch["sequence_lengths"]]),
        "prompt_lengths": batch["prompt_lengths"],
        "gen_labels": batch["gen_labels"][:, trim_start : trim_end] if batch["gen_labels"] is not None else None,
        "cls_labels": batch["cls_labels"],
    }

class Collator:
    def __init__(self, task: Optional[DownstreamTask] = None):
        self.task = task

    def __call__(self, batch: List[Sequence]) -> CollatedBatch:
        token_id_batches = []
        token_type_batches = []
        pos_vec_batches = []
        pos_level_batches = []
        pos_encoding_batches = []
        attention_mask = []
        sequence_lengths = []
        prompt_lengths = []
        gen_label_batches = []
        cls_labels = []

        for sequence in batch:
            token_ids = torch.LongTensor(sequence.token_ids)
            token_id_batches.append(token_ids)
            token_type_batches.append(torch.LongTensor(sequence.token_types))
            pos_vec_batches.append(torch.LongTensor(sequence.pos_vecs))
            pos_level_batches.append(torch.LongTensor(sequence.pos_levels))
            pos_encoding_batches.append(torch.FloatTensor(sequence.pos_encodings))
            attention_mask.append(torch.ones(len(sequence.token_ids)))
            sequence_lengths.append(len(sequence))
            if self.task:
                if is_cls_task(self.task):
                    cls_labels.append(sequence.label)
                else:
                    prompt_lengths.append(sequence.label)
                    gen_label = torch.clone(token_ids)
                    gen_label[:sequence.label] = PADDING_TOKEN_ID
                    gen_label_batches.append(gen_label)

        return {
            "sources": [sequence.name for sequence in batch],
            # The padding values for token ID and type are critical for correct loss computations in the model
            "token_ids": torch.nn.utils.rnn.pad_sequence(token_id_batches, batch_first=True, padding_value=PADDING_TOKEN_ID).to(device),
            "token_types": torch.nn.utils.rnn.pad_sequence(token_type_batches, batch_first=True, padding_value=TokenType.TEXT.value).to(device),
            "pos_vecs": torch.nn.utils.rnn.pad_sequence(pos_vec_batches, batch_first=True).to(device),
            "pos_levels": torch.nn.utils.rnn.pad_sequence(pos_level_batches, batch_first=True).to(device),
            "pos_encodings": torch.nn.utils.rnn.pad_sequence(pos_encoding_batches, batch_first=True).to(device),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device),
            "sequence_lengths": torch.tensor(sequence_lengths), # Must be on CPU
            "prompt_lengths": torch.tensor(prompt_lengths) if prompt_lengths else None, # Must be on CPU
            "gen_labels": torch.nn.utils.rnn.pad_sequence(gen_label_batches, batch_first=True, padding_value=PADDING_TOKEN_ID).to(device) if gen_label_batches else None,
            "cls_labels": torch.tensor(cls_labels).to(device) if cls_labels else None,
        }
