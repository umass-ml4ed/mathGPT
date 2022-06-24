import json
from typing import List, Optional, Dict
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, BatchSampler, distributed
from transformers import GPT2TokenizerFast

from math_tokenize import tokenize_formula, EMPTY_POS_VECTOR, get_empty_pos_encoding, encode_pos
from decode import decode_formula
from data_types import Article, GenTaskSample, AnswerScoringSample, ClassifyTaskSample, Formula, Sequence, CollatedBatch
from constants import TokenType, DownstreamTask, PADDING_TOKEN_ID, EOS_TOKEN, SEP_TOKEN, CLS_TOKEN, FORMULA_IDENTIFIER, DOLLAR_TOK
from utils import device, is_cls_task, TrainOptions

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

def tokenize_sequence(name: str, text: str, formulas: Dict[str, Formula], text_tokenizer: GPT2TokenizerFast, options: TrainOptions):
    decode_formulas = options.baseline
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
        # sequence.pos_encodings += [get_empty_pos_encoding(options.tpe)] * len(text_token_ids)

        # Sequence will end with a text chunk (even if it's an empty string)
        if text_chunk_idx == len(text_chunks) - 1:
            continue

        # Skip formula if it wasn't captured
        if str(text_chunk_idx) not in formulas:
            num_missing_formulas += 1
            continue

        # Tokenize the formula
        formula_sequence = tokenize_formula(formulas[str(text_chunk_idx)]["opt"], options.tpe)

        if decode_formulas:
            # Decode formula back into text, with start and stop latex tokens, and add to the sequence
            formula_text = " $ " + decode_formula(formula_sequence.token_ids, formula_sequence.token_types) + " $ "
            formula_token_ids = text_tokenizer(formula_text)["input_ids"]
            sequence.token_ids += formula_token_ids
            sequence.token_types += [TokenType.TEXT] * len(formula_token_ids)
            sequence.pos_vecs += [EMPTY_POS_VECTOR] * len(formula_token_ids)
            sequence.pos_levels += [0] * len(formula_token_ids)
            # sequence.pos_encodings += [get_empty_pos_encoding(options.tpe)] * len(formula_token_ids)
        else:
            # Add formula start token
            sequence.token_ids.append(0)
            sequence.token_types.append(TokenType.START_FORMULA)
            sequence.pos_vecs.append(EMPTY_POS_VECTOR)
            sequence.pos_levels.append(0)
            # sequence.pos_encodings.append(get_empty_pos_encoding(options.tpe))

            # Add formula
            sequence.token_ids += formula_sequence.token_ids
            sequence.token_types += formula_sequence.token_types
            sequence.pos_vecs += formula_sequence.pos_vecs
            sequence.pos_levels += formula_sequence.pos_levels
            # sequence.pos_encodings += formula_sequence.pos_encodings

            # Add formula end token
            sequence.token_ids.append(0)
            sequence.token_types.append(TokenType.END_FORMULA)
            sequence.pos_vecs.append(EMPTY_POS_VECTOR)
            sequence.pos_levels.append(0)
            # sequence.pos_encodings.append(get_empty_pos_encoding(options.tpe))

    return sequence, num_missing_formulas

class Dataset(TorchDataset):
    def __init__(self):
        self.data: List[Sequence] = []
        self.text_tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        self.num_missing_formulas = 0
        self.trimmed_sequences = 0
        print("Processing data...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class PreTrainDataset(Dataset):
    def __init__(self, article_filenames: List[str], options: TrainOptions, max_seq_len: Optional[int]):
        super().__init__()
        for article_name in tqdm(article_filenames):
            with open(article_name, encoding="utf-8") as article_file:
                article: Article = json.load(article_file)

            article_text = article["text"] + EOS_TOKEN
            sequence, cur_missing_formulas = tokenize_sequence(article_name, article_text, article["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas

            if max_seq_len:
                split_sequences = split_sequence(sequence, max_seq_len)
                self.data += split_sequences
            else:
                self.data.append(sequence)
        print("Missing", self.num_missing_formulas, "formulas")

class PreTrainDatasetPreloaded(Dataset):
    def __init__(self, articles: List[Article], options: TrainOptions, max_seq_len: Optional[int]):
        super().__init__()
        for article in tqdm(articles):
            article_text = article["text"]
            sequence, cur_missing_formulas = tokenize_sequence("", article_text, article["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas

            if max_seq_len:
                split_sequences = split_sequence(sequence, max_seq_len)
                self.data += split_sequences
            else:
                self.data.append(sequence)
        print("Missing", self.num_missing_formulas, "formulas")

class GenTaskDataset(Dataset):
    def __init__(self, samples: List[GenTaskSample], options: TrainOptions, max_seq_len: int):
        super().__init__()
        test_first_eq = False
        for sample in tqdm(samples):
            # Tokenize the prompt and label sequences
            prompt_text = "Question: " + sample["prompt"]["text"]
            prompt_sequence, cur_missing_formulas = tokenize_sequence("", prompt_text, sample["prompt"]["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas
            intermediate_text = SEP_TOKEN + " Summary: "
            intermediate_sequence, _ = tokenize_sequence("", intermediate_text, [], self.text_tokenizer, options)
            label_text = sample["label"]["text"] + EOS_TOKEN
            label_sequence, cur_missing_formulas = tokenize_sequence("", label_text, sample["label"]["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas
            # Sanity check - just generate the first equation of the label with the preceding text given
            if test_first_eq:
                if options.baseline and options.post_proc:
                    start_idx = next((idx for idx, token_id in enumerate(label_sequence.token_ids) if token_id == DOLLAR_TOK), None)
                else:
                    start_idx = next((idx for idx, token_type in enumerate(label_sequence.token_types) if token_type == TokenType.START_FORMULA), None)
                if not start_idx:
                    continue
                headline_start, headline_end = label_sequence.split_at(start_idx + 1)
                intermediate_sequence = intermediate_sequence + headline_start
                if options.baseline and options.post_proc:
                    end_idx = next((idx for idx, token_id in enumerate(headline_end.token_ids) if token_id == DOLLAR_TOK), None)
                else:
                    end_idx = next((idx for idx, token_type in enumerate(headline_end.token_types) if token_type == TokenType.END_FORMULA), None)
                label_sequence = headline_end.split_at(end_idx + 1)[0]
            # Trim the prompt if we go over the max length
            overflow = len(prompt_sequence) + len(intermediate_sequence) + len(label_sequence) - max_seq_len
            if overflow > 0:
                self.trimmed_sequences += 1
                prompt_sequence = split_sequence(prompt_sequence, len(prompt_sequence) - overflow)[0]
            # Concatenate into single sequence, and save the length of the prompt for creating generative labels
            sequence = prompt_sequence + intermediate_sequence + label_sequence
            sequence.meta = {
                "prompt_length": len(prompt_sequence) + len(intermediate_sequence)
            }
            self.data.append(sequence)
        print("Missing", self.num_missing_formulas, "formulas")
        print("Trimmed", self.trimmed_sequences, "long sequences")

class AnswerScoringDataset(Dataset):
    def __init__(self, samples: List[AnswerScoringSample], problems: Dict[str, Article], options: TrainOptions,
                 examples: Optional[List[Sequence]] = None):
        super().__init__()
        self.options = options
        self.problems: Dict[int, Sequence] = {}
        self.example_bank: Dict[int, Dict[int, List[Sequence]]] = {}

        # Static sequences that get added to each sample
        self.qs_prefix_seq = tokenize_sequence("", "Question: ", {}, self.text_tokenizer, self.options)[0]
        self.scores_seq = tokenize_sequence("", " [SEP] Possible scores: Wrong Poor Fair Good Excellent", {}, self.text_tokenizer, self.options)[0]
        self.example_prefix_seq = tokenize_sequence("", " [SEP] Example: ", {}, self.text_tokenizer, self.options)[0]
        self.answer_prefix_seq = tokenize_sequence("", " [SEP] Score this answer: ", {}, self.text_tokenizer, self.options)[0]
        self.cls_seq = tokenize_sequence("", " [CLS]", {}, self.text_tokenizer, self.options)[0]
        self.grade_to_score_seq = {
            0: tokenize_sequence("", " Score: Wrong", {}, self.text_tokenizer, self.options)[0],
            1: tokenize_sequence("", " Score: Poor", {}, self.text_tokenizer, self.options)[0],
            2: tokenize_sequence("", " Score: Fair", {}, self.text_tokenizer, self.options)[0],
            3: tokenize_sequence("", " Score: Good", {}, self.text_tokenizer, self.options)[0],
            4: tokenize_sequence("", " Score: Excellent", {}, self.text_tokenizer, self.options)[0],
        }

        # To avoid exceeding max len, cap answer and problem seq lens to half the remaining space
        max_component_len = (options.max_seq_len - (
            len(self.qs_prefix_seq) + len(self.scores_seq) + len(self.answer_prefix_seq) + len(self.cls_seq)
        )) // 2

        # Process answers
        for sample in tqdm(samples):
            answer_sequence, cur_missing_formulas = tokenize_sequence("", sample["answer"]["text"], sample["answer"]["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas
            if len(answer_sequence) > max_component_len:
                self.trimmed_sequences += 1
                answer_sequence = split_sequence(answer_sequence, max_component_len)[0]
            answer_sequence.meta = {
                "problem_id": sample["problem_id"],
                "problem_log_id": sample["problem_log_id"],
                "label": sample["grade"],
            }
            self.data.append(answer_sequence)

        # Process problems
        for problem_id, problem in problems.items():
            problem_sequence, cur_missing_formulas = tokenize_sequence("", problem["text"], problem["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas
            if len(problem_sequence) > max_component_len:
                self.trimmed_sequences += 1
                problem_sequence = split_sequence(problem_sequence, max_component_len)[0]
            self.problems[int(problem_id)] = problem_sequence

        # Group examples by problem and grade (take own samples to be examples if none explicitly given)
        examples = examples or self.data
        for example in examples:
            cur_problem_examples = self.example_bank.setdefault(example.meta["problem_id"], {grade: [] for grade in range(options.num_classes)})
            cur_problem_examples[example.meta["label"]].append(example)

        print("Missing", self.num_missing_formulas, "formulas")
        print("Trimmed", self.trimmed_sequences, "long sequences")

    def __getitem__(self, index: int):
        # Get current sample and associated problem
        sample = self.data[index]
        problem_id = sample.meta["problem_id"]
        problem_sequence = self.problems[problem_id]

        # Gather one example from the current question for each possible grade, and then several others across grades
        initial_examples: List[Sequence] = []
        additional_examples: List[Sequence] = []
        if problem_id in self.example_bank:
            for grade in range(self.options.num_classes):
                examples = [example for example in self.example_bank[problem_id][grade] if example.meta["problem_log_id"] != sample.meta["problem_log_id"]]
                if examples:
                    examples = random.sample(examples, min(len(examples), 5))
                    initial_examples.append(examples[0])
                    additional_examples.extend(examples[1:])
        else:
            print(problem_id, "not in bank")

        # Shuffle the examples and add them until max_seq_len would be exceeded (keeping the group of one per grade first so no grade is left out)
        random.shuffle(initial_examples)
        random.shuffle(additional_examples)
        example_sequences: List[Sequence] = []
        base_len = len(self.qs_prefix_seq) + len(problem_sequence) + len(self.scores_seq) + len(self.answer_prefix_seq) + len(sample) + len(self.cls_seq)
        for example in initial_examples + additional_examples:
            example_seq = self.example_prefix_seq + example + self.grade_to_score_seq[example.meta["label"]]
            if base_len + len(example_seq) > self.options.max_seq_len:
                break
            example_sequences.append(example_seq)
            base_len += len(example_seq)

        # Construct the final sequence (question, possible scores, examples, answer to be scored) and assign the grade label
        sequence = self.qs_prefix_seq + problem_sequence + self.scores_seq
        for example_seq in example_sequences:
            sequence += example_seq
        sequence += self.answer_prefix_seq + sample + self.cls_seq
        sequence.meta = {"label": sample.meta["label"]}
        return sequence

class ClassifyTaskDataset(Dataset):
    def __init__(self, samples: List[ClassifyTaskSample], options: TrainOptions, max_seq_len: int):
        super().__init__()
        for sample in tqdm(samples):
            # Tokenize sequence and save the label
            text = sample["text"] + CLS_TOKEN
            sequence, cur_missing_formulas = tokenize_sequence("", text, sample["formulas"], self.text_tokenizer, options)
            self.num_missing_formulas += cur_missing_formulas
            sequence.meta = {"label": sample["label"]}
            # Trim the sequence if we go over the max length
            # TODO: this will remove the CLS token...
            if len(sequence) > max_seq_len:
                self.trimmed_sequences += 1
                sequence = split_sequence(sequence, max_seq_len)[0]
            self.data.append(sequence)
        print("Missing", self.num_missing_formulas, "formulas")
        print("Trimmed", self.trimmed_sequences, "long sequences")

def get_data_loader(dataset: Dataset, task: Optional[DownstreamTask], batch_size: int, shuffle: bool, drop_last: bool, options: TrainOptions):
    return DataLoader(
        dataset,
        collate_fn=Collator(options.tpe, task),
        batch_size=1 if options.ddp else batch_size,
        shuffle=not options.ddp and shuffle,
        drop_last=not options.ddp and drop_last,
        batch_sampler=BatchSampler(
            distributed.DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=drop_last
            ),
            batch_size=batch_size,
            drop_last=drop_last
        ) if options.ddp else None
    )

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
    def __init__(self, tpe: str, task: Optional[DownstreamTask] = None):
        self.tpe = tpe
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
            pos_encodings = [
                encode_pos(pos_vec, pos_level, self.tpe)
                    if token_type not in (TokenType.TEXT, TokenType.START_FORMULA, TokenType.END_FORMULA)
                    else get_empty_pos_encoding(self.tpe)
                for token_type, pos_vec, pos_level in zip(sequence.token_types, sequence.pos_vecs, sequence.pos_levels)
            ]
            pos_encoding_batches.append(torch.FloatTensor(pos_encodings))
            attention_mask.append(torch.ones(len(sequence.token_ids)))
            sequence_lengths.append(len(sequence))
            if self.task:
                if is_cls_task(self.task):
                    cls_labels.append(sequence.meta["label"])
                else:
                    prompt_lengths.append(sequence.meta["prompt_length"])
                    gen_label = torch.clone(token_ids)
                    gen_label[:sequence.meta["prompt_length"]] = PADDING_TOKEN_ID
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
