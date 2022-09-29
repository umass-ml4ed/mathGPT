from typing import List, Optional, Dict, Tuple
import json
import random
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, BatchSampler, distributed
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from math_tokenize import tokenize_formula, EMPTY_POS_VECTOR, get_empty_pos_encoding, encode_pos, ExceedMaxDepth
from decode import decode_formula
from data_types import Article, GenTaskSample, AnswerScoringSample, FeedbackTaskSample, ProblemSolvingTaskSample, Formula, Sequence, CollatedBatch
from constants import (
    TokenType, DownstreamTask, TPE, PretrainDataset, PADDING_TOKEN_ID, EOS_TOKEN, SEP_TOKEN, FORMULA_IDENTIFIER, START_FORM_TEXT_TOKS, END_FORM_TEXT_TOK,
    WIKI_DATA, KHAN_DATA, OFEQ_DATA, EXEQ_DATA, AS_ANSWERS, AS_PROBLEMS, FEEDBACK_PROBLEMS, FEEDBACK_SAMPLES, GSM8K_DATA, MATH_DATA, MWP_DATA
)
from utils import device, is_cls_task, text_tokenizer, TrainOptions

def get_article_names(options: TrainOptions):
    if options.dataset == PretrainDataset.WIKI.value:
        data_dir = options.data_dir or WIKI_DATA
        return [os.path.join(data_dir, article_filename) for article_filename in os.listdir(data_dir)]
    if options.dataset == PretrainDataset.KHAN.value:
        data_dir = options.data_dir or KHAN_DATA
        return [
            os.path.join(data_dir, subdir, article_filename)
            for subdir in os.listdir(data_dir)
            for article_filename in os.listdir(os.path.join(data_dir, subdir))
        ]
    return []

def get_probes() -> List[Article]:
    with open("data/probes.json", encoding="utf-8") as probes_file:
        return json.load(probes_file)

def get_headline_data(split: str, options: TrainOptions, fold: int = 0) -> List[GenTaskSample]:
    # Load pre-processed dataset when using the MathGPT model, or using the post_proc option for the baseline model
    pre_processed = not options.baseline or options.post_proc
    if pre_processed:
        with open(os.path.join(OFEQ_DATA, f"{split}.json"), encoding="utf-8") as headlines_file:
            data = json.load(headlines_file)
    else:
        with open(f"../MathSum/OFEQ-10k/post.{split}", encoding="utf-8") as post_file:
            with open(f"../MathSum/OFEQ-10k/title.{split}", encoding="utf-8") as title_file:
                data = [
                    {"prompt": {"text": post, "formulas": {}}, "label": {"text": title, "formulas": {}}}
                    for post, title in zip(post_file, title_file)
                ]
    rng_seed = (fold + 1) * 1000 % 421 # Modulo with prime that gives sufficiently diverse seeds
    random.Random(rng_seed).shuffle(data)
    return data

def get_answer_scoring_data(fold: int = 0) -> Tuple[Dict[str, Article], List[AnswerScoringSample], List[AnswerScoringSample], List[AnswerScoringSample]]:
    with open(AS_PROBLEMS, encoding="utf-8") as problem_file:
        problems: Dict[str, Article] = json.load(problem_file)
    with open(AS_ANSWERS, encoding="utf-8") as answer_file:
        answers: List[AnswerScoringSample] = json.load(answer_file)
    random.Random(221).shuffle(answers)
    # Stratify on problem id so that samples in the training set can be used during test time for meta learning
    answers_np = np.array(answers)
    stratify_labels = np.array([answer["problem_id"] for answer in answers])
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    train_data_idx, test_data_idx = next(split for idx, split in enumerate(skf.split(answers_np, stratify_labels)) if idx == fold)
    train_len = int(.9 * len(train_data_idx))
    return (
        problems,
        answers_np[train_data_idx][:train_len],
        answers_np[train_data_idx][train_len:],
        answers_np[test_data_idx]
    )

def get_feedback_data(fold: int = 0):
    with open(FEEDBACK_PROBLEMS, encoding="utf-8") as problem_file:
        problems: Dict[str, Article] = json.load(problem_file)
    with open(FEEDBACK_SAMPLES, encoding="utf-8") as sample_file:
        samples: List[FeedbackTaskSample] = json.load(sample_file)
    random.Random(221).shuffle(samples)
    # Ensure that no problem appears across train/val/test sets - group answers by problem code before splitting and then expand after splitting
    code_to_samples: Dict[str, List[FeedbackTaskSample]] = {}
    for sample in samples:
        code_to_samples.setdefault(sample["problem_code"], []).append(sample)
    all_codes = list(code_to_samples.keys())
    samples_np = np.array(list(code_to_samples.values()), dtype=object)
    kf = KFold(n_splits=5, shuffle=False)
    train_data_idx, test_data_idx = next(split for idx, split in enumerate(kf.split(all_codes)) if idx == fold)
    train_len = int(.9 * len(train_data_idx))
    def expand(sample_groups: List[List[FeedbackTaskSample]]):
        return [sample for samples in sample_groups for sample in samples]
    return (
        problems,
        expand(samples_np[train_data_idx][:train_len]),
        expand(samples_np[train_data_idx][train_len:]),
        expand(samples_np[test_data_idx])
    )

def get_problem_solving_data(split: str, task: DownstreamTask, ratio: float = 1):
    src_dir = GSM8K_DATA if task == DownstreamTask.GSM8K else MATH_DATA
    with open(os.path.join(src_dir, f"{split}.json"), encoding="utf-8") as data_file:
        data: List[ProblemSolvingTaskSample] = json.load(data_file)
        random.Random(221).shuffle(data)
        return data[:int(len(data) * ratio)], data[int(len(data) * ratio):]

def get_mwp_data(fold: int = 0):
    with open(MWP_DATA, encoding="utf-8") as data_file:
        samples: List[GenTaskSample] = json.load(data_file)
    random.Random(221).shuffle(samples)
    samples_np = np.array(samples)
    kf = KFold(n_splits=5, shuffle=False)
    train_data_idx, test_data_idx = next(split for idx, split in enumerate(kf.split(samples_np)) if idx == fold)
    train_len = int(.9 * len(train_data_idx))
    return (
        samples_np[train_data_idx][:train_len],
        samples_np[train_data_idx][train_len:],
        samples_np[test_data_idx]
    )

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

def tokenize_sequence(name: str, text: str, formulas: Dict[str, Formula], options: TrainOptions):
    sequence = Sequence(name)
    text_chunks = text.split(FORMULA_IDENTIFIER)
    num_missing_formulas = 0
    for text_chunk_idx, text_chunk in enumerate(text_chunks):
        # Ensure that formulas are never perfectly adjacent to simplify decoding constraints
        if text_chunk_idx > 1:
            text_chunk = text_chunk or " "
        # Tokenize the text chunk and add it to the sequence
        text_token_ids: List[int] = text_tokenizer()(text_chunk)["input_ids"]
        sequence.token_ids += text_token_ids
        sequence.token_types += [TokenType.TEXT] * len(text_token_ids)
        sequence.pos_vecs += [EMPTY_POS_VECTOR] * len(text_token_ids)
        sequence.pos_levels += [0] * len(text_token_ids)
        if options.shared_emb:
            sequence.gpt_tokens += [[]] * len(text_token_ids)

        # Sequence will end with a text chunk (even if it's an empty string)
        if text_chunk_idx == len(text_chunks) - 1:
            continue

        # Skip formula if it wasn't captured
        if str(text_chunk_idx) not in formulas:
            num_missing_formulas += 1
            continue

        # Add current formula to the sequence
        formula = formulas[str(text_chunk_idx)]
        if options.baseline:
            # Decode formula back into text, with start and stop latex tokens, and add to the sequence
            if options.post_proc:
                try:
                    formula_sequence = tokenize_formula(formula["opt"], options)
                except ExceedMaxDepth:
                    num_missing_formulas += 1
                    continue
                formula_text = decode_formula(formula_sequence.token_ids, formula_sequence.token_types)
            else:
                formula_text = formula["tex"]
            formula_text = " <m> " + formula_text + " </m> "
            formula_token_ids = text_tokenizer()(formula_text)["input_ids"]
            sequence.token_ids += formula_token_ids
            sequence.token_types += [TokenType.TEXT] * len(formula_token_ids)
            sequence.pos_vecs += [EMPTY_POS_VECTOR] * len(formula_token_ids)
            sequence.pos_levels += [0] * len(formula_token_ids)
            if options.shared_emb:
                sequence.gpt_tokens += [[]] * len(formula_token_ids)
        else:
            # Add formula start token
            sequence.token_ids.append(0)
            sequence.token_types.append(TokenType.START_FORMULA)
            sequence.pos_vecs.append(EMPTY_POS_VECTOR)
            sequence.pos_levels.append(0)
            if options.shared_emb:
                sequence.gpt_tokens.append([])

            # Add formula
            try:
                formula_sequence = tokenize_formula(formula["opt"], options)
            except ExceedMaxDepth:
                num_missing_formulas += 1
                continue
            sequence.token_ids += formula_sequence.token_ids
            sequence.token_types += formula_sequence.token_types
            sequence.pos_vecs += formula_sequence.pos_vecs
            sequence.pos_levels += formula_sequence.pos_levels
            if options.shared_emb:
                sequence.gpt_tokens += formula_sequence.gpt_tokens

            # Add formula end token
            sequence.token_ids.append(0)
            sequence.token_types.append(TokenType.END_FORMULA)
            sequence.pos_vecs.append(EMPTY_POS_VECTOR)
            sequence.pos_levels.append(0)
            if options.shared_emb:
                sequence.gpt_tokens.append([])

    return sequence, num_missing_formulas

class Dataset(TorchDataset):
    def __init__(self):
        self.data: List[Sequence] = []
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
                if options.dataset == PretrainDataset.WIKI.value:
                    article: Article = json.load(article_file)
                    article_text = article["text"] + EOS_TOKEN
                    sequence, cur_missing_formulas = tokenize_sequence(article_name, article_text, article["formulas"], options)
                    self.num_missing_formulas += cur_missing_formulas
                    if max_seq_len:
                        split_sequences = split_sequence(sequence, max_seq_len)
                        self.data += split_sequences
                    else:
                        self.data.append(sequence)

                elif options.dataset == PretrainDataset.KHAN.value:
                    sample: GenTaskSample = json.load(article_file)
                    problem_text = "Problem: " + sample["prompt"]["text"] + " Solution: "
                    problem_sequence, cur_missing_formulas = tokenize_sequence(article_file, problem_text, sample["prompt"]["formulas"], options)
                    self.num_missing_formulas += cur_missing_formulas
                    answer_text = sample["label"]["text"] + EOS_TOKEN
                    answer_sequence, cur_missing_formulas = tokenize_sequence(article_file, answer_text, sample["label"]["formulas"], options)
                    self.num_missing_formulas += cur_missing_formulas
                    sequence = problem_sequence + answer_sequence
                    if len(sequence) > options.max_seq_len:
                        self.trimmed_sequences += 1
                        continue
                    self.data.append(sequence)

        print("Missing", self.num_missing_formulas, "formulas")
        print("Skipped", self.trimmed_sequences, "long sequences")

class PreTrainDatasetPreloaded(Dataset):
    def __init__(self, articles: List[Article], options: TrainOptions, max_seq_len: Optional[int]):
        super().__init__()
        for article in tqdm(articles):
            article_text = article["text"]
            sequence, cur_missing_formulas = tokenize_sequence("", article_text, article["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas

            if max_seq_len:
                split_sequences = split_sequence(sequence, max_seq_len)
                self.data += split_sequences
            else:
                self.data.append(sequence)
        print("Missing", self.num_missing_formulas, "formulas")

class GenTaskDataset(Dataset):
    def __init__(self, samples: List[GenTaskSample], task: DownstreamTask, options: TrainOptions):
        super().__init__()
        min_label_len = 2**31
        for sample in tqdm(samples):
            # Tokenize the prompt and label sequences
            prompt_text = "Question: " + sample["prompt"]["text"]
            prompt_sequence, cur_missing_formulas = tokenize_sequence("", prompt_text, sample["prompt"]["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas
            intermediate_text = SEP_TOKEN + (" Equation: " if task == DownstreamTask.MWP else " Summary: ")
            intermediate_sequence, _ = tokenize_sequence("", intermediate_text, {}, options)
            label_text = sample["label"]["text"] + EOS_TOKEN
            label_sequence, cur_missing_formulas = tokenize_sequence("", label_text, sample["label"]["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas
            min_label_len = min(min_label_len, len(label_sequence))

            # When evaluating particular regions - find each formula/text in the label and set them to be the new labels
            if options.eval_formulas or options.eval_text:
                int_seqs = []
                label_seqs = []
                while True:
                    if options.eval_formulas:
                        if options.baseline:
                            start_idx = next((
                                idx + 3 for idx in range(len(label_sequence) - 2)
                                if label_sequence.token_ids[idx : idx + 3] in START_FORM_TEXT_TOKS
                            ), None)
                        else:
                            start_idx = next((
                                idx + 1 for idx, token_type in enumerate(label_sequence.token_types)
                                if token_type == TokenType.START_FORMULA
                            ), None)
                    else:
                        if options.baseline:
                            start_idx = next((
                                idx + 3 for idx in range(len(label_sequence) - 2)
                                if label_sequence.token_ids[idx : idx + 3] == END_FORM_TEXT_TOK
                            ), None)
                        else:
                            start_idx = next((
                                idx + 1 for idx, token_type in enumerate(label_sequence.token_types)
                                if token_type == TokenType.END_FORMULA
                            ), None)
                    if start_idx is None:
                        break
                    headline_start, headline_end = label_sequence.split_at(start_idx)
                    intermediate_sequence += headline_start
                    if options.eval_formulas:
                        if options.baseline:
                            end_idx = next((
                                idx + 3 for idx in range(len(headline_end) - 2)
                                if headline_end.token_ids[idx : idx + 3] == END_FORM_TEXT_TOK
                            ), None)
                        else:
                            end_idx = next((
                                idx + 1 for idx, token_type in enumerate(headline_end.token_types)
                                if token_type == TokenType.END_FORMULA
                            ), None)
                    else:
                        if options.baseline:
                            end_idx = next((
                                idx + 3 for idx in range(len(headline_end) - 2)
                                if headline_end.token_ids[idx : idx + 3] in START_FORM_TEXT_TOKS
                            ), len(headline_end))
                        else:
                            end_idx = next((
                                idx + 1 for idx, token_type in enumerate(headline_end.token_types)
                                if token_type == TokenType.START_FORMULA
                            ), len(headline_end))
                    cur_label_sequence, label_sequence = headline_end.split_at(end_idx)
                    if options.eval_text and len(cur_label_sequence) <= 2:
                        break
                    int_seqs.append(intermediate_sequence)
                    label_seqs.append(cur_label_sequence)
                    intermediate_sequence += cur_label_sequence
            else:
                int_seqs = [intermediate_sequence]
                label_seqs = [label_sequence]

            # Construct the full sequence(s) and add to the dataset
            for intermediate_sequence, label_sequence in zip(int_seqs, label_seqs):
                # Trim the prompt if we go over the max length
                overflow = len(prompt_sequence) + len(intermediate_sequence) + len(label_sequence) - options.max_seq_len
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
        print("Min label length:", min_label_len)

class AnswerScoringDataset(Dataset):
    def __init__(self, samples: List[AnswerScoringSample], problems: Dict[str, Article], options: TrainOptions,
                 examples: Optional[List[Sequence]] = None):
        super().__init__()
        self.options = options
        self.problems: Dict[int, Sequence] = {}
        self.example_bank: Dict[int, Dict[int, List[Sequence]]] = {}

        # Static sequences that get added to each sample
        self.qs_prefix_seq = tokenize_sequence("", "Question: ", {}, self.options)[0]
        self.scores_seq = tokenize_sequence("", " [SEP] Possible scores: Wrong Poor Fair Good Excellent", {}, self.options)[0]
        self.example_prefix_seq = tokenize_sequence("", " [SEP] Example: ", {}, self.options)[0]
        self.answer_prefix_seq = tokenize_sequence("", " [SEP] Score this answer: ", {}, self.options)[0]
        self.cls_seq = tokenize_sequence("", " [CLS]", {}, self.options)[0]
        self.grade_to_score_seq = {
            0: tokenize_sequence("", " Score: Wrong", {}, self.options)[0],
            1: tokenize_sequence("", " Score: Poor", {}, self.options)[0],
            2: tokenize_sequence("", " Score: Fair", {}, self.options)[0],
            3: tokenize_sequence("", " Score: Good", {}, self.options)[0],
            4: tokenize_sequence("", " Score: Excellent", {}, self.options)[0],
        }

        # To avoid exceeding max len, cap answer and problem seq lens to half the remaining space
        max_component_len = (options.max_seq_len - (
            len(self.qs_prefix_seq) + len(self.scores_seq) + len(self.answer_prefix_seq) + len(self.cls_seq)
        )) // 2

        # Process answers
        for sample in tqdm(samples):
            answer_sequence, cur_missing_formulas = tokenize_sequence("", sample["answer"]["text"], sample["answer"]["formulas"], options)
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
            problem_sequence, cur_missing_formulas = tokenize_sequence("", problem["text"], problem["formulas"], options)
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

class FeedbackDataset(Dataset):
    def __init__(self, samples: List[FeedbackTaskSample], problems: Dict[str, Article], options: TrainOptions):
        super().__init__()
        shortest_feedback = 2**31
        longest_feedback = 0

        # Process problems
        pid_to_seq: Dict[str, Sequence] = {}
        for pid, problem in tqdm(problems.items()):
            problem_text = "Question: " + problem["text"]
            problem_sequence, cur_missing_formulas = tokenize_sequence("", problem_text, problem["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas
            pid_to_seq[pid] = problem_sequence

        # Process samples
        for sample in tqdm(samples):
            problem_sequence = pid_to_seq[sample["problem_id"]]
            answer_text = " [SEP] Answer: " + sample["answer"]["text"] + " [SEP] Feedback: "
            answer_sequence, cur_missing_formulas = tokenize_sequence("", answer_text, sample["answer"]["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas
            feedback_text = sample["feedback"]["text"] + EOS_TOKEN
            feedback_sequence, cur_missing_formulas = tokenize_sequence("", feedback_text, sample["feedback"]["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas
            shortest_feedback = min(shortest_feedback, len(feedback_sequence))
            longest_feedback = max(longest_feedback, len(feedback_sequence))

            # Trim the problem if we go over the max length
            overflow = len(problem_sequence) + len(answer_sequence) + len(feedback_sequence) - options.max_seq_len
            if overflow > 0:
                self.trimmed_sequences += 1
                problem_sequence = split_sequence(problem_sequence, len(problem_sequence) - overflow)[0]

            # Concatenate into single sequence, and save the length of the prompt for creating generative labels
            sequence = problem_sequence + answer_sequence + feedback_sequence
            sequence.meta = {
                "prompt_length": len(problem_sequence) + len(answer_sequence)
            }
            self.data.append(sequence)

        print("Missing", self.num_missing_formulas, "formulas")
        print("Trimmed", self.trimmed_sequences, "long sequences")
        print("Shortest feedback:", shortest_feedback, "Longest feedback:", longest_feedback)

class ProblemSolvingDataset(Dataset):
    def __init__(self, samples: List[ProblemSolvingTaskSample], options: TrainOptions):
        super().__init__()
        shortest_qs = 2**31
        longest_qs = 0
        shortest_sol = 2**31
        longest_sol = 0
        num_empty_prefixes = 0
        no_final_sol = 0

        for sample in tqdm(samples):
            problem_text = "Question: " + sample["problem"]["text"] + " [SEP] Solution: "
            problem_sequence, cur_missing_formulas = tokenize_sequence("", problem_text, sample["problem"]["formulas"], options)
            self.num_missing_formulas += cur_missing_formulas
            if options.eval_final:
                # The final step is the sentence that contains the \\boxed macro (and up to the end of the sequence)
                steps_text = sample["steps"]["text"]
                try:
                    final_formula_idx = next(
                        int(formula_idx) for formula_idx, formula in sample["steps"]["formulas"].items()
                        if "\\boxed" in formula["tex"] or "\\fbox" in formula["tex"]
                    )
                except StopIteration: # In a few cases LaTeXML won't capture the \\boxed macro in the tex representation
                    no_final_sol += 1
                    continue
                final_formula_start = steps_text.find(FORMULA_IDENTIFIER)
                for _ in range(final_formula_idx):
                    final_formula_start = steps_text.find(FORMULA_IDENTIFIER, final_formula_start + 1)
                final_step_start = steps_text.rfind(".", 0, final_formula_start) + 1
                if final_step_start == 0:
                    final_step_start = steps_text.rfind("\n", 0, final_formula_start) + 1
                if final_step_start == 0:
                    num_empty_prefixes += 1

                # Construct sequence of steps up to the final one
                pre_final_steps_text = steps_text[:final_step_start]
                steps_sequence, cur_missing_formulas = tokenize_sequence("", pre_final_steps_text, sample["steps"]["formulas"], options)
                self.num_missing_formulas += cur_missing_formulas

                # Construct sequence with just the final step, rebalance formula indices
                final_step_text = steps_text[final_step_start:] + EOS_TOKEN
                num_pre_final_formulas = pre_final_steps_text.count(FORMULA_IDENTIFIER)
                final_step_formulas = {
                    str(int(form_idx) - num_pre_final_formulas): formula
                    for form_idx, formula in sample["steps"]["formulas"].items()
                    if int(form_idx) >= num_pre_final_formulas
                }
                answer_sequence, cur_missing_formulas = tokenize_sequence("", final_step_text, final_step_formulas, options)
                self.num_missing_formulas += cur_missing_formulas
            else:
                steps_text = sample["steps"]["text"] + " [SEP] Final Answer: "
                steps_sequence, cur_missing_formulas = tokenize_sequence("", steps_text, sample["steps"]["formulas"], options)
                self.num_missing_formulas += cur_missing_formulas

                answer_text = sample["answer"]["text"] + EOS_TOKEN
                answer_sequence, cur_missing_formulas = tokenize_sequence("", answer_text, sample["answer"]["formulas"], options)
                self.num_missing_formulas += cur_missing_formulas

            shortest_qs = min(shortest_qs, len(problem_sequence))
            longest_qs = max(longest_qs, len(problem_sequence))
            shortest_sol = min(shortest_sol, len(steps_sequence))
            longest_sol = max(longest_sol, len(steps_sequence))

            # Concatenate into single sequence, and save the length of the prompt for creating generative labels
            sequence = problem_sequence + steps_sequence + answer_sequence
            if len(sequence) > options.max_seq_len: # TODO: actually trim instead of skip
                self.trimmed_sequences += 1
                continue
            sequence.meta = {
                "prompt_length": len(problem_sequence) + len(steps_sequence) if options.eval_final else len(problem_sequence),
                "level": sample.get("level")
            }
            self.data.append(sequence)

        print("Missing", self.num_missing_formulas, "formulas")
        print("Skipped", self.trimmed_sequences, "overflowed sequences")
        print("Questions: Shortest:", shortest_qs, "Longest:", longest_qs)
        print("Solutions: Shortest:", shortest_sol, "Longest:", longest_sol)
        if options.eval_final:
            print("No Pre-Final Steps:", num_empty_prefixes)
            print("Skipped because no final solution:", no_final_sol)

def get_data_loader(dataset: Dataset, task: Optional[DownstreamTask], batch_size: int, shuffle: bool, drop_last: bool, options: TrainOptions):
    return DataLoader(
        dataset,
        collate_fn=Collator(task, options),
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
        "pos_encodings": batch["pos_encodings"][:, trim_start : trim_end] if batch["pos_encodings"] is not None else None,
        "gpt_tokens": batch["gpt_tokens"][:, trim_start : trim_end] if batch["gpt_tokens"] is not None else None,
        "use_shared_emb": batch["use_shared_emb"][:, trim_start : trim_end] if batch["use_shared_emb"] is not None else None,
        "attention_mask": batch["attention_mask"][:, trim_start : trim_end],
        "sequence_lengths": torch.tensor([min(trim_end - trim_start, max(seq_len - trim_start, 0)) for seq_len in batch["sequence_lengths"]]),
        "prompt_lengths": batch["prompt_lengths"],
        "gen_labels": batch["gen_labels"][:, trim_start : trim_end] if batch["gen_labels"] is not None else None,
        "cls_labels": batch["cls_labels"],
    }

class Collator:
    def __init__(self, task: Optional[DownstreamTask], options: TrainOptions):
        self.options = options
        self.task = task

    def __call__(self, batch: List[Sequence]) -> CollatedBatch:
        token_id_batches = []
        token_type_batches = []
        pos_vec_batches = []
        pos_level_batches = []
        pos_encoding_batches = []
        gpt_token_batches = []
        use_shared_emb_batches = []
        attention_mask = []
        sequence_lengths = []
        prompt_lengths = []
        gen_label_batches = []
        cls_labels = []

        if self.options.shared_emb:
            max_gpt_token_len = max(len(gpt_token_vec) for sequence in batch for gpt_token_vec in sequence.gpt_tokens)

        for sequence in batch:
            token_ids = torch.LongTensor(sequence.token_ids)
            token_id_batches.append(token_ids)
            token_type_batches.append(torch.LongTensor(sequence.token_types))
            pos_vec_batches.append(torch.LongTensor(sequence.pos_vecs))
            pos_level_batches.append(torch.LongTensor(sequence.pos_levels))
            if self.options.tpe != TPE.NONE.value:
                pos_encodings = [
                    encode_pos(pos_vec, pos_level, self.options.tpe)
                        if token_type not in (TokenType.TEXT, TokenType.START_FORMULA, TokenType.END_FORMULA)
                        else get_empty_pos_encoding(self.options.tpe)
                    for token_type, pos_vec, pos_level in zip(sequence.token_types, sequence.pos_vecs, sequence.pos_levels)
                ]
                pos_encoding_batches.append(torch.FloatTensor(pos_encodings))
            if self.options.shared_emb:
                gpt_tokens = torch.tensor(np.array([
                    np.pad(gpt_token_vec, (0, max_gpt_token_len - len(gpt_token_vec)), constant_values=PADDING_TOKEN_ID)
                    for gpt_token_vec in sequence.gpt_tokens
                ]), dtype=torch.long)
                gpt_token_batches.append(gpt_tokens)
                use_shared_emb_batches.append(torch.tensor([len(gpt_token_vec) for gpt_token_vec in sequence.gpt_tokens], dtype=torch.bool))
            attention_mask.append(torch.ones(len(sequence)))
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
            "pos_encodings": torch.nn.utils.rnn.pad_sequence(pos_encoding_batches, batch_first=True).to(device) if pos_encoding_batches else None,
            "gpt_tokens": torch.nn.utils.rnn.pad_sequence(gpt_token_batches, batch_first=True, padding_value=PADDING_TOKEN_ID).to(device) if gpt_token_batches else None,
            "use_shared_emb": torch.nn.utils.rnn.pad_sequence(use_shared_emb_batches, batch_first=True).to(device) if use_shared_emb_batches else None,
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device),
            "sequence_lengths": torch.tensor(sequence_lengths), # Must be on CPU
            "prompt_lengths": torch.tensor(prompt_lengths) if prompt_lengths else None, # Must be on CPU
            "gen_labels": torch.nn.utils.rnn.pad_sequence(gen_label_batches, batch_first=True, padding_value=PADDING_TOKEN_ID).to(device) if gen_label_batches else None,
            "cls_labels": torch.tensor(cls_labels).to(device) if cls_labels else None,
        }
