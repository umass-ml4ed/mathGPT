from typing import List, Callable, Tuple, Optional
import os
import json
from itertools import chain
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from sklearn import metrics
from nlgeval import compute_metrics
import zss

from loading import Dataset, trim_batch, get_data_loader
from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from generate import get_most_likely_predictions, generate
from decode import decode_batch, get_tree, DecodeTreeNode
from pre_process_utils import process_raw_text
from math_tokenize import tokenize_formula
from utils import TrainOptions
from data_types import CollatedBatch, Article, OPT
from constants import PADDING_TOKEN_ID, DownstreamTask

def calculate_ted(labels: List[DecodeTreeNode], preds: List[DecodeTreeNode]):
    """
    Get average tree edit distance across label/pred pairs
    """
    def get_children(tree_node: DecodeTreeNode):
        return tree_node.children
    def get_label(tree_node: DecodeTreeNode):
        return str(tree_node.token_type) + "!" + str(tree_node.token_id)
    def label_dist(label_1: str, label_2: str):
        return 0 if label_1 == label_2 else 1
    return np.mean([
        zss.simple_distance(label, pred, get_children, get_label, label_dist)
        for label, pred in zip(labels, preds)
    ])

def evaluate_lm(model: MathGPTLM, dataset: Dataset, options: TrainOptions):
    """
    Calculate perplexity: e ^ ((1/n) * nll)
    Algorithm from https://huggingface.co/docs/transformers/perplexity
    """
    model.eval()
    # Only 1 sequence can be processed at a time to recover NLL from the cross-entropy loss (because of padding complications)
    data_loader = get_data_loader(dataset, None, 1, False, False, options)
    total_loss = 0.0
    num_batches = 0
    stride = options.stride or options.max_seq_len
    with torch.no_grad():
        nlls: List[torch.Tensor] = []
        total_sequence_length = 0
        for batch in tqdm(data_loader):
            sequence_length = batch["token_ids"].shape[1]
            total_sequence_length += sequence_length

            # Get the sum of the NLL for each token in the sequence, using the stride method
            # Region to left of split point is just for context with no NLL computed, and region to the right contribues to running NLL
            for split_point in range(0, sequence_length, stride):
                start_idx = max(split_point + stride - options.max_seq_len, 0)
                end_idx = min(split_point + stride, sequence_length)
                target_len = end_idx - split_point # This is equal to stride length except maybe shorter for the last iteration
                sub_seq_batch = trim_batch(batch, start_idx, end_idx)
                # Set targets to left of split point to padding so their NLL is not computed
                labels = torch.clone(sub_seq_batch["token_ids"])
                labels[:, :-target_len] = PADDING_TOKEN_ID
                # Run model on batch sub-sequence with altered targets
                loss = model(sub_seq_batch, labels=labels)[0].detach().cpu().numpy()
                total_loss += loss
                num_batches += 1
                # Loss is average NLL over all tokens in the sequence, multiply by number of targets to undo average and retrieve sum
                nlls.append(loss * target_len)

    if options.ddp:
        all_results = [None] * dist.get_world_size()
        dist.all_gather_object(all_results, {
            "total_loss": total_loss,
            "num_batches": num_batches,
            "total_sequence_length": total_sequence_length,
            "nlls": nlls
        })
        total_loss = sum([result["total_loss"] for result in all_results])
        num_batches = sum([result["num_batches"] for result in all_results])
        total_sequence_length = sum([result["total_sequence_length"] for result in all_results])
        nlls = list(chain(*[result["nlls"] for result in all_results]))

    perplexity = np.exp(np.sum(nlls) / total_sequence_length)
    # TODO: see why loss is different here vs. evaluate_lm_accuracy
    return total_loss / num_batches, f"Perplexity: {perplexity:.3f}"

def process_model_output(model: MathGPTBase, dataset: Dataset, task: Optional[DownstreamTask], options: TrainOptions,
                         output_accumulator: Callable[[Tuple, CollatedBatch], None]):
    data_loader = get_data_loader(dataset, task, options.batch_size, False, False, options)
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            model_output = model(batch)
            total_loss += model_output[0].detach().cpu().numpy()
            num_batches += 1
            output_accumulator(model_output, batch)
    return total_loss / num_batches

def evaluate_lm_accuracy(model: MathGPTLM, dataset: Dataset, task: Optional[DownstreamTask], options: TrainOptions):
    """
    Calculate per-token prediction accuracy
    """
    model.eval()
    all_predictions = []
    all_labels = []
    def accumulate_predictions(model_output, batch: CollatedBatch):
        type_preds, token_preds = get_most_likely_predictions(model_output[1])
        # For predictions and targets, stack types and tokens in last dimension
        type_preds = type_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
        token_preds = token_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
        predictions = np.stack([type_preds, token_preds], axis=-1)
        type_targets = batch["token_types"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
        labels = batch["gen_labels"] if batch["gen_labels"] is not None else batch["token_ids"]
        token_targets = labels[:, 1:].contiguous().view(-1).detach().cpu().numpy()
        targets = np.stack([type_targets, token_targets], axis=-1)
        mask = token_targets != PADDING_TOKEN_ID
        all_predictions.append(predictions[mask])
        all_labels.append(targets[mask])

    loss = process_model_output(model, dataset, task, options, accumulate_predictions)
    if options.ddp:
        all_results = [None] * dist.get_world_size()
        dist.all_gather_object(all_results, {
            "loss": loss,
            "all_predictions": all_predictions,
            "all_labels": all_labels
        })
        loss = np.mean([result["loss"] for result in all_results])
        all_predictions = list(chain(*[result["all_predictions"] for result in all_results]))
        all_labels = list(chain(*[result["all_labels"] for result in all_results]))

    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    # Get indices where both type and token match
    match = all_preds_np == all_labels_np
    match = match[:, 0] & match[:, 1]
    accuracy = sum(match) / len(match)
    return loss, f"Accuracy: {accuracy:.3f}"

def evaluate_gen_task(model_name: str, model: MathGPTLM, dataset: Dataset, task: DownstreamTask, options: TrainOptions):
    model.eval()
    compute_ted = options.eval_formulas and not options.baseline
    # Only process one sequence at a time since prompts may have different lengths
    data_loader = get_data_loader(dataset, task, 1, False, False, options)
    all_labels: List[str] = []
    all_predictions: List[str] = []
    all_label_trees: List[DecodeTreeNode] = []
    all_pred_trees: List[DecodeTreeNode] = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            split_point = batch["prompt_lengths"][0]
            gen_batch = generate(model, trim_batch(batch, 0, split_point), options)
            label = trim_batch(batch, split_point, options.max_seq_len)
            pred = trim_batch(gen_batch, split_point, options.max_seq_len)
            all_labels.append(decode_batch(label)[0].replace("\n", " "))
            all_predictions.append(decode_batch(pred)[0].replace("\n", " "))
            if compute_ted:
                all_label_trees.append(get_tree(label["token_ids"][0], label["token_types"][0]))
                all_pred_trees.append(get_tree(pred["token_ids"][0], pred["token_types"][0]))

    if options.ddp:
        all_results = [None] * dist.get_world_size()
        dist.all_gather_object(all_results, {
            "all_predictions": all_predictions,
            "all_labels": all_labels
        })
        all_predictions = list(chain(*[result["all_predictions"] for result in all_results]))
        all_labels = list(chain(*[result["all_labels"] for result in all_results]))

    num_exact_match = sum(pred == label for pred, label in zip(all_predictions, all_labels))
    accuracy = num_exact_match / len(all_labels)
    pred_filename = f"preds_{model_name}.txt"
    label_filename = f"labels_{model_name}.txt"
    with open(pred_filename, "w", encoding="utf-8") as pred_file:
        pred_file.write("\n".join(all_predictions))
    with open(label_filename, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(all_labels))
    metrics = compute_metrics(hypothesis=pred_filename, references=[label_filename], no_skipthoughts=True, no_glove=True)
    avg_ted = calculate_ted(all_label_trees, all_pred_trees) if compute_ted else None
    return 0, f"Exact Match Accuracy: {accuracy:.3f}, BLEU-4: {metrics['Bleu_4']:.3f}, " +\
               f"ROUGE-L: {metrics['ROUGE_L']:.3f}, METEOR: {metrics['METEOR']:.3f}" +\
               (f", TED: {avg_ted:.3f}" if compute_ted else "")

def get_problem_solving_final_answer(full_solution: str):
    if "Final Answer:" in full_solution:
        # Get final answer portion, remove surrounding whitespace, remove commas in numbers
        return full_solution.split("Final Answer:")[1].strip().replace(" , ", "")
    return ""

def evaluate_problem_solving_task(model: MathGPTLM, dataset: Dataset, task: DownstreamTask, options: TrainOptions):
    model.eval()
    # Only process one sequence at a time since prompts may have different lengths
    data_loader = get_data_loader(dataset, task, 1, False, False, options)
    all_labels: List[str] = []
    all_predictions: List[str] = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            split_point = batch["prompt_lengths"][0]
            gen_batch = generate(model, trim_batch(batch, 0, split_point), options)
            label = trim_batch(batch, split_point, options.max_seq_len)
            pred = trim_batch(gen_batch, split_point, options.max_seq_len)
            full_label = decode_batch(label)[0].replace("\n", " ")
            full_pred = decode_batch(pred)[0].replace("\n", " ")
            all_labels.append(get_problem_solving_final_answer(full_label))
            all_predictions.append(get_problem_solving_final_answer(full_pred))
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    return 0, f"Accuracy: {accuracy:.3f}"

def evaluate_cls_task(model: MathGPTClassifier, dataset: Dataset, task: DownstreamTask, options: TrainOptions):
    model.eval()
    all_predictions = []
    all_labels = []
    def accumulate_predictions(model_output, batch: CollatedBatch):
        predictions = torch.nn.Softmax(dim=-1)(model_output[1])
        all_predictions.append(predictions.detach().cpu().numpy())
        all_labels.append(batch["cls_labels"].detach().cpu().numpy())

    loss = process_model_output(model, dataset, task, options, accumulate_predictions)
    if options.ddp:
        all_results = [None] * dist.get_world_size()
        dist.all_gather_object(all_results, {
            "loss": loss,
            "all_predictions": all_predictions,
            "all_labels": all_labels
        })
        loss = np.mean([result["loss"] for result in all_results])
        all_predictions = list(chain(*[result["all_predictions"] for result in all_results]))
        all_labels = list(chain(*[result["all_labels"] for result in all_results]))

    possible_labels = list(range(options.num_classes))
    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    # This is the equivalent of averaging the AUC on each label individually
    auc = metrics.roc_auc_score(all_labels_np, all_preds_np, labels=possible_labels, multi_class="ovr", average="macro")
    all_preds_np = np.argmax(all_preds_np, axis=-1)
    rmse = np.sqrt(metrics.mean_squared_error(all_labels_np, all_preds_np))
    accuracy = metrics.accuracy_score(all_labels_np, all_preds_np)
    kappa = metrics.cohen_kappa_score(all_labels_np, all_preds_np, labels=possible_labels)
    _, _, f1, _ = metrics.precision_recall_fscore_support(all_labels_np, all_preds_np)
    return loss, f"Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, RMSE: {rmse:.3f}, F1: {f1.mean():.3f}"

def evaluate_ted(model_name: str, options_dict: dict):
    batch_size = 40
    options = TrainOptions(options_dict)

    # Load saved labels and predictions
    json_filename = f"results_{model_name}.json"
    if not os.path.exists(json_filename):
        label_filename = f"labels_{model_name}.txt"
        pred_filename = f"preds_{model_name}.txt"
        with open(label_filename, encoding="utf-8") as label_file:
            labels = ["<m> " + label.strip() for label in label_file]
        with open(pred_filename, encoding="utf-8") as pred_file:
            preds = ["<m> " + pred.strip() + ("" if pred.strip().endswith("</m>") else " </m>") for pred in pred_file]

        # Convert sample strings to OPTs via pre-processing pipeline
        err_data = {}
        failed_conversions = []
        processed_labels: List[Article] = []
        processed_preds: List[Article] = []
        for pred_idx, pred in tqdm(enumerate(preds), total=len(preds)):
            try:
                processed_preds.append(process_raw_text([pred], err_data)[0])
            except Exception as exc:
                print(exc)
                failed_conversions.append(pred_idx)
        for batch_start_idx in tqdm(list(range(0, len(labels), batch_size))):
            processed_labels += process_raw_text(labels[batch_start_idx : batch_start_idx + batch_size], err_data)
        print(err_data)
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump({
                "labels": processed_labels,
                "preds": processed_preds,
                "failed": failed_conversions,
            }, json_file, indent=2, ensure_ascii=False)

    # Read file even if immediately written for key string type consistency
    with open(json_filename, encoding="utf-8") as json_file:
        results = json.load(json_file)
        processed_labels = results["labels"]
        processed_preds = results["preds"]
        failed_conversions = results["failed"]

    # Perform post-processing via tokenizer, and then convert back to OPTs and calculate TED
    label_trees: List[DecodeTreeNode] = []
    pred_trees: List[DecodeTreeNode] = []
    missing_formula = []
    for sample_idx, (label, pred) in enumerate(zip(processed_labels, processed_preds)):
        if not label["formulas"] or not pred["formulas"]:
            missing_formula.append(sample_idx)
            continue
        if len(label["formulas"]) > 1 or len(pred["formulas"]) > 1:
            print("More than 1 formula in sample:", sample_idx)
        label_seq = tokenize_formula(label["formulas"]["0"]["opt"], options)
        pred_seq = tokenize_formula(pred["formulas"]["0"]["opt"], options)
        label_trees.append(get_tree(label_seq.token_ids, label_seq.token_types))
        pred_trees.append(get_tree(pred_seq.token_ids, pred_seq.token_types))
    print(f"TED: {calculate_ted(label_trees, pred_trees):.3f}, Failed: {failed_conversions}, Missing formula: {missing_formula}")
