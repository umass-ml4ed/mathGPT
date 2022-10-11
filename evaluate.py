from typing import List, Callable, Tuple, Optional, Dict
import os
import json
import math
from itertools import chain
from functools import reduce
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from sklearn import metrics
from nlgeval import compute_metrics
import zss

from loading import Dataset, GenTaskDataset, FeedbackDataset, trim_batch, get_data_loader, get_headline_data, get_mwp_data, get_feedback_data
from vocabulary import Vocabulary, get_matrix_symbol
from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from generate import get_most_likely_predictions, generate
from decode import decode_batch, get_tree, DecodeTreeNode
from pre_process_utils import process_raw_text
from math_tokenize import tokenize_formula
from utils import TrainOptions
from data_types import CollatedBatch, Article
from constants import PADDING_TOKEN_ID, DownstreamTask, TokenType, SpecialOpToken

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

def trees_equal(tree_1: DecodeTreeNode, tree_2: DecodeTreeNode):
    """
    Test if two decoded trees are equal
    """
    if tree_1.token_id != tree_2.token_id or tree_1.token_type != tree_2.token_type:
        return False
    if len(tree_1.children) != len(tree_2.children):
        return False
    for child_1, child_2 in zip(tree_1.children, tree_2.children):
        if not trees_equal(child_1, child_2):
            return False
    return True

def eval_tree_value(tree_node: DecodeTreeNode, options: TrainOptions) -> float:
    """
    Get numeric value that tree evaluates to
    Just for trees structured like MWP to eq task labels
    """
    if options.num_to_tree and tree_node.token_type == TokenType.OP and tree_node.token_id == SpecialOpToken.NUM_SUB_TREE_HEAD:
        symbol = "".join([Vocabulary.get_symbol(child.token_type, child.token_id) for child in tree_node.children])
        return float(symbol)
    symbol = Vocabulary.get_symbol(tree_node.token_type, tree_node.token_id)
    if tree_node.token_type == TokenType.NUM:
        return float(symbol)
    if symbol == "eq":
        assert len(tree_node.children) == 2
        return eval_tree_value(tree_node.children[1], options)
    if symbol == "plus":
        return sum([eval_tree_value(child, options) for child in tree_node.children])
    if symbol == "minus":
        return eval_tree_value(tree_node.children[0], options) - sum([eval_tree_value(child, options) for child in tree_node.children[1:]])
    if symbol == "times":
        return math.prod([eval_tree_value(child, options) for child in tree_node.children])
    if symbol == "divide":
        if len(tree_node.children) == 0:
            return 1
        return reduce(
            lambda prev, cur: prev / cur,
            [eval_tree_value(child, options) for child in tree_node.children[1:]],
            eval_tree_value(tree_node.children[0], options)
        )
    if symbol == "SUP":
        assert len(tree_node.children) == 2
        return eval_tree_value(tree_node.children[0], options) ** eval_tree_value(tree_node.children[1], options)
    if symbol == "percent":
        assert len(tree_node.children) == 1
        return eval_tree_value(tree_node.children[0], options) / 100
    if symbol == get_matrix_symbol("D"):
        assert len(tree_node.children) == 1
        return eval_tree_value(tree_node.children[0], options)
    if symbol == "limit-from": # Won't occur in gold labels but could occur in preds from copying parsing errors
        return eval_tree_value(tree_node.children[0], options)
    raise Exception(f"Unsupported token: type={tree_node.token_type}, id={tree_node.token_id}, symbol={symbol}")

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
    return total_loss / num_batches, [perplexity], "Perplexity: {:.3f}"

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
    return loss, [accuracy], "Accuracy: {:.3f}"

def evaluate_gen_task(model_name: str, model: MathGPTLM, dataset: Dataset, task: DownstreamTask, fold: int, options: TrainOptions):
    model.eval()
    compute_ted = (options.eval_formulas or task == DownstreamTask.MWP) and not options.baseline
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

    exact_match = [pred == label for pred, label in zip(all_predictions, all_labels)]
    accuracy = sum(exact_match) / len(all_labels)
    postfix = "_formulas" if options.eval_formulas else "_text" if options.eval_text else ""
    pred_filename = f"preds_{model_name}{postfix}_{fold}.txt"
    label_filename = f"labels_{model_name}{postfix}_{fold}.txt"
    with open(pred_filename, "w", encoding="utf-8") as pred_file:
        pred_file.write("\n".join(all_predictions))
    with open(label_filename, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(all_labels))
    metrics = compute_metrics(hypothesis=pred_filename, references=[label_filename], no_skipthoughts=True, no_glove=True)
    results = [accuracy, metrics['Bleu_4'], metrics['ROUGE_L'], metrics['METEOR']]
    template = "Exact Match Accuracy: {:.3f}, BLEU-4: {:.3f}, ROUGE-L: {:.3f}, METEOR: {:.3f}"
    if compute_ted:
        results.append(calculate_ted(all_label_trees, all_pred_trees))
        tree_exact_match = [trees_equal(label_tree, pred_tree) for label_tree, pred_tree in zip(all_label_trees, all_pred_trees)]
        results.append(sum(tree_exact_match) / len(tree_exact_match))
        template += ", TED: {:.3f}, Tree Match: {:.3f}"
        if task == DownstreamTask.MWP:
            pred_tree_vals = []
            for pred_tree in all_pred_trees:
                try:
                    pred_tree_vals.append(eval_tree_value(pred_tree, options))
                except Exception:
                    pred_tree_vals.append(None)
            tree_vals_eq = [
                eval_tree_value(label_tree, options) == pred_tree_val
                for label_tree, pred_tree_val in zip(all_label_trees, pred_tree_vals)
            ]
            results.append(sum(tree_vals_eq) / len(tree_vals_eq))
            text_and_val_match = [tree_val_match and text_match for tree_val_match, text_match in zip(tree_vals_eq, exact_match)]
            results.append(sum(text_and_val_match) / len(text_and_val_match))
            tree_and_val_match = [tree_val_match and tree_match for tree_val_match, tree_match in zip(tree_vals_eq, tree_exact_match)]
            results.append(sum(tree_and_val_match) / len(tree_and_val_match))
            template += ", Value Match: {:.3f}, Text + Value Match {:.3f}, Tree + Value Match {:.3f}"
    return 0, results, template

def get_problem_solving_final_answer(full_solution: str):
    processed_solution = ""
    if "Final Answer:" in full_solution:
        processed_solution = full_solution.split("Final Answer:")[1]
    return processed_solution.strip().replace(" , ", "") # Remove commas from numbers and whitespace around answer

def evaluate_problem_solving_task(model_name: str, model: MathGPTLM, dataset: Dataset, task: DownstreamTask, overwrite_results: bool, options: TrainOptions):
    model.eval()
    postfix = "_final" if options.eval_final else ""
    label_filename = f"labels_{model_name}{postfix}.txt"
    pred_filename = f"preds_{model_name}{postfix}.txt"

    if overwrite_results or not os.path.exists(pred_filename):
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
                label_str = decode_batch(label)[0].replace("\n", " ").strip()
                pred_str = decode_batch(pred)[0].replace("\n", " ").strip()
                if options.eval_final:
                    pred_str = pred_str.split("[SEP] Final Answer:")[0].strip() or " "
                all_labels.append(label_str)
                all_predictions.append(pred_str)

        with open(label_filename, "w", encoding="utf-8") as label_file:
            label_file.write("\n".join(all_labels))
        with open(pred_filename, "w", encoding="utf-8") as pred_file:
            pred_file.write("\n".join(all_predictions))
    else:
        with open(label_filename, encoding="utf-8") as label_file:
            all_labels = label_file.readlines()
        with open(pred_filename, encoding="utf-8") as pred_file:
            all_predictions = pred_file.readlines()

    # Group labels/preds by difficulty level
    level_to_results: Dict[str, Dict[str, List]] = {}
    if task == DownstreamTask.MATH:
        for label, pred, sample in zip(all_labels, all_predictions, dataset):
            cur_level = level_to_results.setdefault(sample.meta["level"], {"labels": [], "preds": []})
            cur_level["labels"].append(label)
            cur_level["preds"].append(pred)
    level_to_results["Overall"] = {"labels": all_labels, "preds": all_predictions}

    if options.eval_final:
        level_to_metrics = {}
        for level, res in sorted(level_to_results.items()):
            label_level_filename = f"labels_{model_name}_{level}.txt"
            pred_level_filename = f"preds_{model_name}_{level}.txt"
            with open(label_level_filename, "w", encoding="utf-8") as label_file:
                label_file.write("\n".join(res["labels"]))
            with open(pred_level_filename, "w", encoding="utf-8") as pred_file:
                pred_file.write("\n".join(res["preds"]))
            level_to_metrics[level] = compute_metrics(hypothesis=pred_level_filename, references=[label_level_filename], no_skipthoughts=True, no_glove=True)
        template = "\n".join([
            level + " - BLEU-4: {:.3f}, ROUGE-L: {:.3f}, METEOR: {:.3f}" for level in sorted(level_to_metrics.keys())
        ])
        results = [mets[stat] for _, mets in sorted(level_to_metrics.items()) for stat in ["Bleu_4", "ROUGE_L", "METEOR"]]
    else:
        template = ", ".join([level + ": {:.3f}" for level in sorted(level_to_results.keys())])
        results = [
            metrics.accuracy_score(
                [get_problem_solving_final_answer(label) for label in res["labels"]],
                [get_problem_solving_final_answer(pred) for pred in res["preds"]]
            )
            for _, res in sorted(level_to_results.items())
        ]

    return 0, results, template

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
    return loss, [accuracy, auc, kappa, rmse, f1.mean()], "Accuracy: {:.3f}, AUC: {:.3f}, Kappa: {:.3f}, RMSE: {:.3f}, F1: {:.3f}"

def evaluate_ct(model_name: str):
    matches: List[Dict[str, int]] = []
    sizes: List[Dict[str, int]] = []
    for fold in range(5):
        cur_matches = {
            "_overall": 0,
            "BUG/ERR": 0,
        }
        matches.append(cur_matches)
        cur_sizes = {
            "_overall": 0,
            "BUG/ERR": 0,
        }
        sizes.append(cur_sizes)
        with open(f"results/preds_{model_name}_{fold}.txt", encoding="utf-8") as pred_file:
            with open(f"results/labels_{model_name}_{fold}.txt", encoding="utf-8") as label_file:
                for pred, label in zip(pred_file, label_file):
                    pred_act_in = pred.split(":", 1)[1].strip()
                    outcome, label_act_in = label.split(":", 1)
                    outcome = outcome.strip()
                    label_act_in = label_act_in.strip()
                    action = label_act_in.split(" ", 1)[0]
                    cur_sizes["_overall"] += 1
                    cur_sizes.setdefault(outcome, 0)
                    cur_sizes[outcome] += 1
                    cur_sizes.setdefault(action, 0)
                    cur_sizes[action] += 1
                    if outcome in ("BUG", "ERROR"):
                        cur_sizes["BUG/ERR"] += 1
                    if pred_act_in == label_act_in:
                        cur_matches["_overall"] += 1
                        cur_matches.setdefault(outcome, 0)
                        cur_matches[outcome] += 1
                        cur_matches.setdefault(action, 0)
                        cur_matches[action] += 1
                        if outcome in ("BUG", "ERROR"):
                            cur_matches["BUG/ERR"] += 1
    print(sizes)
    all_keys = {key for cur_matches in matches for key in cur_matches}
    for key in sorted(all_keys):
        accs = np.array([cur_matches[key] / cur_sizes[key] for cur_matches, cur_sizes in zip(matches, sizes) if key in cur_matches])
        print(f"{key} - avg: {accs.mean():.3f}, std: {accs.std():.3f}")

def evaluate_ted(model_name: str, task: DownstreamTask, options_dict: dict):
    options = TrainOptions(options_dict)
    all_teds = []
    all_accs = []
    all_val_match = []
    folds = range(5) if task in (DownstreamTask.FEEDBACK, DownstreamTask.MWP) else [0]
    for fold in folds:
        # Load saved labels and predictions
        postfix = "_formulas" if options.eval_formulas else ""
        pred_filename = f"results/preds_{model_name}{postfix}_{fold}.txt"
        with open(pred_filename, encoding="utf-8") as pred_file:
            preds = [("<m> " if options.eval_formulas else "") + pred.strip() + ("" if pred.strip().endswith("</m>") else " </m>") for pred in pred_file]

        # Convert sample strings to OPTs via pre-processing pipeline
        batch_size = 100
        err_data = {}
        processed_preds: List[Article] = []
        for batch_start_idx in tqdm(list(range(0, len(preds), batch_size))):
            processed_preds += process_raw_text(preds[batch_start_idx : batch_start_idx + batch_size], err_data)
        print(err_data)

        # Load dataset for targets
        if task == DownstreamTask.HEADLINES:
            headlines = get_headline_data("test", options)
            test_data = GenTaskDataset(headlines, task, options)
        elif task == DownstreamTask.FEEDBACK:
            problems, _, _, test_samples = get_feedback_data(fold)
            test_data = FeedbackDataset(test_samples, problems, options)
        elif task == DownstreamTask.MWP:
            _, _, test_samples = get_mwp_data(fold)
            test_data = GenTaskDataset(test_samples, task, options)
        else:
            raise Exception(f"Unsupported task {task}")

        # Perform post-processing via tokenizer, and then convert back to OPTs and calculate TED
        label_trees: List[DecodeTreeNode] = []
        pred_trees: List[DecodeTreeNode] = []
        failed_conversions = []
        missing_formula = []
        for sample_idx, (label, pred) in enumerate(zip(test_data, processed_preds)):
            if pred is None:
                failed_conversions.append(sample_idx)
                continue
            if not pred["formulas"]:
                missing_formula.append(sample_idx)
                continue
            if len(pred["formulas"]) > 1:
                print("More than 1 formula in sample:", sample_idx)
            pred_seq = tokenize_formula(pred["formulas"][0]["opt"], options)
            label_seq = label.split_at(label.meta["prompt_length"])[1]
            label_trees.append(get_tree(label_seq.token_ids, label_seq.token_types))
            pred_trees.append(get_tree(pred_seq.token_ids, pred_seq.token_types))
        ted = calculate_ted(label_trees, pred_trees)
        acc = sum([trees_equal(label_tree, pred_tree) for label_tree, pred_tree in zip(label_trees, pred_trees)]) / len(label_trees)
        pred_vals = []
        for pred_tree in pred_trees:
            try:
                pred_vals.append(eval_tree_value(pred_tree, options))
            except Exception as exc:
                print(exc)
                pred_vals.append(None)
        val_match = sum([
            eval_tree_value(label_tree, options) == pred_val
            for label_tree, pred_val in zip(label_trees, pred_vals)
        ]) / len(label_trees)
        all_teds.append(ted)
        all_accs.append(acc)
        all_val_match.append(val_match)
        print(f"{fold} - TED: {ted:.3f}, Tree Match: {acc:.3f}, Val Match: {val_match:.3f}, Failed: {failed_conversions}, Missing formula: {missing_formula}")
    teds_np = np.array(all_teds)
    accs_np = np.array(all_accs)
    val_np = np.array(all_val_match)
    print(f"TED - Average: {teds_np.mean():.3f}, STD: {teds_np.std():.3f}")
    print(f"Tree Match - Average: {accs_np.mean():.3f}, STD: {accs_np.std():.3f}")
    print(f"Val Match - Average: {val_np.mean():.3f}, STD: {val_np.std():.3f}")
