from typing import List, Callable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn import metrics

from loading import Collator, trim_batch
from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from generate import get_most_likely_predictions
from utils import TrainOptions
from constants import PADDING_TOKEN_ID, CollatedBatch

def evaluate_lm(model: MathGPTLM, dataset: Dataset, options: TrainOptions):
    """
    Calculate perplexity: e ^ ((1/n) * nll)
    Algorithm from https://huggingface.co/docs/transformers/perplexity
    """
    # TODO: unit test
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=1, # Only 1 sequence can be processed at a time to recover NLL from the cross-entropy loss (because of padding complications)
    )
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
                loss = model(sub_seq_batch, labels=labels)[0]
                total_loss += loss.detach().cpu().numpy()
                num_batches += 1
                # Loss is average NLL over all tokens in the sequence, multiply by number of targets to undo average and retrieve sum
                nlls.append(loss * target_len)
        perplexity = torch.exp(torch.sum(torch.stack(nlls)) / total_sequence_length)
    return total_loss / num_batches, f"Perplexity: {perplexity:.3f}"

def process_model_output(model: MathGPTBase, dataset: Dataset, options: TrainOptions, output_accumulator: Callable[[Tuple, CollatedBatch], None]):
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size
    )
    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            model_output = model(batch)
            total_loss += model_output[0].detach().cpu().numpy()
            num_batches += 1
            output_accumulator(model_output, batch)
    return total_loss / num_batches

def evaluate_lm_accuracy(model: MathGPTLM, dataset: Dataset, options: TrainOptions):
    """
    Calculate per-token prediction accuracy
    """
    # TODO: unit test
    all_predictions = []
    all_labels = []
    def accumulate_predictions(model_output, batch: CollatedBatch):
        type_preds, token_preds = get_most_likely_predictions(model_output[1])
        # For predictions and targets, stack types and tokens in last dimension
        type_preds = type_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
        token_preds = token_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
        predictions = np.stack([type_preds, token_preds], axis=-1)
        type_targets = batch["token_types"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
        token_targets = batch["token_ids"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
        targets = np.stack([type_targets, token_targets], axis=-1)
        mask = batch["attention_mask"][:, 1:].contiguous().view(-1).detach().cpu().numpy() == 1
        all_predictions.append(predictions[mask])
        all_labels.append(targets[mask])

    loss = process_model_output(model, dataset, options, accumulate_predictions)

    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    # Get indices where both type and token match
    match = all_preds_np == all_labels_np
    match = match[:, 0] & match[:, 1]
    accuracy = sum(match) / len(match)
    return loss, f"Accuracy: {accuracy:.3f}"

def evaluate_gen_task(model: MathGPTLM, dataset: Dataset, options: TrainOptions):
    # TODO: unit test
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    def accumulate_predictions(model_output, batch: CollatedBatch):
        # TODO: this doesn't actually seem like the right way to evaluate
        #       we should probably generate a full prediction blind (using nucleus sampling or beam search) and then compare that to the label
        #       what's the proper way to do that? how to fine-tune hyperparameters fairly?
        type_preds, token_preds = get_most_likely_predictions(model_output[1])
        # For predictions and targets, stack types and tokens in last dimension
        type_preds = type_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
        token_preds = token_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
        predictions = np.stack([type_preds, token_preds], axis=-1)
        type_targets = batch["token_types"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
        token_targets = batch["gen_labels"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
        targets = np.stack([type_targets, token_targets], axis=-1)
        mask = token_targets != PADDING_TOKEN_ID # Remove padding as well as region over prompt
        all_predictions.append(predictions[mask])
        all_labels.append(targets[mask])

    loss = process_model_output(model, dataset, options, accumulate_predictions)

    # TODO: evaluate other metrics (BLEU, TED, etc.)
    num_exact_match = sum(1 for pred, label in zip(all_predictions, all_labels) if (pred.shape == label.shape and all(pred == label)))
    accuracy = num_exact_match / len(all_labels)
    return loss, f"Accuracy: {accuracy:.3f}"

def evaluate_cls_task(model: MathGPTClassifier, dataset: Dataset, options: TrainOptions):
    # TODO: unit test
    all_predictions = []
    all_labels = []
    def accumulate_predictions(model_output, batch: CollatedBatch):
        predictions = torch.argmax(model_output[1], dim=-1)
        all_predictions.append(predictions.detach().cpu().numpy())
        all_labels.append(batch["cls_labels"].detach().cpu().numpy())

    loss = process_model_output(model, dataset, options, accumulate_predictions)

    all_preds_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    accuracy = metrics.accuracy_score(all_labels_np, all_preds_np)
    _, _, f1, _ = metrics.precision_recall_fscore_support(all_labels_np, all_preds_np)
    return loss, f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}"
