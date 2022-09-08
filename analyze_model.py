from typing import List, Optional
import math
import torch
import numpy as np

from training import load_model, get_headline_data, get_probes
from loading import PreTrainDatasetPreloaded, GenTaskDataset, get_data_loader
from vocabulary import Vocabulary, MATH_TYPES
from model_baseline import GPTLMBaseline
from utils import text_tokenizer, enum_value_to_member, TrainOptions
from constants import DownstreamTask, TokenType

# Wrapping this in try/catch since compute cluster is missing a relevant library
try:
    from matplotlib import pyplot as plt
except Exception:
    pass

def show_heatmap(weights_array: List[np.ndarray], x_labels: List[str], y_labels: List[str], show_labels: bool):
    """
    Show a heatmap with matplotlib
    Based on example from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    num_cols = min(4, len(weights_array))
    num_rows = math.ceil(len(weights_array) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)
    for weight_array_idx, weights in enumerate(weights_array):
        ax: plt.Axes = axes[weight_array_idx // num_cols][weight_array_idx % num_cols]
        ax.imshow(weights)
        if show_labels:
            ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
            ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
        # for x in range(len(x_labels)):
        #     for y in range(len(y_labels)):
        #         ax.text(x, y, f"{weights[y, x]:.1f}", ha="center", va="center", color="w", fontsize="x-small")
    fig.tight_layout()
    fig.set_size_inches(20, 8)
    plt.show()

def select_heads(layer: torch.Tensor, head_idx: Optional[int]):
    if head_idx is None:
        return layer.mean(dim=0)
    return layer[head_idx]

def visualize_attention(model_name: str, task_str: str, options_dict: dict):
    data_idx = 5
    num_tokens = 50
    exp = 1
    layers = "last"
    heads = "all"
    assert layers in ["all", "first", "last", "avg"]
    assert heads in ["all", "first", "avg"]
    assert not (layers == "all" and heads == "all")

    task = None if task_str == "probes" else enum_value_to_member(task_str, DownstreamTask)
    if model_name == "baseline":
        model = GPTLMBaseline()
        options = TrainOptions({"baseline": True, **options_dict})
    else:
        model, _, options = load_model(model_name, False, task)
        options.update(options_dict)
    if task == None:
        dataset = PreTrainDatasetPreloaded(get_probes()[data_idx : data_idx + 1], options, options.max_seq_len)
    elif task == DownstreamTask.HEADLINES:
        dataset = GenTaskDataset(get_headline_data("val", options)[data_idx : data_idx + 1], task, options)
    data_loader = get_data_loader(dataset, task, 1, False, False, options)
    with torch.no_grad():
        batch = next(iter(data_loader))
        # Attentions is tuple of tensors, shape is (layers x batch x heads x seq x seq)
        attentions: List[torch.Tensor] = model(batch, output_attentions=True)[2]
        if layers == "all":
            head_idx = None if heads == "avg" else 1
            attention_weights = [select_heads(attn_layer[0], head_idx).pow(exp).detach().cpu().numpy() for attn_layer in attentions]
        else:
            src_layer = attentions[0][0] if layers == "first" else attentions[-1][0] if layers == "last" else torch.concat(attentions).mean(dim=0)
            if heads == "all":
                attention_weights = [select_heads(src_layer, head_idx).pow(exp).detach().cpu().numpy() for head_idx in range(src_layer.shape[0])]
            else:
                head_idx = None if heads == "avg" else 1
                attention_weights = [select_heads(src_layer, head_idx).pow(exp).detach().cpu().numpy()]
        x_labels = y_labels = [
            text_tokenizer().decode([int(token_id)]) if token_type in (TokenType.TEXT, TokenType.MATH_TEXT) else (
                Vocabulary.get_symbol(int(token_type), int(token_id)) if token_type in MATH_TYPES else (
                    str(enum_value_to_member(token_type, TokenType)).replace("TokenType.", "")
                )
            )
            for token_id, token_type in zip(batch["token_ids"][0], batch["token_types"][0])
        ]
        if task == DownstreamTask.HEADLINES:
            prompt_len = batch["prompt_lengths"][0]
            attention_weights = [attn[prompt_len:, :prompt_len] for attn in attention_weights]
            x_labels, y_labels = x_labels[:prompt_len], y_labels[prompt_len:]
        attention_weights = [attn[:num_tokens, :num_tokens] for attn in attention_weights]
        x_labels, y_labels = x_labels[:num_tokens], y_labels[:num_tokens]
        show_heatmap(attention_weights, x_labels, y_labels, len(attention_weights) == 1)
