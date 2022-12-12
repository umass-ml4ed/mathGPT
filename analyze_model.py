from typing import List, Optional
import math
import torch
import numpy as np
from sklearn import manifold

from training import load_model
from loading import PreTrainDatasetPreloaded, GenTaskDataset, Collator, get_data_loader, get_headline_data, get_probes, get_mwp_data
from vocabulary import Vocabulary, MATH_TYPES
from model_baseline import GPTLMBaseline
from math_tokenize import tokenize_formula
from symbol_map import SYMBOL_MAP_ANALYSIS, SYMBOL_MAP_DISPLAY
from utils import text_tokenizer, enum_value_to_member, TrainOptions
from data_types import OPT
from constants import DownstreamTask, TokenType, TPE, SpecialOpToken

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
        options = TrainOptions({"baseline": True, **options_dict})
        model = GPTLMBaseline(options)
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

def visualize_tokens(model_name: str):
    # Get most frequently occurring math tokens
    ops = Vocabulary.most_frequent(TokenType.OP, 100)
    # Remove ops that can't be represented as standalone symbols by the baseline model
    # Symbol map has one-to-one operator mapping, and single-character symbols are already in their final form
    # Filter out symbols that map to empty string/whitespace or can't render
    # Filter out "T" type (just assigned op type as a formality)
    ops = [
        op for op in ops
        if (op[1] in SYMBOL_MAP_ANALYSIS or len(op[1]) == 1) and\
            op[0] != "T" and SYMBOL_MAP_ANALYSIS.get(op[1], op[1]) and\
                op[1] != "⏟" and\
                    not SYMBOL_MAP_ANALYSIS.get(op[1], op[1]).isspace()
    ]
    print("Keeping", len(ops), "ops")
    print(ops)
    # print(" ".join([f"${SYMBOL_MAP_ANALYSIS.get(op[1], op[1])}$" for op in ops]))
    # tokens = ops + Vocabulary.most_frequent(TokenType.VAR, 50) + Vocabulary.most_frequent(TokenType.NUM) # Ops, vars and nums
    tokens = ops # Just ops
    # tokens = Vocabulary.most_frequent(TokenType.VAR, 50) # Just vars

    # Create single formula sequence with all those tokens
    formula = ["O", "eq", [ # Arbitrary head node, will get removed
        [type_str, symbol.replace("matrix-", ""), None]
        for type_str, symbol in tokens
    ]]

    if model_name == "baseline":
        options = TrainOptions({"baseline": True})
        model = GPTLMBaseline(options)
    else:
        model, _, options = load_model(model_name, False, None)

    # Get token representations from model
    with torch.no_grad():
        if options.baseline:
            # Take average of token embeddings for each symbol (post mapping)
            embeddings = torch.stack([
                model.gpt2_lm.transformer.wte(
                    torch.LongTensor(text_tokenizer()(SYMBOL_MAP_ANALYSIS.get(symbol, symbol))["input_ids"])
                ).mean(dim=0)
                for (_, symbol, _) in formula[2]
            ]).detach().numpy()
        else:
            options.tpe = TPE.NONE.value # Not visualizing tree positions
            formula_seq = tokenize_formula(formula, options)
            formula_seq = formula_seq.split_at(1)[1] # Remove head node
            batch = Collator(None, options)([formula_seq])
            embeddings = model.get_input_embeddings(batch)
            embeddings = embeddings[~(batch["token_types"] == TokenType.END)]
            embeddings = embeddings.detach().numpy()

    # Transform with TSNE
    # Chosen seeds show visually pleasing representations and represent a variety of patterns
    # rseed = 100 if model_name == "baseline" else 0 if options.baseline else 0
    # rseed = 67 if model_name == "baseline" else 150 if options.baseline else 115
    rseed = 391 if model_name == "baseline" else 250 if options.baseline else 50
    transformer = manifold.TSNE(2, perplexity=10, learning_rate="auto", n_iter=1000, init="pca", random_state=rseed)
    reduced_states = transformer.fit_transform(embeddings)

    # Visualize
    x_vals = reduced_states[:,0]
    y_vals = reduced_states[:,1]
    plt.scatter(x_vals, y_vals, picker=True)
    # Label with display symbols
    for (_, symbol, _), x, y in zip(formula[2], x_vals, y_vals):
        symbol = SYMBOL_MAP_ANALYSIS.get(symbol, symbol)
        symbol = SYMBOL_MAP_DISPLAY.get(symbol, symbol)
        plt.annotate(symbol, (x, y + 1), fontsize=30)
    # Define click handler - print information associated with clicked point
    def onpick(event):
        ind = event.ind
        print(formula[2][ind[0]])
    plt.connect('pick_event', onpick)
    plt.axis("off")
    plt.show()

def traverse_tree(formula: OPT, parent_node: Optional[dict], parent_pos: List[int], child_idx: int):
    cur_node = {
        "symbol": SYMBOL_MAP_ANALYSIS.get(formula[1], formula[1]),
        "pos": parent_pos + [child_idx],
        "parent_node": parent_node
    }
    if not formula[2]:
        return [cur_node]
    assert len(formula[2]) >= 2
    results: List[dict] = traverse_tree(formula[2][0], cur_node, cur_node["pos"], 0)
    results.append(cur_node)
    results += traverse_tree(formula[2][1], cur_node, cur_node["pos"], 1)
    for child_idx in range(2, len(formula[2])):
        cur_node = cur_node.copy()
        cur_node["pos"] = cur_node["pos"].copy()
        cur_node["pos"][-1] += 1
        results.append(cur_node)
        results += traverse_tree(formula[2][child_idx], cur_node, cur_node["pos"], child_idx)
    return results

def visualize_tpe(model_name: str):
    samples, _, _ = get_mwp_data()
    idx, sample = next((idx, sample) for idx, sample in enumerate(samples) if len(sample["label"]["formulas"]["0"]["tex"]) > 20)
    print(sample["label"]["formulas"]["0"]["tex"])
    print(idx)

    if model_name == "baseline":
        options = TrainOptions({"baseline": True})
        model = GPTLMBaseline(options)
    else:
        model, _, options = load_model(model_name, False, None)

    if options.baseline:
        tree_data = traverse_tree(sample["label"]["formulas"]["0"]["opt"], None, [], 0)
        with torch.no_grad():
            tpe = model.gpt2_lm.transformer.wpe(torch.arange(len(tree_data)))
            tpe = tpe.detach().numpy()
    else:
        formula_seq = tokenize_formula(sample["label"]["formulas"]["0"]["opt"], options)
        batch = Collator(None, options)([formula_seq])
        with torch.no_grad():
            tpe = model.get_math_embeddings(batch, torch.ones_like(batch["token_ids"]).type(torch.bool))
            tpe += model.gpt2_lm.transformer.wpe(torch.arange(len(batch["token_ids"])))
            tpe = tpe.detach().numpy()

    # Transform with TSNE
    # rseed = 50 if model_name == "baseline" else 300 # Seeds chosen to put root node roughly in upper left
    rseed = 127 if model_name == "baseline" else 235 if options.baseline else 671 # Vertical orientation
    transformer = manifold.TSNE(2, perplexity=10, learning_rate="auto", n_iter=1000, init="pca", random_state=rseed)
    reduced_states = transformer.fit_transform(tpe)

    # Visualize
    x_vals = reduced_states[:,0]
    y_vals = reduced_states[:,1]
    plt.scatter(x_vals, y_vals)

    if options.baseline:
        # Assign token idx to each node
        for idx in range(len(tree_data)):
            tree_data[idx]["token_idx"] = idx
        # Draw connection between each child and parent
        for idx in range(len(tree_data)):
            if tree_data[idx]["parent_node"] is not None:
                parent_idx = tree_data[idx]["parent_node"]["token_idx"]
                plt.plot(
                    [x_vals[parent_idx], x_vals[idx]],
                    [y_vals[parent_idx], y_vals[idx]],
                    "b"
                )
        # Annotate, done after to render symbols on top of lines
        for idx in range(len(tree_data)):
            # plt.annotate(f'{tree_data[idx]["symbol"]} {tree_data[idx]["pos"]}', (x_vals[idx], y_vals[idx] + 1), fontsize=30)
            plt.annotate(tree_data[idx]["symbol"], (x_vals[idx] - 1, y_vals[idx] + 2), fontsize=30)
    else:
        # Label with position vectors and symbols
        for idx in range(len(formula_seq)):
            token_type = formula_seq.token_types[idx]
            token_id = formula_seq.token_ids[idx]
            gpt_tokens = formula_seq.gpt_tokens[idx]
            pos_vec = formula_seq.pos_vecs[idx]
            pos_level = formula_seq.pos_levels[idx]
            symbol = "E" if token_type == TokenType.END else\
                "N" if token_type == TokenType.OP and token_id == SpecialOpToken.NUM_SUB_TREE_HEAD else\
                    text_tokenizer().decode(gpt_tokens)
            symbol = SYMBOL_MAP_ANALYSIS.get(symbol, symbol)
            # plt.annotate(f'{symbol} {pos_vec[:pos_level + 1]}', (x_vals[idx], y_vals[idx] + 1), fontsize=30)
            plt.annotate(symbol, (x_vals[idx] - 1, y_vals[idx] + 2), fontsize=30)
        # Connect parents to children, keep parent history in stack and use DFS order to do in one pass
        parents = []
        for idx, pos_level in enumerate(formula_seq.pos_levels):
            if parents:
                plt.plot(
                    [x_vals[parents[-1][0]], x_vals[idx]],
                    [y_vals[parents[-1][0]], y_vals[idx]],
                    "b"
                )
            if idx == len(formula_seq) - 1:
                break
            if formula_seq.pos_levels[idx + 1] > pos_level:
                parents.append((idx, pos_level))
            elif formula_seq.pos_levels[idx + 1] < pos_level:
                parents.pop()
    plt.axis("off")
    plt.show()
