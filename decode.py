from typing import List
import re
from dataclasses import dataclass
import torch
from transformers import GPT2TokenizerFast

from data_types import CollatedBatch
from constants import TokenType, SpecialOpToken, EOS_TOKEN_ID, PADDING_TOKEN_ID
from vocabulary import Vocabulary

@dataclass
class DecodeTreeNode:
    token_type: int
    token_id: int
    children: list

def get_tree(token_ids: torch.Tensor, token_types: torch.Tensor) -> DecodeTreeNode:
    """
    Convert DFS-order list of token IDs and types to nested tree structure
    """
    root = DecodeTreeNode(int(token_types[0]), int(token_ids[0]), [])
    ancestors: List[DecodeTreeNode] = []
    for token_id, token_type in zip(token_ids[1:], token_types[1:]):
        if token_type == TokenType.OP:
            new_root = DecodeTreeNode(int(token_type), int(token_id), [])
            root.children.append(new_root)
            ancestors.append(root)
            root = new_root
        elif token_type in (TokenType.VAR, TokenType.NUM):
            root.children.append(DecodeTreeNode(int(token_type), int(token_id), []))
        elif token_type == TokenType.END and ancestors:
            root = ancestors.pop()
    # Return root of formula, may need to look in ancestors in case we're decoding a partial tree
    return ancestors[0] if ancestors else root

def tree_to_text(tree_node: DecodeTreeNode) -> str:
    symbol = Vocabulary.get_symbol(tree_node.token_type, tree_node.token_id)
    # TODO: organize these
    if symbol == "eq":
        symbol = "="
    if symbol == "minus":
        symbol = "-"
    if symbol == "plus":
        symbol = "+"
    if symbol == "times":
        symbol = "\\times"
    if symbol == "partialdiff":
        symbol = "\\partial"
    if symbol == "product":
        symbol = "\\prod"
    if symbol == "conditional":
        symbol = "|"
    if symbol == "in":
        symbol = "\\in"
    if symbol == "leq":
        symbol = "\\leq"
    if symbol == "geq":
        symbol = "\\geq"

    if not tree_node.children:
        return symbol

    if symbol in (str(SpecialOpToken.CERR_OP), str(SpecialOpToken.NUM_SUB_TREE_HEAD)):
        return "".join(tree_to_text(child) for child in tree_node.children[1:])

    if symbol == str(SpecialOpToken.ANON_OP):
        left = tree_to_text(tree_node.children[0])
        return left + " { " + "".join(tree_to_text(child) for child in tree_node.children[1:]) + " } "

    if symbol == "abs":
        return " | " + "".join(tree_to_text(child) for child in tree_node.children) + " | "

    if symbol.startswith("interval("):
        return (" ( " if symbol[9] == "O" else " [ ") +\
            " , ".join(tree_to_text(child) for child in tree_node.children) +\
            (" ) " if symbol[11] == "O" else " ] ")

    if symbol == "differential-d":
        return " \\,d " + "".join(tree_to_text(child) for child in tree_node.children)

    if len(tree_node.children) >= 2:
        if symbol in ("\\times", "<", ">", "\\leq", "\\geq"):
            return f" {symbol} ".join(tree_to_text(child) for child in tree_node.children)
    if len(tree_node.children) == 2:
        left = tree_to_text(tree_node.children[0])
        right = tree_to_text(tree_node.children[1])
        if symbol == "SUB":
            return f" {left} _ {{ {right} }} "
        if symbol == "SUP":
            return f" {left} ^ {{ {right} }} "
        if symbol == "divide":
            return f" \\frac {{ {left} }} {{ {right} }}"
        return f" {left} {symbol} {right} "
    # TODO: handle matrices and matrix rows
    return f" {symbol} " + "".join(tree_to_text(child) for child in tree_node.children)

def decode_formula(token_ids: torch.Tensor, token_types: torch.Tensor):
    """
    Convert a raw OPT sequence into a readable formula string
    """
    scheme = "tree"

    if scheme == "order":
        result = ""
        for token_id, token_type in zip(token_ids, token_types):
            if token_type != TokenType.END:
                result += Vocabulary.get_symbol(int(token_type), int(token_id)) + " "
        return result

    if scheme == "tree":
        tree = get_tree(token_ids, token_types)
        return tree_to_text(tree)

    return ""

def decode_batch(batch: CollatedBatch, text_tokenizer: GPT2TokenizerFast) -> List[str]:
    """
    Given a batch, decode it into human-readable text
    Return text translation for each sequence in the batch
    """
    all_decoded_sequences: List[str] = []
    for seq_idx in range(len(batch["token_ids"])):
        result = ""
        sub_seq_start = sub_seq_end = 0
        is_text = True
        for tok_idx, (token_type, token_id) in enumerate(zip(batch["token_types"][seq_idx], batch["token_ids"][seq_idx])):
            # At start formula, switch to math context, and decode any prior text tokens
            if token_type == TokenType.START_FORMULA:
                if sub_seq_start != sub_seq_end:
                    result += text_tokenizer.decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])
                result += " $ "
                sub_seq_start = sub_seq_end = tok_idx + 1
                is_text = False
                continue

            # At end formula, switch to text context
            if token_type == TokenType.END_FORMULA:
                if sub_seq_start != sub_seq_end:
                    result += decode_formula(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end], batch["token_types"][seq_idx][sub_seq_start : sub_seq_end])
                result += " $ "
                sub_seq_start = sub_seq_end = tok_idx + 1
                is_text = True
                continue

            # Stop decoding at EOS token (do this before incrementing end idx to not print EOS token string)
            if token_type == TokenType.TEXT and token_id == EOS_TOKEN_ID:
                break

            sub_seq_end = tok_idx + 1

        # Decode any trailing tokens at the end
        if sub_seq_start != sub_seq_end:
            if not all(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end] == PADDING_TOKEN_ID): # TODO: probably a more elegant way to handle this
                token_ids = batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end]
                non_padding_idx = token_ids != PADDING_TOKEN_ID # TODO: why would we have padding that is not preceded by EOS?
                if is_text:
                    result += text_tokenizer.decode(token_ids[non_padding_idx])
                else:
                    result += decode_formula(token_ids[non_padding_idx], batch["token_types"][seq_idx][sub_seq_start : sub_seq_end][non_padding_idx])

        result = re.sub(r" +", " ", result)
        all_decoded_sequences.append(result)

    return all_decoded_sequences
