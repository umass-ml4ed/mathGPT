from typing import List
from dataclasses import dataclass
import torch
from transformers import GPT2TokenizerFast

from constants import CollatedBatch, TokenType, EOS_TOKEN_ID, PADDING_TOKEN_ID
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
    # TODO: handle incomplete formulas
    if token_types[0] == TokenType.END:
        # TODO: this happens in Annular_fin and GCD
        print(token_ids, token_types)
        return DecodeTreeNode(TokenType.VAR, 0, [])
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
    return root

def tree_to_text(tree_node: DecodeTreeNode) -> str:
    symbol = Vocabulary.get_symbol(tree_node.token_type, tree_node.token_id)
    # TODO: organize these
    symbol = symbol.replace("normal-", "")
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

    if not tree_node.children:
        return symbol

    if symbol == "SpecialOpToken.ANON_OP": # TODO: better check for this
        left = tree_to_text(tree_node.children[0])
        return left + "{" + "".join(tree_to_text(child) for child in tree_node.children[1:]) + "}"

    if len(tree_node.children) == 2:
        left = tree_to_text(tree_node.children[0])
        right = tree_to_text(tree_node.children[1])
        if symbol == "SUB":
            return f"{left}_{{{right}}}"
        if symbol == "SUP":
            return f"{left}^{{{right}}}"
        if symbol == "divide":
            return f"\\frac{{{left}}}{{{right}}}"
        if symbol == "interval(O-O)":
            return f"({left},{right})"
        return f"{left} {symbol} {right}"
    return symbol + "{" + "".join(tree_to_text(child) for child in tree_node.children) + "}"

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

def decode_batch(batch: CollatedBatch, text_tokenizer: GPT2TokenizerFast) -> List[str]:
    """
    Given a batch, decode it into human-readable text
    Return text translation for each sequence in the batch
    """
    all_decoded_sequences: List[str] = []
    for seq_idx in range(len(batch["token_ids"])):
        result = ""
        sub_seq_start = sub_seq_end = 0
        for tok_idx, (token_type, token_id) in enumerate(zip(batch["token_types"][seq_idx], batch["token_ids"][seq_idx])):
            # At start formula, switch to math context, and decode any prior text tokens
            if token_type == TokenType.START_FORMULA:
                if sub_seq_start != sub_seq_end:
                    result += text_tokenizer.decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])
                result += "$"
                sub_seq_start = sub_seq_end = tok_idx + 1
                continue

            # At end formula, switch to text context
            if token_type == TokenType.END_FORMULA:
                if sub_seq_start != sub_seq_end:
                    result += decode_formula(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end], batch["token_types"][seq_idx][sub_seq_start : sub_seq_end])
                result += "$"
                sub_seq_start = sub_seq_end = tok_idx + 1
                continue

            sub_seq_end = tok_idx + 1

            # Stop decoding at EOS token
            if token_type == TokenType.TEXT and token_id == EOS_TOKEN_ID:
                break

        # Decode any trailing text tokens at the end
        if sub_seq_start != sub_seq_end:
            if not all(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end] == PADDING_TOKEN_ID): # TODO: probably a more elegant way to handle this
                # import pdb; pdb.set_trace()
                text_tokens = batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end]
                text_tokens = text_tokens[text_tokens != PADDING_TOKEN_ID] # TODO: why would we have padding that is not preceded by EOS?
                result += text_tokenizer.decode(text_tokens)

        all_decoded_sequences.append(result)

    return all_decoded_sequences
