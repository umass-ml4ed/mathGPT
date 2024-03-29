from typing import List
import re
from dataclasses import dataclass
import torch

from data_types import CollatedBatch
from utils import text_tokenizer
from constants import TokenType, SpecialOpToken, EOS_TOKEN_ID, PADDING_TOKEN_ID
from vocabulary import Vocabulary, get_matrix_symbol
from symbol_map import SYMBOL_MAP

@dataclass
class DecodeTreeNode:
    token_type: int
    token_id: int
    children: List['DecodeTreeNode']

def get_tree(token_ids: torch.Tensor, token_types: torch.Tensor) -> DecodeTreeNode:
    """
    Convert DFS-order list of token IDs and types to nested tree structure
    Assuming the sequence contains a formula, may exclude start/end formula tokens
    """
    start_idx = next((idx for idx, token_type in enumerate(token_types) if token_type == TokenType.START_FORMULA), -1) + 1
    root = DecodeTreeNode(int(token_types[start_idx]), int(token_ids[start_idx]), [])
    ancestors: List[DecodeTreeNode] = []
    for token_id, token_type in zip(token_ids[start_idx + 1:], token_types[start_idx + 1:]):
        if token_type == TokenType.OP:
            new_root = DecodeTreeNode(int(token_type), int(token_id), [])
            root.children.append(new_root)
            ancestors.append(root)
            root = new_root
        elif token_type in (TokenType.VAR, TokenType.NUM, TokenType.MATH_TEXT):
            root.children.append(DecodeTreeNode(int(token_type), int(token_id), []))
        elif token_type == TokenType.END and ancestors:
            root = ancestors.pop()
        elif token_type == TokenType.END_FORMULA:
            break
    # Return root of formula, may need to look in ancestors in case we're decoding a partial tree
    return ancestors[0] if ancestors else root

# For testing purposes
cur_seq = {}
unhandled_symbols = set()
symbol_to_first_seq = {}

def tree_to_text(tree_node: DecodeTreeNode) -> str:
    # Resolve math text for current node (leaf nodes)
    if Vocabulary.math_text() and tree_node.token_type == TokenType.OP and tree_node.token_id == SpecialOpToken.MATH_TEXT_HEAD:
        return text_tokenizer().decode([child.token_id for child in tree_node.children])

    symbol = Vocabulary.get_symbol(tree_node.token_type, tree_node.token_id)
    children = tree_node.children

    # Resolve math text for op nodes
    if Vocabulary.math_text() and tree_node.token_type == TokenType.OP and tree_node.token_id == SpecialOpToken.ANON_OP and\
        children and children[0].token_type == TokenType.OP and children[0].token_id == SpecialOpToken.MATH_TEXT_HEAD:
        symbol = text_tokenizer().decode([child.token_id for child in children[0].children])
        children = children[1:]

    # Convert symbol to LaTeX macro if mapping exists
    symbol = SYMBOL_MAP.get(symbol, symbol)

    symbol_to_first_seq.setdefault(symbol, cur_seq)

    # Just return symbol for leaf nodes
    if not children:
        if symbol == str(SpecialOpToken.CERR_OP):
            return ""
        return symbol

    # Handle special op nodes
    if symbol == str(SpecialOpToken.CERR_OP):
        return " ".join(tree_to_text(child) for child in children)

    if symbol == str(SpecialOpToken.NUM_SUB_TREE_HEAD):
        return "".join(tree_to_text(child) for child in children)

    if symbol == str(SpecialOpToken.ANON_OP):
        left = tree_to_text(children[0])
        return left + " { " + " ".join(tree_to_text(child) for child in children[1:]) + " } "

    # Handle remaining operators with special meaning
    if symbol == get_matrix_symbol("L") or symbol == "form-seq" or symbol == "and":
        return " , ".join(tree_to_text(child) for child in children)

    # Handles most cases with "and", but commented out since added after training text models
    # if symbol == "and":
    #     fallback_text = " , ".join(tree_to_text(child) for child in children)
    #     try:
    #         assert len(children[0].children) == 2
    #         result = tree_to_text(children[0].children[0])
    #         for child in children:
    #             assert len(child.children) == 2
    #             child.children = child.children[1:]
    #             result += tree_to_text(child)
    #         return result
    #     except AssertionError:
    #         return fallback_text

    if symbol == get_matrix_symbol("D"):
        return " [ " + " , ".join(tree_to_text(child) for child in children) + " ] "

    if symbol == get_matrix_symbol("S"):
        return " \\{ " + " , ".join(tree_to_text(child) for child in children) + " \\} "

    if symbol == get_matrix_symbol("V"):
        return " ( " + " , ".join(tree_to_text(child) for child in children) + " ) "

    if symbol == get_matrix_symbol("M"):
        return " \\begin{matrix} " + " \\\\ ".join(tree_to_text(child) for child in children) + " \\end{matrix} "

    if symbol == get_matrix_symbol("R"):
        return " & ".join(tree_to_text(child) for child in children)

    if symbol == "cases":
        return " \\begin{cases} " +  " \\\\ ".join(
            " & ".join(
                tree_to_text(child)
                for child in children[case_idx : case_idx + 2]
            ) for case_idx in range(0, len(children), 2)
        ) + " \\end{cases} "

    if symbol == "evaluated-at" and len(children) >= 2:
        return tree_to_text(children[0]) + " | _ { " + tree_to_text(children[1]) + " } " + (
            (" ^ { " + tree_to_text(children[2]) + " } ") if len(children) == 3 else "")

    if symbol == "abs":
        return " | " + " ".join(tree_to_text(child) for child in children) + " | "

    if symbol == "norm":
        return " \\| " + " ".join(tree_to_text(child) for child in children) + " \\| "

    if symbol == "conditional-set":
        return " \\{ " + " | ".join(tree_to_text(child) for child in children) + " \\} "

    if symbol == "differential-d":
        return " \\,d " + " ".join(tree_to_text(child) for child in children)

    if symbol == "expectation":
        return " < " + " ".join(tree_to_text(child) for child in children) + " > "

    if symbol == "percent":
        return " ".join(tree_to_text(child) for child in children) + " % "

    if symbol == "factorial":
        return " ".join(tree_to_text(child) for child in children) + " ! "

    if symbol == "double-factorial":
        return " ".join(tree_to_text(child) for child in children) + " !! "

    if symbol == "ceiling":
        return " \\lceil " + " ".join(tree_to_text(child) for child in children) + " \\rceil "

    if symbol == "floor":
        return " \\lfloor " + " ".join(tree_to_text(child) for child in children) + " \\rfloor "

    if symbol == "inner-product":
        return " \\langle " + " \\mid ".join(tree_to_text(child) for child in children) + " \\rangle "

    if symbol == "root":
        return " \\sqrt [ " + tree_to_text(children[-1]) + " ] { " + " ".join(tree_to_text(child) for child in children[:-1]) + " } "

    if symbol == "binomial" and len(children) == 2:
        return " \\binom { " + tree_to_text(children[0]) + " } { " +  tree_to_text(children[1]) + " } "

    if symbol == "continued-fraction" and len(children) == 2:
        return " \\cfrac { " + tree_to_text(children[0]) + " } { " +  tree_to_text(children[1]) + " } "

    if symbol.startswith("interval("):
        return (" ( " if symbol[9] == "O" else " [ ") +\
            " , ".join(tree_to_text(child) for child in children) +\
            (" ) " if symbol[11] == "O" else " ] ")

    if not symbol.startswith("\\"):
        unhandled_symbols.add(symbol)

    if len(children) == 2:
        if symbol == "SUB":
            return f" {tree_to_text(children[0])} _ {{ {tree_to_text(children[1])} }} "
        if symbol == "SUP":
            return f" {tree_to_text(children[0])} ^ {{ {tree_to_text(children[1])} }} "
    if len(children) == 1:
        return f" {symbol} " + tree_to_text(children[0])
    return f" {symbol} ".join(tree_to_text(child) for child in children)

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

def decode_formula_and_cycle(token_ids: torch.Tensor, token_types: torch.Tensor):
    """
    Decode a formula and cycle through the text tokenizer for consistency with baseline output
    Necessary because text tokenizer is not idempotent because of white space stripping
    """
    formula_text = decode_formula(token_ids, token_types)
    return text_tokenizer().decode(text_tokenizer()(formula_text)["input_ids"])

def decode_batch(batch: CollatedBatch) -> List[str]:
    """
    Given a batch, decode it into human-readable text
    Return text translation for each sequence in the batch
    """
    global cur_seq

    all_decoded_sequences: List[str] = []
    for seq_idx in range(len(batch["token_ids"])):
        cur_seq = {}

        result = ""
        sub_seq_start = sub_seq_end = 0
        is_text = True
        for tok_idx, (token_type, token_id) in enumerate(zip(batch["token_types"][seq_idx], batch["token_ids"][seq_idx])):
            # At start formula, switch to math context, and decode any prior text tokens
            if token_type == TokenType.START_FORMULA:
                if sub_seq_start != sub_seq_end:
                    result += text_tokenizer().decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])
                result += " <m> "
                sub_seq_start = sub_seq_end = tok_idx + 1
                is_text = False
                continue

            # At end formula, switch to text context
            if token_type == TokenType.END_FORMULA:
                if sub_seq_start != sub_seq_end:
                    result += decode_formula_and_cycle(
                        batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end],
                        batch["token_types"][seq_idx][sub_seq_start : sub_seq_end]
                    )
                result += " </m> "
                sub_seq_start = sub_seq_end = tok_idx + 1
                is_text = True
                continue

            # Stop decoding at EOS token (do this before incrementing end idx to not print EOS token string)
            if token_type == TokenType.TEXT and token_id == EOS_TOKEN_ID:
                break

            sub_seq_end = tok_idx + 1

        # Decode any trailing tokens at the end
        if sub_seq_start != sub_seq_end:
            token_ids = batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end]
            # Trailing padding can be side effect of beam search and formula-only/text-only eval, different conditions for our model and baseline
            if not all(token_ids == PADDING_TOKEN_ID):
                non_padding_idx = token_ids != PADDING_TOKEN_ID
                if is_text:
                    result += text_tokenizer().decode(token_ids[non_padding_idx])
                else:
                    result += decode_formula_and_cycle(
                        token_ids[non_padding_idx],
                        batch["token_types"][seq_idx][sub_seq_start : sub_seq_end][non_padding_idx]
                    )

        result = re.sub(r" +", " ", result)
        cur_seq["text"] = result
        all_decoded_sequences.append(result)

    return all_decoded_sequences
