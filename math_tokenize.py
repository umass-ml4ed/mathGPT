from typing import List

from vocabulary import Vocabulary
from constants import OPT

# TODO: gather from data or config
max_depth = 10
max_width = 2

def tokenize_formula_rec(formula: OPT, parent_position_str: str, cur_child_num: int, token_ids: List[int], token_types: List[int], positions: List[str]):
    """
    Recursive helper for OPT tokenization, add info for current head and then process children
    """
    token_type, symbol_token = Vocabulary.get_token(formula[0], formula[1])
    position_str = f"{parent_position_str}{cur_child_num}" # TODO: make sure this can handle more than 10 children

    # Add token for current
    token_ids.append(symbol_token)
    token_types.append(token_type)
    positions.append(position_str) # TODO: convert to encoding

    # Process children
    if formula[2]:
        for child_idx, child in enumerate(formula[2]):
            tokenize_formula_rec(child, position_str, child_idx, token_ids, token_types, positions)
    # TODO: append END token as last child (even if no children present)

def tokenize_formula(formula: OPT):
    """
    Given a formula OPT, return in DFS order the token ids, types, and position encodings
    """
    token_ids: List[int] = []
    token_types: List[int] = []
    positions: List[str] = []
    tokenize_formula_rec(formula, "", 0, token_ids, token_types, positions)
    return token_ids, token_types, positions
