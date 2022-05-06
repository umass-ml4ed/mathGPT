from typing import List, Optional, Union
import math
import torch

from vocabulary import Vocabulary
from constants import OPT, TokenType, Sequence, MAX_FORMULA_DEPTH, MAX_FORMULA_WIDTH

POS_ENCODING_REGION_NUM_BITS = math.ceil(math.log2(MAX_FORMULA_WIDTH))
POS_ENCODING_SIZE = MAX_FORMULA_DEPTH * POS_ENCODING_REGION_NUM_BITS * 2

EMPTY_POS_VECTOR = [0] * MAX_FORMULA_DEPTH
EMPTY_POS_ENCODING = [0] * POS_ENCODING_SIZE

def encode_pos(pos_vec: Union[List[int], torch.Tensor], pos_level: int):
    """
    Return encoding of the tree position
    Encoding is multi-hot vector, broken into per-level regions
    Each region contains an encoding of the child position at that level
    This starts with the binary representation of the position, and then converts each bit into a one-hot 2-vector
    """
    # TODO: see if encoding can have bit/bool type to save memory
    encoding = EMPTY_POS_ENCODING.copy()
    for level in range(pos_level + 1):
        region_start_idx = level * (POS_ENCODING_REGION_NUM_BITS * 2)
        for bit in range(POS_ENCODING_REGION_NUM_BITS):
            bit_idx = region_start_idx + bit * 2 # Start of 2-vector representing the child position bit at this level
            active_bit = (pos_vec[level] >> bit) & 1 # 0 or 1, representing this bit of the child position value
            encoding[bit_idx + active_bit] = 1 # Activate either 0 or 1 value for this bit
    return encoding


def tokenize_formula_rec(formula: Optional[OPT], parent_pos: List[int], cur_level: int, cur_child_num: int, sequence: Sequence):
    """
    Recursive helper for OPT tokenization, add info for current head and then process children
    If formula is None, interpret as END token
    """
    # Resolve type and token id from symbol
    if not formula:
        token_type, symbol_token = TokenType.END, 0
    else:
        token_type, symbol_token = Vocabulary.get_token(formula[0], formula[1])

    # Set position
    # TODO: to avoid max depth limit, try just concatting to list and then padding in batch
    #       but think about how this affects generation, might have to rescale vecs on the fly
    pos_vec = parent_pos.copy()
    pos_vec[cur_level] = cur_child_num

    # Add details for current token
    sequence.token_ids.append(symbol_token)
    sequence.token_types.append(token_type)
    sequence.pos_vecs.append(pos_vec)
    sequence.pos_levels.append(cur_level)
    sequence.pos_encodings.append(encode_pos(pos_vec, cur_level))

    if not formula:
        return

    # Process children
    if formula[2]:
        for child_idx, child in enumerate(formula[2]):
            tokenize_formula_rec(child, pos_vec, cur_level + 1, child_idx, sequence)

    # Append END token as last child for all OP tokens
    if token_type == TokenType.OP:
        tokenize_formula_rec(None, pos_vec, cur_level + 1, len(formula[2]) if formula[2] else 0, sequence)

def tokenize_formula(formula: OPT):
    """
    Given a formula OPT, return in DFS order the token ids, types, and position encodings
    """
    sequence = Sequence("")
    tokenize_formula_rec(formula, EMPTY_POS_VECTOR, 0, 0, sequence)
    return sequence
