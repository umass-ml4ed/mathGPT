from typing import List, Optional, Union
import math
from functools import lru_cache
import torch
import numpy as np

from vocabulary import Vocabulary, get_matrix_symbol
from data_types import OPT, Sequence
from utils import TrainOptions
from constants import TokenType, SpecialOpToken, SpecialVarToken, TPE, MAX_FORMULA_DEPTH, MAX_FORMULA_WIDTH, EMB_SIZE

EMPTY_POS_VECTOR = [0] * MAX_FORMULA_DEPTH

POS_ENCODING_REGION_NUM_BITS = math.ceil(math.log2(MAX_FORMULA_WIDTH))
POS_ENCODING_SIZE_FORTE = MAX_FORMULA_DEPTH * POS_ENCODING_REGION_NUM_BITS * 2

_EMPTY_POS_ENCODING_FORTE = [0] * POS_ENCODING_SIZE_FORTE
_EMPTY_POS_ENCODING_SIN = [0] * EMB_SIZE
_EMPTY_POS_ENCODING_RNN = [[0] * MAX_FORMULA_WIDTH] * MAX_FORMULA_DEPTH

def get_empty_pos_encoding(tpe: str):
    if tpe == TPE.FORTE.value:
        return _EMPTY_POS_ENCODING_FORTE
    if tpe in (TPE.SIN_ADD.value, TPE.SIN_PART.value):
        return _EMPTY_POS_ENCODING_SIN
    return _EMPTY_POS_ENCODING_RNN

def encode_pos_forte(pos_vec: Union[List[int], torch.Tensor], pos_level: int):
    """
    Return encoding of the tree position
    Encoding is multi-hot vector, broken into per-level regions
    Each region contains an encoding of the child position at that level
    This starts with the binary representation of the position, and then converts each bit into a one-hot 2-vector
    """
    # TODO: see if encoding can have bit/bool type to save memory
    encoding = _EMPTY_POS_ENCODING_FORTE.copy()
    for level in range(pos_level + 1):
        region_start_idx = level * (POS_ENCODING_REGION_NUM_BITS * 2)
        for bit in range(POS_ENCODING_REGION_NUM_BITS):
            bit_idx = region_start_idx + bit * 2 # Start of 2-vector representing the child position bit at this level
            active_bit = (pos_vec[level] >> bit) & 1 # 0 or 1, representing this bit of the child position value
            encoding[bit_idx + active_bit] = 1 # Activate either 0 or 1 value for this bit
    return encoding

@lru_cache
def sin_encoding_mat(dim: int, max_seq_len: int):
    """
    Get sin position encoding matrix for the given dimensionality
    Code based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    pos(k, 2i) = sin(k / 10000 ^ (2i / d))
    pos(k, 2i + 1) = cos(k / 10000 ^ (2i / d))
    """
    positions = np.arange(max_seq_len).reshape(max_seq_len, 1)
    div_term = np.exp(np.arange(0, dim, 2) * -np.log(10000) / dim) # Using exp and log since numerically more stable than direct computation
    encodings = np.zeros((max_seq_len, dim))
    encodings[:, 0::2] = np.sin(positions * div_term)
    encodings[:, 1::2] = np.cos(positions * div_term)
    return encodings

def encode_pos_sin_part(pos_vec: Union[List[int], torch.Tensor], pos_level: int):
    """
    Return a vector that contains the concatenation of the sinusoidal position encodings for each level in the tree
    """
    encoding_mat = sin_encoding_mat(EMB_SIZE // MAX_FORMULA_DEPTH, 1024) # TODO: use options for max_seq_len
    encoding = np.concatenate([encoding_mat[pos_vec[level]] for level in range(pos_level + 1)])
    return np.pad(encoding, (0, EMB_SIZE - encoding.shape[0])).tolist()

def encode_pos_sin_add(pos_vec: Union[List[int], torch.Tensor], pos_level: int):
    """
    Return a vector that contains the sum of the sinusoidal position encodings for each level in the tree
    """
    encoding_mat = sin_encoding_mat(EMB_SIZE, 1024) # TODO: use options for max_seq_len
    return np.sum([encoding_mat[pos_vec[level]] for level in range(pos_level + 1)], axis=0).tolist()

@lru_cache
def one_hot_matrix():
    """
    A matrix containing one-hot encodings up to the max formula width
    """
    return np.eye(MAX_FORMULA_WIDTH).tolist()

def encode_pos_rnn(pos_vec: Union[List[int], torch.Tensor], pos_level: int):
    """
    Return a 2D matrix where each row is a one-hot encoding of the child position at that level, padded with 0 vectors
    """
    one_hot_mat = one_hot_matrix()
    return [one_hot_mat[pos_vec[level]] for level in range(pos_level + 1)] + [[0] * MAX_FORMULA_WIDTH] * (MAX_FORMULA_DEPTH - pos_level - 1)

def encode_pos(pos_vec: Union[List[int], torch.Tensor], pos_level: int, tpe: str):
    """
    Return the position encoding of a single tree node
    """
    if tpe == TPE.FORTE.value:
        return encode_pos_forte(pos_vec, pos_level)
    if tpe == TPE.SIN_PART.value:
        return encode_pos_sin_part(pos_vec, pos_level)
    if tpe == TPE.SIN_ADD.value:
        return encode_pos_sin_add(pos_vec, pos_level)
    if tpe == TPE.RNN.value:
        return encode_pos_rnn(pos_vec, pos_level)

def tokenize_formula_rec(formula: Optional[OPT], parent_pos: List[int], cur_level: int, cur_child_num: int, options: TrainOptions,
                         sequence: Sequence) -> bool:
    """
    Recursive helper for OPT tokenization, add info for current head and then process children.
    Also do data post-processing.
    If formula is None, interpret as END token.
    Returns if the provided node got added to the running list (was not skipped).
    """

    children = formula and formula[2]

    # First resolve token type and ID - check for end token, then try post-processing rules, then retrieve from vocab
    token_type = None
    if not formula:
        token_type, token_id = TokenType.END, 0
    else:
        # Resolve type and token id from symbol, post-processing rules follow
        type_str, symbol = formula[:2]

        # TODO: should single-character numbers still have the OP head?
        if options.num_to_tree and type_str == "N" and len(symbol) > 1:
            # Convert numbers to sub-trees:
            # Insert a special num op token and set its children to the digits of the number
            token_type, token_id = TokenType.OP, SpecialOpToken.NUM_SUB_TREE_HEAD.value
            children = [["N", char, None] for char in symbol][:MAX_FORMULA_WIDTH - 1]

        elif type_str == "+":
            # Convert all "+" symbols to anonymous operator.
            # TangentCFT will assign the left grandchild, with a sub-type, as the symbol.
            # We'll assume the model doesn't need this hint and can discover the relation via attention.
            token_type, token_id = TokenType.OP, SpecialOpToken.ANON_OP.value

        elif type_str == "E":
            if children:
                # An "E" type with children indicates a cerror tag from the MathML source, which represents a semantic error.
                # Its first child is an identifier for the type of error.
                # Remove this identifier, convert the E tag to special operator, and keep its remaining children.
                children = children[1:]
                token_type, token_id = TokenType.OP, SpecialOpToken.CERR_OP.value
            else:
                # An "E" type with no children indicates a padding value inserted by TangentCFT to make matrices rectangular.
                # Convert to special padding value.
                token_type, token_id = TokenType.VAR, SpecialVarToken.MAT_PAD.value

        elif type_str == "M":
            # Special processing for matrix symbols
            symbol = get_matrix_symbol(symbol)

        # Look up symbol's token type and ID from vocab if not already assigned by post-processing rules
        if token_type is None:
            token_type, token_id = Vocabulary.get_token(type_str, symbol, assign_new=True)

    # Set position
    # TODO: to avoid max depth limit, try just concatting to list and then padding in batch
    #       but think about how this affects generation, might have to rescale vecs on the fly
    pos_vec = parent_pos.copy()
    pos_vec[cur_level] = cur_child_num

    # Add details for current token
    sequence.token_ids.append(token_id)
    sequence.token_types.append(token_type)
    sequence.pos_vecs.append(pos_vec)
    sequence.pos_levels.append(cur_level)
    # sequence.pos_encodings.append(encode_pos(pos_vec, cur_level, tpe))

    # Process children
    child_idx = 0
    if children:
        for child in children:
            if tokenize_formula_rec(child, pos_vec, cur_level + 1, child_idx, options, sequence):
                child_idx += 1

    # Append END token as last child for all OP tokens
    if token_type == TokenType.OP:
        tokenize_formula_rec(None, pos_vec, cur_level + 1, child_idx, options, sequence)

    return True

def tokenize_formula(formula: OPT, options: TrainOptions):
    """
    Given a formula OPT, return in DFS order the token ids, types, and position encodings
    """
    sequence = Sequence("")
    tokenize_formula_rec(formula, EMPTY_POS_VECTOR, 0, 0, options, sequence)
    return sequence
