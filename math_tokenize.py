from typing import List, Optional, Union
import math
import torch

from vocabulary import Vocabulary, get_matrix_symbol
from constants import OPT, TokenType, SpecialOpToken, SpecialVarToken, Sequence, MAX_FORMULA_DEPTH, MAX_FORMULA_WIDTH

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


def tokenize_formula_rec(formula: Optional[OPT], parent_pos: List[int], cur_level: int, cur_child_num: int, sequence: Sequence) -> bool:
    """
    Recursive helper for OPT tokenization, add info for current head and then process children.
    Also do data post-processing.
    If formula is None, interpret as END token.
    Returns if the provided node got added to the running list (was not skipped).
    """

    drop_first_child = False

    # First resolve token type and ID - check for end token, then try post-processing rules, then retrieve from vocab
    token_type = None
    token_id = None
    if not formula:
        token_type, token_id = TokenType.END, 0
    else:
        # Resolve type and token id from symbol, post-processing rules follow
        type_str, symbol = formula[:2]

        if type_str == "+":
            # Convert all "+" symbols to anonymous operator.
            # TangentCFT will assign the left grandchild, with a sub-type, as the symbol.
            # We'll assume the model doesn't need this hint and can discover the relation via attention.
            token_type, token_id = TokenType.OP, SpecialOpToken.ANON_OP.value

        elif type_str == "E":
            if formula[2]:
                # An "E" type with children indicates a cerror tag from the MathML source, which represents a semantic error.
                # Its first child is an identifier for the type of error.
                # Remove this identifier, and convert the E tag to special operator and keep its children.
                drop_first_child = True
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
            token_type, token_id = Vocabulary.get_token(type_str, symbol)

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
    sequence.pos_encodings.append(encode_pos(pos_vec, cur_level))

    # Process children
    child_idx = 0
    if formula and formula[2]:
        children = formula[2][1:] if drop_first_child else formula[2]
        for child in children:
            if tokenize_formula_rec(child, pos_vec, cur_level + 1, child_idx, sequence):
                child_idx += 1

    # Append END token as last child for all OP tokens
    if token_type == TokenType.OP:
        tokenize_formula_rec(None, pos_vec, cur_level + 1, child_idx, sequence)

    return True

def tokenize_formula(formula: OPT):
    """
    Given a formula OPT, return in DFS order the token ids, types, and position encodings
    """
    sequence = Sequence("")
    tokenize_formula_rec(formula, EMPTY_POS_VECTOR, 0, 0, sequence)
    return sequence
