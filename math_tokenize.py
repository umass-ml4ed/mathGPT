from typing import List, Optional

from vocabulary import Vocabulary
from constants import OPT, TokenType, Sequence, MAX_FORMULA_DEPTH

EMPTY_POS_VECTOR = [0] * MAX_FORMULA_DEPTH

def encode_pos(pos_str: str):
    # TODO: idea 1 - for each level pos in str, fill in value in int, and then shift by num bits needed for max num children
    #           issue - number will get huge, embeddings will be very sparse
    #       idea 2 - store mapping from each possible position str to an int, which will be looked up in embedding matrix
    #       idea 3 - store multi-hot vector (using binary idea from FORTE) and then model can use projection to generate embedding
    #       idea 4 - store vector of numbers, and then convert each to a sin/cos representations (width dependent on max_depth, max_width, and embed_size) and concat encodings
    #       idea 5 - each level gets its own sin/cos representation (based on child pos), and add the representations of each level together
    #       idea 6 - train an RNN to generate the embedding, takes child position learnable embeddings as input, goes from top to bottom
    #       note (for all) - for multiple formulas in a sequence, some positions will repeat. do we need to indicate to model exactly which formula is referenced?
    return pos_str

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
    pos = parent_pos.copy()
    pos[cur_level] = cur_child_num

    # Add details for current token
    sequence.token_ids.append(symbol_token)
    sequence.token_types.append(token_type)
    sequence.pos_vecs.append(pos)
    sequence.pos_levels.append(cur_level)

    if not formula:
        return

    # Process children
    if formula[2]:
        for child_idx, child in enumerate(formula[2]):
            tokenize_formula_rec(child, pos, cur_level + 1, child_idx, sequence)

    # Append END token as last child for all OP tokens
    if token_type == TokenType.OP:
        tokenize_formula_rec(None, pos, cur_level + 1, len(formula[2]) if formula[2] else 0, sequence)


def tokenize_formula(formula: OPT):
    """
    Given a formula OPT, return in DFS order the token ids, types, and position encodings
    """
    sequence = Sequence()
    tokenize_formula_rec(formula, EMPTY_POS_VECTOR, 0, 0, sequence)
    return sequence
