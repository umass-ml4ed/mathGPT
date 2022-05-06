from math_tokenize import encode_pos, tokenize_formula, POS_ENCODING_REGION_NUM_BITS
from constants import MAX_FORMULA_DEPTH

def test_encode_pos():
    pos_vec = [0, 1, 5] + ([0] * (MAX_FORMULA_DEPTH - 3))
    pos_level = 2

    encoding = encode_pos(pos_vec, pos_level)

    padding = ([1, 0] * (POS_ENCODING_REGION_NUM_BITS - 3)) # Will use 3 bits to represent numbers below, follow with 0-bit padding
    assert encoding == [1, 0, 1, 0, 1, 0] + padding + [0, 1, 1, 0, 1, 0] + padding + [0, 1, 1, 0, 0, 1] + padding +\
                       [0] * ((MAX_FORMULA_DEPTH - 3) * POS_ENCODING_REGION_NUM_BITS * 2)

def test_tokenize_formula():
    pass # TODO
