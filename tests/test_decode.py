import torch

from decode import DecodeTreeNode, get_tree
from constants import TokenType

def decode_tree_node_to_array(tree_node: DecodeTreeNode):
    return [tree_node.token_type, tree_node.token_id, [decode_tree_node_to_array(child) for child in tree_node.children]]

def test_get_tree():
    token_ids = torch.LongTensor([1, 2, 3, 4, 0, 5, 0])
    token_types = torch.LongTensor([TokenType.OP, TokenType.OP, TokenType.VAR, TokenType.NUM, TokenType.END, TokenType.VAR, TokenType.END])

    tree = get_tree(token_ids, token_types)

    assert decode_tree_node_to_array(tree) == decode_tree_node_to_array(
        DecodeTreeNode(TokenType.OP, 1, [
            DecodeTreeNode(TokenType.OP, 2, [
                DecodeTreeNode(TokenType.VAR, 3, []),
                DecodeTreeNode(TokenType.NUM, 4, [])
            ]),
            DecodeTreeNode(TokenType.VAR, 5, [])
        ])
    )
