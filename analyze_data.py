import os
import json
from typing import List, Dict
from tqdm import tqdm

from constants import Article, OPT, TokenType, TYPE_STR_TO_INT

def analyze_data():
    """
    Gather high-level info on pre-processed data
    """
    type_to_token_to_child_range: Dict[str, Dict[str, List[int]]] = {}
    type_str_to_child_range: Dict[str, List[int]] = {}
    type_to_child_range: Dict[TokenType, List[int]] = {}
    all_error_types = set()
    max_depth = 0
    max_width = 0

    def process_tree(article_name: str, tree_node: OPT, depth: int):
        """
        Update the child range for the given sub-tree, and max depth/width
        """
        nonlocal type_to_token_to_child_range, max_depth, max_width

        num_children = len(tree_node[2]) if tree_node[2] else 0
        max_depth = max(max_depth, depth)
        max_width = max(max_width, num_children)
        token_to_child_range = type_to_token_to_child_range.setdefault(tree_node[0], {})

        if tree_node[0] == "E" and num_children:
            all_error_types.add(tuple(tree_node[2][0][:2]))

        # Update child range for token, type str, and aggregate type
        for key, range_dict in [(tree_node[1], token_to_child_range), (tree_node[0], type_str_to_child_range), (TYPE_STR_TO_INT.get(tree_node[0]), type_to_child_range)]:
            child_range = range_dict.setdefault(key, [num_children, num_children])
            child_range[0] = min(num_children, child_range[0])
            child_range[1] = max(num_children, child_range[1])

        # Process children
        if num_children:
            for child in tree_node[2]:
                if child[0] == "E" and tree_node[0] != "M" and not child[2]:
                    print("Non-mat err w/o children - ", article_name)
                process_tree(article_name,  child, depth + 1)

    # Gather data from each processed article
    for article_name in tqdm(os.listdir("data")):
        article_filepath = os.path.join("data", article_name)
        with open(article_filepath) as article_file:
            article: Article = json.load(article_file)
        for formula in article["formulas"].values():
            process_tree(article_name, formula["opt"], 1)

    # Print results
    for type_str, token_to_child_range in type_to_token_to_child_range.items():
        for token, child_range in token_to_child_range.items():
            if child_range[0] != child_range[1]:
                print(type_str, token, child_range)
    for type_str, child_range in type_str_to_child_range.items():
        print(type_str, child_range)
    for token_type, child_range in type_to_child_range.items():
        print(token_type, child_range)
    print("Error types:")
    print(all_error_types)
    print("Max depth:", max_depth, "Max width:", max_width)
