import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from constants import Article, OPT, TYPE_STR_TO_INT, WIKI_DATA

START_PARENS = ("normal-(", "normal-[", "normal-{")
END_PARENS = ("normal-)", "normal-]", "normal-}")

def process_tree(article_name: str, tree_node: OPT, depth: int, err_found: bool, cat_err_found: bool, stats: dict):
    """
    Update the child range for the given sub-tree, and max depth/width
    """
    num_children = len(tree_node[2]) if tree_node[2] else 0
    stats["max_depth"] = max(stats["max_depth"], depth)
    stats["max_width"] = max(stats["max_width"], num_children)
    token_to_child_range = stats["type_to_token_to_child_range"].setdefault(tree_node[0], {})

    token_to_freq = stats["type_to_token_to_freq"].setdefault(tree_node[0], {})
    token_to_freq.setdefault(tree_node[1], 0)
    token_to_freq[tree_node[1]] += 1

    if num_children:
        stats["num_ops"] += 1

    if tree_node[0] == "T":
        stats["num_text"] += 1
        if num_children:
            stats["num_text_ops"] += 1

    if tree_node[0] == "+":
        stats["num_anon_ops"] += 1
        if tree_node[2][0][1] in ("SUB", "SUP"):
            stats["num_anon_ops_with_type"] += 1

    err_matched = False

    # Check for err case 1
    if num_children > 1:
        for child_idx in range(len(tree_node[2]) - 1):
            e_node = tree_node[2][child_idx + 1]
            if e_node[0] == "E" and e_node[2] and len(e_node[2]) > 3 and e_node[2][0][1] == "fragments" and e_node[2][1][1] in START_PARENS and e_node[2][-1][1] in END_PARENS:
                # Unset E type of child to avoid double counting
                # import pdb; pdb.set_trace()
                e_node[0] = "E_no_more"
                err_found = True
                stats["num_err_ops"] += 1
                stats["num_err_case_1"] += 1
                err_matched = True

    if tree_node[0] == "E" and num_children:
        err_found = True
        stats["num_err_ops"] += 1
        stats["all_error_types"].add(tuple(tree_node[2][0][:2]))

        # Check for err case 2
        if num_children > 3 and tree_node[2][0][1] == "fragments":
            num_ops = 0
            for child in tree_node[2][2:-1]:
                if child[0] in ("U", "O") and not child[2]:
                    num_ops += 1
            if num_ops > 0:
                stats["num_err_case_2"] += 1
                err_matched = True

        # Check for err case 3
        if not err_matched and num_children > 3 and tree_node[2][0][1] == "fragments":
            if tree_node[2][1][1] in START_PARENS and tree_node[2][-1][1] in END_PARENS:
                # import pdb; pdb.set_trace()
                stats["num_err_case_3"] += 1
                err_matched = True

        # if not err_matched:
        #     import pdb; pdb.set_trace()

    if err_matched:
        cat_err_found = True

    # Update child range for token, type str, and aggregate type
    for key, range_dict in [
        (tree_node[1], token_to_child_range),
        (tree_node[0], stats["type_str_to_child_range"]),
        (TYPE_STR_TO_INT.get(tree_node[0]), stats["type_to_child_range"])
    ]:
        child_range = range_dict.setdefault(key, [num_children, num_children])
        child_range[0] = min(num_children, child_range[0])
        child_range[1] = max(num_children, child_range[1])

    # Process children
    if num_children:
        for child in tree_node[2]:
            if child[0] == "E" and tree_node[0] != "M" and not child[2]:
                print("Non-mat err w/o children - ", article_name)
            child_err_found, child_cat_err_found = process_tree(article_name, child, depth + 1, err_found, cat_err_found, stats)
            err_found = err_found or child_err_found
            cat_err_found = cat_err_found or child_cat_err_found

    return err_found, cat_err_found

def analyze_data():
    """
    Gather high-level info on pre-processed data
    """
    stats = {
        "type_to_token_to_child_range": {},
        "type_str_to_child_range": {},
        "type_to_child_range": {},
        "all_error_types": set(),
        "type_to_token_to_freq": {},
        "max_depth": 0,
        "max_width": 0,
        "num_formulas": 0,
        "num_formulas_with_err": 0,
        "num_formulas_with_cat_err": 0,
        "num_ops": 0,
        "num_anon_ops": 0,
        "num_anon_ops_with_type": 0,
        "num_err_ops": 0,
        "num_err_case_1": 0,
        "num_err_case_2": 0,
        "num_err_case_3": 0,
        "num_text": 0,
        "num_text_ops": 0,
    }

    # Gather data from each processed article
    for article_name in tqdm(os.listdir(WIKI_DATA)):
        article_filepath = os.path.join(WIKI_DATA, article_name)
        with open(article_filepath) as article_file:
            article: Article = json.load(article_file)
        for formula in article["formulas"].values():
            stats["num_formulas"] += 1
            has_err, has_cat_err = process_tree(article_name, formula["opt"], 1, False, False, stats)
            if has_err:
                stats["num_formulas_with_err"] += 1
            if has_cat_err:
                stats["num_formulas_with_cat_err"] += 1

    # Print results
    # for type_str, token_to_child_range in type_to_token_to_child_range.items():
    #     for token, child_range in token_to_child_range.items():
    #         if child_range[0] != child_range[1]:
    #             print(type_str, token, child_range)
    for type_str, child_range in stats["type_str_to_child_range"].items():
        print(type_str, child_range, sum(stats["type_to_token_to_freq"][type_str].values()))
    for token_type, child_range in stats["type_to_child_range"].items():
        print(token_type, child_range)
    print("Error types:")
    print(stats["all_error_types"])
    print("Num formulas:", stats["num_formulas"], "with err:", stats["num_formulas_with_err"], "with cat err:", stats["num_formulas_with_cat_err"])
    print("Num ops:", stats["num_ops"])
    print("Num anon ops:", stats['num_anon_ops'], "with type:", stats['num_anon_ops_with_type'])
    print("Num err ops:", stats['num_err_ops'], "case 1:", stats['num_err_case_1'], "case 2:", stats['num_err_case_2'], "case 3:", stats['num_err_case_3'])
    print("Num text tokens:", stats["num_text"], "with children:", stats["num_text_ops"])
    print("Max depth:", stats["max_depth"], "Max width:", stats["max_width"])

    # For relevant types, plot n most frequent types against portion of nodes covered by those types
    for type_str in ["N", "T", "V", "-", "O", "F"]:
        frequencies = sorted(stats["type_to_token_to_freq"][type_str].values(), reverse=True)
        plt.plot(list(range(len(frequencies))), np.cumsum(frequencies) / sum(frequencies))
        plt.title(f"Frequency CDF for {type_str} type")
        plt.show()
