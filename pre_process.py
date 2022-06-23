from functools import lru_cache
from typing import Dict, Optional, List
import os
import re
import json
from subprocess import Popen, PIPE
from copy import copy
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas

from TangentCFT.TangentS.math_tan.math_document import MathDocument
from TangentCFT.TangentS.math_tan.math_extractor import MathExtractor
from TangentCFT.TangentS.math_tan.semantic_symbol import SemanticSymbol

from vocabulary import Vocabulary
from constants import Formula, OPT, FORMULA_IDENTIFIER, GenTaskSample, Article, DATA, WIKI_DATA, AS_DATA

TEX_HEADER = "\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{amssymb}\n\\usepackage{amsmath}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{gensymb}\n"
SAMPLE_SEPARATOR = "NEWMGPTSAMPLE"

all_latexml_errs: List[str] = []
all_tangent_cft_errs: Dict[str, int] = {}

def tree_to_serializable(sem_symbol: SemanticSymbol) -> OPT:
    """
    Convert SemanticSymbol object into serializable OPT format
    Also add new symbols to the vocab as the tree is processed
    """
    # Tag has form <type>!<symbol>
    sym_type, symbol = sem_symbol.tag[0], sem_symbol.tag[2:]
    Vocabulary.add(sym_type, symbol)
    return (
        sym_type,
        symbol,
        [tree_to_serializable(child) for child in sem_symbol.children] if sem_symbol.children else None
    )

def isolate_tex(math_tag: str) -> str:
    """
    Extract the LaTeX from a math tag from the raw text
    """
    parsed_xml = BeautifulSoup(math_tag, "lxml")
    math_root = parsed_xml.find("math") # namespaces have been removed (FWT) # TODO: verify
    application_tex = math_root.find("annotation", {"encoding": "application/x-tex"})
    if not application_tex:
        return ""
    return application_tex.text.strip()

def get_formulas(content: str):
    """
    Given text content, return dictionary with all processed formulas
    """
    # Extract all math tags from the article
    formulas: Dict[int, Formula] = {}
    trees = MathExtractor.math_tokens(content)
    for tree_idx, tree in enumerate(trees):
        # Get content Math ML
        cmml: str = MathExtractor.isolate_cmml(tree)
        sem_tree: Optional[SemanticSymbol] = None
        try:
            sem_tree = MathExtractor.convert_to_semanticsymbol(cmml)
        except Exception as exc:
            sem_tree = None
            all_tangent_cft_errs.setdefault(repr(exc), 0)
            all_tangent_cft_errs[repr(exc)] += 1

        # API could return None or have an exception occur, so just skip this formula in those cases
        if not sem_tree:
            continue

        # Add formula
        formulas[tree_idx] = {
            "opt": tree_to_serializable(sem_tree),
            "tex": isolate_tex(tree)
        }
    return formulas

def process_article(content: str) -> Article:
    """
    Create a sanitized version of the article with text and formulas separated
    Add all encountered math symbols to the vocab
    Article is in HTML format, and formulas are in MathML format in <math> tags
    Use TangentCFT code (https://github.com/BehroozMansouri/TangentCFT) for initial formula conversion
    """
    # Get all formulas in the article
    formulas = get_formulas(content)

    # Extract text and replace <math> tags with identifiers
    text_content = ""
    searchable_content = content
    while True:
        # Find the next math tag in the text yet to be searched
        math_tag_loc = MathExtractor.math_pattern.search(searchable_content)
        if not math_tag_loc:
            text_content += searchable_content
            break

        # Add content up to math formula and add formula identifier
        text_content += searchable_content[:math_tag_loc.start()] + FORMULA_IDENTIFIER
        searchable_content = searchable_content[math_tag_loc.end():]

    # Convert HTML to readable text
    soup = BeautifulSoup(text_content, "lxml")
    body = soup.find("body")
    if not body: # Can happen for texts that reduce to whitespace
        return {"text": "", "formulas": formulas}
    selectors_to_delete = [".footnotes", "footer"]
    for selector in selectors_to_delete:
        item_to_delete = body.find(selector)
        if item_to_delete:
            item_to_delete.decompose()
    text_content = body.get_text()
    text_content = re.sub(r" +", " ", text_content)
    text_content = re.sub(r"\n[\s↩]+", "\n", text_content)
    text_content = text_content.strip()

    return {
        "text": text_content,
        "formulas": formulas
    }

def fix_matrix(formula_text: str):
    final_text = formula_text
    match_found = False
    matrix_types = ["matrix", "pmatrix", "vmatrix", "Vmatrix", "bmatrix", "Bmatrix", "smallmatrix"]
    for matrix_type in matrix_types:
        pmat_start = formula_text.find(f"\\{matrix_type}")
        if pmat_start >= 0:
            # Find starting and ending brackets
            bracket_lvl = 0
            open_bracket_idx = formula_text.find("{", pmat_start)
            for idx in range(open_bracket_idx, len(formula_text)):
                if formula_text[idx] == "{":
                    bracket_lvl += 1
                elif formula_text[idx] == "}":
                    bracket_lvl -= 1
                if bracket_lvl == 0:
                    break

            # Replace with begin/end macros
            final_text = formula_text[:pmat_start] +\
                f" \\begin{{{matrix_type}}} " +\
                formula_text[open_bracket_idx + 1 : idx] +\
                f" \\end{{{matrix_type}}} " +\
                formula_text[idx + 1:]
            match_found = True
            break

    # Run check again in case there were multiple matrix elements
    if match_found:
        final_text = fix_matrix(final_text)
    return final_text

def process_raw_text(src_text_batch: List[str], err_data: Optional[dict] = None) -> List[Article]:
    """
    Extract text and processed formulas from batch of raw text
    """

    # Combine batch and convert formulas to viable LaTeX format
    processed_text = ""
    for batch_idx, src_text in enumerate(src_text_batch):
        searchable_text = src_text.strip()
        form_start = searchable_text.find("<m>")
        while form_start >= 0:
            processed_text += searchable_text[:form_start]
            form_end = searchable_text.find("</m>")
            formula_text = searchable_text[form_start + 3 : form_end]
            formula_text = formula_text.replace(" \\gt ", ">").replace(" \\lt ", "<") # Fix unknown macros
            if "\\newcommand" in formula_text: # \newcommand needs to not be in a math formula to be processed
                processed_text += formula_text
            else:
                formula_text = fix_matrix(formula_text)
                processed_text += f"${formula_text}$"
            searchable_text = searchable_text[form_end + 4:]
            form_start = searchable_text.find("<m>")
        processed_text += searchable_text or " "
        if batch_idx != len(src_text_batch) - 1:
            processed_text += f" {SAMPLE_SEPARATOR} "
    processed_text = processed_text.replace("%", "\\%")

    # Convert LaTeX source with LaTeXML
    temp_filename = "temp.tex"
    temp_xml_filename = "temp.xml"
    math_output_filename = "temp.html"
    with open(temp_filename, "w", encoding="utf-8") as temp_file:
        temp_file.write(TEX_HEADER)
        temp_file.write(processed_text)

    remove_italics = True
    latexml_success = True
    if remove_italics:
        proc = Popen(["latexml", "--dest", temp_xml_filename, temp_filename], stdout=PIPE, stderr=PIPE)
        _, errs = proc.communicate()
        if proc.returncode != 0:
            latexml_success = False
            all_latexml_errs.append(errs.decode("utf-8"))
        if latexml_success:
            with open(temp_xml_filename, encoding="utf-8") as temp_xml_file:
                xml_str: str = temp_xml_file.read()
                xml_str = xml_str.replace("font=\"italic\"", "")
            with open(temp_xml_filename, "w", encoding="utf-8") as temp_xml_file:
                temp_xml_file.write(xml_str)
            proc = Popen(["latexmlpost", "--dest", math_output_filename, "--pmml", "--cmml", "--mathtex", temp_xml_filename], stdout=PIPE, stderr=PIPE)
            _, errs = proc.communicate()
            if proc.returncode != 0:
                latexml_success = False
                all_latexml_errs.append(errs.decode("utf-8"))
    else:
        proc = Popen(["latexmlc", "--dest", math_output_filename, "--pmml", "--cmml", "--mathtex", temp_filename], stdout=PIPE, stderr=PIPE)
        _, errs = proc.communicate()
        if proc.returncode != 0:
            latexml_success = False
            all_latexml_errs.append(errs.decode("utf-8"))

    # Extract text and formulas from processed text
    if latexml_success:
        _, all_content = MathDocument.read_doc_file(math_output_filename)
        # Resolve share elements since TangentCFT doesn't handle them
        # TangentCFT doesn't handle matrix structure <apply><csymbol>matrix</csymbol><matrix>...</matrix></apply>, so collapse to <matrix>...</matrix>
        if "<share" in all_content or "matrix</csymbol>" in all_content:
            soup = BeautifulSoup(all_content, "lxml")
            while True:
                share_el = soup.find("share")
                if not share_el:
                    break
                ref_el = soup.find(id=share_el.attrs["href"][1:])
                if not ref_el:
                    share_el.decompose() # Sometimes LaTeXML assigns a ref that doesn't exist, just destroy the element in that case
                else:
                    share_el.replace_with(copy(ref_el))
            while True:
                matrix_symb_el = soup.find("csymbol", string="matrix")
                if not matrix_symb_el:
                    break
                matrix_el = matrix_symb_el.parent.contents[1]
                matrix_symb_el.parent.replace_with(matrix_el)
            all_content = str(soup)
        articles = [process_article(content) for content in all_content.split(SAMPLE_SEPARATOR)]
        for article in articles:
            exp_num_formulas = article["text"].count(FORMULA_IDENTIFIER)
            if exp_num_formulas != len(article["formulas"]):
                err_data["formulas_missing_from_tangentcft"] += exp_num_formulas - len(article["formulas"])
                err_data["articles_missing_formulas"] += 1
        if len(articles) != len(src_text_batch):
            err_data["formulas_missing_from_latexml_randomly"] += 1
            raise Exception("Failed to split batch!")
        return articles
    if err_data:
        err_data["formulas_missing_from_latexml_failure"] += 1
    raise Exception("LaTeXML failed!")

esc_to_latex = [
    ("&times;", "\\times", True),
    ("&minus;", "-", False),
    ("&amp;", "\\&", False),
    ("&mu;", "\\mu", True),
    ("&gt;", ">", False),
    ("&lt;", "<", False),
    ("&epsilon;", "\\epsilon", True),
    ("&ne;", "\\ne", True),
    ("&pi;", "\\pi", True),
    ("&ge;", "\\ge", True),
    ("&le;", "\\le", True),
    ("&divide;", "/", False),
    ("&ang;", "\\angle", True),
    ("&nbsp;", " ", False),
    ("&oacute;", "\\'o", False),
    ("&aacute;", "\\'a", False),
    ("&iquest;", "?`", False),
    ("&trade;", "TM", False),
    ("&lsquo;", "\\lq", False),
    ("&rsquo;", "\\rq", False),
    ("&loz;", "\\lozenge", True),
    ("&forall;", "\\forall", True),
    ("&bull;", "\\textbullet", False),
    ("&ordm;", " ", False),
    ("&spades;", "\\spadesuit", True),
    ("$diamondsuit;", "\\blacklozenge", True),
    ("&sect;", "", False),
    ("&dagger;", "\\textdagger", False),
    ("&empty;", "\\emptyset", True),
    ("&frasl;", "\\textfraction", False),
    ("&quot;", "\"", False),
    ("&ndash;", "-", False),
    ("&darr;", "\\downarrow", True),
    ("&rdquo;", "''", False),
    ("&zwj;", " ", False),
    ("&hearts;", "\\heartsuit", True),
    ("&iacute;", "\\'i", False),
    ("&copy;", "\\copyright", False),
    ("&hellip;", "\\textellipsis", False),
    ("&raquo;", "\\guillemotright", False),
    ("&laquo;", "\\guillemotleft", False),
    ("&acute;", " ", False),
    ("&cent;", "\\textcent", False),
    ("&ouml;", "\\\"o", False),
    ("&equiv;", "\\equiv", True),
    ("&beta;", "\\beta", True),
    ("&exist;", "\\exists", True),
    ("&notin;", "\\notin", True),
    ("&alpha;", "\\alpha", True),
    ("&prime;", "\\prime", True),
    ("&isin;", "\\in", True),
    ("&uml;", " ", False),
    ("&ntilde;", "\\~{n}", False),
    ("&prod;", "\\prod", True),
    ("&ldquo;", "\"", False),
    ("&lowast;", "\\ast", True),
    ("&eacute;", "\\'e", False),
    ("&sdot;", "\\cdot", True),
    ("&clubs;", "\\clubsuit", True),
    ("&mdash;", "-", False),
    ("&deg;", "\\degree", True),
    ("&auml;", "\\\"a", False),
    ("&rarr;", "\\rightarrow", True),
    ("&lfloor;", "\\lfloor", True),
    ("&uarr;", "\\uparrow", True),
    ("&asymp;", "\\approx", True),
    ("&micro;", "\\textmu", False),
    ("&radic;", "\\surd", True),
    ("&or;", "\\vee", True),
    ("&diams;", "\\blacklozenge", True),
]

latex_math_macros = {latex[1] for latex in esc_to_latex if latex[2]}.union({"^", "_"})

math_word_re = re.compile(r"^([0-9]+([\.,][0-9]+)?|\.[0-9]+|(\\_)+|[0-9\.]*[a-zA-Z][0-9\.]*)$")

punctuation_re = re.compile(r"(\s[0-9a-zA-Z\\_]*)([\.,!?:;]+)([a-zA-Z\\_]*(\s|$))")

math_ops = {"=", "<", ">", "+", "-", "*", "/", "(", ")", "\\{", "\\}", "[", "]"}

latex_ops = {"\\times", "\\ne", "\\ge", "\\le", "\\angle", "\\forall", "\\equiv", "\\exists", "\\notin", "\\in", "\\prod", "\\cdot", "\\degree", "\\approx", "\\surd", "\\vee"}

times_re = re.compile(r"\s([0-9]+([\.,][0-9]+)?|\.[0-9]+|\)|[a-zA-Z])\s*x\s*([0-9]+([\.,][0-9]+)?|\.[0-9]+|\(|[a-zA-Z])\s")

@lru_cache(maxsize=1024) # Cache because many entries have the same question text
def html_to_latex(text: str):
    """
    Convert html text to latex, and do additional cleaning
    """
    # Whoever put this as an answer is a genius hacker who knew how to cause latexml to completely freeze up
    if text == "<p>:(((((((((((((((((((((((((((((((((((((((((((((((</p>":
        return ""
    # Remove broken tokens in the text
    broken_tokens = "âð¥¸¶´ð·ð¤§¦"
    for tok in broken_tokens:
        text = text.replace(tok, " ")
    # Escape special latex characters
    text = text.replace("\\", "\\textbackslash")
    text = text.replace("$", "\\$")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = re.sub(r"_+", "\\_", text) # Multiple underscores are often used to represent blanks - just collapse to one
    # Convert html math tags to latex
    text = re.sub(r"<sup>([^<]*)</sup>", r" ^ { \g<1> } ", text)
    text = re.sub(r"<sub>([^<]*)</sub>", r" _ { \g<1> } ", text)
    # TODO: convert table to matrix
    # Convert html escape codes to latex macros
    for esc, latex, _ in esc_to_latex:
        text = text.replace(esc, f" {latex} ")
    # Remove remaining html tags
    text = BeautifulSoup(text, "lxml").get_text()
    # Put spaces around math operators
    for math_op in math_ops:
        text = text.replace(math_op, f" {math_op} ")
    # Put spaces around punctuation
    text = punctuation_re.sub(r"\g<1> \g<2> \g<3>", text)
    # Handle cases where "x" is used for multiplication
    text = times_re.sub(r" \g<1> \\times \g<3> ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove unnecessary text
    text = text.replace("Modified from EngageNY \\copyright Great Minds Disclaimer", "")
    text = text.replace("Copied for free from openupresources . org", "")
    text = text.replace("copied for free from openupresources . org", "")
    return text

@lru_cache(maxsize=1024) # Cache because many entries have the same question text
def wrap_formulas(text: str):
    """
    Find all formluas in the text and wrap with indicators
    """
    words = text.split(" ")
    final_text = []
    formula_start = None
    last_word_was_op = False
    curly_brace_level = 0

    def try_add_formula(start_idx: int, end_idx: int, words: List[str], final_text: List[str]):
        # We have a valid formula if the length is > 1 or if the single word is a latex math macro
        # Also account for common false positive cases
        false_positive = end_idx - start_idx == 2 and ((
            math_word_re.match(words[start_idx]) and words[start_idx + 1] in ("(", ")", "-")
        ) or (
            words[start_idx] in ("(", ")") and math_word_re.match(words[start_idx + 1])
        ))
        if (end_idx > start_idx + 1 or words[start_idx] in latex_math_macros) and not false_positive:
            final_text.append("<m>")
            final_text.extend(words[start_idx : end_idx])
            final_text.append("</m>")
        else:
            final_text.extend(words[start_idx : end_idx])

    for word_idx, word in enumerate(words):
        var_candidate = math_word_re.match(word)
        math_op = word in math_ops or word in ("{", "}") or word in latex_ops
        latex_var = word in latex_math_macros and not math_op
        in_formula = math_op or latex_var or var_candidate

        # Keep track of curly brace level - shouldn't have issues since only inserted from processing sup/sub html tags
        if word == "{":
            curly_brace_level += 1
        elif word == "}":
            curly_brace_level -= 1

        # End the previous formula if the current word is not a math token or if it's the second variable candidate in a row
        # (and curly braces must match, otherwise resulting latex will be invalid)
        end_prev_formula = formula_start is not None and (
            not in_formula or (var_candidate and not last_word_was_op)
        ) and curly_brace_level == 0
        if end_prev_formula:
            try_add_formula(formula_start, word_idx, words, final_text)
            formula_start = None

        # Start a formula if one is not yet going and current word could be in a formula
        if in_formula and formula_start is None:
            formula_start = word_idx

        # If we aren't part of a potential formula, just add the word to the result
        if formula_start is None:
            final_text.append(word)

        # Record if word was an op
        last_word_was_op = math_op

    # If text ended with a formula then terminate it
    if formula_start is not None:
        try_add_formula(formula_start, word_idx + 1, words, final_text)
    return " ".join(final_text)

def process_wikipedia_data():
    """
    Process all data files in the wikipedia dataset
    """
    print("Gathering articles...")
    article_filenames: List[str] = []
    root_dir = "../NTCIR12_MathIR_WikiCorpus_v2.1.0/MathTagArticles"
    for article_group in os.listdir(root_dir):
        article_group_dir = os.path.join(root_dir, article_group, "Articles")
        if not os.path.isdir(article_group_dir):
            continue
        for _, article in enumerate(os.listdir(article_group_dir)):
            article_filename = os.path.join(article_group_dir, article)
            if article_filename.endswith(".html"):
                article_filenames.append(article_filename)

    print("Processing articles...")
    err_data = {
        "articles_missing_formulas": 0,
        "formulas_missing": 0,
    }
    max_articles = len(article_filenames)
    for article_filename in tqdm(article_filenames[:max_articles]):
        _, content = MathDocument.read_doc_file(article_filename)
        article_data = process_article(content)
        form_diff = article_data["text"].count(FORMULA_IDENTIFIER) - len(article_data["formulas"])
        if form_diff > 0:
            err_data["articles_missing_formulas"] += 1
            err_data["formulas_missing"] += form_diff
        out_filename = os.path.basename(article_filename).replace(".html", ".json")
        with open(os.path.join(WIKI_DATA, out_filename), "w", encoding="utf-8") as out_file:
            json.dump(article_data, out_file, indent=2, ensure_ascii=False)

    # Dump vocab to file
    Vocabulary.dump()

    with open("wiki_errs.json", "w") as err_file:
        json.dump({
            **err_data,
            "all_tangent_cft_errs": all_tangent_cft_errs,
        }, err_file, indent=2)

def process_mathsum_data():
    """
    Process all data files in the MathSum datasets
    """
    batch_size: Optional[int] = 20 # 20 seems fastest!
    err_data: Dict[str, Dict[str, int]] = {}
    root_dir = "../MathSum"
    # for dataset in ("EXEQ-300k", "OFEQ-10k"):
    for dataset in ("OFEQ-10k",):
        print("Processing", dataset)
        for split in ("train", "val", "test"):
            print("Processing", split, "split")
            cur_err_data = err_data[f"{dataset},{split}"] = {
                "articles_missing_formulas": 0,
                "formulas_missing_from_latexml_failure": 0,
                "formulas_missing_from_latexml_randomly": 0,
                "formulas_missing_from_tangentcft": 0,
            }
            post_filename = os.path.join(root_dir, dataset, f"post.{split}")
            title_filename = os.path.join(root_dir, dataset, f"title.{split}")
            out_filename = os.path.join(DATA, dataset, f"{split}.json")
            samples: List[GenTaskSample] = []
            with open(post_filename, encoding="utf-8") as post_file:
                with open(title_filename, encoding="utf-8") as title_file:
                    all_posts = post_file.readlines()
                    all_titles = title_file.readlines()
                    # These samples have invalid syntax that breaks LaTeXML
                    if split == "train" and dataset == "OFEQ-10k":
                        erroneous_samples = [5276, 6707]
                        for sample_num, sample_idx in enumerate(erroneous_samples):
                            all_posts = all_posts[:sample_idx - sample_num] + all_posts[sample_idx - sample_num + 1:]
                            all_titles = all_titles[:sample_idx - sample_num] + all_titles[sample_idx - sample_num + 1:]
                    # Batching speeds things up a lot, but causes a single LaTeXML error to ruin the whole batch
                    if batch_size:
                        for batch_start_idx in tqdm(list(range(0, len(all_posts), batch_size))):
                            cur_batch = all_posts[batch_start_idx : batch_start_idx + batch_size] +\
                                all_titles[batch_start_idx : batch_start_idx + batch_size]
                            cur_batch_size = len(cur_batch) // 2
                            processed_batch = process_raw_text(cur_batch, cur_err_data)
                            samples += [{
                                "prompt": processed_batch[idx],
                                "label": processed_batch[idx + cur_batch_size]
                            } for idx in range(cur_batch_size)]
                    else:
                        for post, title in tqdm(list(zip(all_posts, all_titles))):
                            samples.append({
                                "prompt": process_raw_text([post], cur_err_data)[0],
                                "label": process_raw_text([title], cur_err_data)[0]
                            })

            with open(out_filename, "w", encoding="utf-8") as out_file:
                json.dump(samples, out_file, indent=2, ensure_ascii=False)

    with open("mathsum_errs.json", "w") as err_file:
        json.dump({
            **err_data,
            "all_latexml_errs": all_latexml_errs,
            "all_tangent_cft_errs": all_tangent_cft_errs,
        }, err_file, indent=2)

def process_probes():
    with open("data/probes.txt", encoding="utf-8") as src_prompt_file:
        src_probes = src_prompt_file.readlines()
    err_data = {
        "articles_missing_formulas": 0,
        "formulas_missing_from_latexml_failure": 0,
        "formulas_missing_from_latexml_randomly": 0,
        "formulas_missing_from_tangentcft": 0,
    }
    processed_probes = process_raw_text(src_probes, err_data)
    print(err_data)
    print(all_latexml_errs)
    print(all_tangent_cft_errs)
    with open("data/probes.json", "w", encoding="utf-8") as processed_prompt_file:
        json.dump(processed_probes, processed_prompt_file, indent=2, ensure_ascii=False)

def process_answer_scoring_data():
    df = pandas.read_csv("../qc_full_meta_clean.csv", encoding="utf-8")

    # Do some initial analysis on the dataset
    esc_pat = re.compile(r"&[a-z]*;")
    tag_pat = re.compile(r"<[a-z]*[> /]")
    found_escs = set()
    found_tags = set()
    def match(text):
        for match in esc_pat.findall(text):
            found_escs.add(match)
        for match in tag_pat.findall(text):
            found_tags.add(match)
    df["raw_full_problem"].apply(match)
    df["raw_answer_text"].apply(match)
    print("All escs:", found_escs)
    print("All tags:", found_tags)
    print("Grade Range:", df["grade"].min(), "-", df["grade"].max())
    print("Num problems:", df["problem_id"].unique().size)
    print("Num problem logs:", df["problem_log_id"].unique().size)

    # Convert problem and answer html to latex, which includes identifying and wrapping formulas
    print("Coverting problem HTML...")
    df["full_problem_latex"] = df["raw_full_problem"].apply(html_to_latex)
    print("Extracting problem formulas...")
    df["full_problem_latex"] = df["full_problem_latex"].apply(wrap_formulas)
    print("Coverting answer HTML...")
    df["answer_latex"] = df["raw_answer_text"].apply(html_to_latex)
    print("Extracting answer formulas...")
    df["answer_latex"] = df["answer_latex"].apply(wrap_formulas)

    err_data = {
        "articles_missing_formulas": 0,
        "formulas_missing_from_latexml_failure": 0,
        "formulas_missing_from_latexml_randomly": 0,
        "formulas_missing_from_tangentcft": 0,
    }
    batch_size = 100
    # batch_size = None

    # Do tree conversion on all problems and store in lookup file
    print("Final processing on problem text...")
    raw_problems = list({(row["problem_id"], row["full_problem_latex"]) for _, row in df.iterrows()})
    if batch_size:
        problems = {}
        for batch_start_idx in tqdm(list(range(0, len(raw_problems), batch_size))):
            cur_batch = raw_problems[batch_start_idx : batch_start_idx + batch_size]
            processed_batch = process_raw_text([problem_latex for _, problem_latex in cur_batch], err_data)
            for idx, (problem_id, _) in enumerate(cur_batch):
                problems[problem_id] = processed_batch[idx]
    else:
        problems = {
            problem_id: process_raw_text([problem_latex], err_data)[0]
            for problem_id, problem_latex in tqdm(raw_problems)
        }
    with open(os.path.join(AS_DATA, "problems.json"), "w", encoding="utf-8") as problem_file:
        json.dump(problems, problem_file, indent=2, ensure_ascii=False)

    # Do tree conversion on all answers and save related data
    print("Final processing on answer text...")
    samples = []
    if batch_size:
        for batch_start_idx in tqdm(list(range(0, df.shape[0], batch_size))):
            cur_batch = df["answer_latex"].iloc[batch_start_idx : batch_start_idx + batch_size]
            processed_batch = process_raw_text(cur_batch, err_data)
            samples += [{
                "answer": processed_answer,
                "problem_id": int(df["problem_id"].iloc[batch_start_idx + idx]),
                "problem_log_id": int(df["problem_log_id"].iloc[batch_start_idx + idx]),
                "grade": int(df["grade"].iloc[batch_start_idx + idx]),
            } for idx, processed_answer in enumerate(processed_batch)]
    else:
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            samples.append({
                "answer": process_raw_text([row["answer_latex"]], err_data)[0],
                "problem_id": row["problem_id"],
                "problem_log_id": row["problem_log_id"],
                "grade": row["grade"],
            })
    with open(os.path.join(AS_DATA, "answers.json"), "w", encoding="utf-8") as answer_file:
        json.dump(samples, answer_file, indent=2, ensure_ascii=False)

    # Dump errors
    with open("answer_scoring_errs.json", "w", encoding="utf-8") as err_file:
        json.dump(err_data, err_file, indent=2, ensure_ascii=False)
