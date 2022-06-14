from typing import Dict, Optional, List, Tuple
import os
import re
import json
from subprocess import Popen, PIPE
from copy import copy
from tqdm import tqdm
from bs4 import BeautifulSoup

from TangentCFT.TangentS.math_tan.math_document import MathDocument
from TangentCFT.TangentS.math_tan.math_extractor import MathExtractor
from TangentCFT.TangentS.math_tan.semantic_symbol import SemanticSymbol

from vocabulary import Vocabulary
from constants import Formula, OPT, FORMULA_IDENTIFIER, GenTaskSample, Article, DATA, WIKI_DATA

TEX_HEADER = "\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{amssymb}\n\\usepackage{amsmath}\n\\usepackage[utf8]{inputenc}\n"
SAMPLE_SEPARATOR = "NEW-SAMPLE"

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
