from typing import Dict, Optional, List, Tuple
import os
import re
import json
from subprocess import Popen, PIPE
from tqdm import tqdm
from bs4 import BeautifulSoup

from TangentCFT.TangentS.math_tan.math_document import MathDocument
from TangentCFT.TangentS.math_tan.math_extractor import MathExtractor
from TangentCFT.TangentS.math_tan.semantic_symbol import SemanticSymbol

from vocabulary import Vocabulary
from constants import Formula, OPT, FORMULA_IDENTIFIER, GenTaskSample, Article, DATA, WIKI_DATA

TEX_HEADER = "\\documentclass{article}\n\\usepackage{graphicx}\n\\usepackage{amssymb}\n\\usepackage{amsmath}\n\\usepackage[utf8]{inputenc}\n"

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

def process_article(article_filename: str) -> Article:
    """
    Create a sanitized version of the article with text and formulas separated
    Add all encountered math symbols to the vocab
    Article is in HTML format, and formulas are in MathML format in <math> tags
    Use TangentCFT code (https://github.com/BehroozMansouri/TangentCFT) for initial formula conversion
    """

    # Get raw text data from the article
    _, content = MathDocument.read_doc_file(article_filename)

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

    # Remove all html tags
    soup = BeautifulSoup(text_content, "lxml")
    text_content = soup.get_text()
    text_content = re.sub(r"\s+", " ", text_content)

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

    max_articles = len(article_filenames)
    print("Processing articles...")
    for article_filename in tqdm(article_filenames[:max_articles]):
        article_data = process_article(article_filename)
        out_filename = os.path.basename(article_filename).replace(".html", ".json")
        with open(os.path.join(WIKI_DATA, out_filename), "w", encoding="utf-8") as out_file:
            json.dump(article_data, out_file, indent=2, ensure_ascii=False)

    # Dump vocab to file
    Vocabulary.dump()

def process_raw_text(src_text_batch: List[str], err_data: dict) -> List[Article]:
    """
    Extract text and processed formulas from batch of raw text
    """

    # Extract text and gather formulas
    all_formula_text = ""
    just_text_batch: List[str] = []
    num_formulas_batch: List[int] = []
    for src_text in src_text_batch:
        just_text = ""
        num_formulas = 0
        searchable_text = src_text
        form_start = searchable_text.find("<m>")
        while form_start >= 0:
            just_text += searchable_text[:form_start]
            form_end = searchable_text.find("</m>")
            formula_text = searchable_text[form_start + 3 : form_end]
            if formula_text and not formula_text.isspace(): # There are some empty formulas in the data, remove them here to keep in sync with LaTeXML
                if "\\newcommand" in formula_text: # \newcommand needs to not be in a math formula to be processed
                    all_formula_text += formula_text + "\n"
                else:
                    num_formulas += 1
                    all_formula_text += f"${formula_text}$\n"
                    all_formula_text = all_formula_text.replace(" \\gt ", ">").replace(" \\lt ", "<")
                    just_text += FORMULA_IDENTIFIER
            searchable_text = searchable_text[form_end + 4:]
            form_start = searchable_text.find("<m>")
        just_text += searchable_text or " "
        just_text_batch.append(just_text)
        num_formulas_batch.append(num_formulas)

    # Convert formulas to MathML using LaTeXML
    temp_filename = "temp.tex"
    temp_xml_filename = "temp.xml"
    math_output_filename = "temp.html"
    with open(temp_filename, "w", encoding="utf-8") as temp_file:
        temp_file.write(TEX_HEADER)
        temp_file.write(all_formula_text)

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

    all_formulas: Dict[int, Formula] = {}
    if latexml_success:
        _, content = MathDocument.read_doc_file(math_output_filename)
        if content.count("<math") == sum(num_formulas_batch):
            all_formulas = get_formulas(content)
            err_data["formulas_missing_from_tangentcft"] += sum(num_formulas_batch) - len(all_formulas)
        else:
            import pdb; pdb.set_trace()
            err_data["formulas_missing_from_latexml_randomly"] += sum(num_formulas_batch)
    else:
        err_data["formulas_missing_from_latexml_failure"] += sum(num_formulas_batch)

    # Gather text and formulas for the batch
    formula_start_idx = 0
    articles: List[Article] = []
    for just_text, num_formulas in zip(just_text_batch, num_formulas_batch):
        formulas = {
            (formula_idx - formula_start_idx): all_formulas[formula_idx]
            for formula_idx in range(formula_start_idx, formula_start_idx + num_formulas)
            if formula_idx in all_formulas
        }
        formula_start_idx += num_formulas
        if num_formulas > len(formulas):
            err_data["articles_missing_formulas"] += 1
        articles.append({
            "text": just_text,
            "formulas": formulas
        })
    return articles

def process_mathsum_data():
    """
    Process all data files in the MathSum datasets
    """
    batch_size: Optional[int] = None
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
                    all_posts = post_file.readlines()[:100]
                    all_titles = title_file.readlines()[:100]
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
                            processed_post = process_raw_text([post], cur_err_data)
                            processed_title = process_raw_text([title], cur_err_data)
                            samples.append({
                                "prompt": processed_post,
                                "label": processed_title
                            })

            with open(out_filename, "w", encoding="utf-8") as out_file:
                json.dump(samples, out_file, indent=2, ensure_ascii=False)

    with open("math_sum_errs.json", "w") as err_file:
        json.dump({
            **err_data,
            "all_latexml_errs": all_latexml_errs,
            "all_tangent_cft_errs": all_tangent_cft_errs,
        }, err_file, indent=2)
