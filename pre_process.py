from typing import Dict, Optional, List
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
        except:
            sem_tree = None

        # API could return None or have an exception occur, so just skip this formula in those cases
        if not sem_tree:
            continue

        # Add formula
        formulas[tree_idx] = {
            "opt": tree_to_serializable(sem_tree),
            "tex": isolate_tex(tree) # TODO: might be easier to extract from pmml, already split into tokens
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
        with open(os.path.join(WIKI_DATA, out_filename), "w") as out_file:
            json.dump(article_data, out_file, indent=2)

    # Dump vocab to file
    Vocabulary.dump()

def process_raw_text(src_text: str) -> Article:
    """
    Extract text and processed formulas from raw text
    """
    # Separate text and formulas, insert separator tokens
    just_text = ""
    formula_text = ""
    searchable_text = src_text
    form_start = searchable_text.find("<m>")
    while form_start >= 0:
        form_end = searchable_text.find("</m>")
        formula_text += f"${searchable_text[form_start + 3 : form_end]}$ "
        just_text += searchable_text[:form_start] + FORMULA_IDENTIFIER
        searchable_text = searchable_text[form_end + 4:]
        form_start = searchable_text.find("<m>")
    just_text += searchable_text or " "

    # Convert formulas to MathML using LaTeXML
    temp_filename = "temp.tex"
    math_output_filename = "test.html"
    with open(temp_filename, "w") as temp_file:
        temp_file.write(formula_text)
    Popen(["latexmlc", "--dest", math_output_filename, "--pmml", "--cmml", "--mathtex", temp_filename], stdout=PIPE, stderr=PIPE)
    _, content = MathDocument.read_doc_file(math_output_filename)
    formulas = get_formulas(content)

    return {
        "text": just_text,
        "formulas": formulas
    }

def process_mathsum_data():
    """
    Process all data files in the MathSum datasets
    """
    root_dir = "../MathSum"
    for dataset in ("EXEQ-300k", "OFEQ-10k"):
        print("Processing", dataset)
        for split in ("train", "val", "test"):
            print("Processing", split, "split")
            post_filename = os.path.join(root_dir, dataset, f"post.{split}")
            title_filename = os.path.join(root_dir, dataset, f"title.{split}")
            out_filename = os.path.join(DATA, dataset, f"{split}.json")
            samples: List[GenTaskSample] = []
            with open(post_filename) as post_file:
                with open(title_filename) as title_file:
                    all_posts = post_file.readlines()[:100]
                    all_titles = title_file.readlines()[:100]
                    for post, title in tqdm(list(zip(all_posts, all_titles))):
                        samples.append({
                            "prompt": process_raw_text(post),
                            "label": process_raw_text(title)
                        })

            with open(out_filename, "w") as out_file:
                json.dump(samples, out_file, indent=2)
