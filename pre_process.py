from typing import Dict, Optional
import os
import re
import json
from tqdm import tqdm

from bs4 import BeautifulSoup

from TangentCFT.TangentS.math_tan.math_document import MathDocument
from TangentCFT.TangentS.math_tan.math_extractor import MathExtractor
from TangentCFT.TangentS.math_tan.semantic_symbol import SemanticSymbol

from vocabulary import Vocabulary
from constants import Formula, OPT

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

def process_article(article_filename: str):
    """
    Create a sanitized version of the article with text and formulas separated
    Add all encountered math symbols to the vocab
    """

    # Get raw text data from the article
    _, content = MathDocument.read_doc_file(article_filename)

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
        # TODO: check for 'E!' in equation text, indicating error

    # Extract text and replace <math> tags with identifiers
    text_content = ""
    searchable_content = content
    cur_math_idx = 0
    while True:
        # Find the next math tag in the text yet to be searched
        math_tag_loc = MathExtractor.math_pattern.search(searchable_content)
        if not math_tag_loc:
            text_content += searchable_content
            break

        # Add content up to math formula and add formula identifier
        # TODO: use more unique identifier to avoid possible false positives in articles
        text_content += searchable_content[:math_tag_loc.start()] + f"[{cur_math_idx}]"
        searchable_content = searchable_content[math_tag_loc.end():]
        cur_math_idx += 1

    # Remove all html tags
    soup = BeautifulSoup(text_content, "lxml")
    text_content = soup.get_text()
    text_content = re.sub(r"\s+", " ", text_content)

    # Dump content to file
    out_filename = os.path.basename(article_filename).replace(".html", ".json")
    with open(f"data/{out_filename}", "w") as out_file:
        json.dump({
            "text": text_content,
            "formulas": formulas
        }, out_file, indent=2)

def process_wikipedia_data():
    """
    Process all data files in the wikipedia dataset
    """
    print("Gathering articles...")
    article_filenames = []
    root_dir = "../NTCIR12_MathIR_WikiCorpus_v2.1.0/MathTagArticles"
    for article_group in os.listdir(root_dir):
        article_group_dir = os.path.join(root_dir, article_group, "Articles")
        if not os.path.isdir(article_group_dir):
            continue
        for _, article in enumerate(os.listdir(article_group_dir)):
            article_filename = os.path.join(article_group_dir, article)
            if article_filename.endswith(".html"):
                article_filenames.append(article_filename)

    max_articles = 10000
    print("Processing articles...")
    for article_filename in tqdm(article_filenames[:max_articles]):
        process_article(article_filename)

    # Dump vocab to file
    Vocabulary.dump()
