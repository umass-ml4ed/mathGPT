from typing import Dict, Optional, List, Tuple
import os
import re
import json
from tqdm import tqdm
import pandas

from TangentCFT.TangentS.math_tan.math_document import MathDocument
from mathGPT.data_types import Article

from pre_process_utils import process_article, process_raw_text, html_to_latex, wrap_formulas, all_latexml_errs, all_tangent_cft_errs
from vocabulary import Vocabulary
from data_types import GenTaskSample, AnswerScoringSample
from constants import FORMULA_IDENTIFIER, DATA, WIKI_DATA, AS_PROBLEMS, AS_ANSWERS

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

    # Do tree conversion on all problems and store in lookup file
    print("Final processing on problem text...")
    raw_problems: List[Tuple[int, str]] = list({(row["problem_id"], row["full_problem_latex"]) for _, row in df.iterrows()})
    if batch_size:
        problems: Dict[int, Article] = {}
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
    with open(AS_PROBLEMS, "w", encoding="utf-8") as problem_file:
        json.dump(problems, problem_file, indent=2, ensure_ascii=False)

    # Do tree conversion on all answers and save related data
    print("Final processing on answer text...")
    samples: List[AnswerScoringSample] = []
    if batch_size:
        for batch_start_idx in tqdm(list(range(0, df.shape[0], batch_size))):
            cur_batch = df["answer_latex"].iloc[batch_start_idx : batch_start_idx + batch_size]
            processed_batch = process_raw_text(cur_batch, err_data)
            samples += [{
                "answer": processed_answer,
                "problem_id": int(df["problem_id"].iloc[batch_start_idx + idx]),
                "problem_log_id": int(df["problem_log_id"].iloc[batch_start_idx + idx]),
                "grade": int(df["grade"].iloc[batch_start_idx + idx]) - 1,
            } for idx, processed_answer in enumerate(processed_batch)]
    else:
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            samples.append({
                "answer": process_raw_text([row["answer_latex"]], err_data)[0],
                "problem_id": row["problem_id"],
                "problem_log_id": row["problem_log_id"],
                "grade": row["grade"] - 1,
            })
    with open(AS_ANSWERS, "w", encoding="utf-8") as answer_file:
        json.dump(samples, answer_file, indent=2, ensure_ascii=False)

    # Dump errors
    with open("answer_scoring_errs.json", "w", encoding="utf-8") as err_file:
        json.dump({
            **err_data,
            "all_latexml_errs": all_latexml_errs,
            "all_tangent_cft_errs": all_tangent_cft_errs,
        }, err_file, indent=2, ensure_ascii=False)
