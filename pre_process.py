import pickle
from typing import Dict, Optional, List, Tuple, Set
import os
import re
import json
from tqdm import tqdm
import pandas

from pre_process_utils import (
    process_articles, process_raw_text, html_to_latex, wrap_formulas, remove_calculator_annotations, get_boxed_answer,
    all_latexml_errs, all_tangent_cft_errs
)
from vocabulary import Vocabulary
from data_types import Article, GenTaskSample, AnswerScoringSample, FeedbackTaskSample, ProblemSolvingTaskSample, CTTaskSample
from constants import (
    FORMULA_IDENTIFIER, DATA, WIKI_DATA, AS_PROBLEMS, AS_ANSWERS, FEEDBACK_PROBLEMS, FEEDBACK_SAMPLES, GSM8K_DATA, MATH_DATA, MWP_DATA, KHAN_DATA, CT_DATA
)

# Import after pre_process_utils since module overloading done there
from TangentCFT.TangentS.math_tan.math_document import MathDocument

def dump_errs(err_filename: str, err_data: dict):
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", err_filename), "w", encoding="utf-8") as err_file:
        json.dump({
            **err_data,
            "all_latexml_errs": all_latexml_errs,
            "all_tangent_cft_errs": all_tangent_cft_errs,
        }, err_file, indent=2, ensure_ascii=False)

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
        "formulas_missing": 0,
    }
    max_articles = len(article_filenames)
    for article_filename in tqdm(article_filenames[:max_articles]):
        _, content = MathDocument.read_doc_file(article_filename)
        article_data = process_articles(content)[0]
        form_diff = article_data["text"].count(FORMULA_IDENTIFIER) - len(article_data["formulas"])
        if form_diff > 0:
            err_data["articles_missing_formulas"] += 1
            err_data["formulas_missing"] += form_diff
        out_filename = os.path.basename(article_filename).replace(".html", ".json")
        with open(os.path.join(WIKI_DATA, out_filename), "w", encoding="utf-8") as out_file:
            json.dump(article_data, out_file, indent=2, ensure_ascii=False)

    # Dump vocab to file
    Vocabulary.dump()

    dump_errs("wiki_errs.json", err_data)

def process_mathsum_data(dataset: str):
    """
    Process all data files in the MathSum datasets
    """
    batch_size: Optional[int] = 40
    err_data: Dict[str, Dict[str, int]] = {}
    root_dir = "../MathSum"
    print("Processing", dataset)
    for split in ("train", "val", "test"):
        print("Processing", split, "split")
        cur_err_data = err_data[f"{dataset},{split}"] = {}
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
                        try:
                            processed_batch = process_raw_text(cur_batch, cur_err_data)
                            samples += [{
                                "prompt": processed_batch[idx],
                                "label": processed_batch[idx + cur_batch_size]
                            } for idx in range(cur_batch_size)]
                        except Exception:
                            pass
                else:
                    for post, title in tqdm(list(zip(all_posts, all_titles))):
                        samples.append({
                            "prompt": process_raw_text([post], cur_err_data)[0],
                            "label": process_raw_text([title], cur_err_data)[0]
                        })

        with open(out_filename, "w", encoding="utf-8") as out_file:
            json.dump(samples, out_file, indent=2, ensure_ascii=False)

    dump_errs(f"{dataset}_errs.json", err_data)

def process_probes():
    """
    Process all LM probes
    """
    with open("data/probes.txt", encoding="utf-8") as src_prompt_file:
        src_probes = src_prompt_file.readlines()
    err_data = {}
    processed_probes = process_raw_text(src_probes, err_data)
    print(err_data)
    print(all_latexml_errs)
    print(all_tangent_cft_errs)
    with open("data/probes.json", "w", encoding="utf-8") as processed_prompt_file:
        json.dump(processed_probes, processed_prompt_file, indent=2, ensure_ascii=False)

def process_answer_scoring_data():
    """
    Process all data in the answer scoring dataset
    """
    # df = pandas.read_csv("../qc_full_meta_clean.csv", encoding="utf-8")
    df = pandas.read_csv("../qc_clean.csv", encoding="utf-8")
    # df = pandas.read_csv("../before_rasch.csv", encoding="utf-8")

    # Do some initial analysis on the dataset
    esc_pat = re.compile(r"&[#a-z0-9]*;")
    tag_pat = re.compile(r"<[a-z0-9]*[> /]")
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
    print("Num to keep:", sum(df["keep"]))
    print("Total size:", df.shape[0])

    # Convert problem and answer html to latex, which includes identifying and wrapping formulas
    print("Coverting problem HTML...")
    df["full_problem_latex"] = df["raw_full_problem"].apply(html_to_latex)
    print("Extracting problem formulas...")
    df["full_problem_latex"] = df["full_problem_latex"].apply(wrap_formulas)
    print("Coverting answer HTML...")
    df["answer_latex"] = df["raw_answer_text"].apply(html_to_latex)
    print("Extracting answer formulas...")
    df["answer_latex"] = df["answer_latex"].apply(wrap_formulas)

    err_data = {}
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

    dump_errs("answer_scoring_errs.json", err_data)

def process_feedback_data():
    """
    Process all data in the feedback dataset
    """
    df = pandas.read_csv("../common wrong answer feedback with all parts.csv", encoding="utf-8")

    # Do some initial analysis on the dataset
    esc_pat = re.compile(r"&[#a-z0-9]*;")
    tag_pat = re.compile(r"<[a-z0-9]*[> /]")
    found_escs = set()
    found_tags = set()
    def match(text):
        text_str = str(text)
        for match in esc_pat.findall(text_str):
            found_escs.add(match)
        for match in tag_pat.findall(text_str):
            found_tags.add(match)
    for field in ["body", "cwa_1", "cwa_1_feedback", "cwa_2", "cwa_2_feedback", "cwa_3", "cwa_3_feedback"]:
        df[field].apply(match)
    print("All escs:", found_escs)
    print("All tags:", found_tags)
    print("Unique problems:", df["problem_code"].unique().size, "; with sub-parts:", df["problem_id"].unique().size)

    # Create a copy of the csv with post-processed fields
    # df_proc = df.copy()
    # for field in ["body", "cwa_1_feedback", "cwa_2_feedback", "cwa_3_feedback"]:
    #     df_proc[field] = df_proc[field].apply(lambda val: html_to_latex(val) if isinstance(val, str) else val)
    # df_proc.to_csv("../feedback_proc.csv")
    # return

    # Extract all problems, answers and feedback, and do HTML to LaTeX and formula wrapping
    df.sort_values(["problem_code", "problem_part"]) # Ensure that question parts are adjacent and in order
    seen_problems: Set[int] = set()
    skipped: List[int] = []
    pid_ptext: List[Tuple[int, str]] = []
    unprocessed_samples: List[dict] = []
    cur_problem_code = None
    cur_problem_head_text = ""
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Skip repeats in the dataset - they don't have unique answers/feedback
        if row["problem_id"] in seen_problems:
            continue
        seen_problems.add(row["problem_id"])

        # Process current problem part
        raw_problem = str(row["body"]).replace("\\n", " ").replace("\\r", " ")
        processed_problem_text = wrap_formulas(html_to_latex(raw_problem))
        if not processed_problem_text or processed_problem_text.isspace(): # Skip empty problem bodies - can happen with unparseable images
            skipped.append(row["problem_id"])
            continue

        # Start new problem group if necessary, take problem body of first part to be header for remaining parts
        if row["problem_code"] != cur_problem_code:
            cur_problem_code = row["problem_code"]
            cur_problem_head_text = processed_problem_text
            pid_ptext.append((row["problem_id"], processed_problem_text))
        else:
            pid_ptext.append((row["problem_id"], cur_problem_head_text + processed_problem_text))

        # Process answers
        for cwa_field in ["cwa_1", "cwa_2", "cwa_3"]:
            raw_answer = str(row[cwa_field]).replace("\\n", " ").replace("\\r", " ")
            raw_feedback = str(row[cwa_field + "_feedback"]).replace("\\n", " ").replace("\\r", " ")
            if raw_answer in ("null", "nan", "#ERROR!") or raw_feedback == "nan":
                continue
            unprocessed_samples.append({
                "problem_id": row["problem_id"],
                "problem_code": row["problem_code"],
                "answer": wrap_formulas(html_to_latex(raw_answer)),
                "feedback": wrap_formulas(html_to_latex(raw_feedback)),
            })

    # Do batch LaTeXML/TangentCFT processing, write problems and samples to output files
    err_data = {}
    batch_size = 40
    pid_to_problem: Dict[int, Article] = {}
    for batch_start_idx in tqdm(range(0, len(pid_ptext), batch_size)):
        batch = pid_ptext[batch_start_idx : batch_start_idx + batch_size]
        processed_problems = process_raw_text([tup[1] for tup in batch], err_data)
        for tup, problem in zip(batch, processed_problems):
            pid_to_problem[tup[0]] = problem
    with open(FEEDBACK_PROBLEMS, "w", encoding="utf-8") as problem_file:
        json.dump(pid_to_problem, problem_file, indent=2, ensure_ascii=False)

    samples: List[FeedbackTaskSample] = []
    for batch_start_idx in tqdm(range(0, len(unprocessed_samples), batch_size)):
        batch = unprocessed_samples[batch_start_idx : batch_start_idx + batch_size]
        processed_answers = process_raw_text([sample["answer"] for sample in batch], err_data)
        processed_feedback = process_raw_text([sample["feedback"] for sample in batch], err_data)
        for sample, answer, feedback in zip(batch, processed_answers, processed_feedback):
            samples.append({
                "problem_id": str(sample["problem_id"]),
                "problem_code": str(sample["problem_code"]),
                "answer": answer,
                "feedback": feedback,
            })
    with open(FEEDBACK_SAMPLES, "w", encoding="utf-8") as sample_file:
        json.dump(samples, sample_file, indent=2, ensure_ascii=False)

    print("Skipped", skipped)
    dump_errs("feedback_errs.json", err_data)

def process_gsm8k_data():
    """
    Process all data in the GSM8K dataset
    """
    err_data = {}
    for split in ("train", "test"):
        # Extract all questions/steps/answers from the split
        batch_text = []
        with open(f"../grade-school-math/grade_school_math/data/{split}.jsonl", encoding="utf-8") as src_file:
            for src_line in tqdm(src_file):
                sample = json.loads(src_line)
                batch_text.append(wrap_formulas(html_to_latex(remove_calculator_annotations(sample["question"]))))
                steps, answer = sample["answer"].split("\n####")
                batch_text.append(wrap_formulas(html_to_latex(remove_calculator_annotations(steps))))
                batch_text.append(wrap_formulas(html_to_latex(remove_calculator_annotations(answer))))

        # Batch process LaTeXML/TangentCFT
        batch_size = 30
        samples: List[ProblemSolvingTaskSample] = []
        for batch_start_idx in tqdm(range(0, len(batch_text), batch_size * 3)):
            processed_text = process_raw_text(batch_text[batch_start_idx : batch_start_idx + batch_size * 3], err_data)
            for sample_idx in range(0, len(processed_text), 3):
                samples.append({
                    "problem": processed_text[sample_idx],
                    "steps": processed_text[sample_idx + 1],
                    "answer": processed_text[sample_idx + 2],
                })

        with open(os.path.join(GSM8K_DATA, f"{split}.json"), "w", encoding="utf-8") as out_file:
            json.dump(samples, out_file, indent=2, ensure_ascii=False)

    dump_errs("gsm8k_errs.json", err_data)

def process_math_data():
    """
    Process all data in the MATH dataset
    """
    err_data = {}
    for split in ("train", "test"):
        # Extract all questions/solutions from the split
        print("Split:", split)
        batch_text = []
        levels = []
        for subdir in os.listdir(f"../MATH 2/{split}"):
            print(subdir)
            for problem_filename in tqdm(os.listdir(f"../MATH 2/{split}/{subdir}")):
                with open(f"../MATH 2/{split}/{subdir}/{problem_filename}", encoding="utf-8") as src_file:
                    sample = json.load(src_file)
                    batch_text.append(sample["problem"])
                    batch_text.append(sample["solution"])
                    batch_text.append(get_boxed_answer(sample["solution"]))
                    levels.append(sample["level"])

        # Just assign levels if missing
        # with open(os.path.join(MATH_DATA, f"{split}_backup.json"), encoding="utf-8") as backup_file:
        #     samples = json.load(backup_file)
        #     for sample, level in zip(samples, levels):
        #         sample["level"] = level
        #     with open(os.path.join(MATH_DATA, f"{split}.json"), "w", encoding="utf-8") as out_file:
        #         json.dump(samples, out_file, indent=2, ensure_ascii=False)

        # Batch process LaTeXML/TangentCFT
        batch_size = 20
        samples: List[ProblemSolvingTaskSample] = []
        for batch_start_idx in tqdm(range(0, len(batch_text), batch_size * 3)):
            processed_text = process_raw_text(batch_text[batch_start_idx : batch_start_idx + batch_size * 3], err_data)
            for sample_idx, level in zip(range(0, len(processed_text), 3), levels[batch_start_idx // 3 : batch_start_idx // 3 + batch_size]):
                if None not in processed_text[sample_idx : sample_idx + 3]:
                    samples.append({
                        "problem": processed_text[sample_idx],
                        "steps": processed_text[sample_idx + 1],
                        "answer": processed_text[sample_idx + 2],
                        "level": level
                    })

        with open(os.path.join(MATH_DATA, f"{split}.json"), "w", encoding="utf-8") as out_file:
            json.dump(samples, out_file, indent=2, ensure_ascii=False)

    dump_errs("math_errs.json", err_data)

def process_mwp_data():
    """
    Process all data in the Math23K dataset
    """
    # Get text and equations from samples
    batch_text = []
    answers = []
    with open("../math23k_translated.pkl", "rb") as src_file:
        for sample in tqdm(pickle.load(src_file)):
            batch_text.append(wrap_formulas(html_to_latex(sample["text_en"])))
            batch_text.append("<m> " + sample["equation"] + " </m>")
            answers.append(sample["ans"])

    # Batch process LaTeXML/TangentCFT
    batch_size = 100
    samples: List[GenTaskSample] = []
    err_data = {}
    for batch_start_idx in tqdm(range(0, len(batch_text), batch_size * 2)):
        processed_text = process_raw_text(batch_text[batch_start_idx : batch_start_idx + batch_size * 2], err_data)
        for sample_idx in range(0, len(processed_text), 2):
            samples.append({
                "prompt": processed_text[sample_idx],
                "label": processed_text[sample_idx + 1],
            })
    for sample, answer in zip(samples, answers):
        sample["answer"] = answer
    with open(MWP_DATA, "w", encoding="utf-8") as out_file:
        json.dump(samples, out_file, indent=2, ensure_ascii=False)

    dump_errs("mwp_errs.json", err_data)

def process_khan():
    """
    Process all data in the Khan Academy dataset
    """
    # Gather all problems
    batch_text = []
    filenames = []
    for subdir in tqdm(os.listdir("../amps/khan")):
        if not os.path.isdir(f"../amps/khan/{subdir}"):
            continue
        os.makedirs(os.path.join(KHAN_DATA, subdir), exist_ok=True)
        for problem_filename in os.listdir(f"../amps/khan/{subdir}"):
            with open(f"../amps/khan/{subdir}/{problem_filename}", encoding="utf-8") as problem_file:
                sample = json.load(problem_file)
            filenames.append(f"{subdir}/{problem_filename}")
            batch_text.append(sample["problem"])
            batch_text.append(" ".join(sample["hints"]))

    # Batch process
    batch_size = 50
    err_data = {}
    for batch_start_idx in tqdm(range(0, len(batch_text), batch_size * 2)):
        processed_text = process_raw_text(batch_text[batch_start_idx : batch_start_idx + batch_size * 2], err_data, False)
        for sample_idx, filename in zip(range(0, len(processed_text), 2), filenames[batch_start_idx // 2 : batch_start_idx // 2 + batch_size]):
            if None in processed_text[sample_idx : sample_idx + 2]:
                continue
            with open(os.path.join(KHAN_DATA, filename), "w", encoding="utf-8") as out_file:
                json.dump({
                    "prompt": processed_text[sample_idx],
                    "label": processed_text[sample_idx + 1],
                }, out_file, indent=2, ensure_ascii=False)

    dump_errs("khan_errs.json", err_data)

def process_ct():
    """
    Process Cognitive Tutor dataset
    """
    # Load data and do initial analysis
    df = pandas.read_csv("../ds660.csv", encoding="utf-8")
    # print("Special steps:", df["Step Name"][df["Step Name"].apply(lambda x: "=" not in x if isinstance(x, str) else False)].unique().tolist())
    print("Unique actions:", df["Action"].unique().tolist())
    print("Unique outcomes:", df["Outcome"].unique().tolist())
    print("Unique problems:", df["Problem Name"].unique().size)
    print("Unique students:", df["Anon Student Id"].unique().size)
    student_problem_counts = df[["Anon Student Id", "Problem Name"]].drop_duplicates().groupby("Anon Student Id").count().reset_index()["Problem Name"]
    print("Problems per student- min:", student_problem_counts.min(), "max:", student_problem_counts.max(), "avg:", student_problem_counts.mean())
    # Outcome = nan iff Action = SWITCH
    # Action = nan iff Step Name is not equation or Outcome is HINT
    # Step Name = FinalAnswer and Input = nan iff Outcome is HINT

    # Extract relevant attributes from each row
    problem_id_to_text: Dict[str, str] = {}
    batch_text: List[str] = []
    raw_samples: List[dict] = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not isinstance(row["Step Name"], str) or not ("=" in row["Step Name"] or row["Step Name"] == "FinalAnswer"):
            continue
        if not isinstance(row["Outcome"], str) or row["Outcome"] == "INITIAL_HINT" or row["Outcome"] == "HINT_LEVEL_CHANGE":
            continue
        problem_text = "<m> " + row["Problem Name"].split(" ", 1)[1] + " </m>" # Remove label preceding formula
        problem_id_to_text[row["Problem Name"]] = problem_text
        batch_text.append(
            row["Step Name"]
            if row["Step Name"] == "FinalAnswer" else
            "<m> " + row["Step Name"].rsplit(" ", 1)[0] + " </m>"
        ) # Remove label following formula
        batch_text.append(
            "<m> " + row["Input"] + " </m>"
            if isinstance(row["Input"], str) else "")
        batch_text.append(
            row["Feedback Text"].replace("<expression>", "<m>").replace("</expression>", "</m>")
            if isinstance(row["Feedback Text"], str) else "")
        raw_samples.append({
            "action": row["Action"] if isinstance(row["Action"], str) else "",
            "outcome": row["Outcome"],
            "student_id": row["Anon Student Id"],
            "problem_id": row["Problem Name"]
        })

    batch_size = 50
    err_data = {}

    # Process all problems
    print("Process problems")
    problem_id_to_processed: Dict[str, Article] = {}
    problem_ids = list(problem_id_to_text.keys())
    problem_texts = list(problem_id_to_text.values())
    for batch_start_idx in tqdm(range(0, len(problem_texts), batch_size)):
        processed_text = process_raw_text(problem_texts[batch_start_idx : batch_start_idx + batch_size], err_data)
        for problem, problem_id in zip(processed_text, problem_ids[batch_start_idx : batch_start_idx + batch_size]):
            problem_id_to_processed[problem_id] = problem

    # Process and aggregate all steps by student/problem pairs
    print("Process steps")
    samples: List[CTTaskSample] = []
    cur_sample: Optional[CTTaskSample] = None
    for batch_start_idx in tqdm(range(0, len(batch_text), batch_size * 3)):
        processed_text = process_raw_text(batch_text[batch_start_idx : batch_start_idx + batch_size * 3], err_data)
        for sample_idx, raw_sample in zip(range(0, len(processed_text), 3), raw_samples[batch_start_idx // 3 : batch_start_idx // 3 + batch_size]):
            if not cur_sample or (raw_sample["student_id"], raw_sample["problem_id"]) != (cur_sample["student_id"], cur_sample["problem_id"]):
                cur_sample = {
                    "student_id": raw_sample["student_id"],
                    "problem_id": raw_sample["problem_id"],
                    "problem": problem_id_to_processed[raw_sample["problem_id"]],
                    "steps": []
                }
                samples.append(cur_sample)
            cur_sample["steps"].append({
                "step": processed_text[sample_idx],
                "input": processed_text[sample_idx + 1],
                "feedback": processed_text[sample_idx + 2],
                "action": raw_sample["action"],
                "outcome": raw_sample["outcome"],
            })

    with open(CT_DATA, "w", encoding="utf-8") as out_file:
        json.dump(samples, out_file, indent=2, ensure_ascii=False)

    dump_errs("ct_errs.json", err_data)
