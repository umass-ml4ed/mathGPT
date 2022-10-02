import sys
import numpy as np
import pandas
from tqdm import tqdm
from nlgeval import compute_metrics
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from utils import text_tokenizer

def replace_formulas(sequence: str):
    final_sequence = ""
    start_form_idx = sequence.find(" <m> ")
    end_form_idx = 0
    while start_form_idx >= 0:
        final_sequence += sequence[end_form_idx : start_form_idx] + " <math> "
        end_form_idx = sequence.find(" </m> ", start_form_idx)
        if end_form_idx != -1:
            end_form_idx += 6
        else:
            end_form_idx = len(sequence)
        start_form_idx = sequence.find(" <m> ", end_form_idx)
    final_sequence += sequence[end_form_idx:]
    return final_sequence

def replace_text(sequence: str):
    final_sequence = ""
    start_form_idx = sequence.find(" <m> ")
    end_form_idx = 0
    if start_form_idx > 0:
        final_sequence += "<text>"
    if start_form_idx == -1:
        final_sequence += "<text>"
    while start_form_idx >= 0:
        end_form_idx = sequence.find(" </m> ", start_form_idx)
        if end_form_idx != -1:
            end_form_idx += 6
        else:
            end_form_idx = len(sequence)
        final_sequence += sequence[start_form_idx : end_form_idx]
        if end_form_idx != len(sequence):
            final_sequence += "<text>"
        start_form_idx = sequence.find(" <m> ", end_form_idx)
    return final_sequence

def sanitize(sequence: str):
    final_sequence = ""
    start_form_idx = sequence.find(" <m> ")
    end_form_idx = 0
    while start_form_idx >= 0:
        final_sequence += sequence[end_form_idx : start_form_idx]
        end_form_idx = sequence.find(" </m> ", start_form_idx)
        if end_form_idx != -1:
            end_form_idx += 6
        else:
            end_form_idx = len(sequence)
        final_sequence += text_tokenizer().decode(text_tokenizer()(sequence[start_form_idx : end_form_idx])["input_ids"])
        start_form_idx = sequence.find(" <m> ", end_form_idx)
    final_sequence += sequence[end_form_idx:]
    return final_sequence

def eval_with_substitution(model_name: str, fold: int, eval: str):
    fn = replace_formulas if eval == "text" else replace_text if eval == "math" else sanitize
    with open(f"labels_{model_name}_{fold}.txt", encoding="utf-8") as label_file:
        labels = [fn(label.strip()) for label in label_file.readlines()]
    with open(f"preds_{model_name}_{fold}.txt", encoding="utf-8") as pred_file:
        preds = [fn(pred.strip()) for pred in pred_file.readlines()]
    print("Avg pred len:", np.array([len(pred.split()) for pred in preds]).mean(), "Avg label len:", np.array([len(label.split()) for label in labels]).mean())
    temp_label_file = "labels_temp.txt"
    temp_pred_file = "preds_temp.txt"
    with open(temp_label_file, "w", encoding="utf-8") as label_file:
        label_file.write("\n".join(labels))
    with open(temp_pred_file, "w", encoding="utf-8") as pred_file:
        pred_file.write("\n".join(preds))
    metrics = compute_metrics(hypothesis=temp_pred_file, references=[temp_label_file], no_skipthoughts=True, no_glove=True)
    print(metrics)

def error_analysis(model_1: str, model_2: str):
    with open(f"results/labels_{model_1}.txt", encoding="utf-8") as label_file:
        labels_1 = [label.strip() for label in label_file.readlines()]
    with open(f"results/labels_{model_2}.txt", encoding="utf-8") as label_file:
        labels_2 = [label.strip() for label in label_file.readlines()]
    with open(f"results/preds_{model_1}.txt", encoding="utf-8") as pred_file:
        preds_1 = [pred.strip() for pred in pred_file.readlines()]
    with open(f"results/preds_{model_2}.txt", encoding="utf-8") as pred_file:
        preds_2 = [pred.strip() for pred in pred_file.readlines()]

    # See how many predictions start correctly, up to a certain length
    # results_1 = {5: 0, 10: 0, 20: 0, 100: 0}
    # results_2 = {5: 0, 10: 0, 20: 0, 100: 0}
    # trees_1 = {"less": 0, "more": 0, "eq": 0, "pred_start": 0, "label_start": 0, "eq_start": 0}
    # trees_2 = {"less": 0, "more": 0, "eq": 0, "pred_start": 0, "label_start": 0, "eq_start": 0}
    # for labels, preds, results, trees in [(labels_1, preds_1, results_1, trees_1), (labels_2, preds_2, results_2, trees_2)]:
    #     for label, pred in zip(labels, preds):
    #         if pred.count("<m>") < label.count("<m>"):
    #             trees["less"] += 1
    #         elif pred.count("<m>") > label.count("<m>"):
    #             trees["more"] += 1
    #         else:
    #             trees["eq"] += 1
    #         if "<m>" in pred[:5]:
    #             trees["pred_start"] += 1
    #         if "<m>" in label[:5]:
    #             trees["label_start"] += 1
    #         if pred[:5] == label[:5] and "<m>" in pred[:5]:
    #             trees["eq_start"] += 1
    #         for start in results:
    #             if label[:start] == pred[:start]:
    #                 results[start] += 1
    # same_start = sum(1 for pred_1, pred_2 in zip(preds_1, preds_2) if pred_1[:5] == pred_2[:5])
    # print(results_1, trees_1)
    # print(results_2, trees_2)
    # print(same_start)

    bleu = Bleu(4)
    bleu_1 = [bleu.compute_score({0: [label]}, {0: [pred]})[0][3] for label, pred in tqdm(zip(labels_1, preds_1), total=len(labels_1))]
    bleu_2 = [bleu.compute_score({0: [label]}, {0: [pred]})[0][3] for label, pred in tqdm(zip(labels_2, preds_2), total=len(labels_2))]
    df = pandas.DataFrame({
        "labels": labels_1,
        "preds_1": preds_1,
        "bleu_1": bleu_1,
        "preds_2": preds_2,
        "bleu_2": bleu_2,
    })
    print(f'Avg Lens: {df["preds_1"].apply(len).mean():.3f}, {df["preds_2"].apply(len).mean():.3f}')
    for _, sample in df[(df["bleu_1"] > 0.7) & (df["bleu_2"] < 0.5)][:20].iterrows():
        print(sample["labels"])
        print(sample["preds_1"], f'({sample["bleu_1"]:.3f})')
        print(sample["preds_2"], f'({sample["bleu_2"]:.3f})')
        print("")

if __name__ == "__main__":
    eval_with_substitution(sys.argv[1], sys.argv[2], sys.argv[3])
    # error_analysis(sys.argv[1], sys.argv[2])
