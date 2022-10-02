import argparse
import torch
import torch.multiprocessing as mp

from pre_process import (
    process_wikipedia_data, process_probes, process_mathsum_data, process_answer_scoring_data, process_feedback_data,
    process_gsm8k_data, process_math_data, process_mwp_data, process_khan, process_ct
)
from analyze_data import analyze_wiki, analyze_mathsum, analyze_answer_scoring, analyze_feedback, analyze_vocab, analyze_gsm8k, analyze_math, analyze_mwp
from analyze_model import visualize_attention
from training import pretrain, evaluate_pretrained_lm, test_lm, train_downstream_task, evaluate_downstream_task, test_gen_task, cross_validate_downstream_task
from evaluate import evaluate_ted
from utils import initialize_seeds, device, enum_choices, enum_value_to_member, setup_proc_group, cleanup_proc_group
from vocabulary import Vocabulary
from constants import PretrainDataset, DownstreamTask, TPE, Gen, Optimizer

def bool_type(arg):
    return False if arg == "0" else True

def main():
    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            print("Running on", torch.cuda.device_count(), "GPUs")
        else:
            print("Running on GPU")
    else:
        print("No GPU found")

    initialize_seeds(221)

    parser = argparse.ArgumentParser("MathGPT")
    # Modes
    parser.add_argument("--preprocess_wiki", action="store_true", help="Process raw Wikipedia data and save to JSON files; generate raw vocab file")
    parser.add_argument("--preprocess_khan", action="store_true", help="Process Khan Academy datset")
    parser.add_argument("--preprocess_mathsum", help="Process raw MathSum data and save to JSON files", choices=["OFEQ-10k", "EXEQ-300k"])
    parser.add_argument("--preprocess_answer_scoring", action="store_true", help="Process answer scoring dataset")
    parser.add_argument("--preprocess_feedback", action="store_true", help="Process feedback dataset")
    parser.add_argument("--preprocess_gsm8k", action="store_true", help="Process GSM8K dataset")
    parser.add_argument("--preprocess_math", action="store_true", help="Process MATH dataset")
    parser.add_argument("--preprocess_mwp", action="store_true", help="Process Math23K dataset")
    parser.add_argument("--preprocess_ct", action="store_true", help="Process Cognitive Tutor dataset")
    parser.add_argument("--process_probes", action="store_true", help="Process LM probes and save to JSON files")
    parser.add_argument("--analyze_wiki", action="store_true", help="Produce stats on pre-processed Wikipedia dataset")
    parser.add_argument("--analyze_mathsum", help="Produce stats on pre-processed MathSum dataset", choices=["OFEQ-10k", "EXEQ-300k"])
    parser.add_argument("--analyze_answer_scoring", action="store_true", help="Produce stats on pre-processed answer scoring dataset")
    parser.add_argument("--analyze_feedback", action="store_true", help="Produce stats on pre-processed feedback dataset")
    parser.add_argument("--analyze_gsm8k", action="store_true", help="Produce stats on pre-processed GSM8K dataset")
    parser.add_argument("--analyze_math", action="store_true", help="Produce stats on pre-processed MATH dataset")
    parser.add_argument("--analyze_mwp", action="store_true", help="Produce stats on pre-processed Math23K dataset")
    parser.add_argument("--analyze_vocab", action="store_true", help="Produce stats on the math vocab")
    parser.add_argument("--visualize_attention", help="Visualize model's attention weights", choices=["probes"] + enum_choices(DownstreamTask))
    parser.add_argument("--pretrain", action="store_true", help="Pre-train LM")
    parser.add_argument("--evaluate_lm", action="store_true", help="Evaluate LM performance on test set")
    parser.add_argument("--test_lm", help="Run language generation using given article")
    parser.add_argument("--train_downstream", help="Train downstream task model", choices=enum_choices(DownstreamTask))
    parser.add_argument("--evaluate_downstream", help="Evaluate downstream task model performance on test set", choices=enum_choices(DownstreamTask))
    parser.add_argument("--test_downstream", help="See downstream model output on test samples", choices=enum_choices(DownstreamTask))
    parser.add_argument("--crossval", help="Run cross-validation on downstream task", choices=enum_choices(DownstreamTask))
    parser.add_argument("--evaluate_ted", help="Evaluate TED metric on pred files from formula-only generative task", choices=enum_choices(DownstreamTask))
    # Config
    parser.add_argument("--name", help="Name of current model/experiment, used for saving/loading model and config")
    parser.add_argument("--checkpoint_name", help="Name of model to resume training for")
    parser.add_argument("--pretrained_name", help="Name of pre-trained LM for initializing downstream model parameters")
    parser.add_argument("--dataset", help="Dataset to pre-train on", choices=enum_choices(PretrainDataset))
    parser.add_argument("--data_dir", help="Override default data directory for pre-training")
    parser.add_argument("--vocab_file", help="Override default vocab file")
    parser.add_argument("--split", type=float, help="Portion of data to use in train set during pre-training")
    parser.add_argument("--optim", help="Optimizer to use", choices=enum_choices(Optimizer))
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--epochs", type=int, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Maximum number of sequences per batch")
    parser.add_argument("--grad_accum_batches", type=int, help="Number of batches to accumulate gradients for")
    parser.add_argument("--max_seq_len", type=int, help="Maximum length, in tokens, of any sequence")
    parser.add_argument("--stride", type=int, help="Stride for computing perplexity with sliding context window")
    parser.add_argument("--amp", type=bool_type, help="Use automated mixed precision during training")
    parser.add_argument("--gen", help="Algorithm to use for generation", choices=enum_choices(Gen))
    parser.add_argument("--ns_p", type=float, help="P parameter for nucleus sampling")
    parser.add_argument("--beam_width", type=int, help="Width to use in beam search decoding")
    parser.add_argument("--min_gen_len", type=int, help="Minimum length for generated sequences")
    parser.add_argument("--eval_formulas", type=bool_type, help="For generative tasks, only treat the labels' formulas as targets")
    parser.add_argument("--eval_text", type=bool_type, help="For generative tasks, only treat the labels' text regions as targets")
    parser.add_argument("--eval_final", type=bool_type, help="For problem solving tasks, only generate the final step of the solution")
    parser.add_argument("--baseline", type=bool_type, help="Use baseline GPT-2 model")
    parser.add_argument("--post_proc", type=bool_type, help="For baseline - if true, train on post-processed and decoded formulas, else train on original formulas")
    parser.add_argument("--joint", type=bool_type, help="When true, model type/token probability jointly, otherwise model token probability directly")
    parser.add_argument("--use_type_embs", type=bool_type, help="Add type-specific embeddings to input token embeddings")
    parser.add_argument("--tpe", help="Scheme to use for tree position encodings", choices=enum_choices(TPE))
    parser.add_argument("--ddp", type=bool_type, help="Use DistributedDataParallel")
    parser.add_argument("--num_to_tree", type=bool_type, default=True, help="Convert numeric symbols into sub-trees")
    parser.add_argument("--sd_to_tree", type=bool_type, help="When using num_to_tree, if single digits should convert to sub-trees, otherwise be single tokens")
    parser.add_argument("--math_text", type=bool_type, default=True, help="Convert unseen math tokens to sub-trees with tokens from GPT encoder")
    parser.add_argument("--shared_emb", type=bool_type, help="Math token embeddings derived from corresponding text embeddings")
    parser.add_argument("--cdt", type=bool_type, help="Apply decoding constraint masks during training")
    parser.add_argument("--freeze_wte", type=bool_type, help="Freeze word token embeddings")
    parser.add_argument("--init_math_pred", type=bool_type, help="Initialize the math prediction layer with values from the pre-trained text prediction layer")
    parser.add_argument("--lmhb", type=bool_type, help="Use bias for the math LM head")

    args = parser.parse_args()

    if args.ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(
            main_worker,
            nprocs=world_size,
            args=(world_size, args),
            join=True
        )
    else:
        main_worker(0, 1, args)

def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    if args.ddp:
        setup_proc_group(rank, world_size)

    arg_dict = {arg: val for arg, val in vars(args).items() if val is not None}

    # Set these now since need to be known before vocab is loaded
    if "vocab_file" in arg_dict:
        Vocabulary.override_vocab_file(arg_dict["vocab_file"])
    if "num_to_tree" in arg_dict:
        Vocabulary.set_num_to_tree(arg_dict["num_to_tree"])
    if "math_text" in arg_dict:
        Vocabulary.set_math_text(arg_dict["math_text"])

    if args.preprocess_wiki:
        process_wikipedia_data()
    if args.preprocess_khan:
        process_khan()
    if args.preprocess_mathsum:
        process_mathsum_data(args.preprocess_mathsum)
    if args.preprocess_answer_scoring:
        process_answer_scoring_data()
    if args.preprocess_feedback:
        process_feedback_data()
    if args.preprocess_gsm8k:
        process_gsm8k_data()
    if args.preprocess_math:
        process_math_data()
    if args.preprocess_mwp:
        process_mwp_data()
    if args.preprocess_ct:
        process_ct()
    if args.process_probes:
        process_probes()
    if args.analyze_wiki:
        analyze_wiki()
    if args.analyze_mathsum:
        analyze_mathsum(args.analyze_mathsum)
    if args.analyze_answer_scoring:
        analyze_answer_scoring()
    if args.analyze_feedback:
        analyze_feedback()
    if args.analyze_gsm8k:
        analyze_gsm8k()
    if args.analyze_math:
        analyze_math()
    if args.analyze_mwp:
        analyze_mwp()
    if args.analyze_vocab:
        analyze_vocab()
    if args.visualize_attention:
        visualize_attention(args.name, args.visualize_attention, arg_dict)
    if args.pretrain:
        pretrain(args.name, args.checkpoint_name, args.pretrained_name, arg_dict)
    if args.evaluate_lm:
        evaluate_pretrained_lm(args.name, arg_dict)
    if args.test_lm:
        test_lm(args.name, args.test_lm, arg_dict)
    if args.train_downstream:
        train_downstream_task(args.name, args.checkpoint_name, args.pretrained_name, enum_value_to_member(args.train_downstream, DownstreamTask), arg_dict)
    if args.evaluate_downstream:
        evaluate_downstream_task(args.name, enum_value_to_member(args.evaluate_downstream, DownstreamTask), False, arg_dict)
    if args.test_downstream:
        test_gen_task(args.name, enum_value_to_member(args.test_downstream, DownstreamTask), arg_dict)
    if args.crossval:
        cross_validate_downstream_task(args.name, args.checkpoint_name, args.pretrained_name, enum_value_to_member(args.crossval, DownstreamTask), arg_dict)
    if args.evaluate_ted:
        evaluate_ted(args.name, enum_value_to_member(args.evaluate_ted, DownstreamTask), arg_dict)

    if args.ddp:
        cleanup_proc_group()

if __name__ == "__main__":
    main()
