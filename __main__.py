import argparse
import torch

from pre_process import process_wikipedia_data, process_mathsum_data, process_probes
from analyze_data import analyze_data
from training import pretrain, evaluate_pretrained_lm, test_lm, train_downstream_task, evaluate_downstream_task, test_gen_task
from utils import TrainOptions, initialize_seeds, device, enum_choices, enum_value_to_member
from constants import DownstreamTask

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
    parser.add_argument("--preprocess_mathsum", action="store_true", help="Process raw MathSum data and save to JSON files")
    parser.add_argument("--process_probes", action="store_true", help="Process LM probes and save to JSON files")
    parser.add_argument("--analyze_data", action="store_true", help="Produce stats on pre-processed dataset")
    parser.add_argument("--pretrain", action="store_true", help="Pre-train LM")
    parser.add_argument("--evaluate_lm", action="store_true", help="Evaluate LM performance on test set")
    parser.add_argument("--test_lm", help="Run language generation using given article")
    parser.add_argument("--train_downstream", help="Train downstream task model", choices=enum_choices(DownstreamTask))
    parser.add_argument("--evaluate_downstream", help="Evaluate downstream task model performance on test set", choices=enum_choices(DownstreamTask))
    parser.add_argument("--test_downstream", help="See downstream model output on test samples", choices=enum_choices(DownstreamTask))
    # Config
    parser.add_argument("--name", help="Name of current model/experiment, used for saving/loading model and config")
    parser.add_argument("--pretrained_name", help="Name of pre-trained LM for initializing model parameters")
    parser.add_argument("--epochs", type=int, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Maximum number of sequences per batch")
    parser.add_argument("--grad_accum_batches", type=int, help="Number of batches to accumulate gradients for")
    parser.add_argument("--max_seq_len", type=int, help="Maximum length, in tokens, of any sequence")
    parser.add_argument("--stride", type=int, help="Stride for computing perplexity with sliding context window")
    parser.add_argument("--amp", type=bool_type, help="Use automated mixed precision during training")
    parser.add_argument("--ns_p", type=float, help="P parameter for nucleus sampling")
    parser.add_argument("--use_type_embs", type=bool_type, help="Add type-specific embeddings to input token embeddings")

    args = parser.parse_args()
    arg_dict = {arg: val for arg, val in vars(args).items() if val is not None}

    if args.preprocess_wiki:
        process_wikipedia_data()
    if args.preprocess_mathsum:
        process_mathsum_data()
    if args.process_probes:
        process_probes()
    if args.analyze_data:
        analyze_data()
    if args.pretrain:
        pretrain(args.name, args.pretrained_name, arg_dict)
    if args.evaluate_lm:
        evaluate_pretrained_lm(args.name, arg_dict)
    if args.test_lm:
        test_lm(args.name, args.test_lm, arg_dict)
    if args.train_downstream:
        train_downstream_task(args.name, args.pretrained_name, enum_value_to_member(args.train_downstream, DownstreamTask), TrainOptions(arg_dict))
    if args.evaluate_downstream:
        evaluate_downstream_task(args.name, enum_value_to_member(args.evaluate_downstream, DownstreamTask))
    if args.test_downstream:
        test_gen_task(args.name, enum_value_to_member(args.test_downstream, DownstreamTask))

if __name__ == "__main__":
    main()
