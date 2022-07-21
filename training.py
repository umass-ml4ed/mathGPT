import time
import os
import json
from typing import Optional, List, Tuple, Union, Dict
import random
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm
# from neptune.new.run import Run
from sklearn.model_selection import StratifiedKFold
import numpy as np

from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from model_baseline import GPTLMBaseline, GPTClassifierBaseline
from loading import Dataset, PreTrainDataset, PreTrainDatasetPreloaded, GenTaskDataset, AnswerScoringDataset, FeedbackDataset, trim_batch, get_data_loader
from evaluate import evaluate_lm, evaluate_lm_accuracy, evaluate_cls_task, evaluate_gen_task
from generate import generate
from decode import decode_batch
from utils import TrainOptions, device, is_cls_task, new_neptune_run
from data_types import Article, GenTaskSample, AnswerScoringSample, FeedbackTaskSample
from constants import DownstreamTask, Checkpoint, DOWNSTREAM_TASK_TO_NUM_CLASSES, WIKI_DATA, OFEQ_DATA, AS_PROBLEMS, AS_ANSWERS, FEEDBACK_DATA

def get_article_names(data_dir_override: Optional[str]):
    data_dir = data_dir_override or WIKI_DATA
    return [os.path.join(data_dir, article_filename) for article_filename in os.listdir(data_dir)]

def get_probes() -> List[Article]:
    with open("data/probes.json", encoding="utf-8") as probes_file:
        return json.load(probes_file)

def get_headline_data(split: str, options: TrainOptions) -> List[GenTaskSample]:
    # Load pre-processed dataset when using the MathGPT model, or using the post_proc option for the baseline model
    pre_processed = not options.baseline or options.post_proc
    if pre_processed:
        with open(os.path.join(OFEQ_DATA, f"{split}.json"), encoding="utf-8") as headlines_file:
            return json.load(headlines_file)
    else:
        with open(f"../MathSum/OFEQ-10k/post.{split}", encoding="utf-8") as post_file:
            with open(f"../MathSum/OFEQ-10k/title.{split}", encoding="utf-8") as title_file:
                return [
                    {"prompt": {"text": post, "formulas": {}}, "label": {"text": title, "formulas": {}}}
                    for post, title in zip(post_file, title_file)
                ]

def get_answer_scoring_data() -> Tuple[Dict[str, Article], List[AnswerScoringSample], List[AnswerScoringSample], List[AnswerScoringSample]]:
    with open(AS_PROBLEMS, encoding="utf-8") as problem_file:
        problems: Dict[str, Article] = json.load(problem_file)
    with open(AS_ANSWERS, encoding="utf-8") as answer_file:
        answers: List[AnswerScoringSample] = json.load(answer_file)
    # Stratify on problem id so that samples in the training set can be used during test time for meta learning
    answers_np = np.array(answers)
    stratify_labels = np.array([answer["problem_id"] for answer in answers])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    train_data_idx, test_data_idx = next(skf.split(answers_np, stratify_labels))
    test_len = len(test_data_idx) // 2
    return problems, answers_np[train_data_idx], answers_np[test_data_idx][:test_len], answers_np[test_data_idx][test_len:]

def get_feedback_data():
    with open(FEEDBACK_DATA, encoding="utf-8") as feedback_file:
        samples: List[FeedbackTaskSample] = json.load(feedback_file)
    random.Random(221).shuffle(samples)
    train_len = int(.8 * len(samples))
    test_len = int(.1 * len(samples))
    return samples[:train_len], samples[train_len : -test_len], samples[-test_len:]

def load_options(model_name: str):
    with open(f"{model_name}.json", encoding="utf-8") as config_file:
        return TrainOptions(json.load(config_file))

def load_model(model_name: str, ddp: bool, task: Optional[DownstreamTask] = None):
    print("Loading model...")
    options = load_options(model_name)
    if is_cls_task(task):
        if options.baseline:
            model = GPTClassifierBaseline(options).to(device)
        else:
            model = MathGPTClassifier(options).to(device)
    else:
        if options.baseline:
            model = GPTLMBaseline().to(device)
        else:
            model = MathGPTLM(options).to(device)
    checkpoint: Checkpoint = torch.load(f"{model_name}.pt", map_location=device)
    if "model_state_dict" not in checkpoint: # Backward compatability
        model.load_state_dict(checkpoint)
        checkpoint = None
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint["model_state_dict"] # Free up memory
    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    return model, checkpoint, options

def evaluate_model(model: MathGPTBase, dataset: Dataset, task: Optional[DownstreamTask], options: TrainOptions):
    if not task:
        return evaluate_lm(model, dataset, options)
    if is_cls_task(task):
        return evaluate_cls_task(model, dataset, task, options)
    return evaluate_lm_accuracy(model, dataset, task, options)

def train(model: Union[MathGPTBase, DDP], model_name: str, train_loader: DataLoader, validation_dataset: Dataset, options: TrainOptions,
          run = None, task: Optional[DownstreamTask] = None, checkpoint: Optional[Checkpoint] = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        del checkpoint["optimizer_state_dict"]
    if not options.amp:
        torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    # Scaler prevents gradient underflow when using fp16 precision
    scaler = GradScaler() if options.amp else None
    if checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        del checkpoint["scaler_state_dict"]

    if run:
        run["name"] = model_name
        run["options"] = options.as_dict()
        run["task"] = str(task)

    starting_epoch = checkpoint["epoch"] + 1 if checkpoint else 0
    best_metric = None
    best_stats = None
    cur_stats = None
    best_epoch = starting_epoch

    if checkpoint:
        torch.random.set_rng_state(checkpoint["rng_state"].cpu())

    print("Training...")
    for epoch in range(starting_epoch, options.epochs):
        if options.ddp:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        start_time = time.time()
        model.train() # Set model to training mode
        train_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_loader):
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = model(batch)[0]
                scaler.scale(loss).backward()
            else:
                loss = model(batch)[0]
                loss.backward()
            train_loss += float(loss.detach().cpu().numpy())
            num_batches += 1
            if num_batches % options.grad_accum_batches == 0 or num_batches == len(train_loader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / num_batches
        val_loss, results = evaluate_model(model, validation_dataset, task, options)
        if run:
            run["train/loss"].log(avg_train_loss)
            run["val/loss"].log(val_loss)
            run["val/metrics"].log(results)
        print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.3f}, Val Loss: {val_loss:.3f}, {results}, Time: {time.time() - start_time:.2f}")

        # Save model for best validation metric
        if not best_metric or val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            best_stats = cur_stats
            if not options.ddp or torch.cuda.current_device() == 0:
                print("Saving model")
                if options.ddp:
                    model_state_dict = {param.replace("module.", ""): val for param, val in model.state_dict().items()}
                else:
                    model_state_dict = model.state_dict()
                torch.save({
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                    "rng_state": torch.random.get_rng_state(),
                    "epoch": epoch,
                }, f"{model_name}.pt")
                with open(f"{model_name}.json", "w", encoding="utf-8") as config_file:
                    json.dump(options.as_dict(), config_file, indent=4)
            if options.ddp:
                dist.barrier() # Wait for main process to finish saving

        # Stop training if we haven't improved in a while
        if options.patience and (epoch - best_epoch >= options.patience):
            print("Early stopping")
            break

    return best_stats

def pretrain(model_name: str, checkpoint_name: Optional[str], options_dict: dict):
    if checkpoint_name:
        model, checkpoint, options = load_model(checkpoint_name, options_dict.get("ddp", False))
        options.update(options_dict)
    else:
        checkpoint = None
        options = TrainOptions(options_dict)
        if options.baseline:
            model = GPTLMBaseline().to(device)
        else:
            model = MathGPTLM(options).to(device)
        if options.ddp:
            model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    articles = get_article_names(options.data_dir)
    split_point = int(len(articles) * options.split)
    train_data = PreTrainDataset(articles[:split_point], options, options.max_seq_len)
    val_data = PreTrainDataset(articles[split_point:], options, max_seq_len=None)
    train_loader = get_data_loader(train_data, None, options.batch_size, True, True, options)
    main_proc = not options.ddp or torch.cuda.current_device() == 0
    train(model, model_name, train_loader, val_data, options, checkpoint=checkpoint)
    results = evaluate_pretrained_lm(model_name, options.as_dict())
    # Create run after training/eval to avoid using up hours
    run = new_neptune_run() if main_proc else None
    if run:
        run["results"] = results
        run.stop()

def evaluate_pretrained_lm(model_name: str, test_options: dict):
    model, _, options = load_model(model_name, test_options.get("ddp", False))
    options.update(test_options)

    articles = get_article_names(options.data_dir)
    split_point = int(len(articles) * options.split)
    test_articles = articles[split_point:]
    dataset = PreTrainDataset(test_articles, options, max_seq_len=None)
    loss, results = evaluate_lm(model, dataset, options)
    print(f"Loss: {loss:.3f}, {results}")
    return results

def test_lm(model_name: str, test_article: str, test_options: dict):
    model, _, options = load_model(model_name, test_options.get("ddp", False))
    options.update(test_options)

    wiki = False
    if wiki:
        dataset = PreTrainDataset([test_article], options, options.max_seq_len // 2)
        data_loader = get_data_loader(dataset, None, 1, False, False, options)
        with torch.no_grad():
            data_loader_it = iter(data_loader)
            gen_batch = next(data_loader_it)
            gen_batch_len = len(gen_batch["token_ids"])
            prompt_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]
            gen_batch = generate(model, gen_batch, options)
            pred_text = decode_batch(trim_batch(gen_batch, gen_batch_len, options.max_seq_len), dataset.text_tokenizer)[0]
            followup_batch = next(data_loader_it)
            og_text = decode_batch(followup_batch, dataset.text_tokenizer)[0]

            print("Prompt:", prompt_text)
            print("OG Text:", og_text)
            print("Prediction:", pred_text)
            print("")
    else:
        dataset = PreTrainDatasetPreloaded(get_probes(), options, options.max_seq_len)
        data_loader = get_data_loader(dataset, None, 1, False, False, options)
        with torch.no_grad():
            for batch in data_loader:
                prompt_text = decode_batch(batch, dataset.text_tokenizer)[0]
                gen_batch = generate(model, batch, options)
                pred_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]

                print("Prompt:", prompt_text)
                print("Prediction:", pred_text)
                print("")

def train_downstream_task(model_name: str, checkpoint_name: Optional[str], pretrained_name: Optional[str], task: DownstreamTask, options_dict: dict):
    # Create/load model and config
    if checkpoint_name:
        model, checkpoint, options = load_model(checkpoint_name, options_dict.get("ddp", False), task)
        options.update(options_dict)
    else:
        checkpoint = None
        if pretrained_name:
            options = load_options(pretrained_name)
            options.update(options_dict)
        else:
            options = TrainOptions(options_dict)
        if is_cls_task(task):
            options.num_classes = DOWNSTREAM_TASK_TO_NUM_CLASSES.get(task)
            if options.baseline:
                model = GPTClassifierBaseline(options).to(device)
            else:
                model = MathGPTClassifier(options).to(device)
        else:
            if options.baseline:
                model = GPTLMBaseline().to(device)
            else:
                model = MathGPTLM(options).to(device)
        if pretrained_name:
            print("Loading pre-trained model...")
            checkpoint: Checkpoint = torch.load(f"{pretrained_name}.pt", map_location=device)
            model.load_pretrained(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
            checkpoint = None
        if options.ddp:
            model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    # Load and process data
    if task == DownstreamTask.HEADLINES:
        train_data = GenTaskDataset(get_headline_data("train", options), options)
        val_data = GenTaskDataset(get_headline_data("val", options), options)
    elif task == DownstreamTask.ANSWER_SCORING:
        problems, train_samples, val_samples, _ = get_answer_scoring_data()
        train_data = AnswerScoringDataset(train_samples, problems, options)
        val_data = AnswerScoringDataset(val_samples, problems, options, train_data.data)
    elif task == DownstreamTask.FEEDBACK:
        train_samples, val_samples, _ = get_feedback_data()
        train_data = FeedbackDataset(train_samples, options)
        val_data = FeedbackDataset(val_samples, options)
    else:
        raise Exception(f"Unsupported task {task}")
    train_loader = get_data_loader(train_data, task, options.batch_size, True, True, options)

    # Start training
    main_proc = not options.ddp or torch.cuda.current_device() == 0
    run = new_neptune_run() if main_proc else None
    train(model, model_name, train_loader, val_data, options, run, task, checkpoint=checkpoint)
    results = evaluate_downstream_task(model_name, task, options.as_dict())
    if run:
        run["results"] = results
        run.stop()

def evaluate_downstream_task(model_name: str, task: DownstreamTask, eval_options: dict):
    model, _, options = load_model(model_name, eval_options.get("ddp", False), task)
    options.update(eval_options)

    if task == DownstreamTask.HEADLINES:
        headlines = get_headline_data("test", options)
        test_data = GenTaskDataset(headlines, options)
        _, results = evaluate_gen_task(model_name, model, test_data, task, options)
    elif task == DownstreamTask.ANSWER_SCORING:
        problems, train_samples, _, test_samples = get_answer_scoring_data()
        train_data = AnswerScoringDataset(train_samples, problems, options)
        test_data = AnswerScoringDataset(test_samples, problems, options, train_data.data)
        _, results = evaluate_cls_task(model, test_data, task, options)
    elif task == DownstreamTask.FEEDBACK:
        _, _, test_samples = get_feedback_data()
        test_data = FeedbackDataset(test_samples, options)
        _, results = evaluate_gen_task(model_name, model, test_data, task, options)
    else:
        raise Exception(f"Unsupported task {task}")

    print(results)
    return results

def test_gen_task(model_name: str, task: DownstreamTask, test_options: dict):
    model, _, options = load_model(model_name, test_options.get("ddp", False), task)
    options.update(test_options)
    start_idx = 0
    samples_to_try = 5

    if task == DownstreamTask.HEADLINES:
        samples = get_headline_data("test", options)
        dataset = GenTaskDataset(samples[start_idx : start_idx + samples_to_try], options)
    elif task == DownstreamTask.FEEDBACK:
         _, _, samples = get_feedback_data()
         dataset = FeedbackDataset(samples[start_idx : start_idx + samples_to_try], options)
    else:
        raise Exception(f"Unsupported task {task}")
    data_loader = get_data_loader(dataset, task, 1, False, False, options)
    with torch.no_grad():
        for batch in data_loader:
            split_point = batch["prompt_lengths"][0]
            gen_batch = trim_batch(batch, 0, split_point)
            prompt_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]
            gen_batch = generate(model, gen_batch, options)
            pred_text = decode_batch(trim_batch(gen_batch, split_point, options.max_seq_len), dataset.text_tokenizer)[0]
            og_text = decode_batch(trim_batch(batch, split_point, options.max_seq_len), dataset.text_tokenizer)[0]

            print("Prompt:", prompt_text)
            print("OG Text:", og_text)
            print("Prediction:", pred_text)
            print("")
