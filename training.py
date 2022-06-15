import time
import os
import json
from typing import Optional, List, Union
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from neptune.new.run import Run

from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from loading import PreTrainDataset, PreTrainDatasetPreloaded, GenTaskDataset, ClassifyTaskDataset, Collator, trim_batch, get_data_loader
from evaluate import evaluate_lm, evaluate_lm_accuracy, evaluate_cls_task, evaluate_gen_task
from generate import generate
from decode import decode_batch
from utils import TrainOptions, device, is_cls_task, new_neptune_run
from constants import DownstreamTask, Article, DOWNSTREAM_TASK_TO_NUM_CLASSES, WIKI_DATA, OFEQ_DATA

def get_article_names():
    return [os.path.join(WIKI_DATA, article_filename) for article_filename in os.listdir(WIKI_DATA)]

def save_model(model: MathGPTBase, model_name: str, options: TrainOptions):
    if options.ddp:
        state_dict = {param.replace("module.", ""): val for param, val in model.state_dict().items()}
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, f"{model_name}.pt")
    with open(f"{model_name}.json", "w", encoding="utf-8") as config_file:
        json.dump(options.as_dict(), config_file, indent=4)

def load_model(model_name: str, ddp: bool, task: Optional[DownstreamTask] = None):
    print("Loading model...")
    with open(f"{model_name}.json", encoding="utf-8") as config_file:
        options = TrainOptions(json.load(config_file))
    if is_cls_task(task):
        model = MathGPTClassifier(options).to(device)
    else:
        model = MathGPTLM(options).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    return model, options

def evaluate_model(model: MathGPTBase, dataset: Dataset, task: Optional[DownstreamTask], options: TrainOptions):
    if not task:
        return evaluate_lm(model, dataset, options)
    if is_cls_task(task):
        return evaluate_cls_task(model, dataset, task, options)
    return evaluate_lm_accuracy(model, dataset, task, options)

def train(model: Union[MathGPTBase, DDP], model_name: str, train_loader: DataLoader, validation_dataset: Dataset, options: TrainOptions,
          run: Optional[Run] = None, task: Optional[DownstreamTask] = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    # Scaler prevents gradient underflow when using fp16 precision
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if options.amp else None

    if run:
        run["name"] = model_name
        run["options"] = options.as_dict()
        run["task"] = str(task)

    best_metric = None
    best_stats = None
    cur_stats = None
    best_epoch = 0

    print("Training...")
    for epoch in range(options.epochs):
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

        model.eval() # Set model to evaluation mode
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
                save_model(model, model_name, options)
            if options.ddp:
                dist.barrier() # Wait for main process to finish saving

        # Stop training if we haven't improved in a while
        if options.patience and (epoch - best_epoch >= options.patience):
            print("Early stopping")
            break

    return best_stats

def pretrain(model_name: str, pretrained_name: str, options_dict: dict):
    if pretrained_name:
        model, options = load_model(pretrained_name, options_dict.get("ddp", False))
        options.update(options_dict)
    else:
        options = TrainOptions(options_dict)
        model = MathGPTLM(options).to(device)
        if options.ddp:
            model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    articles = get_article_names()
    # TODO: should actually not set max seq len for val set if we're calculating perplexity
    dataset = PreTrainDataset(articles, options.tpe, options.max_seq_len)
    train_data = Subset(dataset, list(range(0, int(len(dataset) * .9))))
    val_data = Subset(dataset, list(range(int(len(dataset) * .9), len(dataset))))
    train_loader = get_data_loader(train_data, None, options.batch_size, True, True, options.ddp)
    main_proc = not options.ddp or torch.cuda.current_device() == 0
    run = new_neptune_run() if main_proc else None
    train(model, model_name, train_loader, val_data, options, run)
    if run:
        run.stop()

def evaluate_pretrained_lm(model_name: str, test_options: dict):
    model, options = load_model(model_name, test_options.get("ddp", False))
    options.update(test_options)

    # TODO: get test split from a file
    # TODO: try excluding test articles that contain UNKs, see how many are left out
    articles = get_article_names()
    test_articles = articles[int(len(articles) * .9):]
    dataset = PreTrainDataset(test_articles, options.tpe, max_seq_len=None)
    loss, results = evaluate_lm(model, dataset, options)
    print(f"Loss: {loss:.3f}, {results}")

def test_lm(model_name: str, test_article: str, test_options: dict):
    model, options = load_model(model_name, test_options.get("ddp", False))
    options.update(test_options)

    wiki = False
    if wiki:
        dataset = PreTrainDataset([test_article], options.tpe, options.max_seq_len // 2)
        data_loader = get_data_loader(dataset, None, 1, False, False, options.ddp)
        with torch.no_grad():
            data_loader_it = iter(data_loader)
            gen_batch = next(data_loader_it)
            gen_batch_len = len(gen_batch["token_ids"])
            prompt_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]
            generate(model, gen_batch, options)
            pred_text = decode_batch(trim_batch(gen_batch, gen_batch_len, options.max_seq_len), dataset.text_tokenizer)[0]
            followup_batch = next(data_loader_it)
            og_text = decode_batch(followup_batch, dataset.text_tokenizer)[0]

            print("Prompt:", prompt_text)
            print("OG Text:", og_text)
            print("Prediction:", pred_text)
            print("")
    else:
        with open("data/probes.json", encoding="utf-8") as probes_file:
            probes: List[Article] = json.load(probes_file)
        dataset = PreTrainDatasetPreloaded(probes, options.tpe, options.max_seq_len)
        data_loader = get_data_loader(dataset, None, 1, False, False, options.ddp)
        with torch.no_grad():
            for batch in data_loader:
                prompt_text = decode_batch(batch, dataset.text_tokenizer)[0]
                generate(model, batch, options)
                pred_text = decode_batch(batch, dataset.text_tokenizer)[0]

                print("Prompt:", prompt_text)
                print("Prediction:", pred_text)
                print("")

def train_downstream_task(model_name: str, pretrained_name: str, task: DownstreamTask, options: TrainOptions):
    if is_cls_task(task):
        # TODO: get data files for the given task
        options.num_classes = DOWNSTREAM_TASK_TO_NUM_CLASSES.get(task)
        dataset = ClassifyTaskDataset([], options.tpe, options.max_seq_len)
        model = MathGPTClassifier(options).to(device)
    else:
        with open(os.path.join(OFEQ_DATA, "train.json"), encoding="utf-8") as headlines_file:
            train_headlines = json.load(headlines_file)
        with open(os.path.join(OFEQ_DATA, "val.json"), encoding="utf-8") as headlines_file:
            val_headlines = json.load(headlines_file)
        train_data = GenTaskDataset(train_headlines, options.tpe, options.max_seq_len)
        val_data = GenTaskDataset(val_headlines, options.tpe, options.max_seq_len)
        model = MathGPTLM(options).to(device)
    model.load_pretrained(pretrained_name)
    if options.ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
    train_loader = get_data_loader(train_data, task, options.batch_size, True, True, options.ddp)
    main_proc = not options.ddp or torch.cuda.current_device() == 0
    run = new_neptune_run() if main_proc else None
    train(model, model_name, train_loader, val_data, options, run, task)
    results = evaluate_downstream_task(model_name, task, options.as_dict())
    if run:
        run["results"] = results
        run.stop()

def evaluate_downstream_task(model_name: str, task: DownstreamTask, eval_options: dict):
    model, options = load_model(model_name, eval_options.get("ddp", False), task)
    options.update(eval_options)

    # TODO: get data files for the given task
    if is_cls_task(task):
        dataset = ClassifyTaskDataset([], options.tpe, options.max_seq_len)
        _, results = evaluate_cls_task(model, dataset, task, options)
    else:
        with open(os.path.join(OFEQ_DATA, "test.json"), encoding="utf-8") as headlines_file:
            headlines = json.load(headlines_file)
        dataset = GenTaskDataset(headlines, options.tpe, options.max_seq_len)
        _, results = evaluate_gen_task(model, dataset, task, options)
    print(results)
    return results

def test_gen_task(model_name: str, task: DownstreamTask, test_options: dict):
    model, options = load_model(model_name, test_options.get("ddp", False), task)
    options.update(test_options)
    samples_to_try = 5

    with open(os.path.join(OFEQ_DATA, "test.json"), encoding="utf-8") as headlines_file:
        headlines = json.load(headlines_file)[:samples_to_try]
    dataset = GenTaskDataset(headlines, options.tpe, options.max_seq_len)
    data_loader = get_data_loader(dataset, task, 1, False, False, options.ddp)
    with torch.no_grad():
        for batch in data_loader:
            split_point = batch["prompt_lengths"][0]
            gen_batch = trim_batch(batch, 0, split_point)
            prompt_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]
            generate(model, gen_batch, options)
            pred_text = decode_batch(trim_batch(gen_batch, split_point, options.max_seq_len), dataset.text_tokenizer)[0]
            og_text = decode_batch(trim_batch(batch, split_point, options.max_seq_len), dataset.text_tokenizer)[0]

            print("Prompt:", prompt_text)
            print("OG Text:", og_text)
            print("Prediction:", pred_text)
            print("")
