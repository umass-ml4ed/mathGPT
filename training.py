import time
import os
import json
from typing import Optional, List
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from mathGPT.constants import Article

from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from loading import PreTrainDataset, PreTrainDatasetPreloaded, GenTaskDataset, ClassifyTaskDataset, Collator, trim_batch
from evaluate import evaluate_lm, evaluate_lm_accuracy, evaluate_cls_task, evaluate_gen_task
from generate import generate
from decode import decode_batch
from utils import TrainOptions, device, is_cls_task, new_neptune_run
from constants import DownstreamTask, DOWNSTREAM_TASK_TO_NUM_CLASSES, WIKI_DATA, OFEQ_DATA

def get_article_names():
    return [os.path.join(WIKI_DATA, article_filename) for article_filename in os.listdir(WIKI_DATA)]

def save_model(model: MathGPTBase, model_name: str, options: TrainOptions):
    torch.save(model.state_dict(), f"{model_name}.pt")
    with open(f"{model_name}.json", "w", encoding="utf-8") as config_file:
        json.dump(options.__dict__, config_file, indent=4)

def load_model(model_name: str, task: Optional[DownstreamTask] = None):
    with open(f"{model_name}.json", encoding="utf-8") as config_file:
        options = TrainOptions(json.load(config_file))
    if is_cls_task(task):
        model = MathGPTClassifier(options).to(device)
    else:
        model = MathGPTLM(options).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    return model, options

def evaluate_model(model: MathGPTBase, dataset: Dataset, task: Optional[DownstreamTask], options: TrainOptions):
    if not task:
        return evaluate_lm(model, dataset, options)
    if is_cls_task(task):
        return evaluate_cls_task(model, dataset, task, options)
    return evaluate_lm_accuracy(model, dataset, task, options)

def train(model: MathGPTBase, model_name: str, train_loader: DataLoader, validation_dataset: Dataset, options: TrainOptions, task: Optional[DownstreamTask] = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    # Scaler prevents gradient underflow when using fp16 precision
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if options.amp else None
    # Split computations across GPUs if multiple available
    use_dp = torch.cuda.device_count() > 1
    model_dp = torch.nn.parallel.DataParallel(model) if use_dp else None

    run = new_neptune_run()
    run["name"] = model_name
    run["options"] = options.__dict__
    run["task"] = str(task)

    best_metric = None
    best_stats = None
    cur_stats = None
    best_epoch = 0

    print("Training...")
    for epoch in range(options.epochs):
        start_time = time.time()
        model.train() # Set model to training mode
        train_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_loader):
            if scaler:
                with torch.cuda.amp.autocast():
                    if use_dp:
                        loss = model_dp(batch)[0]
                        loss = loss.mean()
                    else:
                        loss = model(batch)[0]
                scaler.scale(loss).backward()
            else:
                if use_dp:
                    loss = model_dp(batch)[0]
                    loss = loss.mean()
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
        run["train/loss"].log(avg_train_loss)
        run["val/loss"].log(val_loss)
        run["val/metrics"].log(results)
        print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.3f}, Val Loss: {val_loss:.3f}, {results}, Time: {time.time() - start_time:.2f}")

        # Save model for best validation metric
        if not best_metric or val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            best_stats = cur_stats
            print("Saving model")
            # TODO: consider saving info for checkpointing - epoch, seed, etc. (look up examples)
            save_model(model, model_name, options)

        # Stop training if we haven't improved in a while
        if options.patience and (epoch - best_epoch >= options.patience):
            print("Early stopping")
            break

    run.stop()
    return best_stats

def pretrain(model_name: str, pretrained_name: str, options_dict: dict):
    if pretrained_name:
        model, options = load_model(pretrained_name)
        options.update(options_dict)
    else:
        model = MathGPTLM(options).to(device)
        options = TrainOptions(options_dict)
    articles = get_article_names()
    dataset = PreTrainDataset(articles, options.max_seq_len)
    train_data = Subset(dataset, list(range(0, int(len(dataset) * .9))))
    val_data = Subset(dataset, list(range(int(len(dataset) * .9), len(dataset))))
    train_loader = DataLoader(
        train_data,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=True
    )
    train(model, model_name, train_loader, val_data, options)

def evaluate_pretrained_lm(model_name: str, test_options: dict):
    model, options = load_model(model_name)
    options.update(test_options)

    # TODO: get test split from a file
    # TODO: try excluding test articles that contain UNKs, see how many are left out
    articles = get_article_names()
    test_articles = articles[int(len(articles) * .9):]
    dataset = PreTrainDataset(test_articles, max_seq_len=None)
    loss, results = evaluate_lm(model, dataset, options)
    print(f"Loss: {loss:.3f}, {results}")

def test_lm(model_name: str, test_article: str, test_options: dict):
    model, options = load_model(model_name)
    options.update(test_options)

    wiki = False
    if wiki:
        dataset = PreTrainDataset([test_article], options.max_seq_len // 2)
        data_loader = DataLoader(
            dataset,
            collate_fn=Collator(),
            batch_size=1
        )
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
        dataset = PreTrainDatasetPreloaded(probes, options.max_seq_len)
        data_loader = DataLoader(
            dataset,
            collate_fn=Collator(),
            batch_size=1
        )
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
        dataset = ClassifyTaskDataset([], options.max_seq_len)
        model = MathGPTClassifier(options).to(device)
    else:
        with open(os.path.join(OFEQ_DATA, "train.json"), encoding="utf-8") as headlines_file:
            train_headlines = json.load(headlines_file)
        with open(os.path.join(OFEQ_DATA, "val.json"), encoding="utf-8") as headlines_file:
            val_headlines = json.load(headlines_file)
        train_data = GenTaskDataset(train_headlines, options.max_seq_len)
        val_data = GenTaskDataset(val_headlines, options.max_seq_len)
        model = MathGPTLM(options).to(device)
    model.load_pretrained(pretrained_name)
    train_loader = DataLoader(
        train_data,
        collate_fn=Collator(task),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False
    )
    train(model, model_name, train_loader, val_data, options, task)

def evaluate_downstream_task(model_name: str, task: DownstreamTask, eval_options: dict):
    model, options = load_model(model_name, task)
    options.update(eval_options)

    # TODO: get data files for the given task
    if is_cls_task(task):
        dataset = ClassifyTaskDataset([], options.max_seq_len)
        evaluate_cls_task(model, dataset, task, options)
    else:
        with open(os.path.join(OFEQ_DATA, "test.json"), encoding="utf-8") as headlines_file:
            headlines = json.load(headlines_file)
        dataset = GenTaskDataset(headlines, options.max_seq_len)
        _, results = evaluate_gen_task(model, dataset, task, options)
        print(results)

def test_gen_task(model_name: str, task: DownstreamTask):
    model, options = load_model(model_name, task)
    samples_to_try = 5

    with open(os.path.join(OFEQ_DATA, "test.json"), encoding="utf-8") as headlines_file:
        headlines = json.load(headlines_file)[:samples_to_try]
    dataset = GenTaskDataset(headlines, options.max_seq_len)
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(task),
        batch_size=1
    )
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
