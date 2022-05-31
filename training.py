import time
import os
import json
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from model_math_gpt import MathGPTBase, MathGPTLM, MathGPTClassifier
from loading import PreTrainDataset, GenTaskDataset, ClassifyTaskDataset, Collator
from evaluate import evaluate_lm, evaluate_cls_task, evaluate_gen_task
from generate import generate
from decode import decode_batch
from utils import TrainOptions, device, is_cls_task
from constants import DownstreamTask, DOWNSTREAM_TASK_TO_NUM_CLASSES, WIKI_DATA

def get_article_names():
    return [os.path.join(WIKI_DATA, article_filename) for article_filename in os.listdir(WIKI_DATA)]

def save_model(model: MathGPTBase, model_name: str, options: TrainOptions):
    torch.save(model.state_dict(), f"{model_name}.pt")
    with open(f"{model_name}.json", "w") as config_file:
        json.dump(options.__dict__, config_file, indent=4)

def load_model(model_name: str, task: Optional[DownstreamTask] = None):
    with open(f"{model_name}.json") as config_file:
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
        return evaluate_cls_task(model, dataset, options)
    return evaluate_gen_task(model, dataset, options)

def train(model: MathGPTBase, model_name: str, train_loader: DataLoader, validation_dataset: Dataset, options: TrainOptions, task: Optional[DownstreamTask] = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.lr, weight_decay=options.weight_decay)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
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
            loss = model(batch)[0]
            loss.backward()
            train_loss += float(loss.detach().cpu().numpy())
            num_batches += 1
            if num_batches % options.grad_accum_batches == 0 or num_batches == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        model.eval() # Set model to evaluation mode
        val_loss, results = evaluate_model(model, validation_dataset, task, options)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss / num_batches:.3f}, Val Loss: {val_loss:.3f}, {results}, Time: {time.time() - start_time:.2f}")

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

    return best_stats

def pretrain(model_name: str, options: TrainOptions):
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
    model = MathGPTLM(options).to(device)
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

    dataset = PreTrainDataset([test_article], options.max_seq_len // 2)
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=1
    )

    with torch.no_grad():
        data_loader_it = iter(data_loader)
        gen_batch = next(data_loader_it)
        prompt_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]
        generate(model, gen_batch, options.max_seq_len)
        pred_text = decode_batch(gen_batch, dataset.text_tokenizer)[0]
        followup_batch = next(data_loader_it)
        og_text = decode_batch(followup_batch, dataset.text_tokenizer)[0]

        print("Prompt:", prompt_text)
        print("OG Text:", og_text)
        print("Prediction:", pred_text)

def train_downstream_task(model_name: str, pretrained_name: str, task: DownstreamTask, options: TrainOptions):
    # TODO: get data files for the given task
    if is_cls_task(task):
        options.num_classes = DOWNSTREAM_TASK_TO_NUM_CLASSES.get(task)
        dataset = ClassifyTaskDataset([], options.max_seq_len)
        model = MathGPTClassifier(options)
    else:
        dataset = GenTaskDataset([], options.max_seq_len)
        model = MathGPTLM(options)
    model.load_pretrained(pretrained_name)
    train_data = Subset(dataset, list(range(0, int(len(dataset) * .9))))
    val_data = Subset(dataset, list(range(int(len(dataset) * .9), len(dataset))))
    train_loader = DataLoader(
        train_data,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False
    )
    train(model, model_name, train_loader, val_data, options)

def evaluate_downstream_task(model_name: str, task: DownstreamTask):
    model, options = load_model(model_name, task)

    # TODO: get data files for the given task
    if is_cls_task(task):
        dataset = ClassifyTaskDataset([], options.max_seq_len)
    else:
        dataset = GenTaskDataset([], options.max_seq_len)

    evaluate_model(model, dataset, task, options)
