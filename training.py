import time
import json
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from typing import Optional

from model_math_gpt import MathGPT
from loading import load_articles, Dataset, Collator, trim_batch
from generate import generate, decode_batch
from utils import TrainOptions, device
from constants import Mode

def save_model(model: MathGPT, model_name: str, options: TrainOptions):
    torch.save(model.state_dict(), f"{model_name}.pt")
    with open(f"{model_name}.json", "w") as config_file:
        json.dump(options.__dict__, config_file, indent=4)

def load_model(model_name: str):
    with open(f"{model_name}.json") as config_file:
        options = TrainOptions(json.load(config_file))
    model = MathGPT().to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    return model, options

def evaluate_model(model: MathGPT, validation_loader: DataLoader, mode: Mode):
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(validation_loader):
            loss, _, type_preds, token_preds = model(batch)
            # TODO: put all this logic into a function, and then write a unit test
            # For predictions and targets, stack types and tokens in last dimension
            type_preds = type_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
            token_preds = token_preds[:, :-1].contiguous().view(-1).detach().cpu().numpy()
            predictions = np.stack([type_preds, token_preds], axis=-1)
            type_targets = batch["token_types"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
            token_targets = batch["token_ids"][:, 1:].contiguous().view(-1).detach().cpu().numpy()
            targets = np.stack([type_targets, token_targets], axis=-1)
            mask = batch["attention_mask"][:, 1:].contiguous().view(-1).detach().cpu().numpy() == 1
            if mode == Mode.PRETRAIN:
                all_predictions.append(predictions[mask])
                all_labels.append(targets[mask])
            total_loss += float(loss.detach().cpu().numpy())
            num_batches += 1

    if mode == Mode.PRETRAIN:
        all_preds_np = np.concatenate(all_predictions, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        # Get indices where both type and token match
        match = all_preds_np == all_labels_np
        match = match[:, 0] & match[:, 1]
        accuracy = sum(match) / len(match)
        return total_loss / num_batches, accuracy

def train(model: MathGPT, mode: Mode, model_name: str, train_loader: DataLoader, validation_loader: Optional[DataLoader], options: TrainOptions):
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
        if mode == Mode.PRETRAIN:
            # train_loss, train_acc = evaluate_model(model, train_loader, mode)
            train_loss = 0
            train_acc = 0
            val_loss, val_acc = evaluate_model(model, validation_loader, mode) if validation_loader else (0, 0)
            cur_stats = [epoch, val_loss]
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Accuracy: {train_acc:.3f}, "
                  f"Val Loss: {val_loss:.3f}, Accuracy: {val_acc:.3f}, "
                  f"Time: {time.time() - start_time:.2f}")

        # Save model for best validation metric
        if not best_metric or val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            best_stats = cur_stats
            print("Saving model")
            # TODO: consider saving info for checkpointing - epoch, seed, etc. (look up examples)
            # TODO: saving is slow - see if we can just save best model in memory and then save to file later
            save_model(model, model_name, options)

        # Stop training if we haven't improved in a while
        if options.patience and (epoch - best_epoch >= options.patience):
            print("Early stopping")
            break

    return best_stats

def pretrain(model_name: str, options: TrainOptions):
    articles = load_articles()
    dataset = Dataset(articles, options.max_seq_len)
    train_data = Subset(dataset, list(range(0, int(len(dataset) * .9))))
    val_data = Subset(dataset, list(range(int(len(dataset) * .9), len(dataset))))
    train_loader = DataLoader(
        train_data,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=True
    )
    validation_loader = DataLoader(
        val_data,
        collate_fn=Collator(),
        batch_size=options.batch_size
    ) if val_data is not None else None

    model = MathGPT().to(device)
    train(model, Mode.PRETRAIN, model_name, train_loader, validation_loader, options)

def test_lm(model_name: str, test_article: str):
    model, options = load_model(model_name)

    with open(test_article) as test_article_file:
        article = json.load(test_article_file)
        article["name"] = test_article_file
    dataset = Dataset([article], options.max_seq_len)
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=len(dataset)
    )

    with torch.no_grad():
        trim_point = int(options.max_seq_len * .9)
        for batch in data_loader:
            # Generate new sequences given first half of original sequence as input
            gen_batch = trim_batch(batch, 0, trim_point)
            generate(model, gen_batch, options.max_seq_len)

            # Decode the generated sequences and compare to the original
            prompts_decoded = decode_batch(trim_batch(batch, 0, trim_point), dataset.text_tokenizer)
            og_decoded = decode_batch(trim_batch(batch, trim_point, options.max_seq_len), dataset.text_tokenizer)
            preds_decoded = decode_batch(trim_batch(gen_batch, trim_point, options.max_seq_len), dataset.text_tokenizer)
            for prompt, og_text, pred_text in zip(prompts_decoded, og_decoded, preds_decoded):
                print("Prompt:", prompt)
                print("OG Text:", og_text)
                print("Prediction:", pred_text)
                print("")
