import time
import json
from typing import List
import torch
import numpy as np
from sklearn import metrics
from transformers import GPT2TokenizerFast
from mathGPT.constants import CollatedBatch, TokenType
from tqdm import tqdm
from mathGPT.vocabulary import Vocabulary

from model_math_gpt import MathGPT
from loading import load_articles, Dataset, Collator, trim_batch
from utils import TrainOptions, device
from constants import Mode

USE_PATIENCE = False

def load_from_config(model_name: str):
    # TODO: open config file and create TrainOptions
    # TODO: open .pt file and create corresponding model
    # TODO: return model and options
    pass

def evaluate_model(model, validation_loader: torch.utils.data.DataLoader, mode: Mode):
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(validation_loader):
            loss, _, type_preds, token_preds = model(batch)
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

def train(model, mode: Mode, model_name: str, train_loader, validation_loader, lr=1e-4, weight_decay=1e-6, epochs=200, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    best_metric = None
    best_stats = None
    cur_stats = None
    best_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()
        model.train() # Set model to training mode
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            loss = model(batch)[0]
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu().numpy())
            num_batches += 1

        model.eval() # Set model to evaluation mode
        if mode == Mode.PRETRAIN:
            train_loss, train_acc = evaluate_model(model, train_loader, mode)
            val_loss, val_acc = evaluate_model(model, validation_loader, mode) if validation_loader else (0, 0, 0, 0)
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
            # TODO: try saving full config for simpler loading
            # also consider saving info for checkpointing - epoch, seed, etc. (look up examples)
            # TODO: saving is slow - see if we can just save best model in memory and then save to file later
            torch.save(model.state_dict(), f"{model_name}.pt")

        # Stop training if we haven't improved in a while
        if USE_PATIENCE and (epoch - best_epoch >= patience):
            print("Early stopping")
            break

    return best_stats

def pretrain(model_name: str, options: TrainOptions):
    articles = load_articles()[:100]
    dataset = Dataset(articles, options.max_seq_len)
    train_data = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * .8))))
    val_data = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * .8), len(dataset))))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        val_data,
        collate_fn=Collator(),
        batch_size=options.batch_size
    ) if val_data is not None else None

    model = MathGPT().to(device)
    train(model, Mode.PRETRAIN, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs, patience=10)

# TODO: move code to another file
def decode_batch(batch: CollatedBatch, text_tokenizer: GPT2TokenizerFast) -> List[str]:
    all_decoded_sequences: List[str] = []
    for seq_idx in range(len(batch["token_ids"])):
        result = ""
        is_text = True # Assumption: sequence will always either start with a text token, or start formula token
        sub_seq_start = sub_seq_end = 0
        for tok_idx, (token_type, token_id) in enumerate(zip(batch["token_types"][seq_idx], batch["token_ids"][seq_idx])):
            # At start formula, switch to math context, and decode any prior text tokens
            if token_type == TokenType.START_FORMULA:
                is_text = False
                if sub_seq_start != sub_seq_end:
                    result += text_tokenizer.decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])
                sub_seq_start = sub_seq_end = tok_idx + 1
                continue

            # At end formula, switch to text context
            if token_type == TokenType.END_FORMULA:
                is_text = True
                # TODO: decode the OPT and add to result
                sub_seq_start = sub_seq_end = tok_idx + 1
                continue

            # NOTE: this is temporary until we implement tree decoding
            if not is_text:
                result += Vocabulary.get_symbol(int(token_type), int(token_id)) + " "

            sub_seq_end = tok_idx + 1

            # TODO: handle EOS tokens

        # Decode any trailing text tokens at the end
        if sub_seq_start != sub_seq_end:
            result += text_tokenizer.decode(batch["token_ids"][seq_idx][sub_seq_start : sub_seq_end])

        all_decoded_sequences.append(result)

    return all_decoded_sequences


def test_lm(model_name: str, test_article: str, options: TrainOptions):
    # TODO: maybe load options from config
    with open(test_article) as test_article_file:
        article = json.load(test_article_file)
    dataset = Dataset([article], options.max_seq_len)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=len(dataset)
    )
    model = MathGPT().to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))

    # TODO: detailed decoding code should move elsewhere
    with torch.no_grad():
        trim_point = options.max_seq_len // 2
        for batch in data_loader:
            batch_size = batch["token_ids"].shape[0]
            gen_batch = trim_batch(batch, 0, trim_point)
            for _ in range(trim_point, options.max_seq_len):
                # TODO: extract and pass past_key_values
                _, _, type_preds, token_preds = model(gen_batch)
                # TODO: apply temperature
                # TODO: from transformers import tf_top_k_top_p_filtering
                gen_batch["token_ids"] = torch.concat([gen_batch["token_ids"], token_preds[:, -1].unsqueeze(1)], dim=-1)
                gen_batch["token_types"] = torch.concat([gen_batch["token_types"], type_preds[:, -1].unsqueeze(1)], dim=-1)
                # TODO: math positions
                gen_batch["attention_mask"] = torch.concat([gen_batch["attention_mask"], torch.ones(batch_size).unsqueeze(1).to(device)], dim=-1)
                # TODO: can we make faster by removing sequences in the batch that hit EOS? or set attention mask to 0 after EOS?

            prompts_decoded = decode_batch(trim_batch(batch, 0, trim_point), dataset.text_tokenizer)
            og_decoded = decode_batch(trim_batch(batch, trim_point, options.max_seq_len), dataset.text_tokenizer)
            preds_decoded = decode_batch(trim_batch(gen_batch, trim_point, options.max_seq_len), dataset.text_tokenizer)

            for prompt, og_text, pred_text in zip(prompts_decoded, og_decoded, preds_decoded):
                print("Prompt:", prompt)
                print("OG Text:", og_text)
                print("Prediction:", pred_text)
                print("")
