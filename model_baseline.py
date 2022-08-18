from typing import Optional, Dict
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions as GPTOutput

from loading import CollatedBatch
from utils import device, TrainOptions
from constants import TokenType, PADDING_TOKEN_ID, EMB_SIZE

class GPTLMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, batch: CollatedBatch, labels: Optional[torch.Tensor] = None, output_attentions: bool = False):
        # Passing padding tokens to the model will break it when looking up embeddings, so just use a valid dummy value
        # Not an issue since we're passing explicit labels that still have the padding
        input_ids = batch["token_ids"]
        input_ids[input_ids == PADDING_TOKEN_ID] = 0
        if labels is None:
            if batch["gen_labels"] is not None:
                # This is hacky but when generating a new label prediction, gen_labels will just contain the padding region over the prompt
                labels = input_ids if torch.all(batch["gen_labels"] == PADDING_TOKEN_ID) else batch["gen_labels"]
            else:
                labels = input_ids

        output: GPTOutput = self.gpt2_lm(input_ids=input_ids, labels=labels, output_attentions=output_attentions)
        probs = {token_type: torch.zeros(input_ids.shape[0], input_ids.shape[1], 1).to(device) for token_type in TokenType}
        probs[TokenType.TEXT] = nn.Softmax(dim=-1)(output.logits)
        return output.loss, probs, output.attentions

class GPTClassifierBaseline(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.classifier_head = nn.Linear(EMB_SIZE, options.num_classes)

    def forward(self, batch: CollatedBatch):
        batch_size = batch["token_ids"].shape[0]

        # Passing padding tokens to the model will break it when looking up embeddings, so just use a valid dummy value
        # Not an issue since we use sequence_lengths to determine the output to use for the classifier head
        input_ids = batch["token_ids"]
        input_ids[input_ids == PADDING_TOKEN_ID] = 0

        # Get GPT output
        gpt_output: GPTOutput = self.gpt2(input_ids=input_ids, attention_mask=batch["attention_mask"])

        # Get prediction from last token in input per sequence
        pred_states = gpt_output.last_hidden_state[torch.arange(batch_size), batch["sequence_lengths"] - 1]
        predictions: torch.Tensor = self.classifier_head(pred_states)
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss: torch.Tensor = loss_fn(predictions, batch["cls_labels"])

        return loss, predictions
