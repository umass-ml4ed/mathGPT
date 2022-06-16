import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions as GPTOutput

from loading import CollatedBatch
from utils import device
from constants import TokenType, PADDING_TOKEN_ID

class GPTBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, batch: CollatedBatch):
        # Passing padding tokens to the model will break it when looking up embeddings, so just use a valid dummy value
        # Not an issue since we're passing explicit labels that still have the padding
        input_ids = batch["token_ids"]
        input_ids[input_ids == PADDING_TOKEN_ID] = 0
        # This is hacky but when generating a new label prediction, gen_labels will just contain the padding region over the prompt
        labels = input_ids if torch.all(batch["gen_labels"] == PADDING_TOKEN_ID) else batch["gen_labels"]

        output: GPTOutput = self.gpt2_lm(input_ids=input_ids, labels=labels)
        probs = {token_type: torch.zeros(input_ids.shape[0], input_ids.shape[1], 1).to(device) for token_type in TokenType}
        probs[TokenType.TEXT] = nn.Softmax(dim=-1)(output.logits)
        return output.loss, probs
