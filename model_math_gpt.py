import torch
from transformers import GPT2Model, GPT2LMHeadModel

from constants import CollatedBatch, TokenType
from utils import device

# Leverages pre-trained GPT2 from transformers library
# https://huggingface.co/docs/transformers/model_doc/gpt2
# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py

EMB_SIZE = 768
TEXT_VOCAB_SIZE = 50257

USE_LM = False

class MathGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
        if not USE_LM:
            self.transformer: GPT2Model = self.gpt2_lm.transformer
            self.text_token_pred_layer: torch.nn.Linear = self.gpt2_lm.lm_head

        # TODO: create embeddings for different math tokens
        # TODO: create embeddings/encodings for math positions

    def forward(self, batch: CollatedBatch):
        # TODO: distinguish generation (keep last prediction) vs training (remove last prediction)
        if USE_LM:
            gpt_output = self.gpt2_lm(
                input_ids=batch["token_ids"],
                labels=batch["token_ids"]
            )
            target_tokens_shifted = batch["token_ids"][:, 1:].contiguous()
            attn_mask_shifted = batch["attention_mask"][:, 1:].contiguous()
            return gpt_output.loss, torch.argmax(gpt_output.logits[:, :-1], dim=-1), target_tokens_shifted, attn_mask_shifted

        batch_size, max_seq_len = batch["token_ids"].shape

        # Input tensor to store embeddings for each sequence in batch
        input_embeddings = torch.zeros((batch_size, max_seq_len, EMB_SIZE)).to(device)

        # Get indices of all text tokens in each sequence
        text_idxs = batch["token_types"] == TokenType.TEXT.value

        # Extract token ids of all text tokens, convert to embeddings, and copy to input embedding tensor
        input_embeddings[text_idxs] = self.transformer.wte(batch["token_ids"][text_idxs])

        # TODO: lookup embeddings from different vocabs and do numeric conversion

        # TODO: construct input by summing the different embeddings together
        # token embedding as calculated above
        # type indicator embedding - learnable embedding per type, experiment with granularity (text/math, text/switch/math, text/var/num/op, etc.)
        # math pos - learnable or 0 for non-math tokens, sin/cos encoding or linear projection for math tokens (think about commonality between these across formulas)

        gpt_output = self.transformer(
            # past_key_values=[], # TODO: for speeding up decoding
            use_cache=False, # TODO: set to True for decoding, but otherwise runs out of memory
            output_attentions=False,
            # attention_mask=batch["attention_mask"], # TODO: when do we need this?
            inputs_embeds=input_embeddings,
            return_dict=True
        )

        # Simple implementation: just do text tokens
        logits = self.text_token_pred_layer(gpt_output.last_hidden_state)
        # Have logit at time step t predict token at time step t+1, thus shifting
        logits_shifted = logits[:, :-1].contiguous() # Contiguous is to reorganize memory layout to allow flattening
        target_tokens_shifted = batch["token_ids"][:, 1:].contiguous()
        attn_mask_shifted = batch["attention_mask"][:, 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss(reduction="none")(logits_shifted.view(-1, TEXT_VOCAB_SIZE), target_tokens_shifted.view(-1))
        # loss *= attn_mask_shifted.view(-1) # TODO: this seems necessary, but why doesn't GPT2LMHeadModel do it?
        # TODO: compare attention masking loss to setting labels to -100 (or other negative value) since CE Loss should not take those into account (apparently)
        # loss *= text_idxs[:, 1:].view(-1)
        return loss.mean(), torch.argmax(logits_shifted, dim=-1), target_tokens_shifted, attn_mask_shifted

        # TODO: run gpt_output.hidden states through linear layer(s) to get token/type activations
        # option 1: single output vector with an entry for every token across all types
        # option 2: single output vector with an entry for every type, followed by subsequent token vector per type (can be implemented as single token vector with regions)

        # TODO: constrain outputs (mask activations)
        # text can go to text or switch context
        # switch context must go to math if after text, otherwise must go to text if after math
        # math can go to math or switch context
        # enforce math tree rules (if possible)
        # handle math END tokens - could be explicit, could also be implicit (probability of popping n times and then generating next child)

        # TODO: calculate token probabilities, look at numGPT and FORTE papers to verify their implementations
        # option 1: softmax across all possible tokens
        # option 2: joint between type and token - P(type, token) = P(token|type) * P(type)
        #   P(type) taken from softmax, P(token|type) = 0 if token not in type, else taken from softmax

        # TODO: calculate loss

        # TODO: calculate most likely predictions
