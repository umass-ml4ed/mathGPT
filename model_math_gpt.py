from typing import Dict, List, Tuple
import torch
from torch import nn
from transformers import GPT2Model, GPT2LMHeadModel

from vocabulary import Vocabulary
from constants import CollatedBatch, TokenType
from utils import device

# Leverages pre-trained GPT2 from transformers library
# https://huggingface.co/docs/transformers/model_doc/gpt2
# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py

TEXT_ONLY = False

EMB_SIZE = 768
TEXT_VOCAB_SIZE = 50257
NUM_TYPES = len(TokenType)

# Map of type to allowed types for next token
ALLOWED_TRANSITIONS: Dict[TokenType, List[TokenType]] = {
    TokenType.TEXT: [TokenType.TEXT, TokenType.START_FORMULA],
    TokenType.START_FORMULA: [TokenType.VAR, TokenType.NUM, TokenType.OP],
    TokenType.END_FORMULA: [TokenType.TEXT], # TODO: we could maybe get away with no end_formula token, since it's implicit from the tree stack being empty
    # TODO: additional logic - if vars and nums are leaves, and can't gen end_formula if stack is not empty
    TokenType.VAR: [TokenType.END_FORMULA, TokenType.VAR, TokenType.NUM, TokenType.OP],
    TokenType.NUM: [TokenType.END_FORMULA, TokenType.VAR, TokenType.NUM, TokenType.OP],
    TokenType.OP: [TokenType.END_FORMULA, TokenType.VAR, TokenType.NUM, TokenType.OP],
}

def get_inverted_allowed_transitions():
    """
    Get map of type to list of types that can precede it
    """
    allowed_transitions_inv: Dict[TokenType, List[TokenType]] = {}
    for token_type, allowed_types in ALLOWED_TRANSITIONS.items():
        for allowed_type in allowed_types:
            allowed_transitions_inv.setdefault(allowed_type, []).append(token_type)
    return allowed_transitions_inv

class MathGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Extract pre-trained GPT2 transformer and text prediction head
        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
        self.transformer: GPT2Model = self.gpt2_lm.transformer
        self.text_token_pred_layer: nn.Linear = self.gpt2_lm.lm_head

        # Create type embeddings
        # TODO: ensure these are sufficiently small to avoid confusing the pre-trained model
        self.type_embeddings = nn.Embedding(NUM_TYPES, EMB_SIZE) # TODO: also try just have text, math, and transition type embeddings

        # TODO: not great that we gotta assign str(Type.value)... see if ModuleList fixes this
        # Create token embedding matrix for each type
        self.token_embeddings = nn.ModuleDict({
            str(token_type.value): nn.Embedding(Vocabulary.num_tokens_in_type(token_type), EMB_SIZE)
            for token_type in TokenType
            if token_type not in (TokenType.TEXT, TokenType.START_FORMULA, TokenType.END_FORMULA)
        })
        # Use pre-trained text token embeddings
        self.token_embeddings[str(TokenType.TEXT.value)] = self.transformer.wte
        # Single embedding for formula start/end tokens each
        self.token_embeddings[str(TokenType.START_FORMULA.value)] = nn.Embedding(1, EMB_SIZE)
        self.token_embeddings[str(TokenType.END_FORMULA.value)] = nn.Embedding(1, EMB_SIZE)

        # TODO: create embeddings/encodings for math positions (try learnable embeddings for each tree position)

        # Predictive type layer for generation
        self.type_pred_layer = nn.Linear(EMB_SIZE, NUM_TYPES)

        # Predictive token layers for generation
        # TODO: better handling for start/end formula
        self.type_to_token_pred_layer = nn.ModuleDict({
            str(token_type.value): nn.Linear(EMB_SIZE, 1 if token_type in (TokenType.START_FORMULA, TokenType.END_FORMULA) else Vocabulary.num_tokens_in_type(token_type))
            for token_type in TokenType
            if token_type not in (TokenType.TEXT,)#, TokenType.START_FORMULA, TokenType.END_FORMULA)
        })
        # Use pre-trained predictive head for text tokens
        self.type_to_token_pred_layer[str(TokenType.TEXT.value)] = self.gpt2_lm.lm_head
        # Formula start/stop types only have a single token each, so no need for predicting them

    def get_input_embeddings(self, batch: CollatedBatch) -> torch.Tensor:
        """
        Return tensor (batch_size x max_seq_len x emb_size) containing input embeddings
        At each time step per sequence, add the type, token, and math position embeddings
        """
        # Start with type embeddings
        input_embeddings = self.type_embeddings(batch["token_types"])

        # Add token embeddings for each type
        for token_type in TokenType:
            type_idxs = batch["token_types"] == token_type
            input_embeddings[type_idxs] += self.token_embeddings[str(token_type.value)](batch["token_ids"][type_idxs])

        # TODO: add math position embeddings

        return input_embeddings

    def forward(self, batch: CollatedBatch): # TODO: return type
        # import pdb; pdb.set_trace()
        batch_size, max_seq_len = batch["token_ids"].shape

        gpt_output = self.transformer( # TODO: typedef for output
            # past_key_values=[], # TODO: for speeding up decoding
            use_cache=False, # TODO: set to True for decoding, but otherwise runs out of memory
            output_attentions=False,
            # attention_mask=batch["attention_mask"], # TODO: when do we need this?
            inputs_embeds=self.get_input_embeddings(batch),
            return_dict=True
        )

        if TEXT_ONLY:
            # Simple implementation: just do text tokens
            logits = self.text_token_pred_layer(gpt_output.last_hidden_state)
            # Have logit at time step t predict token at time step t+1, thus shifting
            logits_shifted = logits[:, :-1].contiguous() # Contiguous is to reorganize memory layout to allow flattening
            target_tokens_shifted = batch["token_ids"][:, 1:].contiguous()
            attn_mask_shifted = batch["attention_mask"][:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(reduction="none")(logits_shifted.view(-1, TEXT_VOCAB_SIZE), target_tokens_shifted.view(-1))
            # loss *= attn_mask_shifted.view(-1) # TODO: this seems necessary, but why doesn't GPT2LMHeadModel do it?
            # TODO: compare attention masking loss to setting labels to -100 (or other negative value) since CE Loss should not take those into account (apparently)
            # loss *= text_idxs[:, 1:].view(-1)
            return loss.mean(), logits, torch.argmax(logits_shifted, dim=-1), target_tokens_shifted, attn_mask_shifted

        # Map type to indices in batch that match that type
        type_idxs = {
            token_type: batch["token_types"] == token_type
            for token_type in TokenType
        }

        # Calculate P(type) for each possible type
        # Create mask with allowed (False) and unallowed (True) types for the following token at each index
        type_mask = torch.full((batch_size, max_seq_len, NUM_TYPES), True).to(device)
        for token_type, allowed_types in ALLOWED_TRANSITIONS.items():
            for allowed_type in allowed_types:
                type_mask[:, :, allowed_type][type_idxs[token_type]] = False
        # Get predicted probability of types at each index
        type_preds = self.type_pred_layer(gpt_output.last_hidden_state)
        type_preds[type_mask] = -torch.inf
        type_probs = nn.Softmax(dim=-1)(type_preds)

        # Calculate P(token, type) for all types/tokens
        type_to_token_probs: Dict[TokenType, torch.Tensor] = {}
        for token_type, allowed_types in get_inverted_allowed_transitions().items():
            # Get all indices that are allowed to transition to the current type
            token_idxs = type_idxs[allowed_types[0]]
            for allowed_type in allowed_types[1:]:
                token_idxs = token_idxs | type_idxs[allowed_type]
            # Get predicted probability of tokens in the type, i.e. P(token|type)
            token_preds = self.type_to_token_pred_layer[str(token_type.value)](gpt_output.last_hidden_state[token_idxs])
            token_probs = nn.Softmax(dim=-1)(token_preds)
            # Multiply P(type) * P(token|type) to get P(token, type)
            type_to_token_probs[token_type] = torch.zeros((batch_size, max_seq_len, token_probs.shape[-1])).to(device)
            type_to_token_probs[token_type][token_idxs] = token_probs * type_probs[:, :, token_type][token_idxs].unsqueeze(1)

        # Calculate cross-entropy loss
        # TODO: see if we have to do anything with padding regions
        loss = torch.Tensor([0]).to(device)
        shifted_target_tokens = batch["token_ids"][:, 1:]
        for token_type in TokenType:
            # Get indices that have a target of this type
            shifted_type_idx = type_idxs[token_type][:, 1:]
            # Skip type if it doesn't exist as a target, going through with calculations results in nan loss
            if not torch.any(shifted_type_idx):
                continue
            # Get token probabilities for this type where the target matches
            selected_probs = type_to_token_probs[token_type][:, :-1][shifted_type_idx]
            # Add cross-entropy loss
            # TODO: seems like we can sometimes get invalid values here, maybe somehow assigning 0 probability to target
            log_probs = torch.log(selected_probs)
            loss += nn.NLLLoss()(log_probs, shifted_target_tokens[shifted_type_idx])

        # Calculate most likely predictions
        predicted_types, predicted_tokens = get_collapsed_predictions(type_to_token_probs)

        return loss, type_to_token_probs, predicted_types, predicted_tokens

        # TODO: look at numGPT and FORTE papers to verify their token probability calculations

def get_collapsed_predictions(type_to_token_probs: Dict[TokenType, torch.Tensor]) -> Tuple[torch.LongTensor, torch.LongTensor]:
    batch_size, max_seq_len = type_to_token_probs[TokenType.TEXT].shape[:2]
    predicted_types = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    predicted_tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    max_token_probs = torch.zeros((batch_size, max_seq_len)).to(device)
    for token_type in TokenType:
        # Get most likely token for this type
        max_values, max_indices = torch.max(type_to_token_probs[token_type], dim=-1)
        # Find indices where most likely token is higher than previously most likely token
        new_highest_prob_idx = max_values > max_token_probs
        # For those indices, set the predicted token and type
        predicted_tokens[new_highest_prob_idx] = max_indices[new_highest_prob_idx]
        predicted_types[new_highest_prob_idx] = token_type
        # Update highest seen probabilities
        max_token_probs = torch.maximum(max_token_probs, max_values)
    return predicted_types, predicted_tokens
