from typing import Dict, List, Optional
import torch
from torch import nn
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions as GPTOutput

from vocabulary import Vocabulary
from math_tokenize import POS_ENCODING_SIZE
from constants import CollatedBatch, TokenType, PADDING_TOKEN_ID
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
    TokenType.END_FORMULA: [TokenType.TEXT],
    TokenType.VAR: [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END],
    TokenType.NUM: [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END],
    TokenType.OP: [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END],
    TokenType.END: [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END],
    # Note that terminal tokens can only generate END_FORMULA if they are the last in a formula, so these cases are handled manually
}

def get_inverted_allowed_transitions():
    """
    Get map of type to list of types that can precede it
    """
    allowed_transitions_inv: Dict[TokenType, List[TokenType]] = {token_type: [] for token_type in TokenType}
    for token_type, allowed_types in ALLOWED_TRANSITIONS.items():
        for allowed_type in allowed_types:
            allowed_transitions_inv[allowed_type].append(token_type)
    return allowed_transitions_inv

class MathGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # Extract pre-trained GPT2 transformer and text prediction head
        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
        self.transformer: GPT2Model = self.gpt2_lm.transformer
        self.text_token_pred_layer: nn.Linear = self.gpt2_lm.lm_head

        # Create type embeddings
        # TODO: ensure these are sufficiently small (magnitude) to avoid confusing the pre-trained model
        self.type_embeddings = nn.Embedding(NUM_TYPES, EMB_SIZE) # TODO: also try just have text, math, and transition type embeddings

        # TODO: not great that we gotta assign str(Type.value)... see if ModuleList fixes this
        # Create token embedding matrix for each type
        self.token_embeddings = nn.ModuleDict({
            str(token_type.value): nn.Embedding(Vocabulary.num_tokens_in_type(token_type), EMB_SIZE)
            for token_type in TokenType
            if token_type not in (TokenType.TEXT, TokenType.START_FORMULA, TokenType.END_FORMULA, TokenType.END)
        })
        # Use pre-trained text token embeddings
        self.token_embeddings[str(TokenType.TEXT.value)] = self.transformer.wte
        # Single embedding each for special single-tokens types
        self.token_embeddings[str(TokenType.START_FORMULA.value)] = nn.Embedding(1, EMB_SIZE)
        self.token_embeddings[str(TokenType.END_FORMULA.value)] = nn.Embedding(1, EMB_SIZE)
        self.token_embeddings[str(TokenType.END.value)] = nn.Embedding(1, EMB_SIZE)

        # Linear projection to convert raw math pos encoding into one that can be added to the input embeddings
        # Not using bias so that the encoding is 0 for non-math tokens
        self.math_embedding_projection = nn.Linear(POS_ENCODING_SIZE, EMB_SIZE, bias=False)

        # Predictive type layer for generation
        self.type_pred_layer = nn.Linear(EMB_SIZE, NUM_TYPES)

        # Predictive token layers for generation
        self.type_to_token_pred_layer = nn.ModuleDict({
            str(token_type.value): nn.Linear(EMB_SIZE, Vocabulary.num_tokens_in_type(token_type))
            for token_type in TokenType
            if token_type not in (TokenType.TEXT, TokenType.START_FORMULA, TokenType.END_FORMULA, TokenType.END)
        })
        # Use pre-trained predictive head for text tokens
        self.type_to_token_pred_layer[str(TokenType.TEXT.value)] = self.gpt2_lm.lm_head
        # Special single-tokens types only have a single token each, so no need for predicting them

    def get_input_embeddings(self, batch: CollatedBatch) -> torch.Tensor:
        """
        Return tensor (batch_size x max_seq_len x emb_size) containing input embeddings
        At each time step per sequence, add the type, token, and math position embeddings
        """
        # Start with type embeddings
        input_embeddings = self.type_embeddings(batch["token_types"])

        # Add token embeddings for each type
        for token_type in TokenType:
            type_idxs = (batch["token_types"] == token_type) & (batch["token_ids"] != PADDING_TOKEN_ID)
            input_embeddings[type_idxs] += self.token_embeddings[str(token_type.value)](batch["token_ids"][type_idxs])

        # Add math position encodings
        input_embeddings += self.math_embedding_projection(batch["pos_encodings"])

        return input_embeddings

    def get_prediction_masks(self, batch: CollatedBatch):
        """
        Get relevant masks for type and token prediction
        Returns (map of type to mask of tokens that have that type, mask of tokens that must end a formula)
        """
        # Map type to indices in batch that match that type
        type_idxs = {
            token_type: batch["token_types"] == token_type
            for token_type in TokenType
        }
        # Find idxs that are the last of a formula. Two ways a formula can end:
        # 1) if it starts with an OP, then with its last child (END token at level == 1)
        # 2) if it starts with a terminal token, then with that token (level == 0)
        # Note we can't just look for END_FORMULA, because we might be generating in which case the END_FORMULA token may not exist yet
        final_formula_token_idx = type_idxs[TokenType.END] & (batch["pos_levels"] == 1)
        final_formula_token_idx |= (type_idxs[TokenType.VAR] | type_idxs[TokenType.NUM]) & (batch["pos_levels"] == 0)
        return type_idxs, final_formula_token_idx

    def get_type_probs(self, gpt_output: GPTOutput, type_idxs: Dict[TokenType, torch.Tensor], final_formula_token_idx: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of generating each type for each token in the batch, including masking to constrain to possible transitions
        """
        # Get raw prediction values from model output
        type_preds = self.type_pred_layer(gpt_output.last_hidden_state)
        # Create mask with allowed (False) and unallowed (True) types for the following token at each index
        type_mask = torch.full(type_preds.shape, True).to(device)
        for token_type, allowed_types in ALLOWED_TRANSITIONS.items():
            for allowed_type in allowed_types:
                type_mask[:, :, allowed_type][type_idxs[token_type]] = False
        # Tokens that finish a formula can only generate END_FORMULA
        just_end_form_token = torch.full((len(TokenType),), True).to(device)
        just_end_form_token[TokenType.END_FORMULA] = False
        type_mask[final_formula_token_idx] = just_end_form_token
        # Get predicted probability of types at each index
        type_preds[type_mask] = -torch.inf
        type_probs = nn.Softmax(dim=-1)(type_preds)
        return type_probs

    def get_token_probs(self, gpt_output: GPTOutput, type_probs: torch.Tensor, type_idxs: Dict[TokenType, torch.Tensor], final_formula_token_idx: torch.Tensor):
        """
        For each type, calculate the joint probability of generating that type and the tokens in that type
        """
        batch_size, max_seq_len = type_probs.shape[:2]
        type_to_token_probs: Dict[TokenType, torch.Tensor] = {}
        for token_type, allowed_types in get_inverted_allowed_transitions().items():
            # Get all indices that are allowed to transition to the current type
            if token_type == TokenType.END_FORMULA:
                token_idxs = final_formula_token_idx
            else:
                token_idxs = type_idxs[allowed_types[0]]
                for allowed_type in allowed_types[1:]:
                    token_idxs = token_idxs | type_idxs[allowed_type]
            # Get predicted probability of tokens in the type, i.e. P(token|type)
            if token_type in (TokenType.START_FORMULA, TokenType.END_FORMULA, TokenType.END):
                token_probs = torch.ones((batch_size, max_seq_len, 1)).to(device)[token_idxs]
            else:
                token_preds = self.type_to_token_pred_layer[str(token_type.value)](gpt_output.last_hidden_state[token_idxs])
                token_probs = nn.Softmax(dim=-1)(token_preds)
            # Multiply P(type) * P(token|type) to get P(token, type)
            type_to_token_probs[token_type] = torch.zeros((batch_size, max_seq_len, token_probs.shape[-1])).to(device)
            type_to_token_probs[token_type][token_idxs] = token_probs * type_probs[:, :, token_type][token_idxs].unsqueeze(1)
        return type_to_token_probs

    def get_prediction_loss(self, type_to_token_probs: Dict[TokenType, torch.Tensor], type_idxs: Dict[TokenType, torch.Tensor],
                            batch: CollatedBatch, labels: Optional[torch.Tensor]):
        """
        Calculate the cross-entropy prediction loss given the token probabilities
        """
        losses: List[torch.Tensor] = []
        shifted_target_tokens = (batch["token_ids"] if labels is None else labels)[:, 1:]
        if shifted_target_tokens.shape[1] == 0:
            # If the sequence has a length of 1, there are no targets to predict, so return a loss of 0
            return torch.tensor(0.0).to(device)

        for token_type in TokenType:
            # Get indices that have a target of this type
            shifted_target_idx = type_idxs[token_type][:, 1:]
            # Exclude padding regions, including them would add extra length to the vector that we get the mean of
            shifted_target_idx &= shifted_target_tokens != PADDING_TOKEN_ID
            # Skip type if it doesn't exist as a target, going through with calculations results in nan loss
            if not torch.any(shifted_target_idx):
                continue
            # Get token probabilities for this type where the target matches
            selected_probs = type_to_token_probs[token_type][:, :-1][shifted_target_idx]
            # Add cross-entropy loss, take average at end to weigh each token equally
            log_probs = torch.log(selected_probs)
            loss_fn = nn.NLLLoss(reduction="none")
            loss: torch.Tensor = loss_fn(log_probs, shifted_target_tokens[shifted_target_idx])
            if any(torch.isnan(l) for l in loss):
                import pdb; pdb.set_trace()
                print("nan loss!")
                loss = loss[~loss.isnan()]
                with open("debug.txt", "a") as debug_file:
                    debug_file.write(f"{batch}\n\n{token_type}\n\n{loss.isnan().nonzero()}\n\n{selected_probs}\n\n{type_to_token_probs}\n\n{type_idxs}\n\n")
            losses.append(loss)
        return torch.concat(losses, dim=0).mean()

    def forward(self, batch: CollatedBatch, labels: torch.Tensor = None): # TODO: return type
        # Run inputs through model
        gpt_output: GPTOutput = self.transformer(
            # past_key_values=[], # TODO: for speeding up decoding
            use_cache=False, # TODO: set to True for decoding, but otherwise runs out of memory
            output_attentions=False,
            # attention_mask=batch["attention_mask"], # TODO: this doesn't seem to make a difference with padding, might be performance-related
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
            return loss.mean(), logits, torch.argmax(logits_shifted, dim=-1), target_tokens_shifted, attn_mask_shifted

        # Get relevant masks
        type_idxs, final_formula_token_idx = self.get_prediction_masks(batch)

        # Calculate P(type) for each possible type
        type_probs = self.get_type_probs(gpt_output, type_idxs, final_formula_token_idx)

        # Calculate P(token, type) for all types/tokens
        type_to_token_probs = self.get_token_probs(gpt_output, type_probs, type_idxs, final_formula_token_idx)

        # Calculate cross-entropy loss
        loss = self.get_prediction_loss(type_to_token_probs, type_idxs, batch, labels)

        return loss, type_to_token_probs
