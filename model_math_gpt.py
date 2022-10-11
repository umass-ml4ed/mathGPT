from typing import Dict, List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions as GPTOutput

from loading import trim_batch
from vocabulary import Vocabulary, UNK_MAP, MATH_TYPES
from math_tokenize import POS_ENCODING_SIZE_FORTE
from data_types import CollatedBatch
from constants import (
    TokenType, TPE, SpecialOpToken, PastKeyValues,
    MODEL_SIZE_TO_EMB_SIZE, MODEL_SIZE_TO_NAME, PADDING_TOKEN_ID, MAX_FORMULA_DEPTH, MAX_FORMULA_WIDTH, TEXT_VOCAB_SIZE
)
from utils import device, TrainOptions, text_tokenizer

# Leverages pre-trained GPT2 from transformers library
# https://huggingface.co/docs/transformers/model_doc/gpt2
# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py

# Map of type to allowed types for next token
ALLOWED_TRANSITIONS: Dict[TokenType, List[TokenType]] = {
    TokenType.TEXT: [TokenType.TEXT, TokenType.START_FORMULA],
    TokenType.END_FORMULA: [TokenType.TEXT],
    # Remaining transition rules depend on model settings or local context
}

def apply_unks(token_ids: torch.Tensor, batch: CollatedBatch):
    """
    Convert values that exceed vocab size to UNK
    """
    new_token_ids = token_ids.clone()
    for token_type in MATH_TYPES:
        new_token_ids[(batch["token_types"] == token_type) & (new_token_ids >= Vocabulary.num_tokens_in_type(token_type))] = UNK_MAP[token_type]
    return new_token_ids

def get_target_tokens(batch: CollatedBatch, labels: Optional[torch.Tensor]):
    """
    Get target tokens for generative task
    """
    # Get appropriate targets
    if labels is not None:
        target_tokens = labels
    elif batch["gen_labels"] is not None:
        target_tokens = batch["gen_labels"]
    else:
        target_tokens = batch["token_ids"]
    return apply_unks(target_tokens, batch)

def init_small_weights(module):
    """
    Init the weights of layers in the perturbation network - should be sufficiently small
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight, 0.1)

class MathGPTBase(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options
        self.num_types = len(TokenType) if options.math_text else len(TokenType) - 1
        self.emb_size = MODEL_SIZE_TO_EMB_SIZE[options.model_size]

        # Extract pre-trained GPT2 transformer and text prediction head
        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(MODEL_SIZE_TO_NAME[options.model_size])
        self.transformer: GPT2Model = self.gpt2_lm.transformer
        if self.options.freeze_wte:
            self.transformer.wte.weight.requires_grad = False

        # Create type embeddings
        self.type_embeddings = nn.Embedding(self.num_types, self.emb_size) # TODO: also try just have text, math, and transition type embeddings
        init_small_weights(self.type_embeddings)

        # Create token embedding matrix for each type
        self.token_embeddings = nn.ModuleDict({
            str(token_type.value): nn.Embedding(Vocabulary.num_tokens_in_type(token_type), self.emb_size)
            for token_type in MATH_TYPES
        })
        # Use pre-trained text token embeddings
        self.token_embeddings[str(TokenType.TEXT.value)] = self.transformer.wte
        if self.options.math_text:
            self.token_embeddings[str(TokenType.MATH_TEXT.value)] = self.transformer.wte
        # Single embedding each for special single-tokens types
        self.token_embeddings[str(TokenType.START_FORMULA.value)] = nn.Embedding(1, self.emb_size)
        self.token_embeddings[str(TokenType.END_FORMULA.value)] = nn.Embedding(1, self.emb_size)
        self.token_embeddings[str(TokenType.END.value)] = nn.Embedding(1, self.emb_size)

        if options.shared_emb:
            sep_hidden_size = 10
            self.shared_emb_perturbation = nn.Sequential(
                nn.Linear(self.emb_size, sep_hidden_size, bias=False),
                nn.Tanh(),
                nn.Linear(sep_hidden_size, self.emb_size, bias=False)
            )
            self.shared_emb_perturbation.apply(init_small_weights)

        # Linear projection to convert raw math pos encoding into one that can be added to the input embeddings
        if options.tpe == TPE.FORTE.value:
            self.math_embedding_projection = nn.Linear(POS_ENCODING_SIZE_FORTE, self.emb_size, bias=False)
        elif options.tpe == TPE.RNN.value:
            self.math_embedding_model = RNNPosEncoder(self.emb_size)

        # Set allowed transitions based on settings
        self.allowed_transitions = {src: allowed.copy() for src, allowed in ALLOWED_TRANSITIONS.items()}
        if options.num_to_tree and options.sd_to_tree:
            self.allowed_transitions[TokenType.NUM] = [TokenType.NUM, TokenType.END]
            self.allowed_transitions[TokenType.VAR] = [TokenType.VAR, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.OP] = [TokenType.VAR, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.END] = [TokenType.VAR, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.START_FORMULA] = [TokenType.VAR, TokenType.OP]
        else:
            self.allowed_transitions[TokenType.NUM] = [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.VAR] = [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.OP] = [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.END] = [TokenType.VAR, TokenType.NUM, TokenType.OP, TokenType.END]
            self.allowed_transitions[TokenType.START_FORMULA] = [TokenType.VAR, TokenType.NUM, TokenType.OP]
        if options.math_text:
            self.allowed_transitions[TokenType.MATH_TEXT] = [TokenType.MATH_TEXT, TokenType.END]

    def get_math_embeddings(self, batch: CollatedBatch, math_idxs: torch.Tensor) -> torch.Tensor:
        """
        Get math position encodings for the batch
        """
        if self.options.tpe == TPE.FORTE.value:
            return self.math_embedding_projection(batch["pos_encodings"][math_idxs])
        if self.options.tpe == TPE.RNN.value:
            return self.math_embedding_model(batch, math_idxs)
        return batch["pos_encodings"][math_idxs]

    def get_input_embeddings(self, batch: CollatedBatch) -> torch.Tensor:
        """
        Return tensor (batch_size x max_seq_len x emb_size) containing input embeddings
        At each time step per sequence, add the type, token, and math position embeddings
        """
        # Start with type embeddings
        if self.options.use_type_embs:
            input_embeddings = self.type_embeddings(batch["token_types"])
        else:
            batch_size, max_seq_len = batch["token_ids"].shape
            input_embeddings = torch.zeros((batch_size, max_seq_len, self.emb_size)).to(device)

        # Add token embeddings for each type
        token_ids = apply_unks(batch["token_ids"], batch)
        non_padded_idx = token_ids != PADDING_TOKEN_ID
        for token_type in TokenType:
            if not self.options.math_text and token_type == TokenType.MATH_TEXT:
                continue
            type_idxs = (batch["token_types"] == token_type) & non_padded_idx
            if self.options.shared_emb:
                type_idxs &= ~batch["use_shared_emb"]
            input_embeddings[type_idxs] += self.token_embeddings[str(token_type.value)](token_ids[type_idxs])

        if self.options.shared_emb:
            # Get GPT token counterparts for math tokens (batch * seq, gpt token ids)
            selected_tokens = batch["gpt_tokens"][batch["use_shared_emb"]]
            # Get GPT token embeddings, excluding the padding regions
            text_emb_mask = selected_tokens != PADDING_TOKEN_ID
            text_embs = torch.zeros((selected_tokens.shape[0], selected_tokens.shape[1], self.emb_size)).to(device)
            text_embs[text_emb_mask] = self.token_embeddings[str(TokenType.TEXT.value)](selected_tokens[text_emb_mask])
            # Get average of GPT token embeddings for each math token
            emb_avgs = torch.sum(text_embs, dim=1) / torch.sum(text_emb_mask, dim=-1).unsqueeze(1)
            # Add the embedding averages and their perturbations to the input embeddings
            perturbations = self.shared_emb_perturbation(emb_avgs)
            input_embeddings[batch["use_shared_emb"]] += emb_avgs + perturbations

        # Add math position encodings
        if self.options.tpe != TPE.NONE.value:
            math_idxs = ~(
                (batch["token_types"] == TokenType.TEXT) |\
                (batch["token_types"] == TokenType.START_FORMULA) |\
                (batch["token_types"] == TokenType.END_FORMULA)
            )
            if torch.any(math_idxs):
                input_embeddings[math_idxs] += self.get_math_embeddings(batch, math_idxs)

        return input_embeddings

    def get_transformer_output(self, batch: CollatedBatch, output_attentions: bool = False,
                               decoding: bool = False, past_key_values: PastKeyValues = None):
        # Run inputs through model
        gpt_output: GPTOutput = self.transformer(
            output_attentions=output_attentions,
            inputs_embeds=self.get_input_embeddings(batch),
            past_key_values=past_key_values,
            use_cache=decoding,
            return_dict=True
        )
        return gpt_output

class MathGPTLM(MathGPTBase):
    def __init__(self, options: TrainOptions):
        super().__init__(options)

        if options.joint:
            # Predictive type layer for generation
            self.type_pred_layer = nn.Linear(self.emb_size, self.num_types)

            # Predictive token layers for generation
            self.type_to_token_pred_layer = nn.ModuleDict({
                str(token_type.value): nn.Linear(self.emb_size, Vocabulary.num_tokens_in_type(token_type))
                for token_type in MATH_TYPES
            })
            # Use pre-trained predictive head for text tokens
            self.type_to_token_pred_layer[str(TokenType.TEXT.value)] = self.gpt2_lm.lm_head
            if self.options.math_text:
                self.type_to_token_pred_layer[str(TokenType.MATH_TEXT.value)] = self.gpt2_lm.lm_head
            # For joint probability modeling, single-token types only rely on the probability of their corresponding type
            # TODO: implement init_math_pred
        else:
            self.text_pred_layer = self.gpt2_lm.lm_head
            self.type_to_size = {
                TokenType.OP: Vocabulary.num_tokens_in_type(TokenType.OP),
                TokenType.VAR: Vocabulary.num_tokens_in_type(TokenType.VAR),
                TokenType.NUM: Vocabulary.num_tokens_in_type(TokenType.NUM),
                TokenType.END: 1,
                TokenType.START_FORMULA: 1,
                TokenType.END_FORMULA: 1,
            }
            if self.options.init_math_pred:
                self.math_pred_layer = nn.Linear(self.emb_size, sum(self.type_to_size.values()), bias=False)
                with torch.no_grad():
                    start_idx = 0
                    for token_type in TokenType:
                        if token_type in (TokenType.TEXT, TokenType.MATH_TEXT):
                            continue
                        if token_type in MATH_TYPES:
                            for token_id in range(self.type_to_size[token_type]):
                                if not Vocabulary.is_special_token(token_type, token_id):
                                    symbol = Vocabulary.get_symbol(token_type, token_id)
                                    if symbol: # Symbol can sometimes be empty string, which has no corresponding text tokens
                                        text_tokens = text_tokenizer()(symbol)["input_ids"]
                                        avg_pred_weights = self.text_pred_layer.weight[text_tokens].mean(dim=0)
                                        self.math_pred_layer.weight[start_idx + token_id] = avg_pred_weights
                        start_idx += self.type_to_size[token_type]
            else:
                self.math_pred_layer = nn.Linear(self.emb_size, sum(self.type_to_size.values()), bias=self.options.lmhb)

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

    def get_type_mask(self, type_idxs: Dict[TokenType, torch.Tensor], final_formula_token_idx: torch.Tensor, batch: CollatedBatch):
        """
        Create mask with allowed (False) and unallowed (True) types for the following token at each index
        """
        batch_size, max_seq_len = batch["token_ids"].shape
        type_mask = torch.full((batch_size, max_seq_len, self.num_types), True).to(device)
        for token_type, allowed_types in self.allowed_transitions.items():
            for allowed_type in allowed_types:
                type_mask[:, :, allowed_type][type_idxs[token_type]] = False
        # Tokens that finish a formula can only generate END_FORMULA
        type_mask[final_formula_token_idx] = ~F.one_hot(torch.tensor(TokenType.END_FORMULA), num_classes=self.num_types).type(torch.bool).to(device)
        # Math text head must produce a math text token
        if self.options.math_text:
            mth_idx = (batch["token_ids"] == SpecialOpToken.MATH_TEXT_HEAD) & type_idxs[TokenType.OP]
            type_mask[mth_idx] = ~F.one_hot(torch.tensor(TokenType.MATH_TEXT), num_classes=self.num_types).type(torch.bool).to(device)
        # Num tree head must produce a num token
        if self.options.num_to_tree:
            nth_idx = (batch["token_ids"] == SpecialOpToken.NUM_SUB_TREE_HEAD) & type_idxs[TokenType.OP]
            type_mask[nth_idx] = ~F.one_hot(torch.tensor(TokenType.NUM), num_classes=self.num_types).type(torch.bool).to(device)
        # Prevent nodes from exceeding max depth, which happens if an OP token is produced in the maximum level
        op_unallowed_idx = ((batch["pos_levels"] == MAX_FORMULA_DEPTH - 1) & ~type_idxs[TokenType.END]) |\
                           ((batch["pos_levels"] == MAX_FORMULA_DEPTH - 2) & type_idxs[TokenType.OP])
        op_blocked_mask = type_mask.clone()
        op_blocked_mask[:, :, TokenType.OP] = True
        type_mask[op_unallowed_idx] = op_blocked_mask[op_unallowed_idx]
        # Prevent nodes from exceeding max width by forcing the last possible child node to be an END token
        cur_level_mask = F.one_hot(batch["pos_levels"], num_classes=batch["pos_vecs"].shape[2]).type(torch.bool)
        upper_level_mask = F.one_hot(torch.clamp(batch["pos_levels"] - 1, min=0), num_classes=batch["pos_vecs"].shape[2]).type(torch.bool)
        must_end_idx = (
            (type_idxs[TokenType.VAR] | type_idxs[TokenType.NUM] | type_idxs[TokenType.MATH_TEXT]) &\
            (batch["pos_vecs"][cur_level_mask].view(batch_size, -1) == MAX_FORMULA_WIDTH - 2)
        ) | (
            type_idxs[TokenType.END] &\
            (batch["pos_vecs"][upper_level_mask].view(batch_size, -1) == MAX_FORMULA_WIDTH - 2)
        )
        type_mask[must_end_idx] = ~F.one_hot(torch.tensor(TokenType.END), num_classes=self.num_types).type(torch.bool).to(device)
        return type_mask

    def get_type_probs(self, gpt_output: GPTOutput, type_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Calculate the probability of generating each type for each token in the batch, including masking to constrain to possible transitions
        """
        # Get raw prediction values from model output
        type_preds = self.type_pred_layer(gpt_output.last_hidden_state)
        # Apply mask and get predicted probability of types at each index
        if type_mask is not None:
            type_preds[type_mask] = -torch.inf
        type_probs = nn.Softmax(dim=-1)(type_preds)
        return type_probs

    def get_token_probs(self, gpt_output: GPTOutput, type_probs: torch.Tensor):
        """
        For each type, calculate the joint probability of generating that type and the tokens in that type
        """
        batch_size, max_seq_len = type_probs.shape[:2]
        type_to_token_probs: Dict[TokenType, torch.Tensor] = {}
        for token_type in TokenType:
            if not self.options.math_text and token_type == TokenType.MATH_TEXT:
                continue

            # Get predicted probability of tokens in the type, i.e. P(token|type)
            if token_type in (TokenType.START_FORMULA, TokenType.END_FORMULA, TokenType.END):
                token_probs = torch.ones((batch_size, max_seq_len, 1)).to(device)
            else:
                token_preds = self.type_to_token_pred_layer[str(token_type.value)](gpt_output.last_hidden_state)
                token_probs = nn.Softmax(dim=-1)(token_preds)
            # Multiply P(type) * P(token|type) to get P(token, type)
            type_to_token_probs[token_type] = token_probs * type_probs[:, :, token_type].unsqueeze(-1)
        return type_to_token_probs

    def get_prediction_loss(self, type_to_token_probs: Dict[TokenType, torch.Tensor], type_idxs: Dict[TokenType, torch.Tensor],
                            target_tokens: torch.Tensor):
        """
        Calculate the cross-entropy prediction loss given the token probabilities
        """
        if target_tokens.shape[1] == 1 or torch.all(target_tokens == PADDING_TOKEN_ID):
            # If the sequence has a length of 1, there are no targets to predict, so return a loss of 0
            # If we have a full padding region (this is the case when generating from a prompt) then no loss can be computed
            return torch.tensor(0.0).to(device)

        losses: List[torch.Tensor] = []
        shifted_target_tokens = target_tokens[:, 1:]
        for token_type in TokenType:
            if not self.options.math_text and token_type == TokenType.MATH_TEXT:
                continue

            # Get indices that have a target of this type
            shifted_target_idx = type_idxs[token_type][:, 1:]
            # Exclude padding regions, including them would add extra length to the vector that we get the mean of
            shifted_target_idx &= shifted_target_tokens != PADDING_TOKEN_ID
            # Skip type if it doesn't exist as a target, going through with calculations results in nan loss
            if not torch.any(shifted_target_idx):
                continue
            # Get token probabilities for this type where the target matches
            selected_probs = type_to_token_probs[token_type][:, :-1][shifted_target_idx]
            # Clamp to avoid probability of 0 in unlucky cases (will break autograd)
            selected_probs = selected_probs.clamp(min=1e-15)
            # Add cross-entropy loss, take average at end to weigh each token equally
            log_probs = torch.log(selected_probs)
            loss_fn = nn.NLLLoss(reduction="none")
            loss: torch.Tensor = loss_fn(log_probs, shifted_target_tokens[shifted_target_idx])
            losses.append(loss)
        return torch.concat(losses, dim=0).mean()

    def get_token_activations(self, gpt_output: GPTOutput, type_mask: Optional[torch.Tensor]):
        # Get activations from linear layers
        text_activations = self.text_pred_layer(gpt_output.last_hidden_state)
        math_activations = self.math_pred_layer(gpt_output.last_hidden_state)
        # Apply text transition mask
        if type_mask is not None:
            if self.options.math_text:
                text_activations[type_mask[:, :, TokenType.TEXT] & type_mask[:, :, TokenType.MATH_TEXT]] = -torch.inf
            else:
                text_activations[type_mask[:, :, TokenType.TEXT]] = -torch.inf
        # Apply type-specific transition mask
        math_activation_parts: List[torch.Tensor] = []
        start_idx = 0
        for token_type in TokenType:
            if token_type in (TokenType.TEXT, TokenType.MATH_TEXT):
                continue
            activation = math_activations[:, :, start_idx : start_idx + self.type_to_size[token_type]]
            if type_mask is not None:
                activation[type_mask[:, :, token_type]] = -torch.inf
            math_activation_parts.append(activation)
            start_idx += self.type_to_size[token_type]
        return torch.concat([text_activations] + math_activation_parts, dim=-1)

    def get_token_probs_from_activations(self, token_activations: torch.Tensor, type_mask: torch.Tensor):
        type_to_token_probs: Dict[TokenType, torch.Tensor] = {}
        token_probs = nn.Softmax(dim=-1)(token_activations)
        type_to_token_probs[TokenType.TEXT] = token_probs[:, :, :TEXT_VOCAB_SIZE]
        if self.options.math_text:
            type_to_token_probs[TokenType.MATH_TEXT] = type_to_token_probs[TokenType.TEXT].clone()
            # Re-apply type mask since text and math text types share activation vector
            type_to_token_probs[TokenType.TEXT][type_mask[:, :, TokenType.TEXT]] = 0
            type_to_token_probs[TokenType.MATH_TEXT][type_mask[:, :, TokenType.MATH_TEXT]] = 0
        start_idx = TEXT_VOCAB_SIZE
        for token_type in TokenType:
            if token_type in (TokenType.TEXT, TokenType.MATH_TEXT):
                continue
            type_to_token_probs[token_type] = token_probs[:, :, start_idx : start_idx + self.type_to_size[token_type]]
            start_idx += self.type_to_size[token_type]
        return type_to_token_probs

    def get_prediction_loss_from_activations(self, token_activations: torch.Tensor, type_idxs: Dict[TokenType, torch.Tensor],
                                             target_tokens: torch.Tensor):
        padding_idx = target_tokens == PADDING_TOKEN_ID
        if target_tokens.shape[1] == 1 or torch.all(padding_idx):
            # If the sequence has a length of 1, there are no targets to predict, so return a loss of 0
            # If we have a full padding region (this is the case when generating from a prompt) then no loss can be computed
            return torch.tensor(0.0).to(device)

        # Adjust the token IDs to reference the indices in the extended activation vector
        target_tokens = target_tokens.clone()
        start_idx = TEXT_VOCAB_SIZE
        for token_type in TokenType:
            if token_type in (TokenType.TEXT, TokenType.MATH_TEXT):
                continue
            target_tokens[type_idxs[token_type] & ~padding_idx] += start_idx
            start_idx += self.type_to_size[token_type]
        shifted_target_tokens = target_tokens[:, 1:]
        # Loss function implicitly applies softmax, log, and mean, and ignores padding regions
        loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=PADDING_TOKEN_ID)
        loss = loss_fn(token_activations[:, :-1].reshape(-1, token_activations.shape[2]), shifted_target_tokens.reshape(-1))
        return loss

    def forward(self, batch: CollatedBatch, labels: torch.Tensor = None, output_attentions: bool = False,
                decoding: bool = False, past_key_values: PastKeyValues = None):
        if past_key_values is not None:
            batch = trim_batch(batch, batch["token_ids"].shape[1] - 1, batch["token_ids"].shape[1])
            if labels is not None:
                labels = labels[:, -1:]

        # Get GPT output
        gpt_output = self.get_transformer_output(batch, output_attentions, decoding, past_key_values)

        # Get relevant masks
        type_idxs, final_formula_token_idx = self.get_prediction_masks(batch)

        # Get type masks that constrain predictions to permitted transitions
        if self.options.cdt or not self.training:
            type_mask = self.get_type_mask(type_idxs, final_formula_token_idx, batch)
        else:
            type_mask = None

        if self.options.joint:
            # Calculate P(type) for each possible type
            type_probs = self.get_type_probs(gpt_output, type_mask)

            # Calculate P(token, type) for all types/tokens
            type_to_token_probs = self.get_token_probs(gpt_output, type_probs)

            # Calculate cross-entropy loss
            loss = self.get_prediction_loss(type_to_token_probs, type_idxs, get_target_tokens(batch, labels))
        else:
            # Get activations for all text and math tokens
            token_activations = self.get_token_activations(gpt_output, type_mask)

            # Calculate cross-entropy loss
            loss = self.get_prediction_loss_from_activations(token_activations, type_idxs, get_target_tokens(batch, labels))

            # Get token probabilities from the activations
            if self.training: # Don't do this in training mode as it will use up excess GPU memory
                type_to_token_probs = None
            else:
                type_to_token_probs = self.get_token_probs_from_activations(token_activations, type_mask)

        return loss, type_to_token_probs, gpt_output.attentions, gpt_output.past_key_values

class MathGPTClassifier(MathGPTBase):
    def __init__(self, options: TrainOptions):
        super().__init__(options)
        self.classifier_head = nn.Linear(self.emb_size, options.num_classes)

    def forward(self, batch: CollatedBatch):
        batch_size = batch["token_ids"].shape[0]

        # Get GPT output
        gpt_output = self.get_transformer_output(batch)

        # Get prediction from last token in input per sequence
        pred_states = gpt_output.last_hidden_state[torch.arange(batch_size), batch["sequence_lengths"] - 1]
        predictions: torch.Tensor = self.classifier_head(pred_states)
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss: torch.Tensor = loss_fn(predictions, batch["cls_labels"])

        return loss, predictions

class RNNPosEncoder(nn.Module):
    hidden_size = 50

    def __init__(self, emb_size: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=MAX_FORMULA_WIDTH,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_size, emb_size)

    def forward(self, batch: CollatedBatch, math_idxs: torch.Tensor):
        # Unroll the batch's position encodings, and only process math tokens
        rnn_input = batch["pos_encodings"][math_idxs].view(-1, MAX_FORMULA_DEPTH, MAX_FORMULA_WIDTH)
        sequence_lengths = batch["pos_levels"][math_idxs].cpu() + 1

        # Run through RNN
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths=sequence_lengths, batch_first=True, enforce_sorted=False)
        _, final_hidden_state = self.rnn(packed_input)

        # Pproject to embedding size
        return self.output_layer(final_hidden_state.view(-1, self.hidden_size))
