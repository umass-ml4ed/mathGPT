from typing import Dict, List, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions as GPTOutput

from vocabulary import Vocabulary
from math_tokenize import POS_ENCODING_SIZE_FORTE
from constants import CollatedBatch, TokenType, TPE, EMB_SIZE, PADDING_TOKEN_ID, MAX_FORMULA_DEPTH, MAX_FORMULA_WIDTH, TEXT_VOCAB_SIZE
from utils import device, TrainOptions

# Leverages pre-trained GPT2 from transformers library
# https://huggingface.co/docs/transformers/model_doc/gpt2
# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/gpt2/modeling_gpt2.py

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

class MathGPTBase(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options

        # Extract pre-trained GPT2 transformer and text prediction head
        self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")
        self.transformer: GPT2Model = self.gpt2_lm.transformer

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
        if options.tpe == TPE.FORTE.value:
            self.math_embedding_projection = nn.Linear(POS_ENCODING_SIZE_FORTE, EMB_SIZE, bias=False)

    def load_pretrained(self, pretrained_state_dict: Dict[str, torch.Tensor]):
        state_dict = self.state_dict()
        for param_name, param_val in pretrained_state_dict.items():
            state_dict[param_name] = param_val
        self.load_state_dict(state_dict)

    def get_math_embeddings(self, batch: CollatedBatch) -> torch.Tensor:
        if self.options.tpe == TPE.FORTE.value:
            return self.math_embedding_projection(batch["pos_encodings"])
        return batch["pos_encodings"]

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
            input_embeddings = torch.zeros((batch_size, max_seq_len, EMB_SIZE)).to(device)

        # Add token embeddings for each type
        for token_type in TokenType:
            type_idxs = (batch["token_types"] == token_type) & (batch["token_ids"] != PADDING_TOKEN_ID)
            input_embeddings[type_idxs] += self.token_embeddings[str(token_type.value)](batch["token_ids"][type_idxs])

        # Add math position encodings
        math_idxs = ~((batch["token_types"] == TokenType.TEXT) | (batch["token_types"] == TokenType.START_FORMULA) | (batch["token_types"] == TokenType.END_FORMULA))
        input_embeddings[math_idxs] += self.get_math_embeddings(batch)[math_idxs]

        return input_embeddings

    def get_transformer_output(self, batch: CollatedBatch):
        # Run inputs through model
        gpt_output: GPTOutput = self.transformer(
            # past_key_values=[], # TODO: for speeding up decoding
            use_cache=False, # TODO: set to True for decoding, but otherwise runs out of memory
            output_attentions=False,
            # attention_mask=batch["attention_mask"], # TODO: this doesn't seem to make a difference with padding, might be performance-related
            inputs_embeds=self.get_input_embeddings(batch),
            return_dict=True
        )
        return gpt_output

class MathGPTLM(MathGPTBase):
    def __init__(self, options: TrainOptions):
        super().__init__(options)

        if options.joint:
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
            # For joint probability modeling, single-token types only rely on the probability of their corresponding type
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
            self.math_pred_layer = nn.Linear(EMB_SIZE, sum(self.type_to_size.values()))

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
        type_mask = torch.full((batch_size, max_seq_len, NUM_TYPES), True).to(device)
        for token_type, allowed_types in ALLOWED_TRANSITIONS.items():
            for allowed_type in allowed_types:
                type_mask[:, :, allowed_type][type_idxs[token_type]] = False
        # Tokens that finish a formula can only generate END_FORMULA
        type_mask[final_formula_token_idx] = ~F.one_hot(torch.tensor(TokenType.END_FORMULA), num_classes=len(TokenType)).type(torch.bool).to(device)
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
            (type_idxs[TokenType.VAR] | type_idxs[TokenType.NUM]) &\
            (batch["pos_vecs"][cur_level_mask].view(batch_size, -1) == MAX_FORMULA_WIDTH - 2)
        ) | (
            type_idxs[TokenType.END] &\
            (batch["pos_vecs"][upper_level_mask].view(batch_size, -1) == MAX_FORMULA_WIDTH - 2)
        )
        type_mask[must_end_idx] = ~F.one_hot(torch.tensor(TokenType.END), num_classes=len(TokenType)).type(torch.bool).to(device)
        return type_mask

    def get_type_probs(self, gpt_output: GPTOutput, type_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of generating each type for each token in the batch, including masking to constrain to possible transitions
        """
        # Get raw prediction values from model output
        type_preds = self.type_pred_layer(gpt_output.last_hidden_state)
        # Apply mask and get predicted probability of types at each index
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
                            batch: CollatedBatch, labels: Optional[torch.Tensor]):
        """
        Calculate the cross-entropy prediction loss given the token probabilities
        """
        losses: List[torch.Tensor] = []
        if batch["gen_labels"] is not None:
            shifted_target_tokens = batch["gen_labels"][:, 1:]
        elif labels is not None:
            shifted_target_tokens = labels[:, 1:]
        else:
            shifted_target_tokens = batch["token_ids"][:, 1:]
        if shifted_target_tokens.shape[1] == 0 or torch.all(shifted_target_tokens == PADDING_TOKEN_ID):
            # If the sequence has a length of 1, there are no targets to predict, so return a loss of 0
            # If we have a full padding region (this is the case when generating from a prompt) then no loss can be computed
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
            # Clamp to avoid probability of 0 in unlucky cases (will break autograd)
            selected_probs = selected_probs.clamp(min=1e-15)
            # Add cross-entropy loss, take average at end to weigh each token equally
            log_probs = torch.log(selected_probs)
            loss_fn = nn.NLLLoss(reduction="none")
            loss: torch.Tensor = loss_fn(log_probs, shifted_target_tokens[shifted_target_idx])
            if any(torch.isinf(l) for l in loss):
                print("inf loss!")
                with open("debug.txt", "a") as debug_file:
                    debug_file.write(f"{batch}\n\n{token_type}\n\n{loss.isinf().nonzero()}\n\n{selected_probs}\n\n{type_to_token_probs}\n\n{type_idxs}\n\n")
                loss = loss[~loss.isinf()]
            losses.append(loss)
        return torch.concat(losses, dim=0).mean()

    def get_token_activations(self, gpt_output: GPTOutput, type_mask: torch.Tensor):
        # Get activations from linear layers
        text_activations = self.text_pred_layer(gpt_output.last_hidden_state)
        math_activations = self.math_pred_layer(gpt_output.last_hidden_state)
        # Apply type-specific transition mask
        text_activations[type_mask[:, :, TokenType.TEXT]] = -torch.inf
        math_activation_parts: List[torch.Tensor] = []
        start_idx = 0
        for token_type in TokenType:
            if token_type == TokenType.TEXT:
                continue
            activation = math_activations[:, :, start_idx : start_idx + self.type_to_size[token_type]]
            activation[type_mask[:, :, token_type]] = -torch.inf
            math_activation_parts.append(activation)
            start_idx += self.type_to_size[token_type]
        return torch.concat([text_activations] + math_activation_parts, dim=-1)

    def get_token_probs_from_activations(self, token_activations: torch.Tensor):
        type_to_token_probs: Dict[TokenType, torch.Tensor] = {}
        token_probs = nn.Softmax(dim=-1)(token_activations)
        type_to_token_probs[TokenType.TEXT] = token_probs[:, :, :TEXT_VOCAB_SIZE]
        start_idx = TEXT_VOCAB_SIZE
        for token_type in TokenType:
            if token_type == TokenType.TEXT:
                continue
            type_to_token_probs[token_type] = token_probs[:, :, start_idx : start_idx + self.type_to_size[token_type]]
            start_idx += self.type_to_size[token_type]
        return type_to_token_probs

    def get_prediction_loss_from_activations(self, token_activations: torch.Tensor, batch: CollatedBatch, labels: Optional[torch.Tensor]):
        # Grab the appropriate target tensor
        if batch["gen_labels"] is not None:
            target_tokens = batch["gen_labels"]
        elif labels is not None:
            target_tokens = labels
        else:
            target_tokens = batch["token_ids"]
        if target_tokens.shape[1] == 1 or torch.all(target_tokens == PADDING_TOKEN_ID):
            # If the sequence has a length of 1, there are no targets to predict, so return a loss of 0
            # If we have a full padding region (this is the case when generating from a prompt) then no loss can be computed
            return torch.tensor(0.0).to(device)

        # Adjust the token IDs to reference the indices in the extended activation vector
        target_tokens = target_tokens.clone()
        start_idx = TEXT_VOCAB_SIZE
        for token_type in TokenType:
            if token_type == TokenType.TEXT:
                continue
            target_tokens[batch["token_types"] == token_type] += start_idx
            start_idx += self.type_to_size[token_type]
        shifted_target_tokens = target_tokens[:, 1:]
        # Loss function implicitly applies softmax, log, and mean, and ignores padding regions
        loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=PADDING_TOKEN_ID)
        loss = loss_fn(token_activations[:, :-1].reshape(-1, token_activations.shape[2]), shifted_target_tokens.reshape(-1))
        return loss

    def forward(self, batch: CollatedBatch, labels: torch.Tensor = None):
        # Get GPT output
        gpt_output = self.get_transformer_output(batch)

        # Get relevant masks
        type_idxs, final_formula_token_idx = self.get_prediction_masks(batch)

        # Get type masks that constrain predictions to permitted transitions
        type_mask = self.get_type_mask(type_idxs, final_formula_token_idx, batch)

        if self.options.joint:
            # Calculate P(type) for each possible type
            type_probs = self.get_type_probs(gpt_output, type_mask)

            # Calculate P(token, type) for all types/tokens
            type_to_token_probs = self.get_token_probs(gpt_output, type_probs)

            # Calculate cross-entropy loss
            loss = self.get_prediction_loss(type_to_token_probs, type_idxs, batch, labels)
        else:
            # Get activations for all text and math tokens
            token_activations = self.get_token_activations(gpt_output, type_mask)

            # Calculate cross-entropy loss
            loss = self.get_prediction_loss_from_activations(token_activations, batch, labels)

            # Get token probabilities from the activations
            type_to_token_probs = self.get_token_probs_from_activations(token_activations)

        return loss, type_to_token_probs

class MathGPTClassifier(MathGPTBase):
    def __init__(self, options: TrainOptions):
        super().__init__(options)

        self.classifier_head = nn.Linear(EMB_SIZE, options.num_classes)

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
