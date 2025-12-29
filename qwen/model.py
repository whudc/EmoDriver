import math, os
from typing import Any, Callable, List, Optional, Tuple, Union, Unpack
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.generation import GenerationMixin
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding, Qwen3MLP, apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from accelerate.hooks import remove_hook_from_submodules
from peft.config import PeftConfig
from safetensors.torch import save_file as safe_save_file
from peft.peft_model import PeftModelForCausalLM
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_peft_model_state_dict,
    PeftType,
    _get_batch_size
)
from transformers import logging

logger = logging.get_logger(__name__)

@dataclass
class CausalLMOutputWithPastWithModel(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window, 
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states



class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }



class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.hidden_size = config.hidden_size
        self.special_token_id = config.special_token_dict['<map>']
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        map_feats: torch.FloatTensor = None,
        map_masks: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if past_key_values is not None:
            input_ids_clone = input_ids.clone()
            input_ids = input_ids[:, -1:]

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if map_feats is not None and past_key_values is None:
            new_tokens_num = input_ids.shape[1] + map_feats.shape[1]
            special_toks_mask = input_ids == self.special_token_id
            special_toks_loc = torch.where(special_toks_mask)[1]
            seq_length = seq_length + map_feats.shape[1]
            seq_length_with_past = seq_length_with_past + map_feats.shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
    
        if map_feats is not None and past_key_values is None:
            new_inputs_embeds = torch.zeros((batch_size, new_tokens_num, inputs_embeds.shape[-1]), device=input_ids.device).to(inputs_embeds.dtype)
            new_inputs_attention_mask = torch.zeros((batch_size, new_tokens_num), dtype=torch.bool, device=input_ids.device).to(inputs_embeds.dtype)
            new_labels = torch.zeros((batch_size, new_tokens_num), dtype=torch.long, device=input_ids.device)
            
            position_ids_bs = position_ids.shape[0]
            new_position_ids = torch.zeros((position_ids_bs, new_tokens_num), dtype=torch.long, device=input_ids.device)
            
            for b in range(batch_size):
                new_inputs_embeds[b, :special_toks_loc[b]+1] = inputs_embeds[b, :special_toks_loc[b]+1]
                new_inputs_embeds[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = map_feats[b]
                new_inputs_embeds[b, special_toks_loc[b]+map_feats.shape[1]+1:] = inputs_embeds[b, special_toks_loc[b]+1:]
                
                if labels is not None:
                    new_labels[b, :special_toks_loc[b]+1] = labels[b, :special_toks_loc[b]+1]
                    new_labels[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = -100
                    new_labels[b, special_toks_loc[b]+map_feats.shape[1]+1:] = labels[b, special_toks_loc[b]+1:]
                
                if b < position_ids_bs:
                    new_position_ids[b, :special_toks_loc[b]+1] = position_ids[b, :special_toks_loc[b]+1]
                    new_position_ids[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = torch.arange(position_ids[b, special_toks_loc[b]]+1, position_ids[b, special_toks_loc[b]]+map_feats.shape[1]+1)
                    new_position_ids[b, special_toks_loc[b]+map_feats.shape[1]+1:] = position_ids[b, special_toks_loc[b]+1:] + map_feats.shape[1]
                
                if attention_mask is not None:
                    new_inputs_attention_mask[b, :special_toks_loc[b]+1] = attention_mask[b, :special_toks_loc[b]+1]
                    new_inputs_attention_mask[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = ~map_masks[b]
                    new_inputs_attention_mask[b, special_toks_loc[b]+map_feats.shape[1]+1:] = attention_mask[b, special_toks_loc[b]+1:]
            
            inputs_embeds = new_inputs_embeds
            attention_mask = new_inputs_attention_mask
            labels = new_labels
            position_ids = new_position_ids

        if map_feats is not None and past_key_values is not None:
            special_toks_loc = torch.where(input_ids_clone == self.special_token_id)[1]
            new_inputs_attention_mask = torch.zeros((batch_size, seq_length_with_past), dtype=torch.bool, device=input_ids.device).to(inputs_embeds.dtype)
            for b in range(batch_size):
                new_inputs_attention_mask[b, :special_toks_loc[b]+1] = attention_mask[b, :special_toks_loc[b]+1]
                new_inputs_attention_mask[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = ~ map_masks[b]
                new_inputs_attention_mask[b, special_toks_loc[b]+map_feats.shape[1]+1:] = attention_mask[b, special_toks_loc[b]+1:]
            attention_mask = new_inputs_attention_mask
            position_ids += map_feats.shape[1]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        ), labels


class Qwen3ForCausalLM(GenerationMixin, Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keep_in_fp32_modules = ['map_adapter']

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Add Map adapter layers
        self.map_insize = config.map_insize
        self.map_adapter = nn.Linear(self.map_insize, config.hidden_size, bias=False)


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        map_feats: Optional[torch.FloatTensor] = None,
        map_masks: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPastWithModel]:

        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if map_feats is not None:
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
            map_feats = self.map_adapter(map_feats)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, labels = self.model(
            input_ids=input_ids,
            labels=labels,
            map_feats=map_feats,
            map_masks=map_masks,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithModel(
            loss=loss,
            logits=logits,
            labels=labels,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, map_feats, map_masks, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "map_feats": map_feats,
                "map_masks": map_masks,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]
        
        # Resize label weight
        if hasattr(self, "weighted_mask"):
            try:
                number_weight = self.config.number_weight
            except:
                number_weight = 1.0
            weighted_mask = torch.ones(self.config.vocab_size, dtype=torch.float32)
            if number_weight > 1:
                number_tokens = [
                    448,
                    29900,
                    29889,
                    29896,
                    29906,
                    29941,
                    29946,
                    29945,
                    29953,
                    29955,
                    29947,
                    29929,
                ]  # -0.123456789
                weighted_mask[number_tokens] = number_weight
            self.weighted_mask = weighted_mask

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds




class MapPeftModelForCausalLM(PeftModelForCausalLM):
    def __init__(self, *args, **kwargs):
        # 调用父类（PeftModelForCausalLM）的__init__方法
        super().__init__(*args, **kwargs)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )
            
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            # overwrite past_kv in kwargs
            # some archs require max_cache_len to re-initialize the cache
            if input_ids is not None:
                max_cache_len = input_ids.shape[1] + peft_config.num_virtual_tokens
            else:
                max_cache_len = inputs_embeds.shape[1] + peft_config.num_virtual_tokens
            kwargs["past_key_values"] = self.get_prompt(batch_size, max_cache_len=max_cache_len)
            return self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        elif peft_config.peft_type == PeftType.CPT:
            return self._cpt_forward(input_ids, inputs_embeds, peft_config, task_ids, batch_size, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        # save map adapter weight
        model_names = kwargs.get("state_dict", None).keys()
        map_adapter_name = None
        for model_name in model_names:
            if 'map_adapter' in model_name:
                map_adapter_name = model_name
                break
                
        # save global dict
        embed_tokens_name = None
        for model_name in model_names:
            if 'embed_tokens' in model_name:
                embed_tokens_name = model_name
                break
        
        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)
        
        if map_adapter_name is None:
            raise ValueError(f"Cannot find map adapter in the model state dict.")
        else:
            map_adapter_weight_dict = dict()
            map_adapter_weight_dict[map_adapter_name] = kwargs.get("state_dict", None)[map_adapter_name]
            if safe_serialization:
                safe_save_file(
                    map_adapter_weight_dict,
                    os.path.join(save_directory, 'map_adapter.safetensors'),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(map_adapter_weight_dict, os.path.join(save_directory, 'map_adapter.bin'))
        
        if embed_tokens_name is None:
            raise ValueError(f"Cannot find embed_tokens in the model state dict.")
        else:
            embed_tokens_weight_dict = dict()
            embed_tokens_weight_dict[embed_tokens_name] = kwargs.get("state_dict", None)[embed_tokens_name]
            if safe_serialization:
                safe_save_file(
                    embed_tokens_weight_dict,
                    os.path.join(save_directory, 'embed_tokens.safetensors'),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(embed_tokens_weight_dict, os.path.join(save_directory, 'embed_tokens.bin'))

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if safe_serialization:
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        from qwen.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # 如何kwargs中存在output_attentions, output_hidden_states和return_dict，则删除
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        
        # load map adapter with copy paramter
        try:
            loaded_weight = torch.load(os.path.join(model_id, 'map_adapter.bin'))
        except:
            print(' error in load map adapter')
            loaded_weight = {}
        for name, param in model.named_parameters():
            if name in loaded_weight.keys():
                param.data.copy_(loaded_weight[name])
                # param.requires_grad = is_trainable
                print(f"Load map adapter weight {name} from {model_id}")
        
        # load embed_tokens with copy paramter
        try:
            loaded_weight = torch.load(os.path.join(model_id, 'embed_tokens.bin'))
        except:
            print(' error in load embed tokens')
            loaded_weight = {}
        for name, param in model.named_parameters():
            if name in loaded_weight.keys():
                param.data.copy_(loaded_weight[name])
                # param.requires_grad = is_trainable
                print(f"Load embed_tokens weight {name} from {model_id}")
        
        return model