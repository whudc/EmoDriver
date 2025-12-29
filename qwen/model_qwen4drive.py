import math, os
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union, List

import torch
import torch.distributions as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils import logging, ModelOutput
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding, Qwen3MLP, apply_rotary_pos_emb, eager_attention_forward
from safetensors.torch import save_file as safe_save_file
from safetensors.torch import load_file
from peft.peft_model import PeftModelForCausalLM
from peft.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_peft_model_state_dict
)

from gameformer.predictor import Encoder as GameformerEncoder
from gameformer.predictor import LLMEnhancedGameFormer
from gameformer.predictor_adapter import LLMEnhancedGameFormer_Adapter
from gameformer.predictor_modules import CrossTransformer
from gameformer.train_utils import *

from peft import set_peft_model_state_dict

from qwen.model import Qwen3RMSNorm, Qwen3DecoderLayer

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "Qwen3Config"

@dataclass
class BaseModelOutputWithPastDrive(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loca: Optional[List[Tuple[int]]] = None


@dataclass
class CausalLMOutputWithPastWithModel(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    llm_regression_loss: Optional[torch.FloatTensor] = None
    llm_multi_head_loss: Optional[torch.FloatTensor] = None
    urban_loss: Optional[torch.FloatTensor] = None
    v_a_loss: Optional[torch.FloatTensor] = None
    neighbour_lane_loss: Optional[torch.FloatTensor] = None
    acc_class_loss: Optional[torch.FloatTensor] = None
    lane_change_loss: Optional[torch.FloatTensor] = None
    traffic_light_loss: Optional[torch.FloatTensor] = None
    gameformer_loss: Optional[torch.FloatTensor] = None
    gmm_loss: Optional[torch.FloatTensor] = None
    gameformer_planner_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    predictions: Optional[Tuple[torch.FloatTensor]] = None
    plan: Optional[Tuple[torch.FloatTensor]] = None
    llm_plan: Optional[Tuple[torch.FloatTensor]] = None

class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Qwen3Model):
            module.gradient_checkpointing = value

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
    ) -> Union[Tuple, BaseModelOutputWithPastDrive]:
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
    
        loca = []
        if map_feats is not None and past_key_values is None:
            new_inputs_embeds = torch.zeros((batch_size, new_tokens_num, inputs_embeds.shape[-1]), device=input_ids.device).to(inputs_embeds.dtype)
            new_inputs_attention_mask = torch.zeros((batch_size, new_tokens_num), dtype=torch.bool, device=input_ids.device).to(inputs_embeds.dtype)
            new_labels = torch.zeros((batch_size, new_tokens_num), dtype=torch.long, device=input_ids.device)
            
            position_ids_bs = position_ids.shape[0]
            new_position_ids = torch.zeros((position_ids_bs, new_tokens_num), dtype=torch.long, device=input_ids.device)
            
            for b in range(batch_size):
                return_special_toks_loc = special_toks_loc
                new_inputs_embeds[b, :special_toks_loc[b]+1] = inputs_embeds[b, :special_toks_loc[b]+1]
                new_inputs_embeds[b, special_toks_loc[b]+1:special_toks_loc[b]+map_feats.shape[1]+1] = map_feats[b]
                new_inputs_embeds[b, special_toks_loc[b]+map_feats.shape[1]+1:] = inputs_embeds[b, special_toks_loc[b]+1:]
                loca.append((special_toks_loc[b]+1, special_toks_loc[b]+map_feats.shape[1]+1))
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

        return BaseModelOutputWithPastDrive(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        ), labels, new_inputs_attention_mask, return_special_toks_loc


class Qwen3ForCausalLM(GenerationMixin, Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keep_in_fp32_modules = ['map_adapter',
                            #  'waypoints_fc',
                             'waypoints_predictor',
                            #  'waypoints_output',
                             'map_encoder',
                             'gameformer',
                             'ego_v_a_predictor',
                             'neighbour_lane',
                             'acc_classification',
                             'lane_change',
                             'traffic_light',
                             'feature_adpter']

    _keep_small_lr_modules = [
            'gameformer',
        ]
    adapter_name_list = _keep_in_fp32_modules

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.feature_len = config.feature_len
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Add Map adapter layers
        self.map_insize = config.map_insize
        self.map_adapter = nn.Linear(self.map_insize, config.hidden_size, bias=False)

        self.waypoints_predictor = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, self.feature_len*2))

        # regression
        self.ego_v_a_predictor = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 4))
        # classification
        self.neighbour_lane = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 2),
                                                    nn.Sigmoid())
        self.acc_classification = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 3),
                                                    nn.Softmax(dim=-1))
        self.lane_change = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 1),
                                                    nn.Sigmoid())
        self.traffic_light = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                                    nn.ELU(),
                                                    nn.Dropout(0.1),
                                                    nn.Linear(256, 4),
                                                    nn.Softmax(dim=-1))
            
        self.use_all_tokens = config.use_all_tokens
        self.adapter_fusion = config.adapter_fusion
        self.llm_inf_step = config.llm_inf_step

        if self.adapter_fusion:
            self.gameformer =  LLMEnhancedGameFormer_Adapter(encoder_layers=3, decoder_levels=2, modalities=6, neighbors=10) # this
        else:
            self.gameformer =  LLMEnhancedGameFormer(encoder_layers=3, decoder_levels=2, modalities=6, neighbors=10)
        self.map_encoder = GameformerEncoder(layers=3)
        self.feature_adpter = nn.Linear(self.model.config.hidden_size, 256)
            
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def reset_trainable_param(self):
        for name, param in self.named_parameters():
            if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in self._keep_in_fp32_modules):
                param.requires_grad = True
        
        if self.adapter_fusion:
            for name, param in self.gameformer.named_parameters():
                if 'llm_adapt_attention' in name:
                    param.requires_grad = True

        if self.config.enable_lora:
            for name, param in self.model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

    def reinit_weights(self):
        init_module_list = [name for name in self.adapter_name_list if hasattr(self, name)]
        for name in init_module_list:
            print(f"Reinit {name} weights")
            for module in getattr(self, name).modules():
                # if isinstance(module, nn.LSTM):
                #     import pdb; pdb.set_trace()
                if hasattr(module, '_reset_parameters'):
                    module._reset_parameters()
                elif hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                elif hasattr(module, 'flatten_parameters'):
                    module.flatten_parameters()
                # else:
                #     print(f"Module {module} has no reset_parameters or _reset_parameters method")
        
    def resume_from_checkpoint(self ,ckpt_dir, gameformer_ckpt=False):
        if gameformer_ckpt:
            weights = torch.load(ckpt_dir, map_location=torch.device('cpu'))

            self.gameformer.load_state_dict(weights, strict=False)

            processed_weights = OrderedDict()
            for key, value in weights.items():
                if key.startswith("encoder."):
                    new_key = key[len("encoder."):]
                    processed_weights[new_key] = value
            if len(processed_weights) == 0:
                processed_weights = weights
            self.map_encoder.load_state_dict(processed_weights, strict=False)
        # elif lora_ckpt:
        #     weights = torch.load(ckpt_dir)
        #     set_peft_model_state_dict(self, weights)
        #     print('LoRA pretrain weights have been loaded')
        else:
            if os.path.isdir(ckpt_dir):
                ckpt_ls = os.listdir(ckpt_dir)
                for ckpt in ckpt_ls:
                    if '.bin' in ckpt:
                        weights = torch.load(ckpt_dir+'/'+ckpt, map_location=torch.device('cpu'))
                        module = getattr(self, ckpt.split('.')[0], None)
                        if ckpt == 'embed_tokens.bin':
                            self.load_state_dict(weights, strict=False)
                        elif module is None:
                            print("%s could not be loaded successfully"%str(ckpt))
                        else:
                            try:
                                module.load_state_dict(weights, strict=True)
                                module.to(self.model.device)
                            except:
                                print("%s could not be loaded successfully"%str(ckpt))
            else:
                weights = torch.load(ckpt_dir)
                self.gameformer.load_state_dict(weights, strict=True)
        self.map_encoder.to(self.model.device)
        self.to(self.model.device)

    def reload_mapencoder_weights(self):
        self.reinit_weights()
        if self.config.mapEncoder_pretrain_weight is None:
            return
        pretrain_weights = torch.load(self.config.mapEncoder_pretrain_weight, map_location=torch.device('cpu'))
        self.gameformer.load_state_dict(pretrain_weights, strict=False)
        self.gameformer.to(self.model.device)
        processed_weights = OrderedDict()
        for key, value in pretrain_weights.items():
            if key.startswith("encoder."):
                new_key = key[len("encoder."):]
                processed_weights[new_key] = value
        if len(processed_weights) == 0:
            processed_weights = pretrain_weights
        self.map_encoder.load_state_dict(processed_weights, strict=False)
        self.map_encoder.to(self.model.device)

    def cuda(self, *args, **kwargs):
        return nn.Module.cuda(self, *args, **kwargs)

    def to(self, *args, **kwargs):
        return nn.Module.to(self, *args, **kwargs)

    def half(self, *args):
        return nn.Module.half(self)

    def float(self, *args):
        return nn.Module.float(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        map_feats: Optional[torch.FloatTensor] = None,
        map_masks: Optional[torch.FloatTensor] = None,
        urban_features = None,
        urban_avails = None,
        ego_agent_past: Optional[torch.FloatTensor] = None,
        neighbor_agents_past: Optional[torch.FloatTensor] = None,
        map_lanes: Optional[torch.FloatTensor] = None,
        map_crosswalks: Optional[torch.FloatTensor] = None,
        route_lanes: Optional[torch.FloatTensor] = None,
        ego_future: Optional[torch.FloatTensor] = None,
        neighbors_future: Optional[torch.FloatTensor] = None,
        cur_iter: Optional[torch.LongTensor] = 1,
        ego_v_a: Optional[torch.FloatTensor] = None,
        neighbour_lane: Optional[torch.FloatTensor] = None,
        acc_classification: Optional[torch.FloatTensor] = None,
        lane_change: Optional[torch.FloatTensor] = None,
        traffic_light: Optional[torch.FloatTensor] = None,
        ego_lane_flag: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        inference = False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, CausalLMOutputWithPastWithModel]:

        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # gt for gameformer
        # import ipdb; ipdb.set_trace()
        if not inference:
            ego_future_gt = ego_future
            neighbors_future_gt = neighbors_future
            neighbors_future_valid_gt = torch.ne(neighbors_future_gt[..., :2], 0)
        if map_feats is not None:
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
            map_feats = self.map_adapter(map_feats)
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
        if ego_agent_past is not None:
            assert map_feats is None
            raw_map_vector = {
                'ego_agent_past': ego_agent_past.to(self.map_adapter.weight.dtype), #[1, 21, 7]
                'neighbor_agents_past': neighbor_agents_past.to(self.map_adapter.weight.dtype),
                'map_lanes': map_lanes.to(self.map_adapter.weight.dtype), # [16, 40, 50, 7]
                'map_crosswalks': map_crosswalks.to(self.map_adapter.weight.dtype),
                'route_lanes': route_lanes.to(self.map_adapter.weight.dtype), # [16, 10, 50, 3]
            }
            encoder_outputs = self.map_encoder(raw_map_vector)
            map_feats, map_masks = encoder_outputs['encoding'], encoder_outputs['mask']
            if torch.isnan(map_feats).any():
                import pdb; pdb.set_trace()
            map_feats = self.map_adapter(map_feats)
            map_feats = map_feats.to(self.map_adapter.weight.dtype)
        else:
            raise NotImplementedError()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        outputs, labels, new_inputs_attention_mask, feature_position = self.model(
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
        ego_plan = None
        level_k_outputs = None
        hidden_states = outputs.last_hidden_state
        
        if isinstance(cur_iter, int):
            cur_iter = cur_iter
        else:
            cur_iter = getattr(cur_iter, "index", 0)

        if cur_iter % self.llm_inf_step != 0:
            hidden_states = self.prev_hidden_states
        else:
            self.prev_hidden_states = hidden_states
        
        # use query feature instead of direct hidden_states
        ########
        if self.use_all_tokens:
            pooling_features = hidden_states.mean(dim=1)
            hidden_states = pooling_features
            predicted_feature = self.feature_adpter(pooling_features)
        else:
            hidden_states = hidden_states[:, -1, :]
            predicted_feature = self.feature_adpter(hidden_states)
            
        # loss for llm hidden feature
        predicted_waypoints = self.waypoints_predictor(hidden_states)
        predicted_waypoints = predicted_waypoints.reshape(predicted_waypoints.shape[0], self.feature_len, 2)
        if not inference:
            waypoints_loss = F.smooth_l1_loss(predicted_waypoints, ego_future[..., :2])
            waypoints_loss += F.smooth_l1_loss(predicted_waypoints[:, -1], ego_future[:, -1, :2])

        predicted_ego_v_a = self.ego_v_a_predictor(hidden_states)
        predicted_ego_v_a = predicted_ego_v_a.reshape(predicted_ego_v_a.shape[0], 4)
        if not inference:
            v_a_loss = F.smooth_l1_loss(predicted_ego_v_a, ego_v_a)
        
        predicted_neighbour_lane = self.neighbour_lane(hidden_states)
        predicted_neighbour_lane = predicted_neighbour_lane.reshape(predicted_neighbour_lane.shape[0], 2)
        if not inference:
            neighbour_lane_loss = F.binary_cross_entropy(predicted_neighbour_lane, torch.tensor(neighbour_lane.squeeze(-1), dtype=torch.float32))
        
        pred_acc_classification = self.acc_classification(hidden_states)
        pred_acc_classification = pred_acc_classification.reshape(pred_acc_classification.shape[0], 3)
        if not inference:
            acc_class_loss = F.cross_entropy(pred_acc_classification, torch.tensor(acc_classification, dtype=torch.float32))
        
        pred_lane_change = self.lane_change(hidden_states)
        pred_lane_change = pred_lane_change.reshape(pred_lane_change.shape[0], 1)
        if not inference:
            lane_change_loss = F.binary_cross_entropy(pred_lane_change, torch.tensor(lane_change, dtype=torch.float32))
        
        pred_traffic_light = self.traffic_light(hidden_states)
        pred_traffic_light = pred_traffic_light.reshape(pred_traffic_light.shape[0], 4)
        if not inference:
            traffic_light_loss = F.cross_entropy(pred_traffic_light, torch.tensor(traffic_light, dtype=torch.float32))
        
        if not inference:
            llm_multi_head_loss = v_a_loss + neighbour_lane_loss + acc_class_loss + lane_change_loss + traffic_light_loss
            llm_loss = 0.5*waypoints_loss + 0.5*llm_multi_head_loss
        else:
            llm_loss = None
            waypoints_loss = None
            llm_multi_head_loss = None
            v_a_loss = None
            neighbour_lane_loss = None
            acc_class_loss = None
            lane_change_loss = None
            traffic_light_loss = None
            gameformer_loss = None
            gmm_loss = None
            plan_loss = None
        
        if len(predicted_feature.shape)<3:
            llm_feature = predicted_feature.unsqueeze(1)
        else:
            llm_feature = predicted_feature

        input_t = (raw_map_vector, llm_feature)
        level_k_outputs, ego_plan = self.gameformer(input_t)
        
        if not inference:
            gmm_loss, results = level_k_loss(level_k_outputs, ego_future_gt[..., :2], neighbors_future_gt[:,:self.gameformer.neighbors,:,:2], neighbors_future_valid_gt[:,:self.gameformer.neighbors,...])
            gmm_loss = torch.abs(gmm_loss)
            plan_loss = planning_loss(ego_plan, ego_future_gt[..., :2])
            gameformer_loss = gmm_loss + plan_loss
            loss = gameformer_loss + llm_loss
        # print("hidden_states.shape =", hidden_states.shape)

        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = self.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithModel(
            loss=loss if not inference else None,
            llm_loss=llm_loss,
            llm_regression_loss=waypoints_loss,
            llm_multi_head_loss=llm_multi_head_loss,
            urban_loss=None,
            v_a_loss=v_a_loss,
            neighbour_lane_loss=neighbour_lane_loss,
            acc_class_loss=acc_class_loss,
            lane_change_loss=lane_change_loss,
            traffic_light_loss=traffic_light_loss,
            gameformer_loss=gameformer_loss,
            gmm_loss=gmm_loss,
            gameformer_planner_loss=plan_loss,
            logits=logits,
            labels=labels,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            predictions = level_k_outputs,
            plan = ego_plan,
            llm_plan = predicted_waypoints,
        )

    def save_pretrained(
            self,
            save_directory: str,
            safe_serialization: bool = False,
            selected_adapters: Optional[List[str]] = None,
            **kwargs: Any,
    ):
        r"""
        This function saves the map adapteer, along with the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        # save map adapter weight
        model_names = kwargs.get("state_dict", None).keys()
        for name in self.adapter_name_list:
            module = getattr(self, name, None)
            if module and any(param.requires_grad for param in module.parameters()):
                if safe_serialization:
                    safe_save_file(
                        module.state_dict(),
                        os.path.join(save_directory, f'{name}.safetensors'),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(module.state_dict(), os.path.join(save_directory, f'{name}.bin'))
                print(f"Save {name}")

        # save global dict
        embed_tokens_name = None
        for model_name in model_names:
            if 'embed_tokens' in model_name:
                embed_tokens_name = model_name
                break

        os.makedirs(save_directory, exist_ok=True)

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

    def load_weights(self, model_id):
        if model_id is None:
            print('!!!!  No model id, not loaded at all')
            return
        if self.config.mapEncoder_pretrain_weight is None:
            self.config.mapEncoder_pretrain_weight = os.path.join(model_id, f'map_encoder.bin')
        self.reload_mapencoder_weights()
        for map_encoder_name in self.adapter_name_list:
            # if 'gameformer' in map_encoder_name:
            #     import pdb; pdb.set_trace()
            try:
                loaded_weight = torch.load(os.path.join(model_id, f'{map_encoder_name}.bin'))
                new_weight = OrderedDict()
                for k in loaded_weight:
                    new_weight[f'{map_encoder_name}.{k}'] = loaded_weight[k]
                loaded_weight = new_weight
            except:
                print(f'Error in load {map_encoder_name}')
                continue
            # print(f'Success in load {map_encoder_name}')

            for name, param in self.named_parameters():
                if name in loaded_weight.keys():
                    # if 'motion' in name:
                    #     import pdb; pdb.set_trace()
                    param.data.copy_(loaded_weight[name])
                    del loaded_weight[name]
                    print(f"Load {map_encoder_name} weight {name} from {model_id}")
            
            if len(loaded_weight.keys())!=0:
                for k in loaded_weight.keys():
                    print('%s has not been successfully loaded!!!!!!!!!!!!!'%str(k))

        try:
            loaded_weight = torch.load(os.path.join(model_id, 'embed_tokens.bin'))
        except:
            print(' error in load embed tokens')
            loaded_weight = {}
        for name, param in self.named_parameters():
            if name in loaded_weight.keys():
                param.data.copy_(loaded_weight[name])
                # param.requires_grad = is_trainable
                print(f"Load embed_tokens weight {name} from {model_id}")
                

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



class ModelWithLoRA(PeftModelForCausalLM):
    def __init__(self, model, peft_config, num_vector_tokens=64):
        super().__init__(model, peft_config)
        self.num_vector_tokens = num_vector_tokens
        self.to(model.device)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastWithModel]:

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return outputs
    
    def resume_lora_from_checkpoint(self, ckpt_path):
        weights = load_file(ckpt_path) 
        set_peft_model_state_dict(self, weights)
        print('LoRA pretrain weights have been loaded')


    def load_weights(self, model_id):
        if model_id is None:
            print('!!!!  No model id, not loaded at all')
            return
        lora_ckpt = os.path.join(model_id, "adapter_model.safetensors")
        # lora_weights = torch.load(lora_ckpt)
        lora_weights = load_file(lora_ckpt)
        set_peft_model_state_dict(self, lora_weights)
        print('LoRA weights have been loaded')

        global_ckpt_ = [dir_path for dir_path in os.listdir(model_id) if 'global' in dir_path][0]
        global_ckpt_ = os.path.join(model_id, global_ckpt_)
        model_states = [dir_path for dir_path in os.listdir(global_ckpt_) if 'model' in dir_path][0]
        model_ckpt_ = os.path.join(global_ckpt_, model_states)
        model_weights = torch.load(model_ckpt_)['module']
        for name, param in self.named_parameters():
                if name in model_weights.keys():
                    param.data.copy_(model_weights[name])
                    del model_weights[name]
                    print(f"Load {name} weight from {model_ckpt_}")

        if len(model_weights.keys())!=0:
                for k in model_weights.keys():
                    print('%s has not been successfully loaded!!!!!!!!!!!!!'%str(k))
        self.to(self.model.device)


        
