import torch
try:
    from .predictor_modules import *
except:
    from predictor_modules import *
from typing import Callable, List, Optional, Union
# from gameformer.layers.transformer import TransformerEncoderLayer, _get_activation_fn

# class FourierEmbedding(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
#         super(FourierEmbedding, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
#         self.mlps = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
#                     nn.LayerNorm(hidden_dim),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(hidden_dim, hidden_dim),
#                 )
#                 for _ in range(input_dim)
#             ]
#         )
#         self.to_out = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#     def forward(
#         self,
#         continuous_inputs: Optional[torch.Tensor],
#     ) -> torch.Tensor:
#         x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
#         x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
#         continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
#         for i in range(self.input_dim):
#             continuous_embs[i] = self.mlps[i](x[..., i, :])
#         x = torch.stack(continuous_embs).sum(dim=0)
#         return self.to_out(x)

# class PointsEncoder(nn.Module):
#     def __init__(self, feat_channel, encoder_channel):
#         super().__init__()
#         self.encoder_channel = encoder_channel
#         self.first_mlp = nn.Sequential(
#             nn.Linear(feat_channel, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 256),
#         )
#         self.second_mlp = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, self.encoder_channel),
#         )

#     def forward(self, x, mask=None):
#         """
#         x : B M 3
#         mask: B M
#         -----------------
#         feature_global : B C
#         """

#         bs, n, _ = x.shape
#         device = x.device

#         # print(x[mask].shape)
#         x_valid = self.first_mlp(x[mask])  # B n 256
#         x_features = torch.zeros(bs, n, 256, device=device)
#         x_features[mask] = x_valid

#         pooled_feature = x_features.max(dim=1)[0]
#         x_features = torch.cat(
#             [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
#         )


#         x_features_valid = self.second_mlp(x_features[mask])
#         res = torch.zeros(bs, n, self.encoder_channel, device=device)
#         res[mask] = x_features_valid

#         res = res.max(dim=1)[0]
#         return res


# class ReflineEncoder(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.r_pos_emb = FourierEmbedding(3, dim, 64)
#         self.r_encoder = PointsEncoder(6, dim)

#     def forward(self, ref_data):
#         r_position = ref_data["position"]  # [B, R, P, 3]
#         r_vector = ref_data["vector"]      # [B, R, P, 2]
#         r_orientation = ref_data["orientation"]
#         r_valid_mask = ref_data["valid_mask"]

#         bs, R, P, _ = r_position.shape
#         r_valid_mask = r_valid_mask.view(bs * R, P)

#         # 组合几何特征
#         r_feature = torch.cat([
#             r_position - r_position[..., 0:1, :2],
#             r_vector,
#             torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1)
#         ], dim=-1)
#         r_feature = r_feature.reshape(bs * R, P, -1)

#         # 提取每条参考线的全局embedding
#         r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)

#         # 加上位置embedding
#         r_pos = torch.cat([r_position[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
#         r_emb = r_emb + self.r_pos_emb(r_pos)
#         return r_emb  # [B, R, D]


# class SafeTransformerEncoderLayer(nn.Module):
#     __constants__ = ['batch_first', 'norm_first']
#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(SafeTransformerEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             activation = _get_activation_fn(activation)

#         # We can't test self.activation in forward() in TorchScript,
#         # so stash some information about it instead.
#         if activation is F.relu or isinstance(activation, torch.nn.ReLU):
#             self.activation_relu_or_gelu = 1
#         elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
#             self.activation_relu_or_gelu = 2
#         else:
#             self.activation_relu_or_gelu = 0
#         self.activation = activation

#     def __setstate__(self, state):
#         super(SafeTransformerEncoderLayer, self).__setstate__(state)
#         if not hasattr(self, 'activation'):
#             self.activation = F.relu


#     def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
#                 src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         r"""Pass the input through the encoder layer.

#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """

#         if src_key_padding_mask is not None:
#             _skpm_dtype = src_key_padding_mask.dtype
#             if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
#                 raise AssertionError(
#                     "only bool and floating types of key_padding_mask are supported")
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#         why_not_sparsity_fast_path = ''
#         if not src.dim() == 3:
#             why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
#         elif self.training:
#             why_not_sparsity_fast_path = "training is enabled"
#         elif not self.self_attn.batch_first :
#             why_not_sparsity_fast_path = "self_attn.batch_first was not True"
#         elif not self.self_attn._qkv_same_embed_dim :
#             why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
#         elif not self.activation_relu_or_gelu:
#             why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
#         elif not (self.norm1.eps == self.norm2.eps):
#             why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
#         elif src_mask is not None:
#             why_not_sparsity_fast_path = "src_mask is not supported for fastpath"
#         elif src.is_nested and src_key_padding_mask is not None:
#             why_not_sparsity_fast_path = "src_key_padding_mask is not supported with NestedTensor input for fastpath"
#         elif self.self_attn.num_heads % 2 == 1:
#             why_not_sparsity_fast_path = "num_head is odd"
#         elif torch.is_autocast_enabled():
#             why_not_sparsity_fast_path = "autocast is enabled"

#         if not why_not_sparsity_fast_path:
#             tensor_args = (
#                 src,
#                 self.self_attn.in_proj_weight,
#                 self.self_attn.in_proj_bias,
#                 self.self_attn.out_proj.weight,
#                 self.self_attn.out_proj.bias,
#                 self.norm1.weight,
#                 self.norm1.bias,
#                 self.norm2.weight,
#                 self.norm2.bias,
#                 self.linear1.weight,
#                 self.linear1.bias,
#                 self.linear2.weight,
#                 self.linear2.bias,
#             )

#             # We have to use list comprehensions below because TorchScript does not support
#             # generator expressions.
#             if torch.overrides.has_torch_function(tensor_args):
#                 why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
#             elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
#                 why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
#             elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
#                 why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
#                                               "input/output projection weights or biases requires_grad")

#             if not why_not_sparsity_fast_path:
#                 return torch._transformer_encoder_layer_fwd(
#                     src,
#                     self.self_attn.embed_dim,
#                     self.self_attn.num_heads,
#                     self.self_attn.in_proj_weight,
#                     self.self_attn.in_proj_bias,
#                     self.self_attn.out_proj.weight,
#                     self.self_attn.out_proj.bias,
#                     self.activation_relu_or_gelu == 2,
#                     self.norm_first,
#                     self.norm1.eps,
#                     self.norm1.weight,
#                     self.norm1.bias,
#                     self.norm2.weight,
#                     self.norm2.bias,
#                     self.linear1.weight,
#                     self.linear1.bias,
#                     self.linear2.weight,
#                     self.linear2.bias,
#                     # TODO: if src_mask and src_key_padding_mask merge to single 4-dim mask
#                     src_mask if src_mask is not None else src_key_padding_mask,
#                     1 if src_key_padding_mask is not None else
#                     0 if src_mask is not None else
#                     None,
#                 )


#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
#             x = self.norm2(x + self._ff_block(x))

#         return x

#     # self-attention block
#     def _sa_block(self, x: torch.Tensor,
#                   attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)

#     # feed forward block
#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)

class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(agent_dim=11)
        self.ego_encoder = AgentEncoder(agent_dim=7)
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)
        # attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
        #                                              activation=F.gelu, dropout=dropout, batch_first=True)
        self.refline_encoder = ReflineEncoder(dim)
        attention_layer = SafeTransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
       

    # 临时修改fusion_encoder的前向传播
    # def debug_forward(self, scene, mask):
    #     print(f"Input to transformer - mean: {scene.mean()}, std: {scene.std()}")
    #     print(f"mask shape: {mask.shape}, mask sum: {mask.sum()}")
    #     # 逐层检查
    #     x = scene
    #     for i, layer in enumerate(self.fusion_encoder.layers):
    #         x = layer(x, src_key_padding_mask=mask)
    #         print(f"Layer {i} output - mean: {x.mean()}, std: {x.std()}, NaN: {torch.isnan(x).any()}")
    #         if torch.isnan(x).any():
    #             print(f"NaN first appeared in layer {i}")
    #             break
    #     return x


    def forward(self, inputs):
        # agents
        # agent_pos = inputs['ego_agent_past'][:, :, :2]
        # agent_heading = inputs["ego_agent_past"][:, :, 2]
        # print("agent_pos:", agent_pos.shape)
        # print("agent_heading:", agent_heading.shape)
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        ref_lines = inputs['ref_lines']
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)
        
        # agent encoding
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)
        # vector maps
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']
        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)
        # refline encoding
        encoded_reflines = self.refline_encoder(ref_lines)
        # attention fusion encoding
        scene = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)

        # encoding = self.fusion_encoder(scene, src_key_padding_mask=mask)
        # encoding = self.debug_forward(scene, mask)
        encoding = self.fusion_encoder(scene, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'route_lanes': inputs['route_lanes'],
            'ref_emb': encoded_reflines
        }

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, modalities=6, levels=3):
        super(Decoder, self).__init__()
        self.levels = levels
        future_encoder = FutureEncoder()
        self.use_ref = False
        # initial level
        if self.use_ref:
            self.initial_predictor = InitialPredictionRefDecoder(modalities, neighbors)
        else:
            self.initial_predictor = InitialPredictionDecoder(modalities, neighbors)

        # level-k reasoning
        if self.use_ref:
            self.interaction_stage = nn.ModuleList([InteractionRefDecoder(modalities, future_encoder) for _ in range(levels)])
        else:
            self.interaction_stage = nn.ModuleList([InteractionDecoder(modalities, future_encoder) for _ in range(levels)])

    def forward(self, encoder_outputs, llm_feature=None):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1] # 1 21(ego+agents) 5(xyzvxvy)
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']
        ref_emb = encoder_outputs['ref_emb']
        if llm_feature is not None:
            mask = torch.cat([mask, torch.zeros_like(llm_feature[...,0])], dim=1)
            encoding = torch.cat([encoding, llm_feature], dim=1)

        # level 0 decode
        if self.use_ref:
            last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask, ref_emb=ref_emb)
        else:
            last_content, last_level, last_score = self.initial_predictor(current_states, encoding, mask)
        # query_content, predictions, scores
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_score
        
        # level k reasoning
        for k in range(1, self.levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            if self.use_ref:
                last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask, ref_emb=ref_emb)
            else:
                last_content, last_level, last_score = interaction_decoder(current_states, last_level, last_score, last_content, encoding, mask)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_score
        
        env_encoding = last_content[:, 0]

        return decoder_outputs, env_encoding


class NeuralPlanner(nn.Module):
    def __init__(self):
        super(NeuralPlanner, self).__init__()
        self._future_len = 80
        self.route_encoder = VectorMapEncoder(3, 50)
        self.route_fusion = CrossTransformer()
        self.plan_decoder = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))

    def forward(self, env_encoding, route_lanes):
        route_lanes, mask = self.route_encoder(route_lanes)
        mask[:, 0] = False
        env_encoding = torch.max(env_encoding, dim=1, keepdim=True)[0]
        route_encoding = self.route_fusion(env_encoding, route_lanes, route_lanes, mask)
        env_route_encoding = torch.cat([env_encoding, route_encoding], dim=-1)
        plan = self.plan_decoder(env_route_encoding.squeeze(1))
        plan = plan.reshape(plan.shape[0], self._future_len, 2)

        return plan


class GameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10): # 3 2 6 20
        super(GameFormer, self).__init__()
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        route_lanes = encoder_outputs['route_lanes']
        decoder_outputs, env_encoding = self.decoder(encoder_outputs)
        ego_plan = self.planner(env_encoding, route_lanes)

        return decoder_outputs, ego_plan

class LLMEnhancedGameFormer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_levels=3, modalities=6, neighbors=10, share_encoder=None):
        super(LLMEnhancedGameFormer, self).__init__()
        self.neighbors = neighbors
        self.share_encoder = share_encoder
        self.encoder = Encoder(layers=encoder_layers)
        self.decoder = Decoder(neighbors, modalities, decoder_levels)
        self.planner = NeuralPlanner()

    def forward_not_share_encoder(self, inputs, llm_feature):
        # TODO: add fusion module
        encoder_outputs = self.encoder(inputs, swap=self.swap)
        route_lanes = encoder_outputs['route_lanes']
        decoder_outputs, env_encoding = self.decoder(encoder_outputs, llm_feature=llm_feature)
        ego_plan = self.planner(env_encoding, route_lanes)

        return decoder_outputs, ego_plan
    
    def forward_share_encoder(self, encoder_outputs, llm_feature):
        # TODO: add fusion module
        route_lanes = encoder_outputs['route_lanes']
        decoder_outputs, env_encoding = self.decoder(encoder_outputs, llm_feature=llm_feature)
        ego_plan = self.planner(env_encoding, route_lanes)

        return decoder_outputs, ego_plan
    
    def forward(self, input, swap=False):
        self.swap = swap
        if self.share_encoder:
            encoder_outputs, llm_feature = input
            return self.forward_share_encoder(encoder_outputs, llm_feature)
        else:
            raw_inputs, llm_feature = input
            return self.forward_not_share_encoder(raw_inputs, llm_feature)