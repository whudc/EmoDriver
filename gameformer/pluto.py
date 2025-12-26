# import math
# import numpy as np
# import torch
# import torch.nn as nn

# from gameformer.layers.fourier_embedding import FourierEmbedding
# from gameformer.layers.transformer import TransformerEncoderLayer
# from gameformer.layers.mlp_layer import MLPLayer

# from gameformer.modules.agent_encoder import AgentEncoder
# from gameformer.modules.map_encoder import MapEncoder
# from gameformer.modules.static_objects_encoder import StaticObjectsEncoder
# from gameformer.modules.agent_predictor import AgentPredictor
# from gameformer.modules.planning_decoder import PlanningDecoder


# class PlutoPlanner(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dim = 128
#         self.num_heads = 4
#         self.num_modes = 12
#         self.state_channel = 6
#         self.history_channel = 9
#         self.drop_path = 0.2
#         self.dropout = 0.1
#         self.state_dropout = 0.75
#         self.history_steps = 21
#         self.future_steps = 80
#         self.polygon_channel = 6
#         self.radius = 120
#         self.encoder_depth = 4
#         self.decoder_depth = 4
#         self.cat_x = True
#         self.use_hidden_proj = True
#         self.ref_free_traj = True

#         self.pos_emb = FourierEmbedding(3, self.dim, 64)

#         self.agent_encoder = AgentEncoder(
#             state_channel=self.state_channel,
#             history_channel=self.history_channel,
#             dim=self.dim,
#             hist_steps=self.history_steps,
#             drop_path=self.drop_path,
#             use_ego_history=False,
#             state_attn_encoder=True,
#             state_dropout=self.state_dropout,
#         )

#         self.map_encoder = MapEncoder(
#             dim=self.dim,
#             polygon_channel=self.polygon_channel,
#             use_lane_boundary=True,
#         )

#         self.static_objects_encoder = StaticObjectsEncoder(dim=self.dim)

#         self.encoder_blocks = nn.ModuleList(
#             TransformerEncoderLayer(dim=self.dim, num_heads=self.num_heads, drop_path=dp)
#             for dp in [x.item() for x in torch.linspace(0, self.drop_path, self.encoder_depth)]
#         )
#         self.norm = nn.LayerNorm(self.dim)

#         self.agent_predictor = AgentPredictor(dim=self.dim, future_steps=self.future_steps)
#         self.planning_decoder = PlanningDecoder(
#             num_mode=self.num_modes,
#             decoder_depth=self.decoder_depth,
#             dim=self.dim,
#             num_heads=self.num_heads,
#             mlp_ratio=4,
#             dropout=self.dropout,
#             cat_x=self.cat_x,
#             future_steps=self.future_steps,
#         )

#         if self.use_hidden_proj:
#             self.hidden_proj = nn.Sequential(
#                 nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim)
#             )

#         if self.ref_free_traj:
#             self.ref_free_decoder = MLPLayer(self.dim, 2 * self.dim, self.future_steps * 4)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.BatchNorm1d):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Embedding):
#             nn.init.normal_(m.weight, mean=0.0, std=0.02)

#     def forward(self, inputs):
#         data = self.data_change(inputs)
#         agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
#         agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
#         agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
#         polygon_center = data["map"]["polygon_center"]
#         polygon_mask = data["map"]["valid_mask"]

#         bs, A = agent_pos.shape[0:2]

#         position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
#         angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
#         angle = (angle + math.pi) % (2 * math.pi) - math.pi
#         pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

#         agent_key_padding = ~(agent_mask.any(-1))
#         polygon_key_padding = ~(polygon_mask.any(-1))
#         key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

#         x_agent = self.agent_encoder(data)
#         x_polygon = self.map_encoder(data)
#         x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

#         x = torch.cat([x_agent, x_polygon, x_static], dim=1)

#         pos = torch.cat([pos, static_pos], dim=1)
#         pos_embed = self.pos_emb(pos)

#         key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
#         x = x + pos_embed

#         for blk in self.encoder_blocks:
#             x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
#         x = self.norm(x)

#         prediction = self.agent_predictor(x[:, 1:A])

#         ref_line_available = data["reference_line"]["position"].shape[1] > 0

#         if ref_line_available:
#             trajectory, probability = self.planning_decoder(
#                 data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
#             )
#         else:
#             trajectory, probability = None, None

#         out = {
#             "trajectory": trajectory,
#             "probability": probability,  # (bs, R, M)
#             "prediction": prediction,  # (bs, A-1, T, 2)
#         }

#         if self.use_hidden_proj:
#             out["hidden"] = self.hidden_proj(x[:, 0])

#         if self.ref_free_traj:
#             ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
#                 bs, self.future_steps, 4
#             )
#             out["ref_free_trajectory"] = ref_free_traj

#         if not self.training:
#             if self.ref_free_traj:
#                 ref_free_traj_angle = torch.arctan2(
#                     ref_free_traj[..., 3], ref_free_traj[..., 2]
#                 )
#                 ref_free_traj = torch.cat(
#                     [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
#                 )
#                 out["output_ref_free_trajectory"] = ref_free_traj

#             output_prediction = torch.cat(
#                 [
#                     prediction[..., :2] + agent_pos[:, 1:A, None],
#                     torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
#                     + agent_heading[:, 1:A, None, None],
#                     prediction[..., 4:6],
#                 ],
#                 dim=-1,
#             )
#             out["output_prediction"] = output_prediction

#             if trajectory is not None:
#                 r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
#                 probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

#                 angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
#                 out_trajectory = torch.cat(
#                     [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
#                 )

#                 bs, R, M, T, _ = out_trajectory.shape
#                 flattened_probability = probability.reshape(bs, R * M)
#                 best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
#                     torch.arange(bs), flattened_probability.argmax(-1)
#                 ]

#                 out["output_trajectory"] = best_trajectory
#                 out["candidate_trajectories"] = out_trajectory
#             else:
#                 out["output_trajectory"] = out["output_ref_free_trajectory"]
#                 out["probability"] = torch.zeros(1, 0, 0)
#                 out["candidate_trajectories"] = torch.zeros(
#                     1, 0, 0, self.future_steps, 3
#                 )

#         return out

#     def data_change(self, inputs):
#         data = {}
#         # Ego agent
#         position = inputs["ego_agent_past"][..., :2]
#         heading = inputs["ego_agent_past"][..., 2]
#         velocity = inputs["ego_agent_past"][..., 3:5]
#         acceleration = inputs["ego_agent_past"][..., 5:7]
#         T = position.shape[1]
#         shape = np.zeros((T, 2), dtype=np.float64)
#         valid_mask = np.ones(T, dtype=bool)
        
#         for t in range(T):
#             shape[t] = np.array([4.5, 2.0], dtype=np.float64)
#         category = np.array(0, dtype=np.int8)
#         ego_features = {
#             'position': position,
#             'heading': heading,
#             'velocity': velocity,
#             'acceleration': acceleration,
#             'shape': shape,
#             'category': category,
#             'valid_mask': valid_mask
#         }

#         # Other agents
#         neighbor = inputs['neighbor_agents_past']
#         # 根据速度区分动静态物体
#         static_agents, dynamic_agents, _, _ = self.split_static_dynamic_agents(neighbor, v_thresh=0.1)

#         agent_features, agent_tokens, agents_polygon = self.get_agent_features(dynamic_agents)
#         data['agent'] = {}
#         for k in agent_features:
#             data["agent"][k] = np.concatenate(
#                 [ego_features[k][None, ...], agent_features[k]], axis=0
#             )
#         agent_tokens = ["ego"] + agent_tokens

#         data['static_objects'] = self.get_stacic_objects_features(static_agents)

#         data['map'], map_polygon_tokens = self.get_map_features_from_tensors(
#             inputs['map_lanes'], inputs['umap_crosswalks'], inputs['route_lanes']
#         )
        
#         data['reference_line'] = inputs['reference_line']

#         return data

#     def split_static_dynamic_agents(self, neighbor_agent_past, v_thresh=0.1):
#         """
#         neighbor_agent_past: np.ndarray, shape (B, N, T, 11)
#             B: batch size
#             N: max agents
#             T: time steps
#             11 features: x, y, heading, vx, vy, yaw_rate, length, width, one-hot(3)
#         v_thresh: float, 速度阈值, m/s
        
#         Returns:
#             static_mask: np.ndarray, shape (B, N), True 表示静态
#             dynamic_mask: np.ndarray, shape (B, N), True 表示动态
#         """
#         # 取速度 vx, vy
#         vx = neighbor_agent_past[..., 3]  # shape (B, N, T)
#         vy = neighbor_agent_past[..., 4]  # shape (B, N, T)

#         # 计算速度幅值（每个 agent 的时间平均速度）
#         speed = np.sqrt(vx**2 + vy**2)  # shape (B, N, T)
#         mean_speed = np.mean(speed, axis=-1)  # shape (B, N)

#         # 划分静态/动态
#         static_mask = mean_speed < v_thresh
#         dynamic_mask = ~static_mask

#         static_agents = []
#         dynamic_agents = []

#         for b in range(B):
#             static_idx = np.where(static_mask[b])[0]
#             dynamic_idx = np.where(dynamic_mask[b])[0]

#             static_agents.append(neighbor_agent_past[b, static_idx, :, :])
#             dynamic_agents.append(neighbor_agent_past[b, dynamic_idx, :, :])

#         # pad成列表方便返回，也可以保持 ragged array
#         return static_agents, dynamic_agents, static_mask, dynamic_mask


    
#     def get_agent_features_from_tensor(self, neighbor_agent_past, max_agents=None):
#         """
#         neighbor_agent_past: np.ndarray or torch.Tensor, shape (B, N, T, 11)
#             B: batch size
#             N: max agents (e.g., 20)
#             T: number of frames (e.g., 21)
#             11 features: x, y, heading, vx, vy, yaw_rate, length, width, one-hot(3)
#         max_agents: int, optional, 最多选取的 agent 数量
        
#         Returns:
#             agent_features: dict with keys position, heading, velocity, shape, category, valid_mask
#                 - position: (B, N_selected, T, 2)
#                 - heading: (B, N_selected, T)
#                 - velocity: (B, N_selected, T, 2)
#                 - shape: (B, N_selected, T, 2)
#                 - category: (B, N_selected)
#                 - valid_mask: (B, N_selected, T)
#             agent_indices: list of selected agent indices per batch
#         """
#         B, N, T, C = neighbor_agent_past.shape
#         assert C == 11, "Expect 11 features in neighbor_agent_past"

#         if max_agents is None:
#             max_agents = N
#         N_selected = min(N, max_agents)

#         # 初始化输出
#         position = np.zeros((B, N_selected, T, 2), dtype=np.float32)
#         heading = np.zeros((B, N_selected, T), dtype=np.float32)
#         velocity = np.zeros((B, N_selected, T, 2), dtype=np.float32)
#         shape = np.zeros((B, N_selected, T, 2), dtype=np.float32)
#         category = np.zeros((B, N_selected), dtype=np.int8)
#         valid_mask = np.zeros((B, N_selected, T), dtype=bool)
#         polygon = [None] * N
#         agent_indices_list = []

#         for b in range(B):
#             # 取 agent 的有效性：只要 x,y 不全为0就认为是有效 agent
#             agent_valid = np.any(neighbor_agent_past[b, :, :, :2] != 0, axis=1)
#             valid_agent_idx = np.where(agent_valid)[0]
#             if len(valid_agent_idx) == 0:
#                 agent_indices_list.append([])
#                 continue

#             # 按距离排序可以选择最近 agent，这里假设最后一帧是 query
#             last_pos = neighbor_agent_past[b, valid_agent_idx, -1, :2]  # shape (num_valid, 2)
#             distances = np.linalg.norm(last_pos, axis=1)
#             sorted_idx = valid_agent_idx[np.argsort(distances)[:N_selected]]
#             agent_indices_list.append(sorted_idx.tolist())

#             # 填充特征
#             position[b, :len(sorted_idx), :, :] = neighbor_agent_past[b, sorted_idx, :, :2]
#             heading[b, :len(sorted_idx), :] = neighbor_agent_past[b, sorted_idx, :, 2]
#             velocity[b, :len(sorted_idx), :, :] = neighbor_agent_past[b, sorted_idx, :, 3:5]
#             shape[b, :len(sorted_idx), :, :] = neighbor_agent_past[b, sorted_idx, :, 6:8]
#             # one-hot 转 category: [1,0,0]->1, [0,1,0]->2, [0,0,1]->3
#             one_hot = neighbor_agent_past[b, sorted_idx, -1, 8:11]
#             category[b, :len(sorted_idx)] = np.argmax(one_hot, axis=1) + 1
#             valid_mask[b, :len(sorted_idx), :] = np.any(neighbor_agent_past[b, sorted_idx, :, :2] != 0, axis=-1)

#         agent_features = {
#             "position": position,
#             "heading": heading,
#             "velocity": velocity,
#             "shape": shape,
#             "category": category,
#             "valid_mask": valid_mask,
#         }

#         return agent_features, agent_indices_list, polygon

#     def get_stacic_objects_features(self, static_objects):
#         """
#         static_objects: np.ndarray, shape (S, 11)
#             S: number of static objects
#             11 features: x, y, heading, vx, vy, yaw_rate, length, width, one-hot(3)
        
#         Returns:
#             static_features: dict with keys position, heading, velocity, shape, category, valid_mask
#                 - position: (S, 2)
#                 - heading: (S,)
#                 - velocity: (S, 2)
#                 - shape: (S, 2)
#                 - category: (S,)
#                 - valid_mask: (S,)
#         """
#         S = static_objects.shape[0]

#         position = static_objects[:, :2]
#         heading = static_objects[:, 2]
#         velocity = static_objects[:, 3:5]
#         shape = static_objects[:, 6:8]
#         one_hot = static_objects[:, 8:11]
#         category = np.argmax(one_hot, axis=1) + 1
#         valid_mask = np.any(static_objects[:, :2] != 0, axis=1)

#         static_features = {
#             "position": position,
#             "heading": heading,
#             "shape": shape,
#             "category": category,
#             "valid_mask": valid_mask,
#         }

#         return static_features
    
#     def get_map_features_from_tensors(self, map_lanes, umap_crosswalks, route_lanes):
#         """
#         输入：
#             map_lanes: (B, 40, 50, 7)
#             umap_crosswalks: (B, 5, 30, 3)
#             route_lanes: (B, 10, 50, 3)
#         输出：
#             map_features: dict
#             object_ids: list of ints
#         """
#         B = map_lanes.shape[0]
#         M_lane = map_lanes.shape[1]
#         P_lane = map_lanes.shape[2]
#         M_cross = umap_crosswalks.shape[1]
#         P_cross = umap_crosswalks.shape[2]

#         map_features_batch = []
#         object_ids_batch = []

#         polygon_types = ["lane", "crosswalk"]

#         for b in range(B):
#             # 合并对象
#             lanes = map_lanes[b]        # (40,50,7)
#             crosswalks = umap_crosswalks[b]  # (5,30,3)
#             M = M_lane + M_cross
#             object_ids = list(range(M))  # 简单用索引作为 id
#             object_ids_batch.append(object_ids)

#             # 初始化特征数组
#             # point_position: (M, P, 2)
#             point_position = np.zeros((M, P_lane, 2), dtype=np.float32)
#             point_vector = np.zeros((M, P_lane-1, 2), dtype=np.float32)
#             point_orientation = np.zeros((M, P_lane-1), dtype=np.float32)
#             polygon_type = np.zeros(M, dtype=np.int8)
#             polygon_on_route = np.zeros(M, dtype=bool)

#             # 处理车道
#             for i in range(M_lane):
#                 pts = lanes[i, :, :2]  # x,y
#                 point_position[i, :, :] = pts
#                 point_vector[i, :, :] = pts[1:] - pts[:-1]
#                 point_orientation[i, :] = np.arctan2(point_vector[i, :, 1], point_vector[i, :, 0])
#                 polygon_type[i] = 0  # lane
#                 # 判断是否在导航车道 route_lanes
#                 route_match = False
#                 for r in range(route_lanes.shape[1]):
#                     route_pts = route_lanes[b, r, :, :2]
#                     if np.allclose(pts, route_pts, atol=1e-2):
#                         route_match = True
#                         break
#                 polygon_on_route[i] = route_match

#             # 处理人行横道
#             for i in range(M_cross):
#                 idx = M_lane + i
#                 pts = crosswalks[i, :, :2]
#                 point_position[idx, :P_cross, :] = pts
#                 # 人行横道向量/朝向
#                 if P_cross > 1:
#                     point_vector[idx, :P_cross-1, :] = pts[1:] - pts[:-1]
#                     point_orientation[idx, :P_cross-1] = np.arctan2(point_vector[idx, :P_cross-1, 1], point_vector[idx, :P_cross-1, 0])
#                 polygon_type[idx] = 1  # crosswalk
#                 polygon_on_route[idx] = False

#             map_features = {
#                 "point_position": point_position,
#                 "point_vector": point_vector,
#                 "point_orientation": point_orientation,
#                 "polygon_type": polygon_type,
#                 "polygon_on_route": polygon_on_route,
#             }
#             map_features_batch.append(map_features)

#         return map_features_batch, object_ids_batch
