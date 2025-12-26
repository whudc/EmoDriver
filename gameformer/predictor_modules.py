import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(torch.float32))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output
    

class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256))

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs)
        output = torch.max(trajs, dim=-2).values

        return output


class GMMPredictor(nn.Module):
    def __init__(self, modalities=6):
        super(GMMPredictor, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    
    def forward(self, input):
        B, N, M, _ = input.shape
        traj = self.gaussian(input).view(B, N, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        score = self.score(input).squeeze(-1)

        return traj, score


class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InitialPredictionDecoder(nn.Module):
    def __init__(self, modalities, neighbors, dim=256):
        super(InitialPredictionDecoder, self).__init__()
        self._modalities = modalities
        self._agents = neighbors + 1
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor()
        self.register_buffer('modal', torch.arange(modalities).long())

    def forward(self, current_states, encoding, mask):
        N = self._agents
        multi_modal_query = self.multi_modal_query_embedding(self.modal) #[6, 256]
        query = encoding[:, :N, None, :] + multi_modal_query[None, None, :, :] #[16, 11, 6, 256] [B,num_agent,modalities,ch]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask) for i in range(N)], dim=1)
        predictions, scores = self.predictor(query_content) #[16, 11, 6, 80, 4], ([16, 11, 6]
        predictions[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, predictions, scores

class InitialPredictionRefDecoder(nn.Module):
    def __init__(self, modalities, neighbors, dim=256, ref_dim=256):
        super(InitialPredictionRefDecoder, self).__init__()
        self._modalities = modalities
        self._agents = neighbors + 1

        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.query_encoder = CrossTransformer()
        self.predictor = GMMPredictor()
        self.register_buffer('modal', torch.arange(modalities).long())

        self.ref_proj = nn.Linear(ref_dim, dim)
        self.query_fuse = nn.Linear(dim * 2, dim)  # 融合参考线特征

    def forward(self, current_states, encoding, mask, ref_emb=None):
        """
        current_states: [B, N, F]
        encoding: [B, M, D]
        mask: [B, M]
        ref_emb: [B, R, D] (可选)
        """
        N = self._agents

        # 多模态查询初始化
        multi_modal_query = self.multi_modal_query_embedding(self.modal)  # [M, D]
        base_query = encoding[:, :N, None, :] + multi_modal_query[None, None, :, :]  # [B, N, M, D]
        # print("InitialPredictionRefDecoder - base_query before ref:", base_query.shape)

        # === 引入参考线引导 ===
        if ref_emb is not None:
            # 聚合参考线 (可以用 mean / attention)
            ref_context = ref_emb.mean(dim=1)  # [B, D]
            ref_context = self.ref_proj(ref_context)  # [B, D]
            ref_context = ref_context[:, None, None, :].expand(-1, N, self._modalities, -1)
            base_query = self.query_fuse(torch.cat([base_query, ref_context], dim=-1))
        # print("InitialPredictionRefDecoder - base_query:", base_query.shape)
        # === query 编码 ===
        query_content = torch.stack([
            self.query_encoder(base_query[:, i], encoding, encoding, mask)
            for i in range(N)
        ], dim=1)  # [B, N, M, D]
        # print("InitialPredictionRefDecoder - query_content:", query_content.shape)

        # === 预测 ===
        predictions, scores = self.predictor(query_content)  # [B, N, M, T, 4], [B, N, M]
        predictions[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, predictions, scores


class InteractionDecoder(nn.Module):
    def __init__(self, modalities, future_encoder):
        super(InteractionDecoder, self).__init__()
        self.modalities = modalities
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor()

    def forward(self, current_states, actors, scores, last_content, encoding, mask):
        # current_states, last_predictions, last_scores, last_query_content, encoding, mask
        N = actors.shape[1]
        multi_futures = self.future_encoder(actors[..., :2], current_states[:, :N]) #[16, 11, 6, 256]
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2) #[16, 11, 256]
        interaction = self.interaction_encoder(futures, mask[:, :N])
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1)

        query = last_content + multi_futures #[16, 11, 6, 256]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask) for i in range(N)], dim=1)
        trajectories, scores = self.decoder(query_content)
        trajectories[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, trajectories, scores

class InteractionRefDecoder(nn.Module):
    def __init__(self, modalities, future_encoder, dim=256, ref_dim=256):
        super(InteractionRefDecoder, self).__init__()
        self.modalities = modalities
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor()

        # === 新增参考线相关模块 ===
        self.ref_proj = nn.Linear(ref_dim, dim)
        self.query_fuse = nn.Linear(dim * 2, dim)

    def forward(self, current_states, actors, scores, last_content, encoding, mask, ref_emb=None):
        """
        Args:
            current_states: [B, N, F]
            actors: [B, N, M, T, 4]   (上一步的多模态预测轨迹)
            scores: [B, N, M]
            last_content: [B, N, M, D]
            encoding: [B, L, D]
            mask: [B, L]
            ref_emb: [B, R, D]  (可选参考线特征)
        """
        B, N = actors.shape[:2]

        # === Step 1: 融合多模态未来轨迹 ===
        multi_futures = self.future_encoder(actors[..., :2], current_states[:, :N])  # [B, N, M, D]
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)     # [B, N, D]

        # === Step 2: 交互建模 ===
        interaction = self.interaction_encoder(futures, mask[:, :N])  # [B, N, D]
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1)

        # === Step 3: 构建 query（加入参考线引导）===
        query = last_content + multi_futures  # [B, N, M, D]

        if ref_emb is not None:
            # 聚合参考线
            ref_context = ref_emb.mean(dim=1)  # [B, D]
            ref_context = self.ref_proj(ref_context)  # [B, D]
            ref_context = ref_context[:, None, None, :].expand(-1, N, self.modalities, -1)  # [B, N, M, D]

            # 融合参考线信息
            query = self.query_fuse(torch.cat([query, ref_context], dim=-1))  # [B, N, M, D]

        # === Step 4: Query 编码与预测 ===
        query_content = torch.stack([
            self.query_encoder(query[:, i], encoding, encoding, mask)
            for i in range(N)
        ], dim=1)  # [B, N, M, D]

        trajectories, scores = self.decoder(query_content)
        trajectories[..., :2] += current_states[:, :N, None, None, :2]

        return query_content, trajectories, scores
