import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import json
import os
from shapely.geometry import Point, LineString
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import matplotlib.pyplot as plt


def visualize_driving_sample(
    ego, neighbors, map_lanes, map_crosswalks, route_lanes,
    ego_future_gt, neighbors_future_gt, save_path=None, title=None
):
    """
    可视化 DrivingData 中的一个样本

    Args:
        ego: np.ndarray, shape [T, 2 or 3]，自车历史轨迹
        neighbors: np.ndarray, shape [N, T, 2 or 3]，邻居车辆历史轨迹
        map_lanes: list[np.ndarray], 每个车道中心线坐标
        map_crosswalks: list[np.ndarray], 每个斑马线的多边形坐标
        route_lanes: list[np.ndarray], 路径车道线坐标
        ego_future_gt: np.ndarray, shape [T_future, 2 or 3]，自车未来轨迹
        neighbors_future_gt: np.ndarray, shape [N, T_future, 2 or 3]，邻居车辆未来轨迹
        save_path: str，图片保存路径
        title: str，标题
    """

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # ==== 地图层 ====
    for lane in map_lanes:
        lane = np.array(lane)
        plt.plot(lane[:, 0], lane[:, 1], color='lightgray', linewidth=1.0, alpha=0.8)

    for cross in map_crosswalks:
        cross = np.array(cross)
        plt.fill(cross[:, 0], cross[:, 1], color='gray', alpha=0.3)

    # ==== 规划路线 ====
    for rl in route_lanes:
        rl = np.array(rl)
        plt.plot(rl[:, 0], rl[:, 1], color='orange', linewidth=2.5, label='route')

    # ==== 自车轨迹 ====
    plt.plot(ego[:, 0], ego[:, 1], '-o', color='blue', label='ego past', markersize=3)
    plt.plot(ego_future_gt[:, 0], ego_future_gt[:, 1], '-o', color='cyan', label='ego future', markersize=3)

    # ==== 邻居轨迹 ====
    for i, n in enumerate(neighbors):
        plt.plot(n[:, 0], n[:, 1], color='green', linewidth=1, alpha=0.6)
    for i, n_future in enumerate(neighbors_future_gt):
        plt.plot(n_future[:, 0], n_future[:, 1], '--', color='lime', alpha=0.6)

    # ==== 坐标样式 ====
    plt.scatter(ego[-1, 0], ego[-1, 1], c='red', s=50, label='ego current')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    if title:
        plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MAX_REF = 10  # 假设最多10条参考线
def pad_ref_line(ref_line, max_ref=MAX_REF):
    padded = {}
    for key, value in ref_line.items():
        shape = list(value.shape)
        # 用 zeros/padding array 扩展第0维到 max_ref
        new_shape = [max_ref] + shape[1:]
        tmp = np.zeros(new_shape, dtype=value.dtype)
        n = min(shape[0], max_ref)
        tmp[:n] = value[:n]
        padded[key] = tmp
    return padded


class DrivingData(Dataset):
    def __init__(self, data_dir, n_neighbors):
        if data_dir.endswith('.json'):
            with open(data_dir, 'r') as f:
                self.data_list = json.load(f)
                base_nameset = set([os.path.basename(j['map_info']) for j in self.data_list])
                self.data_list = [j['map_info']  for j in self.data_list]
        else:
            self.data_list = glob.glob(data_dir)
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes'] 
        map_lanes = data['lanes']
        map_crosswalks = data['crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]
        
        # build instruction
        instruction = ''
        for i in range(len(data['instruction'][0])):
            instruction += data['instruction'][0][i]
            if 'stop' in data['instruction'][0][i]:
                continue
            instruction += (str(data['instruction'][1][i])+' meters. ')
        
        # visualize_driving_sample(
        #     ego, neighbors, map_lanes, map_crosswalks, route_lanes,
        #     ego_future_gt, neighbors_future_gt,
        #     save_path="./vis", title=f'Sample {idx}'
        # )

        # return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt, instruction

        ref_line = self.compute_reference_lines(ego, route_lanes, ego_future_gt)
        ref_line = pad_ref_line(ref_line, max_ref=MAX_REF)
        # ✅ 转为 float32，保持和模型权重一致
        for k, v in ref_line.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ref_line[k] = v.astype(np.float32)

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt, instruction, ref_line


    def compute_reference_lines(
        self, ego, route_lanes, ego_future_gt, length=100, step=1.0, heading_tol=0.3
    ):
        """
        基于自车状态和规划路线生成参考线特征
        Args:
            ego: ndarray [T_past, 3] or [T_past, 2], 自车过去轨迹
            route_lanes: list of dict/list，每个包含中心线坐标点 (N_i, 2)
            length: 最大参考线长度（米）
            step: 采样间隔
            heading_tol: 自车方向与车道方向的最大夹角（弧度）
        Returns:
            dict:
                {
                    "position": (N_ref, n_points, 2),
                    "vector": (N_ref, n_points, 2),
                    "orientation": (N_ref, n_points),
                    "valid_mask": (N_ref, n_points),
                    "future_projection": (N_ref, 8, 2)
                }
        """
        # Step 1. ego状态
        ego_x = ego[-1, 0]
        ego_y = ego[-1, 1]
        ego_heading = ego[-1, 2]

        ego_point = Point(ego_x, ego_y)

        candidate_paths = []

        # Step 2. 遍历车道并筛选
        for lane in route_lanes:
            lane_points = np.array(lane)
            if lane_points.shape[1] > 2:
                lane_xy = lane_points[:, :2]
            else:
                lane_xy = lane_points
            
            lane_xy = np.asarray(lane_xy)
            if np.any(~np.isfinite(lane_xy)):
                continue
            diffs = np.linalg.norm(np.diff(lane_xy, axis=0), axis=-1)
            mask = np.insert(diffs > 0.1, 0, True)
            lane_xy = lane_xy[mask]

            if len(lane_xy) < 2:
                continue

            line = LineString(lane_xy)
            # print(line)
            try:
                dist = line.distance(ego_point)
            except RuntimeWarning:
                print(f"Invalid lane data example: {lane_xy[:5]}")
                continue
            if dist > 5.0:
                continue

            proj = line.project(ego_point)
            path_coords = np.array(
                [line.interpolate(proj + s).coords[0] for s in np.arange(0, length, step)]
            )

            dx, dy = path_coords[1] - path_coords[0]
            lane_heading = np.arctan2(dy, dx)
            diff_heading = np.abs(
                np.arctan2(np.sin(lane_heading - ego_heading), np.cos(lane_heading - ego_heading))
            )
            if diff_heading > heading_tol:
                continue

            headings = np.arctan2(
                np.gradient(path_coords[:, 1]), np.gradient(path_coords[:, 0])
            )
            path_with_heading = np.concatenate([path_coords, headings[:, None]], axis=1)
            candidate_paths.append(path_with_heading)

        # Step 3. 去重
        merged_paths = []
        for i, p_i in enumerate(candidate_paths):
            keep = True
            for p_j in merged_paths:
                min_len = min(len(p_i), len(p_j))
                diff = np.linalg.norm(p_i[:min_len, :2] - p_j[:min_len, :2], axis=-1)
                if np.max(diff) < 0.5:
                    keep = False
                    break
            if keep:
                merged_paths.append(p_i)

        # Step 4. 构造标准化特征结构
        n_points = int(length / step)
        n_ref = len(merged_paths)

        position = np.zeros((n_ref, n_points, 2), dtype=np.float64)
        vector = np.zeros((n_ref, n_points, 2), dtype=np.float64)
        orientation = np.zeros((n_ref, n_points), dtype=np.float64)
        valid_mask = np.zeros((n_ref, n_points), dtype=bool)
        future_projection = np.zeros((n_ref, 8, 2), dtype=np.float64)

        # ego_future 可选（如果你未来需要接入 _get_reference_line_feature 的 ego_future 逻辑）
        ego_future = ego_future_gt[:, 2]  # 这里默认空
        future_samples = []

        for i, line in enumerate(merged_paths):
            subsample = line[::int(1 / step)][: n_points + 1]  # 保持步长一致
            n_valid = len(subsample)
            if n_valid < 2:
                continue

            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)
            orientation[i, : n_valid - 1] = subsample[:-1, 2]
            valid_mask[i, : n_valid - 1] = True

            # future_projection（如果 ego_future 存在，可扩展）
            if len(ego_future) > 0:
                line_geom = LineString(subsample[:, :2])
                for j, future_sample in enumerate(future_samples):
                    future_projection[i, j, 0] = line_geom.project(future_sample)
                    future_projection[i, j, 1] = line_geom.distance(future_sample)

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
        }



def imitation_loss(gmm, scores, ground_truth):
    B, N = gmm.shape[0], gmm.shape[1] #gmm.shape = [4, 11, 6, 80, 4]
    distance = torch.norm(gmm[:, :, :, :, :2] - ground_truth[:, :, None, :, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)

    mu = gmm[..., :2]
    best_mode_mu = mu[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_mu = best_mode_mu.squeeze(2)
    dx = ground_truth[..., 0] - best_mode_mu[..., 0]
    dy = ground_truth[..., 1] - best_mode_mu[..., 1]

    cov = gmm[..., 2:]
    best_mode_cov = cov[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_cov = best_mode_cov.squeeze(2)
    log_std_x = torch.clamp(best_mode_cov[..., 0], -2, 2)
    log_std_y = torch.clamp(best_mode_cov[..., 1], -2, 2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
    gmm_loss = torch.mean(gmm_loss)

    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, label_smoothing=0.2, reduction='none')
    score_loss = score_loss * torch.ne(ground_truth[:, :, 0, 0], 0)
    score_loss = torch.mean(score_loss)
    
    loss = gmm_loss + score_loss

    return loss, best_mode_mu, best_mode


def level_k_loss(outputs, ego_future, neighbors_future, neighbors_future_valid):
    loss: torch.tensor = 0
    levels = len(outputs.keys()) // 2 
    gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)

    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions']
        scores = outputs[f'level_{k}_scores']
        # print("trajectories:", trajectories.shape)
        # print("neighbors_future_valid:", neighbors_future_valid.shape)

        predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
        plan = trajectories[:, :1]
        trajectories = torch.cat([plan, predictions], dim=1)
        il_loss, future, best_mode = imitation_loss(trajectories, scores, gt_future)
        loss += il_loss 

    return loss, future


def planning_loss(plan, ego_future):
    # print("plan shape:", plan.shape)
    # print("ego_future shape:", ego_future.shape)
    loss = F.smooth_l1_loss(plan, ego_future[..., :2])
    loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1, :2])

    return loss


def motion_metrics(plan_trajectory, prediction_trajectories, ego_future, neighbors_future, neighbors_future_valid):
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    try:
        heading_error = torch.abs(torch.fmod(plan_trajectory[:, :, 2] - ego_future[:, :, 2] + np.pi, 2 * np.pi) - np.pi)
        plannerAHE = torch.mean(heading_error)
        plannerFHE = torch.mean(heading_error[:, -1])
    except:
        plannerAHE = plannerADE
        plannerFHE = plannerFDE
    
    
    
    if prediction_trajectories is not None:
        prediction_trajectories = prediction_trajectories * neighbors_future_valid
        prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)
        predictorADE = torch.mean(prediction_distance, dim=-1)
        predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
        predictorADE = torch.mean(predictorADE)
        predictorFDE = prediction_distance[:, :, -1]
        predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
        predictorFDE = torch.mean(predictorFDE)
        return plannerADE.item(), plannerFDE.item(), plannerAHE.item(), plannerFHE.item(), predictorADE.item(), predictorFDE.item()


    return plannerADE.item(), plannerFDE.item()
