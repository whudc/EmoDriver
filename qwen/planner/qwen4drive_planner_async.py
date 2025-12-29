"""Asynchronous Qwen4DrivePlanner - LLM runs in separate process"""
import logging
import time
from typing import Dict, List
import numpy as np
import torch
import math

from shapely import Point

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import get_current_roadblock_candidates, \
    remove_route_loops

from gameformer.planner import Planner as BaseGFPlanner
from gameformer.planner_utils import *
from gameformer.obs_adapter import *
from gameformer.state_lattice_planner import LatticePlanner

from qwen.planner.async_llm_manager import AsyncLLMManager

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader

from qwen.planner.qwen4drive_planner import (
    get_navigation_by_gt,
    route_roadblock_correction
)


class AsyncOutput:
    def __init__(self, predictions=None, plan=None, llm_plan=None):
        self.predictions = predictions
        self.plan = plan
        self.llm_plan = llm_plan


class AsyncQwen4DrivePlanner(BaseGFPlanner):
    requires_scenario = True
    # Global LLM Manager
    _global_async_llm_manager = None
    _global_manager_lock = None

    def __init__(self,
                 scenario: AbstractScenario,
                 sub_planner: AbstractPlanner = None,
                 enable_pdm_scorer_in_multirefpath=False,
                 disable_refpath=False,
                 ins_mode=None,
                 llm_plan=False,
                 ins_wo_stop=False,
                 lora_r=8,
                 finetune_model_path=None,
                 model_name_or_path=None,
                 near_multiple_vehicles=False,
                 short_ins=-1,
                 llm_inf_step=1,
                 model_cfg=None,
                 model_urban: TorchModuleWrapper = None,
                 async_mode=True,
                 llm_update_interval=1.5,
                 llm_device='cuda:0',
                 gameformer_device='cuda:0'):
        super().__init__(disable_refpath=disable_refpath)

        if isinstance(model_cfg, list):
            model_cfg = {k: v for d in model_cfg for k, v in d.items()}

        self.ins_mode = ins_mode
        self.llm_plan = llm_plan
        self.ins_wo_stop = ins_wo_stop
        self.lora_r = lora_r
        self.finetune_model_path = finetune_model_path
        self.model_name_or_path = model_name_or_path
        self.near_multiple_vehicles = near_multiple_vehicles
        self.short_ins = short_ins

        logging.error(f'Ins mode: {ins_mode}')
        if ins_mode in ['gt', 'plain_ref']:
            ins_mode = None

        model_cfg['ins_mode'] = ins_mode
        model_cfg['ins_wo_stop'] = ins_wo_stop
        model_cfg['lora_r'] = lora_r
        model_cfg['finetune_model_path'] = finetune_model_path
        model_cfg['llm_inf_step'] = llm_inf_step
        model_cfg['near_multiple_vehicles'] = near_multiple_vehicles
        model_cfg['model_name_or_path'] = model_name_or_path
        self._model_cfg = model_cfg
        self.scenario = scenario
        self.sub_planner = sub_planner
        self.enable_pdm_scorer_in_multirefpath = enable_pdm_scorer_in_multirefpath
        self.async_mode = async_mode
        self.llm_update_interval = llm_update_interval
        self.llm_device = llm_device
        self.gameformer_device = gameformer_device
        self.async_llm_manager = None
        self.llm_wait_times = []
        self.gameformer_times = []
        self.total_iterations = 0

        if enable_pdm_scorer_in_multirefpath:
            logging.info('PDM scorer enabled in multi-refpath mode')
            assert sub_planner

        logging.info(f"AsyncQwen4DrivePlanner init: async={async_mode}, "
                    f"llm_interval={llm_update_interval}s, llm_dev={llm_device}, gf_dev={gameformer_device}")

    def name(self) -> str:
        return self.__class__.__name__

    def __getstate__(self):
        """Don't pickle certain objects"""
        state = self.__dict__.copy()
        state.pop('_model', None)
        state.pop('_path_planner', None)
        state.pop('_trajectory_planner', None)
        state.pop('sub_planner', None)
        state.pop('model_urban', None)
        state.pop('_model_loader', None)
        state.pop('async_llm_manager', None)
        return state

    def initialize(self, initialization: PlannerInitialization):
        super().initialize(initialization)
        if self.sub_planner:
            self.sub_planner.initialize(initialization)
        if self.enable_pdm_scorer_in_multirefpath:
            self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length,
                                               return_all_refpath=True)

    def _initialize_model(self):
        if self.async_mode:
            if AsyncQwen4DrivePlanner._global_async_llm_manager is not None:
                logging.info("=" * 60)
                logging.info("Reusing existing LLM Manager (saving ~60s)")
                logging.info("=" * 60)
                self.async_llm_manager = AsyncQwen4DrivePlanner._global_async_llm_manager
                logging.info("LLM Manager reused")
                return

            logging.info("=" * 60)
            logging.info("Initializing new LLM Manager (~60s)...")
            logging.info("=" * 60)

            new_manager = AsyncLLMManager(
                model_config=self._model_cfg,
                update_interval=self.llm_update_interval,
                buffer_size_mb=50,
                llm_device=self.llm_device
            )
            new_manager.start()

            AsyncQwen4DrivePlanner._global_async_llm_manager = new_manager
            self.async_llm_manager = new_manager
            logging.info("Async LLM Manager started")

            logging.info("Sending dummy data for LLM warmup...")
            dummy_features = {
                'ego_agent_past': torch.randn(1, 21, 7).to(self.gameformer_device),
                'neighbor_agents_past': torch.randn(1, 20, 21, 11).to(self.gameformer_device),
                'map_lanes': torch.randn(1, 40, 50, 7).to(self.gameformer_device),
                'map_crosswalks': torch.randn(1, 5, 30, 3).to(self.gameformer_device),
                'route_lanes': torch.randn(1, 10, 50, 3).to(self.gameformer_device),
            }
            dummy_ref_path = np.random.randn(100, 6).astype(np.float32)
            self.async_llm_manager.update_scene(dummy_features, dummy_ref_path, 0)
            logging.info("Dummy data sent")

            logging.info("Waiting for LLM warmup...")
            warmup_start = time.time()
            while time.time() - warmup_start < 120:
                output = self.async_llm_manager.get_latest_output()
                if output is not None and output.get('initialized', False):
                    elapsed = time.time() - warmup_start
                    logging.info(f"LLM warmup complete ({elapsed:.1f}s)")
                    break
                time.sleep(2.0)
            else:
                logging.warning(f"LLM warmup timeout")
        else:
            # Sync mode fallback
            logging.info("Initializing Sync LLM Model...")
            from qwen.planner.qwen4drive import Qwen2DriveModel
            self._model = Qwen2DriveModel(self._model_cfg)

    def _get_prediction(self, features, ref_path, cur_iter):
        if self.async_mode and self.async_llm_manager is not None:

            start_time = time.perf_counter()

            iter_index = cur_iter.index if hasattr(cur_iter, 'index') else cur_iter
            self.async_llm_manager.update_scene(features, ref_path, iter_index)
            output_data = self.async_llm_manager.get_latest_output()

            wait_time = time.perf_counter() - start_time
            self.llm_wait_times.append(wait_time)

            if output_data is None or not output_data.get('initialized', False):
                logging.warning("LLM not ready, using dummy output")
                return None, None, None, None, None

            # Convert output
            predictions = output_data['predictions']
            if self.llm_plan:
                plan = torch.from_numpy(output_data['llm_plan']).to(self.gameformer_device) \
                    if output_data['llm_plan'] is not None else None
            else:
                plan = torch.from_numpy(output_data['plan']).to(self.gameformer_device) \
                    if output_data['plan'] is not None else None

            if self.total_iterations % 10 == 0:
                logging.info(f"[Async] LLM iter {output_data.get('iteration', -1)}, "
                           f"LLM time: {output_data.get('inference_time', 0):.3f}s, wait: {wait_time:.4f}s")

        else:
            start_time = time.perf_counter()
            output = self._model.inference(features, ref_path, cur_iter)
            predictions = output.predictions
            plan = output.llm_plan if self.llm_plan else output.plan
            logging.info(f"[Sync] LLM time: {time.perf_counter() - start_time:.3f}s")

        # Extract level-k predictions
        if predictions is not None:
            K = len(predictions) // 2 - 1
            final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
            final_scores = predictions[f'level_{K}_scores']
        else:
            final_predictions = None
            final_scores = None

        try:
            ego_current = features['ego_agent_past'][:, -1]
            neighbors_current = features['neighbor_agents_past'][:, :, -1]
        except:
            ego_current = None
            neighbors_current = None

        return plan, final_predictions, final_scores, ego_current, neighbors_current

    def get_ego_agent_future(self, ego_state):
        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=self.iteration, num_samples=80, time_horizon=8
        )
        trajectory_absolute_states = [state.rear_axle for state in trajectory_absolute_states]
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            ego_state.rear_axle, trajectory_absolute_states
        )
        return trajectory_relative_poses

    def _get_multi_refpath(self, ego_state, traffic_light_data, observation):
        starting_block = None
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf
        # Find closest roadblock
        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance
            if np.isclose(closest_distance, 0):
                break

        if closest_distance > 7:
            return None
        # Generate paths
        ref_paths = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        if ref_paths is None:
            return []

        ans = []
        for ref_path, cost in ref_paths:
            occupancy = np.zeros(shape=(ref_path.shape[0], 1))
            for data in traffic_light_data:
                id_ = str(data.lane_connector_id)
                if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                    lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                    conn_path = np.array([[p.x, p.y] for p in lane_conn.baseline_path.discrete_path])
                    red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                    occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

            target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
            target_speed = np.clip(target_speed, 3, 15)
            max_speed = annotate_speed(ref_path, target_speed)

            ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1)
            if len(ref_path) < MAX_LEN * 10:
                ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN * 10 - len(ref_path), axis=0), axis=0)

            ans.append((ref_path.astype(np.float32), cost))

        return ans

    def _plan(self, ego_state, history, traffic_light_data, observation, cur_iter):
        start_time = time.perf_counter()

        # Construct features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # Get reference path
        if self.enable_pdm_scorer_in_multirefpath:
            ref_path_set = self._get_multi_refpath(ego_state, traffic_light_data, observation)
            if ref_path_set is None or len(ref_path_set) == 0:
                logging.error('No reference path found')
        else:
            ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)
            if ref_path is None:
                logging.error('No reference path found')

        # Determine instruction path
        if self.ins_mode == 'gt':
            ins_path = self.get_ego_agent_future(ego_state)
        elif self.enable_pdm_scorer_in_multirefpath:
            ins_path = ref_path_set[0][0] if ref_path_set and len(ref_path_set) > 0 else None
        else:
            if self.short_ins != -1 and ref_path is not None:
                ego_future_poses = ref_path[:, :3]
                dis_norm = np.linalg.norm(np.diff(ego_future_poses[:, :-1], n=1, axis=0), axis=1)
                dis_cum = np.cumsum(dis_norm, axis=0)
                dis_cum_num = np.where(dis_cum < self.short_ins)[0][-1]
                ins_path = ref_path[:dis_cum_num, :]
            else:
                ins_path = ref_path

        # Get LLM prediction (async or sync)
        plan, predictions, pred_scores, ego_state_transformed, neighbors_state_transformed = \
            self._get_prediction(features, ins_path, cur_iter)

        if self.disable_refpath:
            print('ref path is disabled !!!!!!!!!!!!')
            ref_path = None

        # GameFormer trajectory planning
        gameformer_start = time.perf_counter()
        with torch.no_grad():
            if self.enable_pdm_scorer_in_multirefpath:
                # Multi-path: score each path with PDM
                max_score = -1
                max_traj = None
                corr_cost = -1
                for ref_path, cost in ref_path_set:
                    plan_r = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                          predictions, plan, pred_scores, ref_path, observation)
                    states = transform_predictions_to_states(plan_r, history.ego_states, self._future_horizon, 0.1)
                    trajectory = InterpolatedTrajectory(states)
                    _, scores = self.sub_planner.compute_planner_trajectory_just_4_get_score(self.current_input, trajectory)
                    curr_score = np.mean(scores)
                    if curr_score > max_score or (abs(curr_score - max_score) < 1e-5 and cost < corr_cost):
                        max_score = curr_score
                        max_traj = trajectory
                        corr_cost = cost

                if max_traj is None:
                    plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                        predictions, plan, pred_scores, None, observation)
                    states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, 0.1)
                    trajectory = InterpolatedTrajectory(states)
                else:
                    logging.info(f"Iter {self.iteration}: Max score={max_score:.3f}, cost={corr_cost:.3f}")
                    trajectory = max_traj
            else:
                # Single path
                plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed,
                                                    predictions, plan, pred_scores, ref_path, observation)
                states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, 0.1)
                trajectory = InterpolatedTrajectory(states)

        gameformer_time = time.perf_counter() - gameformer_start
        self.gameformer_times.append(gameformer_time)

        return trajectory

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point((ego_state.rear_axle.x, ego_state.rear_axle.y)))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance
            if np.isclose(closest_distance, 0):
                break

        if closest_distance > 7:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, self._map_api, self._route_roadblock_dict, scenario=self.scenario
            )
            self._initialize_route_plan(route_roadblock_ids)

    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        self.iteration = iteration
        self.total_iterations += 1

        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        self.current_input = current_input

        # Correct route on first iteration
        if iteration == 0:
            old_ids = [b.id for b in self._route_roadblocks]
            self._route_roadblock_correction(ego_state)
            new_ids = [b.id for b in self._route_roadblocks]
            logging.error(f'\n New route: {new_ids} \n Old route: {old_ids}')

        if self.sub_planner:
            pdm_trajectory = self.sub_planner.compute_planner_trajectory(current_input)

        # Main planning
        trajectory = self._plan(ego_state, history, traffic_light_data, observation, current_input.iteration)

        total_time = time.time() - s
        self._compute_trajectory_runtimes.append(total_time)

        # Log stats every 50 iterations
        if iteration % 50 == 0 and iteration > 0:
            avg_llm_wait = np.mean(self.llm_wait_times[-50:]) if self.llm_wait_times else 0
            avg_gameformer = np.mean(self.gameformer_times[-50:]) if self.gameformer_times else 0
            logging.info(
                f"[Stats] Iter {iteration}: Total={total_time:.3f}s, "
                f"Avg LLM wait={avg_llm_wait:.4f}s, Avg GameFormer={avg_gameformer:.3f}s"
            )

        logging.error(f'Iteration {iteration}: {total_time:.3f}s')

        return trajectory

    def __del__(self):
        if (hasattr(self, 'async_llm_manager') and
            self.async_llm_manager is not None and
            AsyncQwen4DrivePlanner._global_async_llm_manager is self.async_llm_manager):
            try:
                logging.info("Final cleanup - stopping AsyncLLMManager...")
                self.async_llm_manager.stop()
                AsyncQwen4DrivePlanner._global_async_llm_manager = None
            except Exception as e:
                logging.warning(f"Error cleaning up AsyncLLMManager: {e}")
