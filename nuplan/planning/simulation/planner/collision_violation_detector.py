from typing import List, Set
import numpy as np
import numpy.typing as npt
import math
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver,
    ILQRSolverParameters,
    ILQRWarmStartParameters,
)
from nuplan.planning.simulation.controller.tracker.ilqr_optimizer import ILQROptimizer
import copy
# from nuplan.common.actor_state.agent import Agent

# helper function to find the index of a dict in a list of dicts
def build_index(list_of_dicts, key):
    return {d[key]: d for d in list_of_dicts}

class CollisionViolationDetector:
    """Detects time-to-collision violations between ego's predicted trajectory and obstacles."""

    def __init__(
        self,
        prediction_time_horizon: float = 3.0,
        timestep: float = 0.1,
        velocity_threshold: float = -0.1, # only for debugging, change back to 0.1
        max_ttc: float = 0.95,
        collision_distance_threshold: float = 2.0,
        ego_width: float = 2.297,
        ego_length: float = 4.049,

    ):
        """
        :param prediction_time_horizon: Time horizon for trajectory prediction [s]
        :param timestep: Time step between trajectory points [s]
        :param velocity_threshold: Speed threshold below which TTC is considered infinite [m/s]
        :param max_ttc: Maximum TTC value to consider [s]
        :param collision_distance_threshold: Minimum distance to consider collision [m]
        """
        self._prediction_horizon = prediction_time_horizon
        self._timestep = timestep
        self._velocity_threshold = velocity_threshold
        self._max_ttc = max_ttc
        self._collision_distance_threshold = collision_distance_threshold 
        self._ego_width = ego_width
        self._ego_length = ego_length
        self._ego_speed = 0.0

    def detect(self, ego_current, ego_future, filtered_neighbor_agents, static_objects):
        """
        Calculate minimum TTC at each timestep of ego's predicted trajectory.
        
        :param scenario: Scenario object containing:
            - ego_history: List[dict] - Historical ego states
            - ego_prediction: List[StateSE2] - Predicted ego trajectory 
            - obstacle_predictions: List[List[dict]] - Obstacle predictions per timestep
        :return: List of minimum TTC values for each timestep (infinity means no collision risk)
            Each element represents the minimum TTC to any obstacle at that timestep
        """
        #(x, y, cos, sin, vx, vy, ax, ay, ...)
        self.ego_speed = np.hypot(ego_current[0][4], ego_current[0][5])

        # Check if ego speed is above threshold
        if not self._should_calculate_collision():
            return []
        
        # Parse ego predicted trajectory
        ego_trajectory = self._create_ego_trajectory(ego_future)
        
        # Parse dynamic obstacle observations
        dynamic_obstacle_states = self._create_obstacle_states(filtered_neighbor_agents)

        original_obs_indices = build_index(dynamic_obstacle_states, "obs_id")
        
        collision_results = []
        collided_obs_ids = set()
        total_timesteps = min(len(ego_trajectory), int(self._prediction_horizon / self._timestep))
        
        for time_idx in range(total_timesteps):
            ego_state = ego_trajectory[time_idx]
            cur_obs_states = copy.deepcopy(dynamic_obstacle_states)
            cur_obs_states = self._update_cur_obs_states_by_time(cur_obs_states, time_idx)
            for obs_state in cur_obs_states:
                if obs_state['obs_id'] in collided_obs_ids:
                    continue
                if self._in_collision(ego_state, obs_state, time_idx):
                    collided_obs_ids.add(obs_state['obs_id'])
                    origin_obs_state = original_obs_indices[obs_state['obs_id']]
                    collision_results.append({'time_idx': time_idx, 'collided_obs': origin_obs_state, 'ego_state': ego_state})
                    break
        return collision_results

    def _update_cur_obs_states_by_time(self, obstacle_states, time_step):
        """Get synchronized obstacle states for current timestep with fallback to last known state."""
        for obs in obstacle_states:
            obs['x'] += obs['dx'] * time_step
            obs['y'] += obs['dy'] * time_step
        return obstacle_states

    def _should_calculate_collision(self) -> bool:
        """Check if TTC calculation should be performed for current ego state."""
        return self._ego_speed >= self._velocity_threshold

    def _create_ego_trajectory(self, ego_future):
        """
        Parse ego trajectory.
        :param ego_future: Arrary of predicted ego states with:
            - x: float
            - y: float
            - cos: float 
            - sin: float
        :return: Ego trajectory as dict
        """
        trajectory = []
        x_seq = ego_future[:, 0]
        y_seq = ego_future[:, 1]
        dx_seq = x_seq[1:] - x_seq[:-1]
        dy_seq = y_seq[1:] - y_seq[:-1]
        s_seq = np.sqrt(dx_seq ** 2 + dy_seq ** 2)
        s_cum = np.concatenate(([0], np.cumsum(s_seq)))

            
        # Process prediction
        for i in range(ego_future.shape[0]):
            state = {
                'x': ego_future[i][0],
                'y': ego_future[i][1],
                'cos': ego_future[i][2],
                'sin': ego_future[i][3],
                's': s_cum[i],
                'time_step': i * self._timestep,
            }
            trajectory.append(state)
            
        return trajectory

    def _create_obstacle_states(self, focal_agents) :
        """
        Convert raw obstacle history to TrackedObject states.
        
        :param obstacle_history: List of obstacle states per timestep
        :return: List of TrackedObject lists per timestep
        """
        obstacle_states = []

        for i in range(focal_agents.shape[0]): # num of agents
            # (x, y, cos, sin, vx, vy, length, width, type_indicator)
            state = {
                'obs_id':  str(i),
                'x': focal_agents[i][0],
                'y':focal_agents[i][1],
                'cos': focal_agents[i][2],
                'sin': focal_agents[i][3],
                'vx': focal_agents[i][4],
                'vy': focal_agents[i][5],
                'length': focal_agents[i][6],
                'width': focal_agents[i][7],
                'dx': focal_agents[i][4] * self._timestep,
                'dy': focal_agents[i][5] * self._timestep,
            }
            obstacle_states.append(state)
            
        return obstacle_states

    
    def _in_collision(self, ego_state, obs_state, time_index):
        """
        Calculate collision between ego and obstacle.
        
        :param ego_state: Ego's state at current timestep
        :param obs_state: obstacle state with const v extrapolation
        :return: if obs and ego are in collision
        """
        ego_heading = np.arctan2(ego_state['sin'], ego_state['cos'])
        if ego_heading < 0:
            ego_heading += 2 * math.pi
        
        ego_elongated_box_center_pose: npt.NDArray[np.float64] = np.array(
        [
            ego_state['x'],
            ego_state['y'],
            ego_heading,
        ],
        dtype=np.float64,
    )
        ego_elongated_box = OrientedBox(
            StateSE2(*ego_elongated_box_center_pose),
            self._ego_length,# + ego_state['s'],
            self._ego_width,
            1.0,
        )

        obs_heading = np.arctan2(obs_state['sin'], obs_state['cos'])
        if obs_heading < 0:
            obs_heading += 2 * math.pi
        obs_elongated_box_center_pose: npt.NDArray[np.float64] = np.array(
            [
                obs_state['x'],
                obs_state['y'],
                obs_heading,
            ],
            dtype=np.float64,
        )
        ratio = time_index / self._prediction_horizon
        obs_elongated_box = OrientedBox(
            StateSE2(*obs_elongated_box_center_pose),
            obs_state['length'],# + np.hypot(obs_state['dx'] * ratio, obs_state['dy'] * ratio),
            obs_state['width'],
            1.0,
        )
        return in_collision(ego_elongated_box, obs_elongated_box)
    
        

def main():
    import pickle
    record_path = './nuplan/planning/metrics/evaluation_metrics/PostDecision/2025_05_12_16-43-59_2021.09.16.17.40.09_veh-45_02539_02745_1075ba60960c564e.pkl'
    # record_path = './nuplan/planning/metrics/evaluation_metrics/PostDecision/collision.pkl'
    with open(record_path, 'rb') as f:
        record = pickle.load(f)
    inputs = record['inputs']
    predictions = record['predictions']
    # ego_param = record['ego_param']

    for i in range(len(inputs)):
        input = inputs[i]
        prediction = predictions[i]
        ego_past = input['ego_agent_past']
        neighbor_agents_past = input['neighbor_agents_past']
        static_objects = input['static_objects']

        ego_past = ego_past.squeeze(0).cpu().numpy()
        ego_current = ego_past[0:1,:] # [1, 14]
        neighbor_agents_past = neighbor_agents_past.squeeze(0).cpu().numpy()
        neighbor_agents_current = neighbor_agents_past[:,-1,:] #[]
        filtered_neighbor_agents = []
        for idx in range(neighbor_agents_current.shape[0]):
            if np.abs(np.sum(neighbor_agents_current[idx,:])) > 0.001:
                filtered_neighbor_agents.append(neighbor_agents_current[idx,:])
        filtered_neighbor_agents = np.array(filtered_neighbor_agents) # (num_neighbors, 9)

        static_objects = static_objects.squeeze(0).cpu().numpy() # (num_objects, dim) [5, 10], (x, y)

        ego_future = prediction.squeeze(0).squeeze(0).cpu().numpy() # [1, 80, 4] (x, y, cos, sin)
        collision_detector = CollisionViolationDetector()
        result = collision_detector.detect(ego_current, ego_future, filtered_neighbor_agents, static_objects)
        print("{} frame result, {}".format(i, result))

def main_test():
    import pickle
    record_path = './nuplan/planning/metrics/evaluation_metrics/PostDecision/2025_05_12_16-43-59_2021.09.16.17.40.09_veh-45_02539_02745_1075ba60960c564e.pkl'
    # record_path = './nuplan/planning/metrics/evaluation_metrics/PostDecision/collision.pkl'
    with open(record_path, 'rb') as f:
        record = pickle.load(f)
    inputs = record['inputs']
    predictions = record['predictions']
    # ego_param = record['ego_param']

    input = inputs[2]
    prediction = predictions[2]
    ego_past = input['ego_agent_past']
    neighbor_agents_past = input['neighbor_agents_past']
    static_objects = input['static_objects']

    ego_past = ego_past.squeeze(0).cpu().numpy()
    ego_current = ego_past[0:1,:] # [1, 14]
    neighbor_agents_past = neighbor_agents_past.squeeze(0).cpu().numpy()
    neighbor_agents_current = neighbor_agents_past[:,-1,:] #[]
    filtered_neighbor_agents = []
    for idx in range(neighbor_agents_current.shape[0]):
        if np.abs(np.sum(neighbor_agents_current[idx,:])) > 0.001:
            filtered_neighbor_agents.append(neighbor_agents_current[idx,:])
    filtered_neighbor_agents = np.array(filtered_neighbor_agents) # (num_neighbors, 9)

    static_objects = static_objects.squeeze(0).cpu().numpy() # (num_objects, dim) [5, 10], (x, y)

    ego_future = prediction.squeeze(0).squeeze(0).cpu().numpy() # [1, 80, 4] (x, y, cos, sin)
    collision_detector = CollisionViolationDetector()
    result = collision_detector.detect(ego_current, ego_future, filtered_neighbor_agents, static_objects)
    print("{} frame result, {}".format(2, result))


    min_safe_distance = 100.0
    obs_positions = []
    for i in range(len(result)):
        collision_result = result[i]
        collided_obs = collision_result['collided_obs']
        obs_positions.append((collided_obs['x'], collided_obs['y']))
        length = collided_obs['length']
        width = collided_obs['width']
        safe_dist = math.sqrt(length**2 + width**2) / 2
        if safe_dist < min_safe_distance:
            min_safe_distance = safe_dist
    solver_params = ILQRSolverParameters(
            obstacle_positions=obs_positions,
            obstacle_safe_distance=min_safe_distance,
            obstacle_cost_weight=0.1,
            discretization_time=0.2,
            state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
            input_cost_diagonal_entries=[1.0, 10.0],
            state_trust_region_entries=[1.0] * 5,
            input_trust_region_entries=[1.0] * 2,
            max_ilqr_iterations=100,
            convergence_threshold=1e-6,
            max_solve_time=0.05,
            max_acceleration=3.0,
            max_steering_angle=np.pi / 3.0,
            max_steering_angle_rate=0.5,
            min_velocity_linearization=0.01,
        )
    warm_start_params = ILQRWarmStartParameters(
            k_velocity_error_feedback=0.5,
            k_steering_angle_error_feedback=0.05,
            lookahead_distance_lateral_error=15.0,
            k_lateral_error=0.1,
            jerk_penalty_warm_start_fit=1e-4,
            curvature_rate_penalty_warm_start_fit=1e-2,
        )
    ilqr_solver=ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)
    ilqr_optimizer = ILQROptimizer(n_horizon=80, ilqr_solver=ilqr_solver)


    ## TO tianyi: inital state is the current state of the ego vehicle
    #  current_state: DoubleMatrix = np.array(
    #         [
    #             initial_state.rear_axle.x,
    #             initial_state.rear_axle.y,
    #             initial_state.rear_axle.heading,
    #             initial_state.dynamic_car_state.rear_axle_velocity_2d.x,
    #             initial_state.tire_steering_angle,
    #         ]
    #     )
    optimal_trajectory = ilqr_optimizer.optimize_trajectory(initial_state=ego_current, reference_trajectory=ego_future)

if __name__ == "__main__":
    # main()
    main_test()
