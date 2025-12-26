import scipy
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point, LineString
from shapely.geometry.base import CAP_STYLE
import logging
try:
    from common_utils import *
    from bezier_path import calc_4points_bezier_path
    from cubic_spline_planner import calc_spline_course
except:
    from .common_utils import *
    from .bezier_path import calc_4points_bezier_path
    from .cubic_spline_planner import calc_spline_course
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.metrics.utils.expert_comparisons import principal_value


class LatticePlanner: 
    def __init__(self, route_ids, max_len=120, return_all_refpath=False):
        self.target_depth = max_len
        self.candidate_lane_edge_ids = route_ids
        self.max_path_len = max_len
        self.return_all_refpath = return_all_refpath

    def get_candidate_paths(self, edges):
        '''Get candidate paths using depth first search'''
        # get all paths
        paths = []
        for edge in edges:
            paths.extend(self.depth_first_search(edge)) # [[path1(lane1, lane2, ...)], [path2], ...]


        candidate_paths = {}

        # extract path polyline
        for i, path in enumerate(paths):
            path_polyline = [] # all point of a path; [points*N]
            for edge in path:
                path_polyline.extend(edge.baseline_path.discrete_path)

            path_polyline = self.check_path(np.array(path_to_linestring(path_polyline).coords))
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], path_polyline)
            path_polyline = path_polyline[dist_to_ego.argmin():]
            if len(path_polyline) < 3:
                continue

            path_len = len(path_polyline) * 0.25
            polyline_heading = self.calculate_path_heading(path_polyline)
            path_polyline = np.stack([path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1)
            candidate_paths[i] = (path_len, dist_to_ego.min(), path, path_polyline)

        if len(candidate_paths) == 0:
            return None

        # trim paths by length
        self.path_len = max([v[0] for v in candidate_paths.values()])
        acceptable_path_len = MAX_LEN * 0.2 if self.path_len > MAX_LEN * 0.2 else self.path_len
        candidate_paths = {k: v for k, v in candidate_paths.items() if v[0] >= acceptable_path_len}

        # sort paths by distance to ego
        candidate_paths = sorted(candidate_paths.items(), key=lambda x: x[1][1])

        return candidate_paths

    def get_candidate_edges(self, starting_block, ego_state):
        '''Get candidate edges from the starting block'''
        edges = []
        edges_distance = []
        self.ego_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        self.num_edges = len(starting_block.interior_edges)

        for edge in starting_block.interior_edges:
            edges_distance.append(edge.polygon.distance(Point(self.ego_point)))
            if edge.polygon.distance(Point(self.ego_point)) < 4:
                edges.append(edge)
        
        # if no edge is close to ego, use the closest edge
        if len(edges) == 0:
            edges.append(starting_block.interior_edges[np.argmin(edges_distance)])

        return edges

    def plan(self, ego_state, starting_block, observation, traffic_light_data):
        # Get candidate paths
        edges = self.get_candidate_edges(starting_block, ego_state)
        candidate_paths = self.get_candidate_paths(edges)

        if candidate_paths is None:
            return None

        # Get obstacles
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)

        obstacles = []
        vehicles = []
        for obj in objects:
            if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30:
                continue

            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.01:
                    obstacles.append(obj.box)
                else:
                    vehicles.append(obj.box)
            else:
                obstacles.append(obj.box)


        # Generate paths using state lattice
        paths = self.generate_paths(ego_state, candidate_paths)

        # disable lane change in large intersections
        if len(traffic_light_data) > 0:
            self._just_stay_current = True
        elif self.num_edges >= 4 and ego_state.dynamic_car_state.rear_axle_velocity_2d.x <= 3:
            self._just_stay_current = True
        else:
            self._just_stay_current = False

        # Calculate costs and choose the optimal path
        optimal_path = None
        min_cost = np.inf
        path_cost = []
        for path in paths:
            try:
                cost = self.calculate_cost(path, obstacles, vehicles)
            except:
                logging.error('Error in calculating cost !!!!!!!!!!!! skip this path')
                continue
            path_cost.append((path[0], cost))
            if cost < min_cost:
                min_cost = cost
                optimal_path = path[0]

        if self.return_all_refpath:
            ans = []
            for p, c in path_cost:
                p = self.post_process(p, ego_state)
                ans.append((p, c))
            ans = sorted(ans, key=lambda x: x[1])
            return ans

        # Post-process the path
        ref_path = self.post_process(optimal_path, ego_state)

        return ref_path

    def generate_paths(self, ego_state, paths):
        '''Generate paths from state lattice'''
        new_paths = []
        ego_state = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
        
        for _, (path_len, dist, path, path_polyline) in paths:
            if len(path_polyline) > 81:
                sampled_index = np.array([5, 10, 15, 20]) * 4
            elif len(path_polyline) > 61:
                sampled_index = np.array([5, 10, 15]) * 4
            elif len(path_polyline) > 41:
                sampled_index = np.array([5, 10]) * 4
            elif len(path_polyline) > 21:
                sampled_index = [20]
            else:
                print(len(path_polyline))
                print("Error: path too short in lattice planner!")
                sampled_index = [1]
     
            target_states = path_polyline[sampled_index].tolist()
            for j, state in enumerate(target_states):
                first_stage_path = calc_4points_bezier_path(ego_state[0], ego_state[1], ego_state[2], 
                                                            state[0], state[1], state[2], 3, sampled_index[j])[0]
                second_stage_path = path_polyline[sampled_index[j]+1:, :2]
                path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
                new_paths.append((path_polyline, dist, path, path_len))     
        # 检查生成的路径是否为空
        if not new_paths:
            logging.error("Error: No paths generated in lattice planner!")
        # 检查生成的路径长度
        for p in new_paths:
            if len(p[0]) < 10:
                logging.error("Error: Generated path is too short in lattice planner!")
            

        return new_paths

    def calculate_cost(self, path, obstacles, vehicles):
        try:
            # 1. 检查路径有效性
            if path is None or len(path) < 4:
                return np.inf
                
            path_points = path[0]  # 路径点数组
            if len(path_points) < 10:  # 确保路径足够长
                return np.inf
            
            # 2. 路径曲率计算 - 添加边界检查
            if len(path_points) >= 100:
                curvature_sample = path_points[0:100]
            else:
                curvature_sample = path_points
                
            curvature = self.calculate_path_curvature(curvature_sample)
            if len(curvature) > 0:
                curvature_cost = np.max(curvature)
            else:
                curvature_cost = 0

            # 3. 车道变更成本 - 添加默认值
            lane_change = path[1] if len(path) > 1 else 0
            if self._just_stay_current:
                lane_change_cost = 5 * lane_change
            else:
                lane_change_cost = lane_change

            # 4. 目标车道检查 - 添加参数检查
            if len(path) >= 4:
                target_lane_bonus = self.check_target_lane(path_points[0:50:10], path[3], vehicles)
            else:
                target_lane_bonus = 0

            # 5. 障碍物检查 - 确保路径点有效
            if len(path_points) >= 100:
                obstacle_check_points = path_points[0:100:10]
            else:
                obstacle_check_points = path_points[::max(1, len(path_points)//10)]  # 均匀采样
                
            obstacles_cost = self.check_obstacles(obstacle_check_points, obstacles)

            # 6. 边界检查 - 添加路径有效性检查
            if len(path) > 2:
                out_boundary_cost = self.check_out_boundary(path_points[:min(100, len(path_points))], path[2])
            else:
                out_boundary_cost = 0
            
            # 最终成本计算
            cost = (10 * obstacles_cost + 
                    2 * out_boundary_cost + 
                    1 * lane_change_cost + 
                    0.1 * curvature_cost - 
                    5 * target_lane_bonus)

            return cost
            
        except Exception as e:
            logging.error(f'Detailed error in calculate_cost: {e}')
            logging.error(f'Path structure: {[type(x) for x in path] if path else "No path"}')
            raise  # 重新抛出异常以便外层捕获

    def post_process(self, path, ego_state):
        '''Post process the selected path'''
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        ref_path = self.transform_to_ego_frame(spline_path, ego_state)
        ref_path = ref_path[:self.max_path_len*10]

        return ref_path

    def depth_first_search(self, starting_edge, depth=0):
        if depth >= self.target_depth:
            return [[starting_edge]]
        else:
            traversed_edges = []
            child_edges = [edge for edge in starting_edge.outgoing_edges if edge.id in self.candidate_lane_edge_ids]

            if child_edges:
                for child in child_edges:
                    edge_len = len(child.baseline_path.discrete_path) * 0.25
                    traversed_edges.extend(self.depth_first_search(child, depth+edge_len))

            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []

            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)
                    
            return edges_to_return
        
    def check_target_lane(self, path, path_len, vehicles):
        if np.abs(path_len - self.path_len) > 5:
            return 0
        
        expanded_path = LineString(path)
        min_distance_to_vehicles = np.inf

        for v in vehicles:
            d = expanded_path.distance(v.geometry)
            if d < min_distance_to_vehicles:
                min_distance_to_vehicles = d

        if min_distance_to_vehicles < 5:
            return 0

        return 1

    @staticmethod
    def check_path(path):
        refine_path = [path[0]]
        
        for i in range(1, path.shape[0]):
            if np.linalg.norm(path[i] - path[i-1]) < 0.1:
                continue
            else:
                refine_path.append(path[i])
        
        line = np.array(refine_path)

        return line

    @staticmethod
    def calculate_path_heading(path):
        heading = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
        heading = np.append(heading, heading[-1])

        return heading
    
    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0
    
    @staticmethod
    def check_out_boundary(polyline, path):
        line = LineString(polyline).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

        for edge in path:
            left, right = edge.adjacent_edges
            if (left is None and line.intersects(edge.left_boundary.linestring)) or \
                (right is None and line.intersects(edge.right_boundary.linestring)):
                return 1

        return 0

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

        return curvature

    @staticmethod
    def transform_to_ego_frame(path, ego_state):
        ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
        path_x, path_y, path_h, path_k = path[:, 0], path[:, 1], path[:, 2], path[:, 3]
        ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
        ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
        ego_path_h = principal_value(path_h - ego_h)
        ego_path = np.stack([ego_path_x, ego_path_y, ego_path_h, path_k], axis=-1)

        return ego_path
