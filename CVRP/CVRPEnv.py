from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F

from utils import augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    dist: torch.Tensor = None
    # shape: (batch, problem+1, problem+1)


@dataclass
class Step_State:
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, multi)
    current_node: torch.Tensor = None
    # shape: (batch, multi)
    ninf_mask: torch.Tensor = None
    # shape: (batch, multi, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, multi)


class CVRPEnv:
    def __init__(self, multi_width, device):

        # Const @INIT
        ####################################
        self.device = device
        self.vrplib = False
        self.problem_size = None
        self.multi_width = multi_width

        self.depot_xy = None
        self.unscaled_depot_xy = None
        self.node_xy = None
        self.node_demand = None
        self.input_mask = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = None
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, multi)
        self.load = None
        # shape: (batch, multi)
        self.visited_ninf_flag = None
        # shape: (batch, multi, problem+1)
        self.ninf_mask = None
        # shape: (batch, multi, problem+1)
        self.finished = None
        # shape: (batch, multi)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_vrplib_problem(self, instance, aug_factor=1):
        self.vrplib = True
        self.batch_size = 1
        node_coord = torch.FloatTensor(instance['node_coord']).unsqueeze(0).to(self.device)
        demand = torch.FloatTensor(instance['demand']).unsqueeze(0).to(self.device)
        demand = demand / instance['capacity']
        self.unscaled_depot_node_xy = node_coord
        # shape: (batch, problem+1, 2)
        
        min_x = torch.min(node_coord[:, :, 0], 1)[0]
        min_y = torch.min(node_coord[:, :, 1], 1)[0]
        max_x = torch.max(node_coord[:, :, 0], 1)[0]
        max_y = torch.max(node_coord[:, :, 1], 1)[0]
        scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)
        
        # self.depot_node_xy = self.unscaled_depot_node_xy / 1000
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None]
                                        , scaled_depot_node_y[:, :, None]), dim=2)
        depot = self.depot_node_xy[:, instance['depot'], :]
        # shape: (batch, problem+1)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy)
                self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy)
                demand = demand.repeat(8, 1)
            else:
                raise NotImplementedError
        
        self.depot_node_demand = demand
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand[:, 1:]
        self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def load_random_problems(self, batch, aug_factor=1):
        self.batch_size = batch['loc'].shape[0]
        node_coord = batch['loc'].to(self.device)
        demand = batch['demand'].to(self.device)
        depot = batch['depot'].to(self.device)
        if len(depot.shape) == 2:
            depot = depot[:, None, :]
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot = augment_xy_data_by_8_fold(depot)
                node_coord = augment_xy_data_by_8_fold(node_coord)
                demand = demand.repeat(8, 1)
            else:
                raise NotImplementedError
            
        self.depot_node_xy = torch.cat((depot, node_coord), dim=1)
        self.depot_node_demand = torch.cat((torch.zeros(self.batch_size, 1).to(self.device), demand), dim=1)    
            
        self.reset_state.depot_xy = depot
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demand
        self.problem_size = self.reset_state.node_xy.shape[1]
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, multi)
        self.selected_node_list = torch.zeros(size=(self.batch_size, self.multi_width, 0), dtype=torch.long, device=self.device)
        # shape: (batch, multi, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)
        self.load = torch.ones(size=(self.batch_size, self.multi_width), device=self.device)
        # shape: (batch, multi)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        if self.input_mask is not None:
            self.visited_ninf_flag = self.input_mask[:, None, :].expand(self.batch_size, self.multi_width, self.problem_size+1).clone()
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.multi_width, self.problem_size+1), device=self.device)
        # shape: (batch, multi, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.multi_width), dtype=torch.bool, device=self.device)
        # shape: (batch, multi)

        reward = None
        done = False
        return self.reset_state, reward, done

    def reset_width(self, new_width):
        self.multi_width = new_width

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, multi)
        # Dynamic-1
        ####################################
        
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, multi)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, multi, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        # shape: (batch, multi, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, multi, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, multi)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag.scatter_(2, self.selected_node_list, float('-inf'))
        # shape: (batch, multi, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 1e-6
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, multi, problem+1)
        # print(self.load)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, multi, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, multi)
        self.finished = self.finished + newly_finished
        # shape: (batch, multi)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        # returning values
        done = self.finished.all()
        if done:
            if self.vrplib == True:
                reward = self.compute_unscaled_reward()
            else:
                reward = self._get_reward()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_reward(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances

    def compute_unscaled_reward(self, solutions=None, rounding=True):
        if solutions is None:
            solutions = self.selected_node_list
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, multi, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.multi_width, -1, -1)
        # shape: (batch, multi, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, multi, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if rounding == True:
            segment_lengths = torch.round(segment_lengths)
        # shape: (batch, multi, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, multi)
        return -travel_distances

    
    def get_cur_feature(self):
        if self.current_node is None:
            return None, None, None, None
        
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.multi_width, 1, self.problem_size + 1)

        # Compute the relative distance
        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, self.problem_size + 1), 
                                        current_node, dim=2).squeeze(2)
        # shape: (batch, multi, problem)
        # print(cur_dist[0])
        expanded_xy = self.depot_node_xy[:, None, :, :].expand(self.batch_size, self.multi_width, self.problem_size + 1, 2)
        relative_xy = expanded_xy - torch.take_along_dim(expanded_xy, self.current_node[:, :, None, None].expand(
            self.batch_size, self.multi_width, 1, 2), dim=2)
        # shape: (batch, problem, 2)

        relative_x = relative_xy[:, :, :, 0]
        relative_y = relative_xy[:, :, :, 1]

        # Compute the relative coordinates
        cur_theta = torch.atan2(relative_y, relative_x)
        # shape: (batch, multi, problem)

        # Compute the normalized demand. inf generated by division will be masked. 
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.multi_width, -1)
        norm_demand = demand_list / self.load[:, :, None]

        return cur_dist, cur_theta, relative_xy, norm_demand

