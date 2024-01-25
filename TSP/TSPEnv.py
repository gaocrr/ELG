from dataclasses import dataclass
import torch

from utils import augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, multi_width, device):

        # Const @INIT
        ####################################
        self.problem_size = None
        self.pomo_size = multi_width
        self.device = device
        self.tsplib = False

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)
        self.dist = None
        # shape: (batch, problem, problem)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_random_problems(self, problems, aug_factor=1):
        self.batch_size = problems.size(0)
        self.problems = problems.to(self.device)
        self.problem_size = problems.size(1)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError
        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1).to(self.device) # (batch, problem, problem)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)  # (batch_size, pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)
    
    def load_tsplib_problem(self, problems, unscaled_problems, aug_factor=1):
        self.tsplib = True
        self.batch_size = problems.size(0)
        self.problem_size = problems.size(1)
        self.problems = problems
        self.unscaled_problems = unscaled_problems
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1) # (batch, problem, problem)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)  # (batch_size, pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            if self.tsplib == True:
                reward = self.compute_unscaled_distance()
            else:
                reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def get_local_feature(self):
        if self.current_node is None:
            return None, None, None
        
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size)

        cur_dist = torch.take_along_dim(self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, self.problem_size), 
                                        current_node, dim=2).squeeze(2)
        # shape: (batch, multi, problem)

        expanded_xy = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        relative_xy = expanded_xy - torch.take_along_dim(expanded_xy, self.current_node[:, :, None, None].expand(
            self.batch_size, self.pomo_size, 1, 2), dim=2)
        # shape: (batch, problem, multi, 2)

        relative_x = relative_xy[:, :, :, 0]
        relative_y = relative_xy[:, :, :, 1]

        cur_theta = torch.atan2(relative_y, relative_x)
        # shape: (batch, multi, problem)

        return cur_dist, cur_theta, relative_xy

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def compute_unscaled_distance(self, solutions=None):
        if solutions is None:
            solutions = self.selected_node_list
        multi_width = solutions.shape[1]
        # Gather instance in order of tour
        d = self.unscaled_problems[:, None, :, :].expand(self.batch_size, multi_width, self.problem_size, 2)\
            .gather(2, solutions[:, :, :, None].expand(self.batch_size, multi_width, self.problem_size, 2))
        # shape: (batch, multi, problem, 2)

        rolled_seq = d.roll(dims=2, shifts=-1)
        return -torch.round(((d-rolled_seq)**2).sum(3).sqrt()).sum(2)
