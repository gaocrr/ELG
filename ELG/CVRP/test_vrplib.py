import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os

from CVRPModel import CVRPModel, CVRPModel_local
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible


def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()
    ins_feature = env.get_instance_feature()
    while not done:
        cur_dist, cur_theta = env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, cur_theta, ins_feature, eval_type=eval_type)
        # selected, one_step_prob = model(state)
        state, reward, done = env.step(selected)
        actions.append(selected)
        probs.append(one_step_prob)

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward


class VRPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']

        # cuda
        USE_CUDA = config['use_cuda']
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # load trained model
        self.model = CVRPModel(**model_params)
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vrplib_path = '../../VRPLib/Vrp-Set-X/X/' if config['vrplib_set'] == 'X' else "../../VRPLib/Vrp-Set-XXL/Vrp-Set-XXL/XXL/"
        self.repeat_times = 1
        self.aug_factor = config['params']['aug_factor']
        self.vrplib_results = None
        
    def test_on_vrplib(self):
        files = os.listdir(self.vrplib_path)
        vrplib_results = []
        total_time = 0.
        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = self.vrplib_path + '/' + name + '.vrp'
                solution_file = self.vrplib_path + '/' + name + '.sol'
                
                solution = vrplib.read_solution(solution_file)
                optimal = solution['cost']

                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance_file, solution=solution_file)
                total_time += time.time() - start_time

                new_instance_dict = {}
                new_instance_dict['instance'] = name
                new_instance_dict['optimal'] = optimal
                new_instance_dict['record'] = [result_dict]
                vrplib_results.append(new_instance_dict)

                print("Instance Name {}: gap {:.4f}".format(name, result_dict['gap']))
                if 'XXL' in self.vrplib_path:
                    print("cost: {}".format(result_dict['best_cost']))
        if 'XXL' in self.vrplib_path:
            avg_gap = []
            for result in vrplib_results:
                avg_gap.append(result['record'][-1]['gap'])
            
            print("{:.2f}%".format(100 * np.array(avg_gap).mean()))
            print("Average time: {:.2f}s".format(total_time / 4))
        else:
            avg_gap_small = []
            avg_gap_medium = []
            avg_gap_large = []
            total = []
            number = 0
            for result in vrplib_results:
                scale = int(result['record'][-1]['scale'])
                if scale <= 200:
                    avg_gap_small.append(result['record'][-1]['gap'])
                elif scale <= 500:
                    avg_gap_medium.append(result['record'][-1]['gap'])
                else:
                    avg_gap_large.append(result['record'][-1]['gap'])
                total.append(result['record'][-1]['gap'])
                number += 1
            
            print("Average gap on subset of <200: {:.2f}%".format(100 * np.array(avg_gap_small).mean()))
            print("Average gap on subset of 200-500: {:.2f}%".format(100 * np.array(avg_gap_medium).mean()))
            print("Average gap on subset of 500-1000: {:.2f}%".format(100 * np.array(avg_gap_large).mean()))
            print("Average gap total: {:.2f}%".format(100 *(np.array(total).mean())))
            print("Average time: {:.2f}s".format(total_time / number))
            vrplib_results.append({"<200": 100 * np.array(avg_gap_small).mean(), 
            "200-500": 100 * np.array(avg_gap_medium).mean(),
            "500-1000": 100 * np.array(avg_gap_large).mean(),
            "total": 100 *(np.array(total).mean())})
            with open('test_results/' + self.config['name'] + '_' + 'vrplib.json', 'w') as f:
                json.dump(vrplib_results, f)


    def test_on_one_ins(self, name, result_dict, instance, solution):
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, 1000)
        # multiple_width = problem_size

        # Initialize CVRP state
        env = CVRPEnv(multiple_width, self.device)
        env.load_vrplib_problem(instance, aug_factor=self.aug_factor)

        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)

        with torch.no_grad():
            _, _, rewards = rollout(self.model, env, 'greedy')

        # Return
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        # shape: (augmentation, batch, multi)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value

        best_cost = aug_cost

        if result_dict is not None:
            result_dict['best_cost'] = best_cost.cpu().numpy().tolist()[0]
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            # print(best_cost)


if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()