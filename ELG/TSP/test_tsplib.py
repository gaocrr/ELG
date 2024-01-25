import os
import yaml
import time
import pickle
import json
import torch
import numpy as np
from torch.optim import Adam as Optimizer

from generate_data import generate_tsp_data, TSPDataset
from TSPModel import TSPModel, MLP_Local_policy
from TSPEnv import TSPEnv
from utils import rollout, check_feasible

def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()
    while not done:
        cur_dist, cur_theta, scale = env.get_local_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, cur_theta, scale, eval_type='greedy')
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


class TSPLib_Tester:

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
        if config['training'] == 'joint':
            self.model = TSPModel(**model_params)
        elif config['training'] == 'only_local':
            self.model = MLP_Local_policy(**model_params)

        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tsplib_path = 'TSPLib'
        self.repeat_times = 1
        self.aug_factor = config['params']['aug_factor']
        self.tsplib_results = None
        
    def test_on_tsplib(self):
        files = os.listdir(self.tsplib_path)
        tsplib_results = []
        total_time = 0.
        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = self.tsplib_path + '/' + name + '.pkl'
                # load tsplib file
                print(instance_file)
                with open(instance_file, 'rb') as f:
                    instance = pickle.load(f)  
                    optimal = instance[1]

                result_dict = {}
                result_dict['run_idx'] = t
                start_time = time.time()
                self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance)
                total_time += time.time() - start_time

                # update the results of current instance and method
                exist = False
                for result_per_instance in tsplib_results:
                    if result_per_instance['instance'] == name:
                        exist = True
                        for record in result_per_instance['record']:
                            if record['method'] == result_dict['method'] and record['run_idx'] == result_dict['run_idx']:
                                assert 'not necessary experiments!'
                                
                        result_per_instance['record'].append(result_dict)

                if exist == False:
                    new_instance_dict = {}
                    new_instance_dict['instance'] = name
                    new_instance_dict['optimal'] = optimal
                    new_instance_dict['record'] = [result_dict]
                    tsplib_results.append(new_instance_dict)

                print("Instance Name {}: gap {:.4f}".format(name, result_dict['gap']))

        with open('test_results/' + self.config['name'] + '_' + 'tsplib.json', 'w') as f:
            json.dump(tsplib_results, f)
        
        total_cost = []
        opt = []
        number = 0
        small_cost = []
        small_opt = []
        medium_cost = []
        medium_opt = []
        large_cost = []
        large_opt = []
        for result in tsplib_results:
            scale = result['record'][-1]['scale']
            if scale <= 1002:
                opt.append(result['optimal'])
                total_cost.append(result['record'][-1]['best_cost'])
            if scale <= 200:
                small_cost.append(result['record'][-1]['best_cost'])
                small_opt.append(result['optimal'])
            elif scale <= 500:
                medium_cost.append(result['record'][-1]['best_cost'])
                medium_opt.append(result['optimal'])
            elif scale <= 1002:
                large_cost.append(result['record'][-1]['best_cost'])
                large_opt.append(result['optimal'])
            number += 1
        
        print("Total average cost {:.2f}".format(np.array(total_cost).mean()))
        print("Total average gap {:.2f}%".format(100 * ((np.array(total_cost).mean() - np.array(opt).mean()) / np.array(opt).mean())))
        print("Average time: {:.2f}s".format(total_time / number))


    def test_on_one_ins(self, name, result_dict, instance):
        unscaled_points = torch.tensor(instance[0], dtype=torch.float)[None, :, :]
        points = instance[0] / np.max(instance[0])
        test_batch = torch.tensor(points, dtype=torch.float)[None, :, :]
        optimal = instance[1]

        problem_size = test_batch.shape[1]
        pomo_size = problem_size
        batch_size = test_batch.shape[0]

        # initialize env
        env = TSPEnv(pomo_size, self.device)
        env.load_tsplib_problem(test_batch, unscaled_points, self.aug_factor)
        reset_state, reward, done = env.reset()

        self.model.eval()
        self.model.requires_grad_(False)
        if self.config['training'] == 'joint':
            self.model.pre_forward(reset_state)

        _, _, rewards = rollout(self.model, env, 'greedy')

        aug_reward = rewards.reshape(self.aug_factor, 1, pomo_size)
        # shape: (augmentation, batch, pomo)
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
    tester = TSPLib_Tester(config=config)
    tester.test_on_tsplib()