import torch
from torch.optim import Adam as Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import yaml
import wandb
import datetime
import os
import json
from tqdm import trange

from generate_data import generate_vrp_data, VRPDataset
from CVRPModel import CVRPModel, CVRPModel_local
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, Logger, seed_everything


def test_rollout(loader, env, model):
    avg_cost = 0.
    num_batch = 0.
    for batch in loader:
        env.load_random_problems(batch)
        reset_state, _, _ = env.reset()
        model.eval()
        # greedy rollout
        with torch.no_grad():
            model.pre_forward(reset_state)
            solutions, probs, rewards = rollout(model=model, env=env, eval_type='greedy')
        # check feasible
        check_feasible(solutions[0:1], reset_state.node_demand[0:1])
        batch_cost = -rewards.max(1)[0].mean()
        avg_cost += batch_cost
        num_batch += 1.
    avg_cost /= num_batch

    return avg_cost

def validate(model, multiple_width, device):
    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)

    # validation dataset
    val_100 = VRPDataset('data/vrp100_val.pkl', num_samples=1000)
    val_100_loader = DataLoader(val_100, batch_size=1000)
    val_200 = VRPDataset('data/vrp200_val.pkl', num_samples=1000)
    val_200_loader = DataLoader(val_200, batch_size=1000)
    val_500 = VRPDataset('data/vrp500_val.pkl', num_samples=100)
    val_500_loader = DataLoader(val_500, batch_size=10)

    # validate
    val_100_cost = test_rollout(val_100_loader, env, model)
    val_200_cost = test_rollout(val_200_loader, env, model)
    val_500_cost = test_rollout(val_500_loader, env, model)

    avg_cost_list = [val_100_cost.cpu().numpy().tolist(), 
                     val_200_cost.cpu().numpy().tolist(), 
                     val_500_cost.cpu().numpy().tolist()]
    return avg_cost_list
    

def train(model, training, start_steps, train_steps, inner_steps, train_batch_size, problem_size, size, multiple_width, 
keep_num, lr, device, alg, logger, fileLogger, dir_path, log_step):
    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)
    
    if training == 'joint':
        # Only train global policy in the first 180k steps
        for k, v in model.named_parameters():
            if 'local_policies' in k:
                v.requires_grad = False
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Optimizer(params, lr=lr, weight_decay=1e-6)
    else:
        optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=1e-6)

    # REINFORCE training
    for i in trange(train_steps - start_steps + 1):
        model.train()

        # Enable joint training
        if i == 0 and i > 180000 - start_steps and training == 'joint':
            for k, v in model.named_parameters():
                if 'local_policies' in k:
                    v.requires_grad = True
            optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=1e-6)

        if size == 'small':
            # fixed problem size of training samples
            true_problem_size = problem_size
            true_batch_size = train_batch_size

        elif size == 'varying':
            if i <= 200000 - start_steps:
                 # fixed problem size of training samples
                true_problem_size = 100
                true_batch_size = train_batch_size

            else:
                # varying problem size of training samples
                true_problem_size = np.random.randint(100, problem_size)
                true_batch_size = int(train_batch_size * ((100 / true_problem_size)**1.6))
                
        batch = generate_vrp_data(dataset_size=true_batch_size, problem_size=true_problem_size)
        env.load_random_problems(batch)
        reset_state, _, _ = env.reset()
        for j in range(inner_steps):
            model.pre_forward(reset_state)
            solutions, probs, rewards = rollout(model=model, env=env, eval_type='sample')
            # check feasible
            check_feasible(solutions[0:1], reset_state.node_demand[0:1])

            optimizer.zero_grad()
            # POMO
            if alg == 'pomo':
                bl_val = rewards.mean(dim=1)[:, None]
                log_prob = probs.log().sum(dim=1)
                advantage = rewards - bl_val
                J = - advantage * log_prob
                # J = J / (advantage ** 2).mean(dim=1)
                J = J / advantage.max(dim=1)[0][:, None]
                J = J.mean()

            # Risk-seeking REINFORCE
            if alg == 'risk_seeking':
                top_reward, idx = rewards.topk(keep_num, dim=-1, largest=True)
                bl_val = top_reward[:, -1][:, None]
                top_prob = torch.take_along_dim(probs, idx[:, None, :]
                                                .expand(true_batch_size, probs.shape[1], keep_num), dim=2)
                log_prob = top_prob.log().sum(dim=1)
                advantage = top_reward - bl_val
                J = - advantage * log_prob
                # J = J / (advantage ** 2).mean(dim=1)[:, None]
                J = J / advantage.max(dim=1)[0][:, None]
                J = J.mean()

            # print("training length: {:.4f}".format(-rewards.max(1)[0].mean()))
            J.backward()
            optimizer.step()

        # validation and log
        if i * inner_steps % log_step == 0:
            val_info = validate(model, multiple_width, device)
            cost = np.array(val_info)
            if i == 0:
                best_cost = cost
            else:
                best_cost = np.min(np.concatenate([cost[:, None], best_cost[:, None]], axis=1), axis=1)
            fileLogger.log(val_info)
            if logger is not None:
                logger.log({'val_100_cost': val_info[0],
                            'val_300_cost': val_info[1],
                            'val_500_cost': val_info[2]},
                        step=i * inner_steps)   

            checkpoint_dict = {
                    'step': i * inner_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
            torch.save(checkpoint_dict, dir_path + '/model_epoch_{}.pt'.format(int(i * inner_steps / log_step)))
        

if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # params
    name = config['name']
    device = "cuda:{}".format(config['cuda_device_num']) if config['use_cuda'] else 'cpu'
    logger_name = config['logger']
    load_checkpoint = config['load_checkpoint']
    alg = config['params']['alg']
    problem_size = config['params']['problem_size']
    size = config['params']['size']
    multiple_width = config['params']['multiple_width']
    keep_num = config['params']['keep_num']
    start_steps = config['params']['start_steps']
    train_steps = config['params']['train_steps']
    inner_steps = config['params']['inner_steps']
    train_batch_size = config['params']['train_batch_size']
    lr = config['params']['learning_rate']
    log_step = config['params']['log_step']
    model_params = config['model_params']

    seed_everything(config['seed'])
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f'-ts{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'
    dir_path = 'weights/{}_{}_{}'.format(name, ts_name, config['seed'])
    os.mkdir(dir_path)

    log_config = config.copy()
    param_config = log_config['params'].copy()
    log_config.pop('params')
    model_params_config = log_config['model_params'].copy()
    log_config.pop('model_params')
    log_config.update(param_config)
    log_config.update(model_params_config)
    # Initialize logger
    if(logger_name == 'wandb'):
        log_config = config.copy()
        param_config = log_config['params'].copy()
        log_config.pop('params')
        model_params_config = log_config['model_params'].copy()
        log_config.pop('model_params')
        log_config.update(param_config)
        log_config.update(model_params_config)
        logger = wandb.init(project="ELG-CVRP",
                         name=name + ts_name,
                         config=log_config)
    else:
        logger = None
    # Initialize fileLogger
    filename = 'log/{}_{}'.format(name, ts_name)
    fileLogger = Logger(filename, config)

    # Initialize model and baseline
    if config['training'] == 'only_local':
        model = CVRPModel_local(**model_params)
    elif config['training'] == 'joint':
        model = CVRPModel(**model_params)
    
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Training
    train(model=model,
          training=config['training'], 
          start_steps=start_steps,
          train_steps=train_steps,
          inner_steps=inner_steps,
          train_batch_size=train_batch_size, 
          problem_size=problem_size, 
          size=size,
          multiple_width=multiple_width, 
          keep_num=keep_num, 
          lr=lr,
          device=device,
          alg=alg,
          logger=logger,
          fileLogger=fileLogger,
          dir_path=dir_path,
          log_step=log_step)