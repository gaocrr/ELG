import torch
import numpy as np
import json
import random


def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()
    while not done:
        cur_dist, cur_theta, xy = env.get_local_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist=cur_dist, cur_theta=cur_theta, xy=xy, eval_type=eval_type)
        state, reward, done = env.step(selected)
        actions.append(selected)
        probs.append(one_step_prob)

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward

def batched_two_opt_torch(cuda_points, cuda_tour, max_iterations=1000, device="cpu"):
  cuda_tour = torch.cat((cuda_tour, cuda_tour[:, 0:1]), dim=-1)
  iterator = 0
  problem_size = cuda_points.shape[0]
  with torch.inference_mode():
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      # For TSPLib Euc_2D, distance must be integer.
      # A_ij = torch.round(torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1)))
      # A_i_plus_1_j_plus_1 = torch.round(torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1)))
      # A_i_i_plus_1 = torch.round(torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1)))
      # A_j_j_plus_1 = torch.round(torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1)))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, problem_size, rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, problem_size)

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break

  return cuda_tour[:, :-1]

def check_feasible(pi):
   # input shape: (batch, multi, problem)
   pi = pi.squeeze(0)
   return (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all()

def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

class Logger(object):
  def __init__(self, filename, config):
    '''
    filename: a json file
    '''
    self.filename = filename
    self.logger = config
    self.logger['result'] = {}
    self.logger['result']['val_100'] = []
    self.logger['result']['val_200'] = []
    self.logger['result']['val_500'] = []

  def log(self, info):
    '''
    Log validation cost on 3 datasets every log steps
    '''
    self.logger['result']['val_100'].append(info[0].cpu().numpy().tolist())
    self.logger['result']['val_200'].append(info[1].cpu().numpy().tolist())
    self.logger['result']['val_500'].append(info[2].cpu().numpy().tolist())
    print(self.logger)
    with open(self.filename, 'w') as f:
      json.dump(self.logger, f)