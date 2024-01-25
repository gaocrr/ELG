import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import os
import argparse


def generate_tsp_data(batch_size, problem_size, distribution):
    # return torch.rand(size=(dataset_size, problem_size, 2))
    if distribution['data_type'] == 'uniform':
        problems = torch.rand(size=(batch_size, problem_size, 2))
        # problems.shape: (batch, problem, 2)
    elif distribution['data_type'] == 'cluster':
        n_cluster = distribution['n_cluster']
        center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(batch_size)])
        center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
        std = distribution['std']
        for j in range(batch_size):
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            coords = torch.zeros(problem_size, 2)
            for i in range(n_cluster):
                if i < n_cluster - 1:
                    coords[int((problem_size) / n_cluster) * i:int((problem_size) / n_cluster) * (i + 1)] = \
                        torch.cat((torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_x[i], std),
                                torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_y[i], std)),dim=1)
                elif i == n_cluster - 1:
                    coords[int((problem_size) / n_cluster) * i:] = \
                        torch.cat((torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i,1).normal_(mean_x[i], std),
                                torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i,1).normal_(mean_y[i], std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
            problems = coords.unsqueeze(0) if j == 0 else torch.cat((problems, coords.unsqueeze(0)), dim=0)
    elif distribution['data_type'] == 'mixed':
        n_cluster_mix = distribution['n_cluster_mix']
        center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(batch_size)])
        center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
        std = distribution['std']
        for j in range(batch_size):
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
            coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1)
            for i in range(n_cluster_mix):
                if i < n_cluster_mix - 1:
                    coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                        torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)),dim=1)
                elif i == n_cluster_mix - 1:
                    coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                        torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,1).normal_(mean_x[i], std),
                                torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,1).normal_(mean_y[i], std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords).cuda()
            problems = coords.unsqueeze(0) if j == 0 else torch.cat((problems, coords.unsqueeze(0)), dim=0)
    
    return problems


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


class TSPDataset(Dataset):
    def __init__(self, filename=None, size=100, num_samples=10000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            if distribution == None:
                # Sample points randomly in [0, 1] square
                self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            else:
                self.data = generate_tsp_data(num_samples, size, distribution)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":

    data_size = [1000, 1000, 100]
    problem_size = [100, 200, 500]

    data_type = "uniform" # cluster, mixed, uniform
    distribution = {
        "data_type": data_type,  
        "n_cluster": 3,
        "n_cluster_mix": 1,
        "lower": 0.2,
        "upper": 0.8,
        "std": 0.07,
    }

    for i in range(len(problem_size)):
        val_filename = f'data/tsp_{problem_size[i]}_val.pkl'
        test_filename = f'data/tsp_{data_type}{problem_size[i]}_test.pkl'

        # generate validation data
        validation_dataset = generate_tsp_data(data_size[i], problem_size[i], distribution=distribution)
        save_dataset(validation_dataset, val_filename)

        # generate test data
        # test_dataset = generate_tsp_data(data_size[i], problem_size[i], distribution=distribution)
        # save_dataset(test_dataset, test_filename)