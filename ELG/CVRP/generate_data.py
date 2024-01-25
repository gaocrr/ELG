import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import os


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

def generate_vrp_data(dataset_size, problem_size, random_capacity=True):
    if random_capacity == False:
        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {
                    10: 20.,
                    20: 30.,
                    50: 40.,
                    100: 50.,
                    200: 80.,
                    500: 100.,
                    1000: 250.
                }
        data = {
            'loc': torch.FloatTensor(dataset_size, problem_size, 2).uniform_(0, 1),
            # Uniform 1 - 9, scaled by capacities
            'demand': (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float() / CAPACITIES[problem_size],
            'depot': torch.FloatTensor(dataset_size, 1, 2).uniform_(0, 1)
        }

    else:
        # Following the set-X of VRPLib ("New benchmark instances for the capacitated vehicle routing problem") to generate capacity 
        route_length = torch.tensor(np.random.triangular(3, 6, 25, size=dataset_size))
        demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        capacities = torch.ceil(route_length * demand.sum(1) / problem_size)
        data = {
            'loc': torch.FloatTensor(dataset_size, problem_size, 2).uniform_(0, 1),
            # Uniform 1 - 9, scaled by capacities
            'demand': (demand / capacities[:, None]).float(),
            'depot': torch.FloatTensor(dataset_size, 1, 2).uniform_(0, 1)
        }     

    return data

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

def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }

class TSPDataset(Dataset):
    def __init__(self, filename=None, size=100, num_samples=1000000, offset=0):
        super(TSPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]

        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, test=False):
        super(VRPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            if test == True:
                self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
            else:
                self.data = data[offset:offset+num_samples]

        else:
            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.,
                200: 80.,
                500: 100.,
                1000: 250.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(1, 2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    data_size = [5000, 5000, 100]
    problem_size = [100, 200, 500]

    for i in range(len(problem_size)):
        val_filename = f'data/vrp{problem_size[i]}_val.pkl'
        test_filename = f'data/vrp{problem_size[i]}_test.pkl'

        # generate validation data
        validation_dataset = VRPDataset(num_samples=data_size[i], size=problem_size[i], test=True)
        save_dataset(validation_dataset, val_filename)
    # generate test data
    # test_dataset = VRPDataset(size=problem_size, num_samples=data_size)
    # save_dataset(test_dataset, test_filename)