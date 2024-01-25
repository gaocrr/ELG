import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import os
import argparse


def generate_tsp_data(dataset_size, problem_size):
    return torch.rand(size=(dataset_size, problem_size, 2))

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


if __name__ == "__main__":

    data_size = [5000, 5000, 100]
    problem_size = [100, 200, 500]

    for i in range(len(problem_size)):
        val_filename = f'data/tsp_{problem_size[i]}_val.pkl'
        test_filename = f'data/tsp_{problem_size[i]}_test.pkl'

        # generate validation data
        validation_dataset = generate_tsp_data(data_size[i], problem_size[i])
        save_dataset(validation_dataset, val_filename)

        # generate test data
        # test_dataset = generate_tsp_data(data_size, problem_size)
        # save_dataset(test_dataset, test_filename)