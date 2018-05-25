from torch.utils import data
import os
import csv

class Dataset(data.Dataset):
    def __init__(self, ratio=1):
        self.data = list(csv.reader(open(f'{os.path.dirname(os.path.abspath(__file__))}/white-snake-preprocess.csv')))
        self.data = self.data[:int(len(self.data) * ratio)]

    def __getitem__(self, index):
        return (self.data[index], self.data[index + 1])

    def __len__(self):
        return len(self.data) - 1