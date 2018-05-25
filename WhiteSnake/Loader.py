from torch.utils import data
import os
import csv

class Dataset(data.Dataset):
    def __init__(self):
        self.data = list(csv.reader(open(f'{os.path.dirname(os.path.abspath(__file__))}/white-snake-preprocess.csv')))

    def __getitem__(self):
        return self.data[index]

    def __len__(self):
        return len(self.data)