import torch
from tqdm import tqdm

class TensorZip():
    def __init__(self, dataset):
        self.dataset = dataset

    def fromDataset(self, gpu=False, vis=None, message=None):
        if message != None: print(message)
        X, Y = zip(*tqdm(self.dataset))
        X = torch.LongTensor(X)
        Y = torch.LongTensor(Y)

        return X, Y