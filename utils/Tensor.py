import torch
from tqdm import tqdm

class TensorZip():
    def fromDataset(dataset, vis=None, message=None):
        if message != None: print(message)
        X = [0] * len(dataset)
        Y = [0] * len(dataset)

        for i, (x, y) in enumerate(tqdm(dataset)):
            X[i] = x
            Y[i] = y
            if vis != None:
                vis.text('progress', f'目前資料輸入進度: {i + 1}/{len(dataset)}')

        X = torch.Tensor(X).long()
        Y = torch.Tensor(Y).long()

        return X, Y