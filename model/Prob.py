from tqdm import tqdm
from torch.nn import NLLLoss
from config import Env
import torch

options = Env()

class MLE4GEN():
    def __init__(self, gen, dataLoader, vis=None, epochs=150):
        self.gen = gen
        self.gen_optim = torch.optim.Adam(gen.parameters(), lr=1e-2)
        self.epochs = epochs
        self.loader = dataLoader
        self.vis = vis

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0

            for batchIdx, (batchX, batchY) in enumerate(tqdm(self.loader)):
                batch_size, seq_len = batchX.size()
                inp = batchX.permute(1, 0)      # seq_len * batch_size
                target = batchY.permute(1, 0)   # seq_len * batch_size

                # TODO: MLE model complete
                

            tqdm.write(f'Average NLL = {total_loss / options.batch_size}')
            if self.vis != None: self.vis.text('progress', f'目前迭代進度:<br>epochs={epoch + 1}<br>batch={batchIdx + 1}')