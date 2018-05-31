from tqdm import tqdm
from torch.nn import NLLLoss
from config import Env
from torch.autograd import Variable
import torch

class MLE4GEN():
    def __init__(self, gen, dataLoader, vis=None, epochs=150):
        self.gen = gen
        self.gen_optim = torch.optim.Adam(gen.parameters(), lr=1e-2)
        self.epochs = epochs
        self.loader = dataLoader
        self.vis = vis

    def train(self, gpu=False):
        for epoch in tqdm(range(self.epochs)):
            avg_loss = 0

            for batchIdx, (batchX, batchY) in enumerate(tqdm(self.loader)):
                batchX = Variable(batchX)
                batchY = Variable(batchY)

                if gpu:
                    batchX = batchX.cuda()
                    batchY = batchY.cuda()

                self.gen_optim.zero_grad()
                loss = self.gen.NLLLoss(batchX, batchY)
                loss.backward()
                self.gen_optim.step()
                avg_loss += loss.data[0] / batchX.size()[0]
                if self.vis != None: self.vis.text('progress', f'目前迭代進度:<br>epochs={epoch + 1}<br>batch={batchIdx + 1}')

            self.vis.drawLine('loss', epoch + 1, avg_loss)