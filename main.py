from WhiteSnake.Loader import Dataset
from utils.Visualization import CustomVisdom
from utils.Tensor import TensorZip
from config import Env
from tqdm import tqdm
from model.GAN import Generator
from model.Prob import MLE4GEN
import numpy
import torch
import torch.utils.data as Data

options = Env()
WORD_VEC = numpy.load('./WhiteSnake/word2vec.npy').tolist()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    vis = CustomVisdom(name='SeqGAN')
    vis.summary(options=options)

    whiteSnake = Dataset(ratio=options.ratio)
    print(f'收錄{len(whiteSnake)}個pair')

    X, Y = TensorZip.fromDataset(dataset=whiteSnake, vis=vis, message='將每對pair存入X和Y中')
    dataset = Data.TensorDataset(X, Y)
    loader = Data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=options.shuffle, drop_last=True, num_workers=options.core)

    gen = Generator(options.g_emb_dim, options.g_hid_dim, len(whiteSnake.classes), whiteSnake.maxLen())
    if options.use_gpu: gen = gen.cuda()
    
    mle = MLE4GEN(gen, loader, vis=vis)
    mle.train()

if __name__ == '__main__':
    import fire
    fire.Fire()