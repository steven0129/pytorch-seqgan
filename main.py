from WhiteSnake.Loader import Dataset
from utils.Visualization import CustomVisdom
from config import Env
from tqdm import tqdm
import numpy
import torch
import torch.utils.data as Data

options = Env()
WORD_VEC = numpy.load(f'./WhiteSnake/word2vec.npy').tolist()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    vis = CustomVisdom(name=f'SeqGAN')

    configSummary = ''
    for key, value in options.__dict__.items():
        if not key.startswith('__'): configSummary += str(key) + '=' + str(value) + '<br>'
    
    vis.text('config', f'{configSummary}')

    whiteSnake = Dataset(ratio=options.ratio)
    print(f'收錄{len(whiteSnake)}個pair')
    print('將每對pair存入X和Y中')

    X = [0] * len(whiteSnake)
    Y = [0] * len(whiteSnake)

    for i, (x, y) in enumerate(tqdm(whiteSnake)):
        X[i] = x
        Y[i] = y
        vis.text('progress', f'目前資料輸入進度: {i + 1}/{len(whiteSnake)}')

    X = torch.Tensor(X).long()
    Y = torch.Tensor(Y).long()

    dataset = Data.TensorDataset(data_tensor=X, target_tensor=Y)
    loader = Data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=options.shuffle, drop_last=True, num_workers=options.core)

    for epoch in tqdm(range(options.epochs)):
        for index, (batchX, batchY) in enumerate(tqdm(loader)):
            tqdm.write(str(batchX.size()))
            tqdm.write(str(batchY.size()))
            vis.text('progress', f'目前迭代進度:<br>epochs={epoch + 1}<br>batch={index + 1}')

if __name__ == '__main__':
    import fire
    fire.Fire()