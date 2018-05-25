from WhiteSnake.Loader import Dataset
from utils.Visualization import CustomVisdom
from config import Env

options = Env()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(options, k_, v_)

    vis = CustomVisdom(name=f'SeqGAN')

    configSummary = ''
    for key, value in options.__dict__.items():
        if not key.startswith('__'): configSummary += str(key) + '=' + str(value) + '<br>'
    
    vis.text('config', f'{configSummary}')

if __name__ == '__main__':
    import fire
    fire.Fire()