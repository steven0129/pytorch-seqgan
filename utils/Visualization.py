import visdom
import numpy as np

class CustomVisdom(object):
    def __init__(self, name='default', **kwargs):
        self.vis = visdom.Visdom(env=name, **kwargs)
        self.X = {}
        self.Y = {}
    
    def text(self, id, content):
        try:
            self.vis.text(content, win=id)
        except:
            pass

    def drawLine(self, id, x, y):
        try:
            if id not in self.Y:
                self.X[id] = []
                self.Y[id] = []
            self.X[id].append(x)
            self.Y[id].append(y)
            self.vis.line(X=np.array(self.X[id]), Y=np.array(self.Y[id]), win=id)
        except:
            pass