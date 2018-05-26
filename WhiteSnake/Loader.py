from torch.utils import data
from sklearn.preprocessing import LabelEncoder
import os
import csv
import numpy

class Dataset(data.Dataset):
    def __init__(self, ratio=1):
        currentPath = os.path.dirname(os.path.abspath(__file__))

        self.data = list(csv.reader(open(f'{currentPath}/white-snake-preprocess.csv')))
        self.data = list(map(lambda x: x[0], self.data))
        self.data = self.data[:int(len(self.data) * ratio)]
        self.data = list(map(self.__filterNewLine, self.data))
        self.lengths = list(map(len, self.data))
        self.classes = numpy.load(f'{currentPath}/classes.npy').tolist()

    def __getitem__(self, index):
        data1 = self.__spacesAlign(self.__maxLen(), self.data[index])
        data2 = self.__spacesAlign(self.__maxLen(), self.data[index + 1])
        return tuple(map(self.__data2ind, [data1, data2]))

    def __len__(self):
        return len(self.data) - 1

    def __maxLen(self):
        return max(self.lengths)

    def __spacesAlign(self, outLen, myStr):
        return myStr + [' '] * (outLen - len(myStr))

    def __filterNewLine(self, data):
        data = list(filter(lambda x: x != 'n', list(data)))
        data = list(map(lambda x: x.replace('\\', '\\n'), list(data)))

        return data

    def __data2ind(self, data):
        return list(map(self.classes.index, data))