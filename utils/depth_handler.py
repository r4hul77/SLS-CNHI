import pandas as pd
import nptdms
from utils.filehandling import *
import tqdm
import math

class depth_handler:

    def __init__(self, filePath, frame_size):

        self.filePath = filePath
        self.df = nptdms.TdmsFile.read(ret[0]).as_dataframe(time_index=True, absolute_time=True)
        self.frame_size = frame_size
        self.now = 0
        self.resetFlag = False
        self.length = len(self.df)
        self.iters = math.ceil(len(self.df)/(4*self.frame_size))

    def __len__(self):

        return self.iters + 1

    def __iter__(self):
        return self

    def __next__(self):
        if(self.resetFlag):
            self.resetFlag = False
            self.reset()
        if(self.now > self.length):
            self.resetFlag = True
            raise StopIteration
        ret = pd.DataFrame()
        end = self.now + self.frame_size
        if( end > self.length//4):
            end = self.length//4
        for i in range(self.now, end):
            temp = self.df.iloc[i*4 : 4*i+4].sum()
            temp["time_stamp"] = self.df.iloc[4*i+3].name
            ret = ret.append(temp, ignore_index=True)
            pass
        self.now += self.frame_size
        return ret

    def reset(self):
        self.now = 0


if __name__ == "__main__":

    ret = LoadFilesWithExtensions(path = "/home/harsha/Desktop/seedIdentification/SeedSpace2_ImagesSplitted/"
                            , extensions=["tdms"])

    depth = depth_handler(ret[0], 10)

    for i, j in enumerate(depth):
        pass