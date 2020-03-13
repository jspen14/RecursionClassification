import pandas as pd
import matplotlib.pyplot as plt

class logger:
    def __init__(self, writeFile, colName=None):
        if colName == None:
            colName = ["loss", "acc"]
        self.dest = writeFile
        with open(self.dest,'a') as fd:
            fd.write(str(colName)[1:-1]+'\n')

    def log(self, arr):
        with open(self.dest,'a') as fd:
            fd.write(str(arr)[1:-1]+'\n')

    def showStats(self):
        csv = pd.read_csv(self.dest)
        plt.plot(csv)
        plt.show()
