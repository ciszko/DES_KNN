import pandas as pd
import os
from termcolor import colored

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read():
    for file in os.listdir("./datasets/"):

        f = pd.read_csv("./datasets/" + file, header=0)
        count = (f.iloc[:, -1].value_counts())
        print(count)
        all_ = count[0] + count[1]
        prc = count[1] / all_ * 100
        print(f"{file}  {prc}")

df = pd.read_csv("./datasets/high_diamond_ranked_10min.csv", header=0)
cols = list(df.columns)
a, b = cols.index('redGoldPerMin'), cols.index('blueWins')
cols[b], cols[a] = cols[a], cols[b]
df = df[cols]
df = df.drop(["gameId"], 1)

df.to_csv('./datasets/lol_ranked_10min.csv', index=False)
