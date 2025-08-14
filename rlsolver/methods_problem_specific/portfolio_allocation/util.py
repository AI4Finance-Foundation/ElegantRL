import pandas as pd
import numpy as np

def read_csv(file):
    data = pd.read_csv(file)
    if 'Amount' in data.columns:
        vec = data['Amount'].tolist()
    if 'amount' in data.columns:
        vec = data['amount'].tolist()
    return vec

def read_npy(file):
    data = np.load(file).tolist()
    return data

if __name__ == '__main__':
    file = 'data/xxx.csv'
    df = read_csv(file)
