import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Ising:
    def load_data(split=True):
        np.random.seed(123)

        data = pd.read_csv("../data/50/s3_cfg_L50_A0_mc1000000_burn1_tl1.000_tu3.530.csv", header=None)
        X = data.iloc[:, :-1].to_numpy()

        y = data.iloc[:, -1][::50].to_numpy() # take every 50th row

        X = X.reshape(len(y), 50*50)
        y = y.reshape(len(y), 1)
        if split:
            return train_test_split(X, y, test_size=0.2)
        
        return X, y

class Metrics:
    def MSE(y, y_pred):
        N = len(y)
        y = y.ravel()
        y_pred = y_pred.ravel()

        return 1/N * np.sum((y - y_pred)**2)

    def R2(y, y_pred):
        y = y.ravel()
        y_pred = y_pred.ravel()
        mean_y = np.mean(y)
        return 1 - np.sum( (y - y_pred )**2) / np.sum((mean_y - y_pred)**2)

    def accuracy(y, y_pred):
        y = y.ravel()
        y_pred = y_pred.ravel()
        N = len(y)

        y_pred = y_pred > 0.5
        
        return np.sum( y == y_pred) / N


    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
