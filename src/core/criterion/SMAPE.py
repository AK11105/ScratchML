import numpy as np 
import pandas as pd 

def SMAPE(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.clip(denom, 1e-15, None)  # avoid divide-by-zero
    return np.mean(np.abs(y_true - y_pred) / denom) * 100
