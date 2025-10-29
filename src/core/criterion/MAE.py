import numpy as np
import pandas as pd

def MAE(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    
    return np.mean(abs(y_pred - y_true))
