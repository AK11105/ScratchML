import numpy as np
import pandas as pd

def R2(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    
    y_mean = np.mean(y_true)

    res = np.sum(np.square(y_true - y_pred))
    tot = np.sum(np.square(y_true - y_mean))
    
    return 1 - res/tot
