import numpy as np
import pandas as pd

def Hinge(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")

    out = np.mean(np.maximum(0, 1 - y_pred * y_true))
    return out
