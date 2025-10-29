import numpy as np
import pandas as pd

def BCE(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    
    # Clip predictions to prevent log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Compute binary cross-entropy
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce
