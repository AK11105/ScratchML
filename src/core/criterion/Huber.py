import numpy as np
import pandas as pd

def Huber(y_pred, y_true, delta=1.0):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")

    residuals = y_pred - y_true
    residuals = abs(residuals)
    
    quadratic = residuals <= delta
    linear =  residuals > delta 
    
    quadratic_loss = 0.5*residuals**2
    linear_loss = delta*residuals - 0.5*delta**2
    
    loss = quadratic_loss*quadratic + linear*linear_loss
    
    return np.mean(loss)
