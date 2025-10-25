import numpy as np 
import pandas as pd 

def Maxout(X, k=2):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_features = X.shape[1] // k
    out = X.reshape(X.shape[0], n_features, k).max(axis=2)
    return out
