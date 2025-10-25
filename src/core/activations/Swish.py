import numpy as np 
import pandas as pd

from .Sigmoid import sigmoid 

def swish(X):
    out = X * sigmoid(X)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
