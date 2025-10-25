import numpy as np 
import pandas as pd 

def ThresholdedReLU(X, theta=1.0):
    out = np.where(X > theta, X, 0)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
