import numpy as np 
import pandas as pd

def GeLU(X):
    out = 0.5*X*(1+np.tanh(np.sqrt(2/np.pi)*(X + 0.044715*X**3)))
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out