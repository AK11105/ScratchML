import numpy as np 
import pandas as pd 

def HardSigmoid(X):
    out = np.clip(0.2*X + 0.5, 0, 1)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
