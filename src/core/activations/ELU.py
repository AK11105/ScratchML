import numpy as np 
import pandas as  pd 

def ELU(X, alpha=0.01):
    out = np.where(X > 0, X, alpha * (np.exp(X) - 1))
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
