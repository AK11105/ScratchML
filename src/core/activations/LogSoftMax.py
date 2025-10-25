import numpy as np 
import pandas as pd 

def LogSoftmax(X, axis=0):
    X_np = np.array(X)
    X_stable = X_np - np.max(X_np, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(X_stable), axis=axis, keepdims=True))
    out = X_stable - logsumexp
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
