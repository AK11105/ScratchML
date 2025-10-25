import numpy as np
import pandas as pd

def Softmax(X, axis=0):
    # Convert DataFrame to NumPy temporarily
    if isinstance(X, pd.DataFrame):
        data = X.values
        out = np.exp(data - np.max(data, axis=axis, keepdims=True))
        out /= np.sum(out, axis=axis, keepdims=True)
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    
    # NumPy array or Series
    else:
        X = np.array(X)
        e_X = np.exp(X - np.max(X, axis=axis, keepdims=True))
        return e_X / np.sum(e_X, axis=axis, keepdims=True)
