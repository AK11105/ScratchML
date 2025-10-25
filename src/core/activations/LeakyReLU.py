import numpy as np
import pandas as pd

def LeakyReLU(X, alpha=0.01):
    out = np.where(X > 0, X, alpha * X)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
