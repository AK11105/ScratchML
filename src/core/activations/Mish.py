import numpy as np
import pandas as pd

def mish(X):
    out = X * np.tanh(np.log1p(np.exp(X)))

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
