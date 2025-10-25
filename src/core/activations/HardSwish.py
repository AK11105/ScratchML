import numpy as np 
import pandas as pd 

from .HardSigmoid import HardSigmoid

def HardSwish(X):
    out = X * HardSigmoid(X)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out