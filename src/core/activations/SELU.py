import numpy as np
import pandas as pd

def SeLU(X, alpha=1.67326, lambd=1.0507):
    out = np.where(X > 0, lambd * X, lambd * (alpha * np.exp(X) - alpha))
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(out, index=X.index, columns=X.columns)
    return out
