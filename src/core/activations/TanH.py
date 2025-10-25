import numpy as np
import pandas as pd

def tanH(X):
    out = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    return out
