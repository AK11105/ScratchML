import numpy as np
import pandas as pd

def ReLU(X):
    # --- NumPy array ---
    if isinstance(X, np.ndarray):
        return np.maximum(0, X)

    # --- pandas Series ---
    elif isinstance(X, pd.Series):
        return np.maximum(0, X)

    # --- pandas DataFrame ---
    elif isinstance(X, pd.DataFrame):
        result = X.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].apply(lambda col: np.maximum(0, col))
        return result

    # --- Python list ---
    elif isinstance(X, list):
        return [max(0, x) for x in X]

    # --- Unsupported type ---
    else:
        raise TypeError(f"Unsupported type {type(X)}. Expected numpy.ndarray, list, pd.Series, or pd.DataFrame.")
