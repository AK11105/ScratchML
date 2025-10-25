import pandas as pd
import numpy as np

def sigmoid(X):
    # NumPy array
    if isinstance(X, np.ndarray):
        return 1 / (1 + np.exp(-X))
    
    # pandas Series
    elif isinstance(X, pd.Series):
        return 1 / (1 + np.exp(-X))
    
    # pandas DataFrame (apply only to numeric columns)
    elif isinstance(X, pd.DataFrame):
        result = X.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].apply(lambda col: 1 / (1 + np.exp(-col)))
        return result
    
    # Python list
    elif isinstance(X, list):
        arr = np.array(X, dtype=float)
        return list(1 / (1 + np.exp(-arr)))
    
    # Unsupported types
    else:
        raise TypeError(f"Unsupported type {type(X)}. Expected numpy.ndarray, list, pd.Series, or pd.DataFrame.")
