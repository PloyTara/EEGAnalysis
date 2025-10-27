import numpy as np

def normalize_data(data, method="raw"):
    if method == "raw":
        return data
    elif method == "min-max":
        return (data - np.min(data, axis=0)) / (np.ptp(data, axis=0) + 1e-8)
    elif method == "z-score":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
