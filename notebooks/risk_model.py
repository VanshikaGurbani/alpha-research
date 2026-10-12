
import pandas as pd
import numpy as np

def compute_covariance(returns, half_life=None):
    if half_life is None:
        return returns.cov()
    lam = 0.5**(1/half_life)
    w = np.array([lam**(len(returns)-1-i) for i in range(len(returns))])
    w = w / w.sum()
    centered = returns.sub(returns.mean(axis=0), axis=1)
    cov = centered.mul(w, axis=0).T @ centered
    return cov

def factor_correlation_matrix(factor_dict):
    series = {k: v.mean(axis=1) for k, v in factor_dict.items()}
    return pd.DataFrame(series).corr()
