
import pandas as pd
import numpy as np

def winsorize(df, limits=(0.01, 0.99)):
    def _w(row):
        if row.isna().all():
            return row
        lo = row.quantile(limits[0])
        hi = row.quantile(limits[1])
        return row.clip(lo, hi)
    return df.apply(_w, axis=1)

def zscore(df):
    return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

def compute_momentum(prices, lookback=21):
    mom = prices / prices.shift(lookback) - 1
    return zscore(winsorize(mom))

def compute_reversal(returns, lag=1):
    rev = -returns.shift(lag)
    return zscore(winsorize(rev))

def compute_volatility(returns, window=21):
    vol = returns.rolling(window).std()
    return zscore(winsorize(vol))

def combine_factors(factor_dict, weights=None):
    keys = list(factor_dict.keys())
    if weights is None:
        weights = {k: 1/len(keys) for k in keys}
    combined = None
    for k, df in factor_dict.items():
        if combined is None:
            combined = df * weights[k]
        else:
            combined = combined.add(df * weights[k], fill_value=0.0)
    return combined
