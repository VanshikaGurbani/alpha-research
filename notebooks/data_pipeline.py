
import pandas as pd
import numpy as np

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # long format
    if set(['date','ticker','close']).issubset(df.columns):
        df['date'] = pd.to_datetime(df['date'])
        return df.pivot(index='date', columns='ticker', values='close').sort_index()
    # wide format
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
    df = df.set_index(df.columns[0])
    return df.sort_index()

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))

def align_universe(prices: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    valid = prices.count(axis=0) >= min_obs
    return prices.loc[:, valid]
