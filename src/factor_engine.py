"""Vectorized factor computation engine with additional factors."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from typing import Tuple


def winsorize(df: pd.DataFrame, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Winsorize DataFrame row-wise."""
    def _w(row):
        if row.isna().all():
            return row
        lo = row.quantile(limits[0])
        hi = row.quantile(limits[1])
        return row.clip(lo, hi)
    return df.apply(_w, axis=1)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize DataFrame row-wise."""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_momentum(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Compute momentum factor (price change over lookback period)."""
    mom = prices / prices.shift(lookback) - 1
    return zscore(winsorize(mom))


def compute_reversal(returns: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Compute reversal factor (negative lagged returns)."""
    rev = -returns.shift(lag)
    return zscore(winsorize(rev))


def compute_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Compute volatility factor (rolling standard deviation)."""
    vol = returns.rolling(window).std()
    return zscore(winsorize(vol))


def compute_value(prices: pd.DataFrame, market_caps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute value factor (simplified: inverse of price normalized by market cap).
    If market_caps not provided, uses price as proxy.
    """
    if market_caps is None:
        # Use inverse price as simple value proxy
        value = 1.0 / prices
    else:
        # Price-to-market cap ratio (inverse is value)
        value = market_caps / prices
    
    return zscore(winsorize(value))


def compute_quality(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality factor (combination of low volatility and positive returns).
    """
    # Rolling Sharpe-like metric
    rolling_mean = returns.rolling(60).mean()
    rolling_std = returns.rolling(60).std()
    quality = rolling_mean / (rolling_std + 1e-8)
    
    return zscore(winsorize(quality))


def compute_liquidity(returns: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute liquidity factor (Amihud illiquidity measure).
    If volumes not provided, uses return volatility as proxy.
    """
    if volumes is None:
        # Use inverse volatility as liquidity proxy
        vol = returns.rolling(20).std()
        liquidity = 1.0 / (vol + 1e-8)
    else:
        # Amihud illiquidity: abs(return) / volume
        # Lower is more liquid, so we invert
        amihud = np.abs(returns) / (volumes + 1e-8)
        liquidity = 1.0 / (amihud + 1e-8)
    
    return zscore(winsorize(liquidity))


def compute_size(market_caps: Optional[pd.DataFrame] = None, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute size factor (log market cap).
    If market_caps not provided, uses price as proxy.
    """
    if market_caps is None:
        if prices is None:
            raise ValueError("Either market_caps or prices must be provided")
        # Use log price as size proxy
        size = np.log(prices)
    else:
        size = np.log(market_caps)
    
    return zscore(winsorize(size))


def combine_factors(
    factor_dict: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Combine factors with optional weights."""
    keys = list(factor_dict.keys())
    if weights is None:
        weights = {k: 1.0 / len(keys) for k in keys}
    
    combined = None
    for k, df in factor_dict.items():
        if combined is None:
            combined = df * weights.get(k, 0.0)
        else:
            combined = combined.add(df * weights.get(k, 0.0), fill_value=0.0)
    
    return combined


def compute_all_factors(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: Dict,
    market_caps: Optional[pd.DataFrame] = None,
    volumes: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """Compute all enabled factors based on config."""
    factors = {}
    factor_config = config.get('factors', {})
    
    if factor_config.get('momentum', {}).get('enabled', True):
        lookback = factor_config.get('momentum', {}).get('lookback', 21)
        factors['MOM'] = compute_momentum(prices, lookback)
    
    if factor_config.get('reversal', {}).get('enabled', True):
        lag = factor_config.get('reversal', {}).get('lag', 1)
        factors['REV'] = compute_reversal(returns, lag)
    
    if factor_config.get('volatility', {}).get('enabled', True):
        window = factor_config.get('volatility', {}).get('window', 21)
        factors['VOL'] = compute_volatility(returns, window)
    
    if factor_config.get('value', {}).get('enabled', False):
        factors['VAL'] = compute_value(prices, market_caps)
    
    if factor_config.get('quality', {}).get('enabled', False):
        factors['QUAL'] = compute_quality(returns, prices)
    
    if factor_config.get('liquidity', {}).get('enabled', False):
        factors['LIQ'] = compute_liquidity(returns, volumes)
    
    if factor_config.get('size', {}).get('enabled', False):
        factors['SIZE'] = compute_size(market_caps, prices)
    
    return factors

