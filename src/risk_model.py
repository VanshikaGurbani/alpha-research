"""Barra-style risk model implementation."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple


def compute_covariance(returns: pd.DataFrame, half_life: Optional[int] = None) -> pd.DataFrame:
    """
    Compute covariance matrix with optional exponential weighting.
    
    Args:
        returns: DataFrame of returns
        half_life: Half-life for exponential weighting (None for equal weights)
    
    Returns:
        Covariance matrix
    """
    if half_life is None:
        return returns.cov()
    
    # Exponential weights
    n = len(returns)
    lam = 0.5 ** (1.0 / half_life)
    weights = np.array([lam ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()
    
    # Weighted covariance
    centered = returns.sub(returns.mean(axis=0), axis=1)
    cov = centered.mul(weights, axis=0).T @ centered
    
    return cov


def factor_correlation_matrix(factor_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute correlation matrix between factors."""
    series = {k: v.mean(axis=1) for k, v in factor_dict.items()}
    return pd.DataFrame(series).corr()


def compute_factor_exposures(
    returns: pd.DataFrame,
    factors: Dict[str, pd.DataFrame],
    window: int = 60
) -> Dict[str, pd.DataFrame]:
    """
    Compute factor exposures using rolling regression.
    
    Args:
        returns: Stock returns
        factors: Dictionary of factor DataFrames
        window: Rolling window for regression
    
    Returns:
        Dictionary of factor exposure DataFrames
    """
    exposures = {}
    
    for factor_name, factor_df in factors.items():
        exp = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        
        for stock in returns.columns:
            if stock not in factor_df.columns:
                continue
            
            for i in range(window, len(returns)):
                # Rolling regression: returns ~ factor
                y = returns[stock].iloc[i-window:i]
                x = factor_df[stock].iloc[i-window:i]
                
                if x.notna().sum() < 10:
                    continue
                
                # Simple regression coefficient
                valid = x.notna() & y.notna()
                if valid.sum() < 10:
                    continue
                
                x_clean = x[valid]
                y_clean = y[valid]
                
                if x_clean.std() > 1e-8:
                    beta = np.cov(x_clean, y_clean)[0, 1] / np.var(x_clean)
                    exp.loc[returns.index[i], stock] = beta
        
        exposures[factor_name] = exp
    
    return exposures


def compute_barra_risk(
    returns: pd.DataFrame,
    factors: Dict[str, pd.DataFrame],
    half_life: int = 60,
    min_periods: int = 60
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compute Barra-style risk model.
    
    Returns:
        Tuple of (covariance matrix, factor exposures)
    """
    # Compute factor exposures
    exposures = compute_factor_exposures(returns, factors, window=min_periods)
    
    # Compute factor covariance
    factor_returns = {k: v.mean(axis=1) for k, v in factors.items()}
    factor_df = pd.DataFrame(factor_returns)
    factor_cov = compute_covariance(factor_df, half_life)
    
    # Compute stock covariance from factor model
    # Cov = B * F * B' + D (where D is specific risk)
    # Simplified version: use factor exposures and factor covariance
    stock_cov = pd.DataFrame(
        index=returns.columns,
        columns=returns.columns,
        dtype=float
    )
    
    # For each pair of stocks, compute covariance via factors
    for i, stock1 in enumerate(returns.columns):
        for j, stock2 in enumerate(returns.columns):
            if i > j:
                continue
            
            cov_sum = 0.0
            for factor_name in factors.keys():
                if stock1 in exposures[factor_name].columns and stock2 in exposures[factor_name].columns:
                    exp1 = exposures[factor_name][stock1].iloc[-1] if len(exposures[factor_name]) > 0 else 0
                    exp2 = exposures[factor_name][stock2].iloc[-1] if len(exposures[factor_name]) > 0 else 0
                    
                    if factor_name in factor_cov.index and factor_name in factor_cov.columns:
                        factor_var = factor_cov.loc[factor_name, factor_name]
                        cov_sum += exp1 * exp2 * factor_var
            
            stock_cov.loc[stock1, stock2] = cov_sum
            stock_cov.loc[stock2, stock1] = cov_sum
    
    # Add specific risk (idiosyncratic)
    specific_risk = returns.std() ** 2
    np.fill_diagonal(stock_cov.values, stock_cov.values.diagonal() + specific_risk.values)
    
    return stock_cov, exposures


def compute_portfolio_risk(
    weights: pd.DataFrame,
    covariance: pd.DataFrame
) -> pd.Series:
    """
    Compute portfolio risk (volatility) from weights and covariance matrix.
    
    Args:
        weights: DataFrame of portfolio weights (dates x stocks)
        covariance: Covariance matrix (stocks x stocks)
    
    Returns:
        Series of portfolio volatilities
    """
    portfolio_risk = pd.Series(index=weights.index, dtype=float)
    
    for date in weights.index:
        w = weights.loc[date]
        w = w[w.notna()]
        
        if len(w) == 0:
            portfolio_risk.loc[date] = np.nan
            continue
        
        # Align weights with covariance
        common_stocks = w.index.intersection(covariance.index)
        if len(common_stocks) == 0:
            portfolio_risk.loc[date] = np.nan
            continue
        
        w_aligned = w[common_stocks]
        cov_aligned = covariance.loc[common_stocks, common_stocks]
        
        # Portfolio variance = w' * Cov * w
        portfolio_var = w_aligned @ cov_aligned @ w_aligned
        portfolio_risk.loc[date] = np.sqrt(portfolio_var) if portfolio_var > 0 else 0.0
    
    return portfolio_risk

