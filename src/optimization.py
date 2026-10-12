"""Optimization modules for IC weighting and ML-based factor weighting."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


def compute_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    lag: int = 1
) -> pd.Series:
    """
    Compute Information Coefficient (IC) - correlation between factor and forward returns.
    
    Args:
        factor: Factor scores DataFrame
        forward_returns: Forward returns DataFrame
        lag: Number of periods to look forward
    
    Returns:
        Series of IC values over time
    """
    ic_series = pd.Series(index=factor.index, dtype=float)
    
    for date in factor.index:
        if date not in forward_returns.index:
            continue
        
        # Get forward return date
        try:
            forward_date_idx = forward_returns.index.get_loc(date) + lag
            if forward_date_idx >= len(forward_returns):
                continue
            forward_date = forward_returns.index[forward_date_idx]
        except:
            continue
        
        # Get factor and returns for this date
        factor_row = factor.loc[date]
        returns_row = forward_returns.loc[forward_date]
        
        # Align
        common = factor_row.index.intersection(returns_row.index)
        if len(common) < 10:
            continue
        
        factor_aligned = factor_row[common].dropna()
        returns_aligned = returns_row[common].dropna()
        
        common_stocks = factor_aligned.index.intersection(returns_aligned.index)
        if len(common_stocks) < 10:
            continue
        
        factor_clean = factor_aligned[common_stocks]
        returns_clean = returns_aligned[common_stocks]
        
        # Compute correlation
        if factor_clean.std() > 1e-8 and returns_clean.std() > 1e-8:
            ic = np.corrcoef(factor_clean, returns_clean)[0, 1]
            ic_series.loc[date] = ic
    
    return ic_series


def compute_rolling_ic_weights(
    factors: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    window: int = 60,
    lag: int = 1
) -> pd.DataFrame:
    """
    Compute rolling IC-based factor weights.
    
    Args:
        factors: Dictionary of factor DataFrames
        forward_returns: Forward returns DataFrame
        window: Rolling window for IC calculation
        lag: Lag for forward returns
    
    Returns:
        DataFrame of factor weights over time
    """
    # Compute IC for each factor
    ic_dict = {}
    for factor_name, factor_df in factors.items():
        ic_dict[factor_name] = compute_ic(factor_df, forward_returns, lag)
    
    # Combine ICs into DataFrame
    ic_df = pd.DataFrame(ic_dict)
    
    # Compute rolling mean IC
    rolling_ic = ic_df.rolling(window=window, min_periods=10).mean()
    
    # Convert IC to weights (normalize to sum to 1, use absolute IC)
    weights = rolling_ic.abs()
    
    # Normalize row-wise
    row_sums = weights.sum(axis=1)
    weights = weights.div(row_sums, axis=0)
    
    # Fill NaN with equal weights
    weights = weights.fillna(1.0 / len(factors))
    
    return weights


def compute_ml_factor_weights(
    factors: Dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    model_type: str = "ridge",
    alpha: float = 1.0,
    retrain_frequency: int = 20,
    lookback: int = 60,
    lag: int = 1
) -> pd.DataFrame:
    """
    Compute ML-based factor weights using regression.
    
    Args:
        factors: Dictionary of factor DataFrames
        forward_returns: Forward returns DataFrame
        model_type: Type of model (ridge, lasso, elastic_net)
        alpha: Regularization parameter
        retrain_frequency: How often to retrain (in days)
        lookback: Lookback window for training
        lag: Lag for forward returns
    
    Returns:
        DataFrame of factor weights over time
    """
    # Prepare data
    factor_names = list(factors.keys())
    dates = factors[factor_names[0]].index
    
    # Initialize model
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha)
    elif model_type == "elastic_net":
        model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    else:
        model = Ridge(alpha=alpha)
    
    scaler = StandardScaler()
    
    # Store weights
    weights = pd.DataFrame(
        index=dates,
        columns=factor_names,
        dtype=float
    )
    
    # Train and predict in rolling fashion
    for i in range(lookback, len(dates), retrain_frequency):
        train_end = dates[i]
        train_start_idx = max(0, i - lookback)
        train_start = dates[train_start_idx]
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for date in dates[train_start_idx:i]:
            if date not in forward_returns.index:
                continue
            
            # Get forward return
            try:
                forward_date_idx = forward_returns.index.get_loc(date) + lag
                if forward_date_idx >= len(forward_returns):
                    continue
                forward_date = forward_returns.index[forward_date_idx]
            except:
                continue
            
            # Get factor values and returns for all stocks
            factor_values = {}
            for factor_name in factor_names:
                if date in factors[factor_name].index:
                    factor_values[factor_name] = factors[factor_name].loc[date]
            
            if len(factor_values) == 0:
                continue
            
            returns_row = forward_returns.loc[forward_date]
            
            # Align all factors and returns
            common_stocks = returns_row.index
            for fv in factor_values.values():
                common_stocks = common_stocks.intersection(fv.index)
            
            if len(common_stocks) < 10:
                continue
            
            # Stack factor values
            X_row = []
            for factor_name in factor_names:
                if factor_name in factor_values:
                    X_row.extend(factor_values[factor_name][common_stocks].values)
                else:
                    X_row.extend([0] * len(common_stocks))
            
            y_row = returns_row[common_stocks].values
            
            X_list.append(X_row)
            y_list.extend(y_row)
        
        if len(X_list) == 0:
            # Use equal weights
            weights.loc[train_end:] = 1.0 / len(factor_names)
            continue
        
        # Convert to arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Reshape X to be (n_samples * n_stocks, n_factors)
        # Each row in X_list has n_stocks * n_factors elements
        # We need to reshape to (n_samples * n_stocks, n_factors)
        n_factors = len(factor_names)
        n_stocks_per_sample = len(common_stocks) if len(common_stocks) > 0 else 1
        
        if X.shape[1] % n_factors != 0:
            # Fallback: use equal weights
            weights.loc[train_end:] = 1.0 / len(factor_names)
            continue
        
        n_stocks = X.shape[1] // n_factors
        X_reshaped = X.reshape(-1, n_factors)
        
        # Remove rows with all zeros or NaN
        valid = ~(np.isnan(X_reshaped).any(axis=1) | (X_reshaped == 0).all(axis=1))
        X_clean = X_reshaped[valid]
        y_clean = y[valid[:len(y)]]
        
        if len(X_clean) < 10:
            # Use equal weights
            weights.loc[train_end:] = 1.0 / len(factor_names)
            continue
        
        # Scale features
        X_scaled = scaler.fit_transform(X_clean)
        
        # Train model
        try:
            model.fit(X_scaled, y_clean)
            
            # Get factor importance (coefficients)
            coefs = model.coef_
            
            # Normalize to weights (use absolute values, normalize to sum to 1)
            factor_weights = np.abs(coefs)
            if factor_weights.sum() > 0:
                factor_weights = factor_weights / factor_weights.sum()
            else:
                factor_weights = np.ones(n_factors) / n_factors
            
            # Assign weights for this period
            end_idx = min(i + retrain_frequency, len(dates))
            for j in range(i, end_idx):
                for k, factor_name in enumerate(factor_names):
                    if k < len(factor_weights):
                        weights.loc[dates[j], factor_name] = factor_weights[k]
                    else:
                        weights.loc[dates[j], factor_name] = 1.0 / len(factor_names)
        except:
            # Use equal weights on error
            weights.loc[train_end:] = 1.0 / len(factor_names)
    
    # Fill any remaining NaN with equal weights
    weights = weights.fillna(1.0 / len(factor_names))
    
    return weights

