"""Enhanced backtest engine with constraints, transaction costs, and turnover penalties."""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from scipy.optimize import minimize


def compute_portfolio_beta(
    returns: pd.DataFrame,
    market_returns: Optional[pd.Series] = None
) -> pd.Series:
    """
    Compute portfolio beta for each stock.
    If market_returns not provided, uses equal-weighted market.
    """
    if market_returns is None:
        market_returns = returns.mean(axis=1)
    
    betas = pd.Series(index=returns.columns, dtype=float)
    
    for stock in returns.columns:
        stock_ret = returns[stock]
        valid = stock_ret.notna() & market_returns.notna()
        
        if valid.sum() < 20:
            betas[stock] = 1.0  # Default beta
            continue
        
        stock_clean = stock_ret[valid]
        market_clean = market_returns[valid]
        
        if market_clean.std() > 1e-8:
            beta = np.cov(stock_clean, market_clean)[0, 1] / np.var(market_clean)
            betas[stock] = beta
        else:
            betas[stock] = 1.0
    
    return betas


def make_long_short_weights(
    factor: pd.DataFrame,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    sector_neutral: bool = False,
    beta_neutral: bool = False,
    sector_info: Optional[pd.DataFrame] = None,
    betas: Optional[pd.Series] = None,
    max_leverage: float = 1.0
) -> pd.DataFrame:
    """
    Create long-short portfolio weights with optional constraints.
    
    Args:
        factor: Factor scores DataFrame
        long_quantile: Top quantile for longs
        short_quantile: Bottom quantile for shorts
        sector_neutral: Whether to enforce sector neutrality
        beta_neutral: Whether to enforce beta neutrality
        sector_info: DataFrame with columns: ticker, sector
        betas: Series of stock betas
        max_leverage: Maximum leverage (sum of absolute weights)
    
    Returns:
        DataFrame of portfolio weights
    """
    def _w(row):
        if row.isna().all():
            return row * 0
        
        # Initial weights based on quantiles
        q_long = row.quantile(1 - long_quantile)
        q_short = row.quantile(short_quantile)
        w = pd.Series(0.0, index=row.index)
        w[row >= q_long] = 1.0
        w[row <= q_short] = -1.0
        
        # Normalize
        if w.abs().sum() > 0:
            w = w / w.abs().sum()
        
        # Apply constraints if needed
        if sector_neutral and sector_info is not None:
            w = _apply_sector_neutrality(w, sector_info)
        
        if beta_neutral and betas is not None:
            w = _apply_beta_neutrality(w, betas)
        
        # Apply leverage constraint
        if w.abs().sum() > max_leverage:
            w = w / w.abs().sum() * max_leverage
        
        return w
    
    return factor.apply(_w, axis=1)


def _apply_sector_neutrality(weights: pd.Series, sector_info: pd.DataFrame) -> pd.Series:
    """Apply sector-neutral constraint to weights."""
    w = weights.copy()
    
    # Get sector for each stock
    sector_map = dict(zip(sector_info['ticker'], sector_info['sector']))
    
    # Group by sector
    sector_weights = {}
    for stock in w.index:
        sector = sector_map.get(stock, 'Unknown')
        if sector not in sector_weights:
            sector_weights[sector] = []
        sector_weights[sector].append(stock)
    
    # Make each sector neutral
    for sector, stocks in sector_weights.items():
        sector_w = w[stocks]
        if sector_w.abs().sum() > 0:
            # Normalize sector to be neutral
            sector_long = sector_w[sector_w > 0].sum()
            sector_short = sector_w[sector_w < 0].sum()
            
            if sector_long > 0 and sector_short < 0:
                # Scale to make neutral
                scale = min(abs(sector_long), abs(sector_short)) / max(abs(sector_long), abs(sector_short))
                w[stocks] = sector_w * scale
    
    return w


def _apply_beta_neutrality(weights: pd.Series, betas: pd.Series) -> pd.Series:
    """Apply beta-neutral constraint to weights."""
    w = weights.copy()
    
    # Compute portfolio beta
    portfolio_beta = (w * betas).sum()
    
    if abs(portfolio_beta) > 0.01:  # Threshold for neutrality
        # Adjust weights to neutralize beta
        # Find adjustment factor
        beta_long = (w[w > 0] * betas[w > 0]).sum()
        beta_short = (w[w < 0] * betas[w < 0]).sum()
        
        if beta_long > 0 and beta_short < 0:
            # Scale to neutralize
            total_beta = beta_long + beta_short
            if abs(total_beta) > 1e-8:
                adjustment = -total_beta / (beta_long - beta_short) if (beta_long - beta_short) != 0 else 0
                w = w * (1 + adjustment)
    
    return w


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute portfolio turnover (sum of absolute weight changes)."""
    turnover = pd.Series(index=weights.index, dtype=float)
    
    for i in range(1, len(weights)):
        prev_w = weights.iloc[i-1]
        curr_w = weights.iloc[i]
        
        # Align indices
        common = prev_w.index.intersection(curr_w.index)
        if len(common) == 0:
            turnover.iloc[i] = 0.0
            continue
        
        prev_aligned = prev_w[common].fillna(0)
        curr_aligned = curr_w[common].fillna(0)
        
        # Turnover = sum of absolute changes
        turnover.iloc[i] = (curr_aligned - prev_aligned).abs().sum()
    
    turnover.iloc[0] = weights.iloc[0].abs().sum()  # Initial turnover
    return turnover


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    turnover_penalty: float = 0.0
) -> pd.DataFrame:
    """
    Compute portfolio returns with transaction costs and turnover penalty.
    
    Args:
        weights: Portfolio weights DataFrame
        returns: Stock returns DataFrame
        transaction_cost_bps: Transaction cost in basis points per trade
        turnover_penalty: Penalty coefficient for turnover
    
    Returns:
        DataFrame with columns: gross_returns, transaction_costs, turnover, net_returns
    """
    # Align weights with returns (shift by 1 for forward-looking)
    aligned_weights = weights.shift(1).reindex_like(returns)
    
    # Gross returns
    gross_returns = (aligned_weights * returns).sum(axis=1)
    
    # Compute turnover
    turnover = compute_turnover(aligned_weights)
    
    # Transaction costs (as fraction, not bps)
    transaction_costs = turnover * (transaction_cost_bps / 10000.0)
    
    # Turnover penalty
    turnover_penalty_cost = turnover * turnover_penalty
    
    # Net returns
    net_returns = gross_returns - transaction_costs - turnover_penalty_cost
    
    result = pd.DataFrame({
        'gross_returns': gross_returns,
        'transaction_costs': transaction_costs,
        'turnover': turnover,
        'net_returns': net_returns
    })
    
    return result


def _max_dd(nav: pd.Series) -> float:
    """Compute maximum drawdown."""
    peak = nav.cummax()
    dd = nav / peak - 1
    return dd.min()


def performance_stats(
    ret: pd.Series,
    annualization: int = 252
) -> Dict[str, float]:
    """
    Compute comprehensive performance statistics.
    
    Returns:
        Dictionary of performance metrics
    """
    r = ret.dropna()
    if len(r) == 0:
        return {
            "mean_daily": None,
            "vol_daily": None,
            "sharpe": None,
            "cum_return": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None
        }
    
    mu = r.mean()
    vol = r.std()
    sharpe = mu / vol * (annualization ** 0.5) if vol > 0 else None
    
    cum = (1 + r).prod() - 1
    nav = (1 + r).cumprod()
    mdd = _max_dd(nav)
    calmar = cum / abs(mdd) if mdd != 0 else None
    
    # Win rate
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = len(wins) / len(r) if len(r) > 0 else None
    avg_win = wins.mean() if len(wins) > 0 else None
    avg_loss = losses.mean() if len(losses) > 0 else None
    
    return {
        "mean_daily": mu,
        "vol_daily": vol,
        "sharpe": sharpe,
        "cum_return": cum,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }

