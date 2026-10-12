"""Enhanced data pipeline with Yahoo Finance integration and de-spike filters."""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple
from pathlib import Path


def load_prices(path: str) -> pd.DataFrame:
    """Load prices from CSV file."""
    df = pd.read_csv(path)
    # long format
    if set(['date', 'ticker', 'close']).issubset(df.columns):
        df['date'] = pd.to_datetime(df['date'])
        return df.pivot(index='date', columns='ticker', values='close').sort_index()
    # wide format
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0])
    return df.sort_index()


def fetch_yahoo_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    progress: bool = False
) -> pd.DataFrame:
    """
    Fetch price data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), None for today
        progress: Show progress bar
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # Fetch data
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=progress,
        group_by='ticker'
    )
    
    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        # Extract 'Close' prices
        prices = data['Close'].copy()
    else:
        # Single ticker case
        prices = pd.DataFrame({tickers[0]: data['Close']})
    
    prices.index.name = 'date'
    return prices.sort_index()


def get_sp500_tickers() -> List[str]:
    """Get S&P 500 ticker list."""
    try:
        import requests
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()
    except Exception as e:
        # Fallback list - top S&P 500 stocks
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
                'UNH', 'MA', 'PG', 'HD', 'DIS', 'BAC', 'AVGO', 'ADBE', 'NFLX', 'CRM',
                'XOM', 'COST', 'WMT', 'LIN', 'ABBV', 'PEP', 'TMO', 'CSCO', 'ACN', 'DHR']


def despike_returns(
    returns: pd.DataFrame,
    threshold: float = 5.0,
    window: int = 20
) -> pd.DataFrame:
    """
    Remove spikes from returns using rolling z-score method.
    
    Args:
        returns: DataFrame of returns
        threshold: Z-score threshold for spike detection
        window: Rolling window for z-score calculation
    
    Returns:
        DataFrame with spikes replaced by NaN
    """
    returns_clean = returns.copy()
    
    for col in returns.columns:
        ret_series = returns[col]
        rolling_mean = ret_series.rolling(window=window, min_periods=5).mean()
        rolling_std = ret_series.rolling(window=window, min_periods=5).std()
        
        z_scores = (ret_series - rolling_mean) / rolling_std
        
        # Mark spikes
        spikes = np.abs(z_scores) > threshold
        returns_clean.loc[spikes, col] = np.nan
    
    return returns_clean


def despike_prices(
    prices: pd.DataFrame,
    threshold: float = 5.0,
    window: int = 20
) -> pd.DataFrame:
    """
    Remove spikes from prices by converting to returns, despiking, and converting back.
    
    Args:
        prices: DataFrame of prices
        threshold: Z-score threshold for spike detection
        window: Rolling window for z-score calculation
    
    Returns:
        DataFrame with spikes replaced by forward fill
    """
    returns = compute_returns(prices)
    returns_clean = despike_returns(returns, threshold, window)
    
    # Convert back to prices
    prices_clean = prices.copy()
    for col in prices.columns:
        ret_clean = returns_clean[col]
        # Forward fill spikes in returns
        ret_clean = ret_clean.ffill().fillna(0)
        # Reconstruct prices
        prices_clean[col] = prices[col].iloc[0] * np.exp(ret_clean.cumsum())
    
    return prices_clean


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from prices."""
    return np.log(prices / prices.shift(1))


def align_universe(prices: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """Filter stocks with insufficient observations."""
    valid = prices.count(axis=0) >= min_obs
    return prices.loc[:, valid]


def get_sector_info(tickers: List[str]) -> pd.DataFrame:
    """
    Get sector information for tickers from Yahoo Finance.
    
    Returns:
        DataFrame with columns: ticker, sector, industry
    """
    sector_data = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector_data.append({
                'ticker': ticker,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            })
        except:
            sector_data.append({
                'ticker': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown'
            })
    
    return pd.DataFrame(sector_data)

