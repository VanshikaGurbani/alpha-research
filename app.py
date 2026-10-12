"""Streamlit app for Alpha Research Pipeline."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config
from src.data_pipeline import (
    fetch_yahoo_data, get_sp500_tickers, despike_prices,
    compute_returns, align_universe, get_sector_info
)
from src.factor_engine import compute_all_factors, combine_factors
from src.backtest import (
    make_long_short_weights, compute_portfolio_returns,
    performance_stats, compute_portfolio_beta
)
from src.risk_model import factor_correlation_matrix, compute_barra_risk
from src.optimization import compute_rolling_ic_weights, compute_ml_factor_weights

# Page config
st.set_page_config(
    page_title="Alpha Research Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    try:
        st.session_state.config = load_config()
    except:
        st.session_state.config = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'results' not in st.session_state:
    st.session_state.results = {}


def main():
    st.markdown('<div class="main-header">📈 Alpha Research Pipeline</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.selectbox("Data Source", ["Yahoo Finance", "CSV File"])
        
        # Initialize variables
        csv_file = None
        tickers = None
        start_date = None
        end_date = None
        
        if data_source == "Yahoo Finance":
            # Ticker selection
            ticker_input = st.text_input(
                "Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL,AMZN,META"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365*2)
                )
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            
            # Quick select buttons
            st.subheader("Quick Select")
            if st.button("S&P 500 (Sample)"):
                tickers = get_sp500_tickers()[:50]  # Limit for demo
                st.session_state.tickers = tickers
        else:
            csv_file = st.file_uploader("Upload CSV", type=['csv'])
        
        # Factor settings
        st.header("📊 Factors")
        factor_settings = {
            'momentum': st.checkbox("Momentum", value=True),
            'reversal': st.checkbox("Reversal", value=True),
            'volatility': st.checkbox("Volatility", value=True),
            'value': st.checkbox("Value", value=False),
            'quality': st.checkbox("Quality", value=False),
            'liquidity': st.checkbox("Liquidity", value=False),
            'size': st.checkbox("Size", value=False),
        }
        
        # Portfolio settings
        st.header("💼 Portfolio")
        long_quantile = st.slider("Long Quantile", 0.05, 0.5, 0.2, 0.05)
        short_quantile = st.slider("Short Quantile", 0.05, 0.5, 0.2, 0.05)
        sector_neutral = st.checkbox("Sector Neutral", value=False)
        beta_neutral = st.checkbox("Beta Neutral", value=False)
        
        # Backtest settings
        st.header("🔬 Backtest")
        transaction_cost_bps = st.slider("Transaction Cost (bps)", 0, 50, 10)
        turnover_penalty = st.slider("Turnover Penalty", 0.0, 0.01, 0.0, 0.001)
        
        # Optimization
        st.header("🎯 Optimization")
        use_ic_weighting = st.checkbox("IC Weighting", value=False)
        use_ml_weighting = st.checkbox("ML Weighting", value=False)
        
        # Run button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            run_analysis(
                data_source, tickers, start_date, end_date, csv_file,
                factor_settings, long_quantile, short_quantile,
                sector_neutral, beta_neutral, transaction_cost_bps,
                turnover_penalty, use_ic_weighting, use_ml_weighting
            )
    
    # Main content
    if st.session_state.data_loaded:
        display_results()
    else:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to begin.")


def run_analysis(
    data_source, tickers, start_date, end_date, csv_file,
    factor_settings, long_quantile, short_quantile,
    sector_neutral, beta_neutral, transaction_cost_bps,
    turnover_penalty, use_ic_weighting, use_ml_weighting
):
    """Run the full analysis pipeline."""
    
    with st.spinner("Loading data..."):
        # Load data
        if data_source == "Yahoo Finance":
            if not tickers:
                st.error("Please provide ticker symbols")
                return
            
            prices = fetch_yahoo_data(
                tickers,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                progress=False
            )
        else:
            if csv_file is None:
                st.error("Please upload a CSV file")
                return
            prices = pd.read_csv(csv_file)
            if 'date' in prices.columns:
                prices['date'] = pd.to_datetime(prices['date'])
                prices = prices.set_index('date')
        
        if prices.empty:
            st.error("No data loaded. Please check your inputs.")
            return
        
        # Despike
        prices = despike_prices(prices, threshold=5.0)
        
        # Align universe
        prices = align_universe(prices, min_obs=60)
        
        if prices.empty:
            st.error("No stocks meet minimum observation requirements.")
            return
        
        st.success(f"✅ Loaded {len(prices.columns)} stocks with {len(prices)} days of data")
    
    with st.spinner("Computing factors..."):
        # Compute returns
        returns = compute_returns(prices)
        
        # Get sector info if needed
        sector_info = None
        if sector_neutral:
            sector_info = get_sector_info(prices.columns.tolist())
        
        # Get betas if needed
        betas = None
        if beta_neutral:
            market_returns = returns.mean(axis=1)
            betas = compute_portfolio_beta(returns, market_returns)
        
        # Build config dict
        config = {
            'factors': {
                'momentum': {'enabled': factor_settings['momentum'], 'lookback': 21},
                'reversal': {'enabled': factor_settings['reversal'], 'lag': 1},
                'volatility': {'enabled': factor_settings['volatility'], 'window': 21},
                'value': {'enabled': factor_settings['value']},
                'quality': {'enabled': factor_settings['quality']},
                'liquidity': {'enabled': factor_settings['liquidity']},
                'size': {'enabled': factor_settings['size']},
            }
        }
        
        # Compute factors
        factors = compute_all_factors(prices, returns, config)
        
        if not factors:
            st.error("No factors enabled. Please enable at least one factor.")
            return
    
    with st.spinner("Optimizing factor weights..."):
        # Factor weighting
        if use_ic_weighting:
            factor_weights = compute_rolling_ic_weights(factors, returns, window=60)
            # Use latest weights
            latest_weights = factor_weights.iloc[-1].to_dict()
        elif use_ml_weighting:
            factor_weights = compute_ml_factor_weights(factors, returns, model_type="ridge")
            latest_weights = factor_weights.iloc[-1].to_dict()
        else:
            latest_weights = None
        
        # Combine factors
        combined = combine_factors(factors, latest_weights)
    
    with st.spinner("Constructing portfolio..."):
        # Create portfolio
        weights = make_long_short_weights(
            combined,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            sector_neutral=sector_neutral,
            beta_neutral=beta_neutral,
            sector_info=sector_info,
            betas=betas
        )
    
    with st.spinner("Running backtest..."):
        # Backtest
        backtest_results = compute_portfolio_returns(
            weights,
            returns,
            transaction_cost_bps=transaction_cost_bps,
            turnover_penalty=turnover_penalty
        )
        
        # Performance stats
        stats = performance_stats(backtest_results['net_returns'])
        
        # Factor correlations
        factor_corr = factor_correlation_matrix(factors)
    
    # Store results
    st.session_state.results = {
        'prices': prices,
        'returns': returns,
        'factors': factors,
        'combined': combined,
        'weights': weights,
        'backtest_results': backtest_results,
        'stats': stats,
        'factor_corr': factor_corr,
        'sector_info': sector_info,
        'betas': betas
    }
    
    st.session_state.data_loaded = True
    st.success("✅ Analysis complete!")
    st.rerun()


def display_results():
    """Display analysis results."""
    results = st.session_state.results
    
    # Check if results are available
    if not results or 'stats' not in results:
        st.error("No results available. Please run the analysis first.")
        return
    
    # Performance metrics
    st.header("📊 Performance Metrics")
    stats = results.get('stats', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}" if stats['sharpe'] else "N/A")
    with col2:
        st.metric("Cumulative Return", f"{stats['cum_return']*100:.2f}%" if stats['cum_return'] else "N/A")
    with col3:
        st.metric("Max Drawdown", f"{stats['max_drawdown']*100:.2f}%" if stats['max_drawdown'] else "N/A")
    with col4:
        st.metric("Win Rate", f"{stats['win_rate']*100:.2f}%" if stats['win_rate'] else "N/A")
    with col5:
        st.metric("Calmar Ratio", f"{stats['calmar']:.2f}" if stats['calmar'] else "N/A")
    
    # NAV Chart
    st.header("📈 Portfolio NAV")
    if 'backtest_results' in results and 'net_returns' in results['backtest_results']:
        nav = (1 + results['backtest_results']['net_returns'].fillna(0)).cumprod()
        
        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=nav.index,
            y=nav.values,
            mode='lines',
            name='Portfolio NAV',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_nav.update_layout(
            title="Portfolio Net Asset Value",
            xaxis_title="Date",
            yaxis_title="NAV",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_nav, use_container_width=True)
    else:
        st.warning("Backtest results not available for NAV chart.")
    
    # Returns decomposition
    st.header("💰 Returns Decomposition")
    
    if 'backtest_results' in results and results['backtest_results'] is not None:
        backtest = results['backtest_results']
        col1, col2 = st.columns(2)
        
        with col1:
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(
                x=backtest.index,
                y=backtest['gross_returns'].cumsum(),
                mode='lines',
                name='Gross Returns',
                line=dict(color='green')
            ))
            fig_returns.add_trace(go.Scatter(
                x=backtest.index,
                y=backtest['net_returns'].cumsum(),
                mode='lines',
                name='Net Returns',
                line=dict(color='blue')
            ))
            fig_returns.add_trace(go.Scatter(
                x=backtest.index,
                y=-backtest['transaction_costs'].cumsum(),
                mode='lines',
                name='Transaction Costs',
                line=dict(color='red', dash='dash')
            ))
            fig_returns.update_layout(
                title="Cumulative Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=300
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            fig_turnover = go.Figure()
            fig_turnover.add_trace(go.Scatter(
                x=backtest.index,
                y=backtest['turnover'],
                mode='lines',
                name='Turnover',
                line=dict(color='orange')
            ))
            fig_turnover.update_layout(
                title="Portfolio Turnover",
                xaxis_title="Date",
                yaxis_title="Turnover",
                height=300
            )
            st.plotly_chart(fig_turnover, use_container_width=True)
    else:
        st.warning("Backtest results not available for returns decomposition.")
    
    # Factor analysis
    st.header("🔍 Factor Analysis")
    
    # Factor correlations
    st.subheader("Factor Correlations")
    if 'factor_corr' in results and results['factor_corr'] is not None:
        factor_corr = results['factor_corr']
        fig_corr = px.imshow(
            factor_corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Factor Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Factor correlations not available.")
    
    # Factor time series
    st.subheader("Factor Scores Over Time")
    if 'factors' in results:
        factors = results['factors']
        
        factor_ts = pd.DataFrame({
            k: v.mean(axis=1) for k, v in factors.items()
        })
        
        fig_factors = go.Figure()
        for col in factor_ts.columns:
            fig_factors.add_trace(go.Scatter(
                x=factor_ts.index,
                y=factor_ts[col],
                mode='lines',
                name=col
            ))
        fig_factors.update_layout(
            title="Average Factor Scores",
            xaxis_title="Date",
            yaxis_title="Factor Score",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_factors, use_container_width=True)
    else:
        st.warning("Factor data not available.")
    
    # Portfolio composition
    st.header("💼 Portfolio Composition")
    
    if 'weights' in results and not results['weights'].empty:
        # Latest weights
        latest_weights = results['weights'].iloc[-1]
        long_positions = latest_weights[latest_weights > 0].sort_values(ascending=False)
        short_positions = latest_weights[latest_weights < 0].sort_values()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Long Positions")
            if len(long_positions) > 0:
                st.dataframe(
                    pd.DataFrame({
                        'Ticker': long_positions.index[:10],
                        'Weight': long_positions.values[:10]
                    }),
                    use_container_width=True
                )
            else:
                st.info("No long positions")
        
        with col2:
            st.subheader("Top Short Positions")
            if len(short_positions) > 0:
                st.dataframe(
                    pd.DataFrame({
                        'Ticker': short_positions.index[:10],
                        'Weight': short_positions.values[:10]
                    }),
                    use_container_width=True
                )
            else:
                st.info("No short positions")
    else:
        st.warning("Portfolio weights not available.")
    
    # Drawdown chart
    st.header("📉 Drawdown Analysis")
    if 'backtest_results' in results and 'net_returns' in results['backtest_results']:
        nav = (1 + results['backtest_results']['net_returns'].fillna(0)).cumprod()
        peak = nav.cummax()
        drawdown = (nav / peak - 1) * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        fig_dd.update_layout(
            title="Portfolio Drawdown (%)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.warning("Backtest results not available for drawdown analysis.")
    
    # Download results
    st.header("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'backtest_results' in results:
            csv_returns = results['backtest_results'].to_csv()
            st.download_button(
                label="Download Returns",
                data=csv_returns,
                file_name="portfolio_returns.csv",
                mime="text/csv"
            )
        else:
            st.info("Returns not available")
    
    with col2:
        if 'weights' in results and not results['weights'].empty:
            csv_weights = results['weights'].to_csv()
            st.download_button(
                label="Download Weights",
                data=csv_weights,
                file_name="portfolio_weights.csv",
                mime="text/csv"
            )
        else:
            st.info("Weights not available")
    
    with col3:
        if stats:
            csv_stats = pd.DataFrame([stats]).to_csv(index=False)
            st.download_button(
                label="Download Stats",
                data=csv_stats,
                file_name="performance_stats.csv",
                mime="text/csv"
            )
        else:
            st.info("Stats not available")


if __name__ == "__main__":
    main()

