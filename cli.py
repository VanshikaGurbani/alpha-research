"""CLI runner for Alpha Research Pipeline."""

import click
import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config
from src.data_pipeline import (
    fetch_yahoo_data, load_prices, despike_prices,
    compute_returns, align_universe
)
from src.factor_engine import compute_all_factors, combine_factors
from src.backtest import (
    make_long_short_weights, compute_portfolio_returns,
    performance_stats
)
from src.risk_model import factor_correlation_matrix
from src.optimization import compute_rolling_ic_weights, compute_ml_factor_weights


@click.group()
def cli():
    """Alpha Research Pipeline CLI."""
    pass


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Config file path')
@click.option('--output', '-o', default='results', help='Output directory')
@click.option('--tickers', '-t', help='Comma-separated ticker list (overrides config)')
@click.option('--start-date', help='Start date YYYY-MM-DD (overrides config)')
@click.option('--end-date', help='End date YYYY-MM-DD (overrides config)')
def run(config, output, tickers, start_date, end_date):
    """Run the full alpha research pipeline."""
    click.echo("Loading configuration...")
    cfg = load_config(config)
    
    # Override config with CLI args
    if tickers:
        cfg['data']['universe'] = [t.strip() for t in tickers.split(',')]
    if start_date:
        cfg['data']['start_date'] = start_date
    if end_date:
        cfg['data']['end_date'] = end_date
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    click.echo("Loading data...")
    if cfg['data']['source'] == 'yahoo':
        ticker_list = cfg['data'].get('universe', [])
        if not ticker_list:
            click.echo("No tickers specified. Using sample tickers.")
            ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        prices = fetch_yahoo_data(
            ticker_list,
            cfg['data']['start_date'],
            cfg['data'].get('end_date'),
            progress=True
        )
    else:
        prices = load_prices(cfg['data']['csv_path'])
    
    # Despike
    click.echo("Applying de-spike filter...")
    prices = despike_prices(prices, threshold=cfg['data']['despike_threshold'])
    
    # Align universe
    prices = align_universe(prices, min_obs=cfg['data']['min_obs'])
    click.echo(f"Universe: {len(prices.columns)} stocks, {len(prices)} days")
    
    # Compute returns
    returns = compute_returns(prices)
    
    # Compute factors
    click.echo("Computing factors...")
    factors = compute_all_factors(prices, returns, cfg)
    click.echo(f"Computed {len(factors)} factors: {list(factors.keys())}")
    
    # Factor weighting
    factor_weights = None
    opt_config = cfg.get('optimization', {})
    
    if opt_config.get('ml_weighting', False):
        click.echo("Computing ML-based factor weights...")
        factor_weights_df = compute_ml_factor_weights(
            factors,
            returns,
            model_type=opt_config.get('ml_model', 'ridge'),
            retrain_frequency=opt_config.get('ml_retrain_frequency', 20)
        )
        factor_weights = factor_weights_df.iloc[-1].to_dict()
        click.echo(f"ML weights: {factor_weights}")
    elif opt_config.get('ic_window', None):
        click.echo("Computing IC-based factor weights...")
        factor_weights_df = compute_rolling_ic_weights(
            factors,
            returns,
            window=opt_config.get('ic_window', 60)
        )
        factor_weights = factor_weights_df.iloc[-1].to_dict()
        click.echo(f"IC weights: {factor_weights}")
    
    # Combine factors
    combined = combine_factors(factors, factor_weights)
    
    # Portfolio construction
    click.echo("Constructing portfolio...")
    portfolio_config = cfg.get('portfolio', {})
    weights = make_long_short_weights(
        combined,
        long_quantile=portfolio_config.get('long_quantile', 0.2),
        short_quantile=portfolio_config.get('short_quantile', 0.2),
        sector_neutral=portfolio_config.get('sector_neutral', False),
        beta_neutral=portfolio_config.get('beta_neutral', False)
    )
    
    # Backtest
    click.echo("Running backtest...")
    backtest_config = cfg.get('backtest', {})
    backtest_results = compute_portfolio_returns(
        weights,
        returns,
        transaction_cost_bps=backtest_config.get('transaction_cost_bps', 10),
        turnover_penalty=backtest_config.get('turnover_penalty', 0.0)
    )
    
    # Performance stats
    stats = performance_stats(
        backtest_results['net_returns'],
        annualization=backtest_config.get('annualization', 252)
    )
    
    # Factor correlations
    factor_corr = factor_correlation_matrix(factors)
    
    # Save results
    click.echo("Saving results...")
    
    # Save returns
    backtest_results.to_csv(output_dir / 'portfolio_returns.csv')
    
    # Save weights
    weights.to_csv(output_dir / 'portfolio_weights.csv')
    
    # Save stats
    with open(output_dir / 'performance_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save factor correlations
    factor_corr.to_csv(output_dir / 'factor_correlations.csv')
    
    # Save factor weights if available
    if factor_weights:
        pd.Series(factor_weights).to_csv(output_dir / 'factor_weights.csv')
    
    # Print summary
    click.echo("\n" + "="*50)
    click.echo("PERFORMANCE SUMMARY")
    click.echo("="*50)
    click.echo(f"Sharpe Ratio: {stats['sharpe']:.2f}" if stats['sharpe'] else "Sharpe Ratio: N/A")
    click.echo(f"Cumulative Return: {stats['cum_return']*100:.2f}%" if stats['cum_return'] else "Cumulative Return: N/A")
    click.echo(f"Max Drawdown: {stats['max_drawdown']*100:.2f}%" if stats['max_drawdown'] else "Max Drawdown: N/A")
    click.echo(f"Win Rate: {stats['win_rate']*100:.2f}%" if stats['win_rate'] else "Win Rate: N/A")
    click.echo("="*50)
    click.echo(f"\nResults saved to: {output_dir.absolute()}")


@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Config file path')
def validate(config):
    """Validate configuration file."""
    try:
        cfg = load_config(config)
        click.echo(f"✅ Configuration file '{config}' is valid")
        click.echo(f"Data source: {cfg['data']['source']}")
        click.echo(f"Factors enabled: {sum(1 for f in cfg['factors'].values() if f.get('enabled', False))}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_factors():
    """List available factors."""
    factors = [
        "momentum - Price momentum over lookback period",
        "reversal - Mean reversion factor",
        "volatility - Rolling volatility",
        "value - Value factor (inverse price)",
        "quality - Quality factor (Sharpe-like)",
        "liquidity - Liquidity factor (Amihud)",
        "size - Size factor (log market cap)"
    ]
    for factor in factors:
        click.echo(f"  • {factor}")


if __name__ == '__main__':
    cli()

