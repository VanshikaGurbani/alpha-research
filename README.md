# Multi-Factor Alpha Research Pipeline v2

A production-ready multi-factor alpha research pipeline with real-time data integration, advanced risk modeling, and comprehensive visualization.

## 🚀 Features

### Data & Processing
- **Real-time Yahoo Finance integration** - Fetch live market data
- **De-spike filters** - Remove outliers using rolling z-score method
- **Flexible data sources** - Support for CSV files and Yahoo Finance
- **Universe alignment** - Filter stocks by minimum observations

### Factor Engine
- **7 Factor Types**:
  - Momentum - Price momentum over lookback period
  - Reversal - Mean reversion factor
  - Volatility - Rolling volatility
  - Value - Value factor (inverse price/market cap)
  - Quality - Quality factor (Sharpe-like metric)
  - Liquidity - Amihud illiquidity measure
  - Size - Log market cap
- **Vectorized computation** - Fast, efficient factor calculation
- **Winsorization & Z-scoring** - Robust factor normalization

### Portfolio Construction
- **Long-short portfolios** - Quantile-based selection
- **Sector-neutral constraint** - Neutralize sector exposure
- **Beta-neutral constraint** - Neutralize market beta
- **Leverage limits** - Configurable maximum leverage

### Backtesting
- **Transaction costs** - Configurable basis points per trade
- **Turnover penalties** - Penalize excessive trading
- **Comprehensive metrics**:
  - Sharpe Ratio
  - Cumulative Return
  - Maximum Drawdown
  - Calmar Ratio
  - Win Rate
  - Average Win/Loss

### Risk Management
- **Barra-style risk model** - Factor-based risk decomposition
- **Factor exposures** - Rolling regression-based exposures
- **Portfolio risk calculation** - Volatility from covariance matrix
- **Factor correlation analysis**

### Optimization
- **IC-based weighting** - Information Coefficient weighted factors
- **ML-based weighting** - Ridge/Lasso/ElasticNet regression
- **Rolling optimization** - Time-varying factor weights
- **Configurable retrain frequency**

### Visualization & Interface
- **Streamlit web app** - Interactive dashboard
- **Plotly visualizations**:
  - Portfolio NAV curve
  - Returns decomposition
  - Drawdown analysis
  - Factor correlations
  - Factor time series
- **Export capabilities** - Download results as CSV

### Deployment
- **CLI runner** - Command-line interface with config support
- **Docker support** - Containerized deployment
- **YAML configuration** - Flexible, version-controlled config

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd alpha_research_v2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure settings in `config.yaml`

## 🎯 Quick Start

### Streamlit App (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### CLI Runner

Run the pipeline from command line:

```bash
# Using default config
python cli.py run

# With custom options
python cli.py run --tickers AAPL,MSFT,GOOGL --start-date 2020-01-01 --output results
```

### Docker

Build and run with Docker:

```bash
# Build image
docker build -t alpha-research .

# Run container
docker run -p 8501:8501 alpha-research

# Or use docker-compose
docker-compose up
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Data settings**: Source, date range, universe, de-spike threshold
- **Factor settings**: Enable/disable factors, adjust parameters
- **Portfolio settings**: Quantiles, constraints, leverage
- **Backtest settings**: Transaction costs, turnover penalty
- **Optimization settings**: IC window, ML model type, retrain frequency
- **Risk model settings**: Half-life, minimum periods

Example configuration:

```yaml
data:
  source: "yahoo"
  start_date: "2020-01-01"
  min_obs: 60
  despike_threshold: 5.0

factors:
  momentum:
    lookback: 21
    enabled: true
  value:
    enabled: true

portfolio:
  long_quantile: 0.2
  short_quantile: 0.2
  sector_neutral: true
  beta_neutral: true

backtest:
  transaction_cost_bps: 10
  turnover_penalty: 0.0

optimization:
  ic_window: 60
  ml_weighting: true
  ml_model: "ridge"
```

## 📊 Usage Examples

### Basic Analysis

```python
from src.config_loader import load_config
from src.data_pipeline import fetch_yahoo_data, compute_returns
from src.factor_engine import compute_all_factors, combine_factors
from src.backtest import make_long_short_weights, compute_portfolio_returns

# Load config
config = load_config()

# Fetch data
prices = fetch_yahoo_data(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01')
returns = compute_returns(prices)

# Compute factors
factors = compute_all_factors(prices, returns, config)
combined = combine_factors(factors)

# Build portfolio
weights = make_long_short_weights(combined)
backtest = compute_portfolio_returns(weights, returns)
```

### With Optimization

```python
from src.optimization import compute_rolling_ic_weights

# IC-based weighting
ic_weights = compute_rolling_ic_weights(factors, returns, window=60)
latest_weights = ic_weights.iloc[-1].to_dict()
combined = combine_factors(factors, latest_weights)
```

### With Constraints

```python
from src.backtest import make_long_short_weights
from src.data_pipeline import get_sector_info

# Get sector info
sector_info = get_sector_info(prices.columns.tolist())

# Build sector-neutral portfolio
weights = make_long_short_weights(
    combined,
    sector_neutral=True,
    sector_info=sector_info
)
```

## 📁 Project Structure

```
alpha_research_v2/
├── src/                    # Core modules
│   ├── config_loader.py   # Configuration management
│   ├── data_pipeline.py   # Data loading & processing
│   ├── factor_engine.py   # Factor computation
│   ├── backtest.py        # Portfolio construction & backtesting
│   ├── risk_model.py      # Risk modeling
│   └── optimization.py    # Factor weighting optimization
├── notebooks/             # Original demo notebooks
├── data/                   # Data files
├── app.py                 # Streamlit application
├── cli.py                 # CLI runner
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
└── docker-compose.yml     # Docker Compose configuration
```

## 🔧 Advanced Features

### De-spike Filtering

Automatically removes outliers using rolling z-score:

```python
from src.data_pipeline import despike_prices

prices_clean = despike_prices(prices, threshold=5.0, window=20)
```

### Barra Risk Model

Compute factor-based risk:

```python
from src.risk_model import compute_barra_risk

covariance, exposures = compute_barra_risk(returns, factors, half_life=60)
```

### ML Factor Weighting

Use machine learning to optimize factor weights:

```python
from src.optimization import compute_ml_factor_weights

weights = compute_ml_factor_weights(
    factors, returns,
    model_type="ridge",
    alpha=1.0,
    retrain_frequency=20
)
```

## 📈 Performance Considerations

- **Vectorized operations** - Uses NumPy/Pandas vectorization for speed
- **Efficient data structures** - Optimized DataFrame operations
- **Configurable lookbacks** - Adjust window sizes for performance
- **Caching** - Streamlit app caches results in session state

## 🐳 Docker Deployment

The application is fully containerized:

```bash
# Build
docker build -t alpha-research .

# Run
docker run -p 8501:8501 -v $(pwd)/data:/app/data alpha-research

# Or use compose
docker-compose up -d
```

## 📝 Output Files

The CLI generates the following outputs in the results directory:

- `portfolio_returns.csv` - Daily portfolio returns
- `portfolio_weights.csv` - Daily portfolio weights
- `performance_stats.json` - Performance metrics
- `factor_correlations.csv` - Factor correlation matrix
- `factor_weights.csv` - Factor weights (if optimized)

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Add tests for new features
- Update documentation

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Built with:
- pandas, numpy - Data processing
- yfinance - Market data
- streamlit - Web interface
- plotly - Visualizations
- scikit-learn - Machine learning
