# Changelog

## Version 2.0.0 - Production Release

### Major Features Added

#### Data & Processing
- ✅ Real-time Yahoo Finance integration
- ✅ De-spike filters with configurable thresholds
- ✅ Flexible data source support (CSV + Yahoo Finance)
- ✅ S&P 500 ticker list fetching
- ✅ Sector information retrieval

#### Factor Engine
- ✅ 7 factor types: Momentum, Reversal, Volatility, Value, Quality, Liquidity, Size
- ✅ Vectorized factor computation
- ✅ Winsorization and z-scoring
- ✅ Configurable factor parameters

#### Portfolio Construction
- ✅ Sector-neutral constraint
- ✅ Beta-neutral constraint
- ✅ Configurable leverage limits
- ✅ Quantile-based long-short selection

#### Backtesting
- ✅ Transaction costs (basis points)
- ✅ Turnover penalties
- ✅ Comprehensive performance metrics:
  - Sharpe Ratio
  - Cumulative Return
  - Maximum Drawdown
  - Calmar Ratio
  - Win Rate
  - Average Win/Loss

#### Risk Management
- ✅ Barra-style risk model
- ✅ Factor exposure calculation
- ✅ Portfolio risk computation
- ✅ Factor correlation analysis
- ✅ Exponential-weighted covariance

#### Optimization
- ✅ IC-based factor weighting
- ✅ ML-based factor weighting (Ridge/Lasso/ElasticNet)
- ✅ Rolling optimization windows
- ✅ Configurable retrain frequency

#### User Interface
- ✅ Streamlit web application
- ✅ Interactive visualizations (Plotly):
  - Portfolio NAV curve
  - Returns decomposition
  - Drawdown analysis
  - Factor correlations heatmap
  - Factor time series
  - Turnover analysis
- ✅ Real-time configuration
- ✅ Export capabilities (CSV)

#### Deployment
- ✅ CLI runner with config support
- ✅ Docker containerization
- ✅ Docker Compose setup
- ✅ YAML configuration system
- ✅ Setup.py for package installation

### Technical Improvements
- Vectorized operations for performance
- Robust error handling
- Comprehensive documentation
- Modular architecture
- Type hints throughout

### Files Added
- `src/` - Core modules directory
- `app.py` - Streamlit application
- `cli.py` - Command-line interface
- `config.yaml` - Configuration file
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose config
- `setup.py` - Package setup
- `QUICKSTART.md` - Quick start guide
- `CHANGELOG.md` - This file

### Breaking Changes
- Module structure changed from `notebooks/` to `src/`
- Configuration now required (YAML format)
- API changes in factor computation functions

### Migration Guide
If upgrading from v1:
1. Update imports to use `src.` prefix
2. Create `config.yaml` from template
3. Update any custom factor code
4. Review new optimization options

