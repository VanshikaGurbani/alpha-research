# Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Streamlit App (Recommended for beginners)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
Navigate to `http://localhost:8501`

4. **Configure and run:**
   - Select tickers (e.g., "AAPL,MSFT,GOOGL")
   - Choose date range
   - Enable factors
   - Click "Run Analysis"

### Option 2: Docker (Easiest deployment)

1. **Build and run:**
```bash
docker-compose up
```

2. **Access the app:**
Open `http://localhost:8501`

### Option 3: CLI (For automation)

1. **Run with defaults:**
```bash
python cli.py run
```

2. **Run with custom settings:**
```bash
python cli.py run \
  --tickers AAPL,MSFT,GOOGL,AMZN,META \
  --start-date 2020-01-01 \
  --output my_results
```

## 📝 Example Workflows

### Basic Analysis
```bash
# 1. Edit config.yaml to enable factors
# 2. Run CLI
python cli.py run --tickers AAPL,MSFT,GOOGL

# 3. Check results/
ls results/
```

### Advanced Analysis with ML
```bash
# 1. Edit config.yaml:
#    optimization:
#      ml_weighting: true
#      ml_model: "ridge"

# 2. Run
python cli.py run --tickers AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA
```

### Sector-Neutral Portfolio
```bash
# 1. Edit config.yaml:
#    portfolio:
#      sector_neutral: true

# 2. Run (needs sector data from Yahoo Finance)
python cli.py run
```

## 🔧 Common Issues

### Issue: "No module named 'src'"
**Solution:** Make sure you're in the project root directory.

### Issue: Yahoo Finance data fails
**Solution:** 
- Check internet connection
- Try fewer tickers
- Use CSV file instead

### Issue: Docker build fails
**Solution:**
```bash
# Rebuild without cache
docker build --no-cache -t alpha-research .
```

### Issue: Streamlit app is slow
**Solution:**
- Reduce number of tickers
- Use shorter date range
- Disable unused factors

## 📊 Understanding Results

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Cumulative Return**: Total return over period
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable days

### Output Files
- `portfolio_returns.csv`: Daily returns (gross, net, costs)
- `portfolio_weights.csv`: Daily position weights
- `performance_stats.json`: Summary statistics
- `factor_correlations.csv`: Factor relationships

## 🎯 Next Steps

1. **Experiment with factors**: Enable different combinations
2. **Adjust parameters**: Modify lookback windows, quantiles
3. **Add constraints**: Try sector/beta neutral portfolios
4. **Optimize**: Use IC or ML weighting
5. **Analyze**: Review factor correlations and exposures

## 💡 Tips

- Start with 5-10 stocks for faster testing
- Use 1-2 years of data initially
- Enable factors one at a time to understand each
- Check factor correlations to avoid redundancy
- Monitor turnover - high turnover = high costs

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Review `config.yaml` for all options
- Examine `notebooks/alpha_research_demo.ipynb` for examples

