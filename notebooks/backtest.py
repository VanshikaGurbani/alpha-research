
import pandas as pd
import numpy as np

def make_long_short_weights(factor, long_quantile=0.2, short_quantile=0.2):
    def _w(row):
        if row.isna().all():
            return row * 0
        q_long = row.quantile(1 - long_quantile)
        q_short = row.quantile(short_quantile)
        w = pd.Series(0.0, index=row.index)
        w[row >= q_long] = 1.0
        w[row <= q_short] = -1.0
        if w.abs().sum() > 0:
            w = w / w.abs().sum()
        return w
    return factor.apply(_w, axis=1)

def compute_portfolio_returns(weights, returns):
    aligned = weights.shift(1).reindex_like(returns)
    return (aligned * returns).sum(axis=1)

def _max_dd(nav):
    peak = nav.cummax()
    dd = nav/peak - 1
    return dd.min()

def performance_stats(ret, annualization=252):
    r = ret.dropna()
    if len(r) == 0:
        return {"mean_daily": None, "vol_daily": None, "sharpe": None,
                "cum_return": None, "max_drawdown": None}
    mu = r.mean()
    vol = r.std()
    sharpe = mu/vol * (annualization**0.5) if vol > 0 else None
    cum = (1 + r).prod() - 1
    nav = (1 + r).cumprod()
    mdd = _max_dd(nav)
    return dict(mean_daily=mu, vol_daily=vol, sharpe=sharpe,
                cum_return=cum, max_drawdown=mdd)
