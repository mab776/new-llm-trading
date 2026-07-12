import sys, json, time
from llm_trading_bot.config import load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
import opt.fastbt as fb

config = load_config("config.json")
configure_cache(config.data_cache.ttl_seconds)
ds = config.data_source
symbol = ds.exchange_symbol
data = fetch_multi_timeframe(symbol, config.trading.timeframes,
    start_date=config.backtesting.start_date, end_date=config.backtesting.end_date,
    warmup_periods=config.backtesting.warmup_periods, source=ds.source, market=ds.market)
t0=time.time()
pre = fb.precompute(data, config.trading.primary_timeframe, config.backtesting.warmup_periods)
t1=time.time()
r = fb.simulate(pre, config, config.backtesting.start_date, config.backtesting.end_date)
t2=time.time()
print(json.dumps({"return_pct": r.return_pct, "trades": r.trades, "win_rate": round(r.win_rate,2),
  "profit_factor": r.profit_factor, "max_dd_pct": r.max_dd_pct, "sharpe": r.sharpe,
  "precompute_s": round(t1-t0,2), "sim_s": round(t2-t1,3)}, indent=2))
print("ENGINE BASELINE: return 12.56, trades 75, win_rate 29.33, pf 1.51, max_dd 10.95, sharpe 1.92", file=sys.stderr)
