import sys, json, time, io, contextlib
from llm_trading_bot.config import load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from llm_trading_bot.backtesting import BacktestEngine

cfg_path = sys.argv[1] if len(sys.argv)>1 else "config.json"
config = load_config(cfg_path)
configure_cache(config.data_cache.ttl_seconds)
ds = config.data_source
symbol = ds.exchange_symbol if ds.source!="yfinance" else config.trading.yfinance_symbol
t0=time.time()
data = fetch_multi_timeframe(symbol, config.trading.timeframes,
    start_date=config.backtesting.start_date, end_date=config.backtesting.end_date,
    warmup_periods=config.backtesting.warmup_periods, source=ds.source, market=ds.market)
t1=time.time()
for tf,df in data.items():
    print(f"{tf}: {len(df)} rows  {df.index[0]} -> {df.index[-1]}", file=sys.stderr)
engine = BacktestEngine(config)
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    result = engine.run(data, config.trading.primary_timeframe)
t2=time.time()
s=result.stats
print(json.dumps({
 "return_pct": s.total_return_pct, "final_balance": s.final_balance,
 "trades": s.total_trades, "win_rate": s.win_rate, "profit_factor": s.profit_factor,
 "max_dd_pct": s.max_drawdown_pct, "sharpe": s.sharpe_ratio,
 "avg_win": s.avg_win, "avg_loss": s.avg_loss,
 "load_s": round(t1-t0,2), "run_s": round(t2-t1,2)
}, indent=2))
