"""
OpenWebUI Financial Data Injection Filter

This is a STANDALONE file designed to be copy-pasted directly into OpenWebUI's
filter environment. It has NO external dependencies on the llm_trading_bot package.

When a user sends a message in OpenWebUI, this filter:
1. Detects if the message is about crypto/market analysis
2. Fetches current OHLCV data from Yahoo Finance
3. Calculates all technical indicators
4. Injects the pre-calculated data into the user message
5. The LLM then sees accurate numbers instead of hallucinating

This is the "Financial Data Injection" concept — the LLM reasons over real data.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these for your setup
# ──────────────────────────────────────────────────────────────────────

FILTER_CONFIG = {
    "enabled": True,
    "symbol": "BTC-USD",
    "timeframes": ["1h", "4h", "1d"],
    "trigger_keywords": [
        "btc", "bitcoin", "crypto", "market", "analysis", "trade",
        "trading", "bullish", "bearish", "long", "short", "price",
        "signal", "setup", "entry", "target", "support", "resistance",
    ],
    "always_inject": False,  # If True, inject data into every message
}


# ──────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS (self-contained — no external imports)
# ──────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series):
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_h, prev_l = high.shift(1), low.shift(1)
    plus_dm = (high - prev_h).clip(lower=0)
    minus_dm = (prev_l - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_val = _atr(high, low, close, period)
    plus_di = 100 * _ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * _ema(minus_dm, period) / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = _ema(dx, period)
    return adx_val, plus_di, minus_di


def _stochastic(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _bollinger(series, period=20, std_dev=2.0):
    mid = _sma(series, period)
    std = series.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std


def _obv(close, volume):
    return (volume * np.sign(close.diff())).cumsum()


def _vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def _pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    return {
        "pivot": pivot,
        "s1": 2 * pivot - high,
        "s2": pivot - (high - low),
        "r1": 2 * pivot - low,
        "r2": pivot + (high - low),
    }


def _aggregate_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df["group"] = df.index.floor("4h")
    agg = df.groupby("group").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    agg.index.name = "Datetime"
    return agg


# ──────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────

def _fetch_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    import yfinance as yf

    yf_interval = "1h" if timeframe in ("1h", "4h") else "1d"
    if yf_interval == "1h":
        days = 90
    else:
        days = 365

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(
        interval=yf_interval,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    if df.empty:
        return df

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if timeframe == "4h":
        df = _aggregate_4h(df)

    return df


# ──────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def _compute_analysis(df: pd.DataFrame, timeframe: str) -> dict:
    """Calculate all indicators and return a dict for the report."""
    if len(df) < 50:
        return {"error": f"Insufficient data ({len(df)} candles)"}

    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    result = {"timeframe": timeframe}

    # Price
    result["price"] = round(float(c.iloc[-1]), 2)
    if len(c) >= 2:
        prev = float(c.iloc[-2])
        result["change_pct"] = round((float(c.iloc[-1]) - prev) / prev * 100, 2)

    # Trend
    result["ema_9"] = round(float(_ema(c, 9).iloc[-1]), 2)
    result["ema_21"] = round(float(_ema(c, 21).iloc[-1]), 2)
    result["ema_50"] = round(float(_ema(c, 50).iloc[-1]), 2)
    if len(c) >= 200:
        result["ema_200"] = round(float(_ema(c, 200).iloc[-1]), 2)

    adx_s, pdi, mdi = _adx(h, l, c)
    if not pd.isna(adx_s.iloc[-1]):
        result["adx"] = round(float(adx_s.iloc[-1]), 1)
        result["plus_di"] = round(float(pdi.iloc[-1]), 1)
        result["minus_di"] = round(float(mdi.iloc[-1]), 1)

    ml, ms, mh = _macd(c)
    result["macd"] = round(float(ml.iloc[-1]), 2)
    result["macd_signal"] = round(float(ms.iloc[-1]), 2)
    result["macd_hist"] = round(float(mh.iloc[-1]), 2)

    # Momentum
    rsi = _rsi(c)
    result["rsi_14"] = round(float(rsi.iloc[-1]), 1)
    sk, sd = _stochastic(h, l, c)
    if not pd.isna(sk.iloc[-1]):
        result["stoch_k"] = round(float(sk.iloc[-1]), 1)
        result["stoch_d"] = round(float(sd.iloc[-1]), 1)

    # Volume
    vol_sma = _sma(v, 20)
    if not pd.isna(vol_sma.iloc[-1]) and float(vol_sma.iloc[-1]) > 0:
        result["volume_ratio"] = round(float(v.iloc[-1]) / float(vol_sma.iloc[-1]), 2)
    obv_s = _obv(c, v)
    obv_sma = _sma(obv_s, 20)
    if not pd.isna(obv_sma.iloc[-1]):
        result["obv_trend"] = "accumulation" if float(obv_s.iloc[-1]) > float(obv_sma.iloc[-1]) else "distribution"

    # Volatility
    atr_s = _atr(h, l, c)
    result["atr_14"] = round(float(atr_s.iloc[-1]), 2)
    result["atr_pct"] = round(float(atr_s.iloc[-1]) / float(c.iloc[-1]) * 100, 2)
    bb_up, bb_mid, bb_low = _bollinger(c)
    if not pd.isna(bb_up.iloc[-1]):
        result["bb_upper"] = round(float(bb_up.iloc[-1]), 2)
        result["bb_middle"] = round(float(bb_mid.iloc[-1]), 2)
        result["bb_lower"] = round(float(bb_low.iloc[-1]), 2)
        bb_range = float(bb_up.iloc[-1]) - float(bb_low.iloc[-1])
        if bb_range > 0:
            result["bb_position"] = round((float(c.iloc[-1]) - float(bb_low.iloc[-1])) / bb_range, 2)

    # CCI (Commodity Channel Index)
    tp = (h + l + c) / 3
    tp_sma = _sma(tp, 20)
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_s = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))
    if not pd.isna(cci_s.iloc[-1]):
        result["cci_20"] = round(float(cci_s.iloc[-1]), 1)

    # ROC (Rate of Change)
    if len(c) > 10:
        roc_val = float((c.iloc[-1] - c.iloc[-11]) / c.iloc[-11] * 100)
        result["roc_10"] = round(roc_val, 2)

    # BB width
    if not pd.isna(bb_up.iloc[-1]):
        bb_w = (float(bb_up.iloc[-1]) - float(bb_low.iloc[-1])) / float(bb_mid.iloc[-1]) * 100
        result["bb_width"] = round(bb_w, 2)

    # S/R from previous candle
    if len(df) >= 2:
        prev = df.iloc[-2]
        pivots = _pivot_points(float(prev["High"]), float(prev["Low"]), float(prev["Close"]))
        result["pivot"] = round(pivots["pivot"], 2)
        result["support_1"] = round(pivots["s1"], 2)
        result["support_2"] = round(pivots["s2"], 2)
        result["resistance_1"] = round(pivots["r1"], 2)
        result["resistance_2"] = round(pivots["r2"], 2)

    # ── 5-Category Weighted Scoring (mirrors scoring.py) ──
    # Weights: trend=0.30, momentum=0.25, volume=0.15, S/R=0.20, risk=0.10
    WEIGHTS = {"trend": 0.30, "momentum": 0.25, "volume": 0.15,
               "support_resistance": 0.20, "risk": 0.10}

    # --- TREND (max ±100) ---
    trend_score = 0.0
    ema9 = result.get("ema_9", 0)
    ema21 = result.get("ema_21", 0)
    ema50 = result.get("ema_50", 0)
    if ema9 > ema21 > ema50:
        trend_score += 30
    elif ema9 < ema21 < ema50:
        trend_score -= 30
    else:
        trend_score += 10 if ema9 > ema21 else -10
    if "ema_200" in result:
        trend_score += 15 if result["price"] > result["ema_200"] else -15
    adx_val = result.get("adx", 0)
    if adx_val > 40:
        adx_mult = 1.0
    elif adx_val > 25:
        adx_mult = 0.7
    elif adx_val > 20:
        adx_mult = 0.4
    else:
        adx_mult = 0.1
    pdi = result.get("plus_di", 0)
    mdi = result.get("minus_di", 0)
    if pdi and mdi:
        trend_score += 20 * adx_mult if pdi > mdi else -20 * adx_mult
    macd_h = result.get("macd_hist", 0)
    trend_score += 15 if macd_h > 0 else -15
    ml_val = result.get("macd", 0)
    ms_val = result.get("macd_signal", 0)
    trend_score += 5 if ml_val > ms_val else -5
    trend_score = max(-100, min(100, trend_score))

    # --- MOMENTUM (max ±100) ---
    mom_score = 0.0
    rsi_val = result.get("rsi_14", 50)
    if rsi_val > 70:
        mom_score -= 20
    elif rsi_val > 60:
        mom_score += 15
    elif rsi_val > 40:
        pass  # neutral
    elif rsi_val > 30:
        mom_score -= 15
    else:
        mom_score += 20
    sk_val = result.get("stoch_k")
    sd_val = result.get("stoch_d")
    if sk_val is not None and sd_val is not None:
        if sk_val > 80:
            mom_score -= 10
        elif sk_val < 20:
            mom_score += 10
        elif sk_val > sd_val:
            mom_score += 10
        else:
            mom_score -= 10
    cci_val = result.get("cci_20")
    if cci_val is not None:
        if cci_val > 100:
            mom_score += 10
        elif cci_val < -100:
            mom_score -= 10
    roc_val = result.get("roc_10")
    if roc_val is not None:
        if roc_val > 5:
            mom_score += 15
        elif roc_val > 0:
            mom_score += 5
        elif roc_val > -5:
            mom_score -= 5
        else:
            mom_score -= 15
    mom_score = max(-100, min(100, mom_score))

    # --- VOLUME (max ±100) ---
    vol_score = 0.0
    vr = result.get("volume_ratio")
    if vr is not None:
        if vr > 2.0:
            vol_score += 30
        elif vr > 1.5:
            vol_score += 20
        elif vr > 1.0:
            vol_score += 5
        elif vr > 0.5:
            vol_score -= 10
        else:
            vol_score -= 25
    chg_pct = result.get("change_pct", 0)
    if vr is not None:
        price_up = chg_pct > 0
        high_vol = vr > 1.0
        if price_up and high_vol:
            vol_score += 20
        elif not price_up and high_vol:
            vol_score -= 20
    obv_t = result.get("obv_trend")
    if obv_t == "accumulation":
        vol_score += 15
    elif obv_t == "distribution":
        vol_score -= 15
    vol_score = max(-100, min(100, vol_score))

    # --- SUPPORT/RESISTANCE (max ±100) ---
    sr_score = 0.0
    price = result.get("price", 0)
    s1 = result.get("support_1")
    r1 = result.get("resistance_1")
    if price and s1 and r1:
        dist_sup = (price - s1) / price * 100
        dist_res = (r1 - price) / price * 100
        if dist_sup > 0:
            sr_ratio = dist_res / dist_sup
        else:
            sr_ratio = 0
        if dist_sup < 1.0:
            sr_score += 25
        elif dist_res < 1.0:
            sr_score -= 25
        if sr_ratio > 3:
            sr_score += 25
        elif sr_ratio > 2:
            sr_score += 15
        elif sr_ratio > 1:
            sr_score += 5
        else:
            sr_score -= 15
    bbp = result.get("bb_position")
    if bbp is not None:
        if bbp > 0.95:
            sr_score -= 15
        elif bbp < 0.05:
            sr_score += 15
    sr_score = max(-100, min(100, sr_score))

    # --- RISK (max ±100) ---
    risk_score = 0.0
    atr_pct = result.get("atr_pct", 0)
    if atr_pct > 8:
        risk_score -= 40
    elif atr_pct > 5:
        risk_score -= 20
    elif atr_pct > 2:
        risk_score += 10
    elif atr_pct > 0.5:
        risk_score += 5
    else:
        risk_score -= 30
    if adx_val < 15:
        risk_score -= 30
    elif adx_val < 20:
        risk_score -= 15
    else:
        risk_score += 10
    bb_w = result.get("bb_width", 5)
    if bb_w < 2:
        risk_score -= 10
    risk_score = max(-100, min(100, risk_score))

    # --- WEIGHTED COMPOSITE ---
    weighted_total = (
        trend_score * WEIGHTS["trend"]
        + mom_score * WEIGHTS["momentum"]
        + vol_score * WEIGHTS["volume"]
        + sr_score * WEIGHTS["support_resistance"]
        + risk_score * WEIGHTS["risk"]
    )
    composite = max(-100, min(100, weighted_total))

    result["category_scores"] = {
        "trend": round(trend_score, 1),
        "momentum": round(mom_score, 1),
        "volume": round(vol_score, 1),
        "support_resistance": round(sr_score, 1),
        "risk": round(risk_score, 1),
    }
    result["composite_score"] = round(composite, 1)
    if composite > 10:
        result["bias"] = "BULLISH"
    elif composite < -10:
        result["bias"] = "BEARISH"
    else:
        result["bias"] = "NEUTRAL"

    return result


def _format_analysis_text(analyses: list[dict]) -> str:
    """Format multi-timeframe analysis into a readable injection block."""
    lines = [
        "=" * 60,
        "FINANCIAL DATA INJECTION — Pre-calculated Technical Analysis",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        "WARNING: These are real calculated values. Do NOT modify or invent numbers.",
        "=" * 60,
    ]

    for a in analyses:
        if "error" in a:
            lines.append(f"\n[{a['timeframe'].upper()}] Error: {a['error']}")
            continue

        tf = a.get("timeframe", "?").upper()
        lines.append(f"\n{'─' * 40}")
        lines.append(f"  {tf} TIMEFRAME")
        lines.append(f"{'─' * 40}")
        lines.append(f"  Price: ${a.get('price', 0):,.2f} ({a.get('change_pct', 0):+.2f}%)")
        comp = a.get('composite_score', 0)
        lines.append(f"  Bias: {a.get('bias', 'N/A')} (composite: {comp:+.1f})")
        cats = a.get("category_scores", {})
        if cats:
            lines.append(f"  Category Scores: trend={cats.get('trend', 0):+.0f}×0.30  "
                         f"momentum={cats.get('momentum', 0):+.0f}×0.25  "
                         f"volume={cats.get('volume', 0):+.0f}×0.15  "
                         f"S/R={cats.get('support_resistance', 0):+.0f}×0.20  "
                         f"risk={cats.get('risk', 0):+.0f}×0.10")

        lines.append(f"\n  Trend:")
        lines.append(f"    EMA 9/21/50: ${a.get('ema_9', 0):,.2f} / ${a.get('ema_21', 0):,.2f} / ${a.get('ema_50', 0):,.2f}")
        if "ema_200" in a:
            lines.append(f"    EMA 200: ${a['ema_200']:,.2f}")
        if "adx" in a:
            lines.append(f"    ADX: {a['adx']:.1f} (+DI: {a.get('plus_di', 0):.1f}, -DI: {a.get('minus_di', 0):.1f})")
        lines.append(f"    MACD: {a.get('macd', 0):.2f} / Signal: {a.get('macd_signal', 0):.2f} / Hist: {a.get('macd_hist', 0):.2f}")

        lines.append(f"\n  Momentum:")
        lines.append(f"    RSI(14): {a.get('rsi_14', 0):.1f}")
        if "stoch_k" in a:
            lines.append(f"    Stochastic K/D: {a['stoch_k']:.1f} / {a['stoch_d']:.1f}")
        if "cci_20" in a:
            lines.append(f"    CCI(20): {a['cci_20']:.1f}")
        if "roc_10" in a:
            lines.append(f"    ROC(10): {a['roc_10']:.2f}%")

        lines.append(f"\n  Volume:")
        if "volume_ratio" in a:
            lines.append(f"    Volume Ratio: {a['volume_ratio']:.2f}x average")
        if "obv_trend" in a:
            lines.append(f"    OBV Trend: {a['obv_trend']}")

        lines.append(f"\n  Volatility:")
        lines.append(f"    ATR(14): ${a.get('atr_14', 0):,.2f} ({a.get('atr_pct', 0):.2f}%)")
        if "bb_upper" in a:
            lines.append(f"    BB: ${a['bb_lower']:,.2f} / ${a['bb_middle']:,.2f} / ${a['bb_upper']:,.2f}")
            if "bb_position" in a:
                lines.append(f"    BB Position: {a['bb_position']:.2f}")

        if "pivot" in a:
            lines.append(f"\n  Support/Resistance:")
            lines.append(f"    Pivot: ${a['pivot']:,.2f}")
            lines.append(f"    S1: ${a['support_1']:,.2f}  S2: ${a['support_2']:,.2f}")
            lines.append(f"    R1: ${a['resistance_1']:,.2f}  R2: ${a['resistance_2']:,.2f}")

    lines.append(f"\n{'=' * 60}")
    lines.append("END FINANCIAL DATA INJECTION")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# OPENWEBUI FILTER CLASS
# ──────────────────────────────────────────────────────────────────────

class Filter:
    """
    OpenWebUI inlet/outlet filter for Financial Data Injection.

    - Inlet: intercepts user messages, detects market-related queries,
      fetches real-time data, calculates indicators, and injects them
      into the message before the LLM sees it.
    - Outlet: optionally formats the response.
    """

    class Valves:
        """Filter configuration exposed in OpenWebUI settings."""
        enabled: bool = True
        symbol: str = "BTC-USD"
        timeframes: str = "1h,4h,1d"  # comma-separated
        always_inject: bool = False
        trigger_keywords: str = "btc,bitcoin,crypto,market,analysis,trade,trading,bullish,bearish,long,short,price,signal"

    def __init__(self):
        self.valves = self.Valves()

    def _should_inject(self, message: str) -> bool:
        """Check if the message warrants data injection."""
        if self.valves.always_inject:
            return True
        if not self.valves.enabled:
            return False
        msg_lower = message.lower()
        keywords = [k.strip() for k in self.valves.trigger_keywords.split(",")]
        return any(kw in msg_lower for kw in keywords)

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Intercept the user message and inject financial data if relevant.
        """
        try:
            messages = body.get("messages", [])
            if not messages:
                return body

            last_message = messages[-1]
            if last_message.get("role") != "user":
                return body

            user_content = last_message.get("content", "")

            if not self._should_inject(user_content):
                return body

            # Fetch and analyze data
            timeframes = [t.strip() for t in self.valves.timeframes.split(",")]
            analyses = []
            for tf in timeframes:
                try:
                    df = _fetch_data(self.valves.symbol, tf)
                    if not df.empty:
                        analysis = _compute_analysis(df, tf)
                        analyses.append(analysis)
                except Exception as e:
                    analyses.append({"timeframe": tf, "error": str(e)})

            if analyses:
                injection = _format_analysis_text(analyses)
                # Prepend the data injection to the user message
                enhanced_content = f"{injection}\n\n---\n\nUser Question: {user_content}"
                messages[-1]["content"] = enhanced_content

            body["messages"] = messages

        except Exception as e:
            # Never break the chat — silently fail
            print(f"[Financial Filter] Error: {e}")

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Post-process the assistant response (optional formatting).
        Currently a pass-through — can be extended for response formatting.
        """
        return body
