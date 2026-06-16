"""
UG Checklist v6 — Web App (Flask)
3 điều kiện:
  DK1: Xu hướng PA pivot + EMA backup
  DK2: Vùng KC/HT + R:R >= 2.0 (gộp lại)
  DK3: Nến đảo chiều (tất cả dùng nến đã đóng [1])
Score: /3
"""
import os, warnings, logging, secrets
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from datetime import date, timedelta
import pandas as pd
import numpy as np

app = Flask(__name__)

_secret = os.environ.get("SECRET_KEY")
if not _secret:
    log.warning("⚠️  SECRET_KEY chưa set → session mất khi restart!")
    _secret = secrets.token_hex(32)
app.secret_key = _secret
app.permanent_session_lifetime = timedelta(hours=12)

from auth import login_required, do_login

# ═══════════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════════
def safe(v):
    if v is None: return None
    if isinstance(v, bool): return bool(v)
    try:
        if pd.isna(v): return None
    except: pass
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating):
        return None if (np.isinf(v) or np.isnan(v)) else float(v)
    if isinstance(v, float):
        return None if (np.isinf(v) or np.isnan(v)) else v
    return v

# ═══════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════
def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def find_pivot_high(series, left, right):
    result = pd.Series(np.nan, index=series.index)
    arr = series.values
    for i in range(left, len(arr) - right):
        w = arr[i-left: i+right+1]
        if arr[i] == w.max() and arr[i] > arr[i-1]:
            result.iloc[i] = arr[i]
    return result

def find_pivot_low(series, left, right):
    result = pd.Series(np.nan, index=series.index)
    arr = series.values
    for i in range(left, len(arr) - right):
        w = arr[i-left: i+right+1]
        if arr[i] == w.min() and arr[i] < arr[i-1]:
            result.iloc[i] = arr[i]
    return result

# ═══════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════
def _normalize(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    rename = {"time":"date","tradingdate":"date","open":"open",
              "high":"high","low":"low","close":"close","volume":"volume"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if "date" not in df.columns:
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date","open","high","low","close","volume"]].dropna(subset=["close"])

def get_data(symbol, start, end=None, timeframe="1D"):
    end = end or date.today().strftime("%Y-%m-%d")
    interval = {"1D":"1D","1W":"1W"}.get(timeframe,"1D")

    try:
        from vnstock import stock_historical_data
        import inspect
        sig = inspect.signature(stock_historical_data)
        df = (stock_historical_data(symbol, start, end, resolution=interval, type="stock")
              if "resolution" in sig.parameters else
              stock_historical_data(symbol, start, end, interval=interval, asset_type="stock"))
        df = _normalize(df)
        if not df.empty:
            log.info(f"vnstock OK: {symbol} {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"vnstock: {e}")

    try:
        from vnstock3 import Vnstock
        df = Vnstock().stock(symbol=symbol, source="VCI").quote.history(
            start=start, end=end, interval=interval)
        df = _normalize(df)
        if not df.empty:
            log.info(f"vnstock3 OK: {symbol} {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"vnstock3: {e}")

    try:
        import yfinance as yf, socket
        socket.setdefaulttimeout(20)
        ticker = symbol+".VN" if not symbol.endswith(".VN") else symbol
        df = _normalize(yf.download(ticker, start=start, end=end,
                                    auto_adjust=True, progress=False,
                                    timeout=15).reset_index())
        if not df.empty:
            log.info(f"yfinance OK: {ticker} {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"yfinance: {e}")

    raise RuntimeError(f"Không lấy được data cho '{symbol}'.")

def get_intraday_h4(symbol, start, end=None):
    """Lấy data khung H4 cho DK3. Thử yfinance 4h → vnstock 1H gộp → fallback D."""
    end = end or date.today().strftime("%Y-%m-%d")

    # 1. yfinance 4h (đúng khung H4)
    try:
        import yfinance as yf, socket
        socket.setdefaulttimeout(20)
        ticker = symbol+".VN" if not symbol.endswith(".VN") else symbol
        df = yf.download(ticker, period="60d", interval="4h",
                         progress=False, timeout=15)
        df = _normalize(df.reset_index())
        if not df.empty and len(df) > 20:
            log.info(f"H4 yfinance OK: {symbol} {len(df)} nến")
            return df, "H4"
    except Exception as e:
        log.warning(f"H4 yfinance: {e}")

    # 2. vnstock 1H rồi gộp 4 nến thành H4
    try:
        from vnstock import stock_historical_data
        import inspect
        sig = inspect.signature(stock_historical_data)
        df = (stock_historical_data(symbol, start, end, resolution="1H", type="stock")
              if "resolution" in sig.parameters else
              stock_historical_data(symbol, start, end, interval="1H", asset_type="stock"))
        df = _normalize(df)
        if not df.empty and len(df) > 20:
            # Gộp 1H → 4H: lấy mỗi 4 nến
            df = df.set_index("date")
            agg = df.resample("4H").agg({"open":"first","high":"max",
                                          "low":"min","close":"last","volume":"sum"}).dropna()
            agg = agg.reset_index()
            if not agg.empty:
                log.info(f"H4 vnstock(1H→4H) OK: {symbol} {len(agg)} nến")
                return agg, "H4(1H)"
    except Exception as e:
        log.warning(f"H4 vnstock 1H: {e}")

    # 3. Fallback: dùng khung ngày
    try:
        df = get_data(symbol, start, end, timeframe="1D")
        if not df.empty:
            log.info(f"H4 fallback Daily: {symbol} {len(df)} nến")
            return df, "D(fallback)"
    except Exception as e:
        log.warning(f"H4 fallback: {e}")

    return pd.DataFrame(), "N/A"

# ═══════════════════════════════════════════════════════════════
#  CHECKLIST v6
# ═══════════════════════════════════════════════════════════════
def run_checklist(df_ltf, df_htf, cfg, df_entry=None):
    SWING   = cfg["swing_len"]
    ZLEN    = cfg["zone_len"]
    ZATR    = cfg["zone_atr_mult"]
    BODY    = cfg["body_ratio"]
    WICK    = cfg["wick_ratio"]
    ATRM    = cfg["atr_mult"]
    UVOL    = cfg["use_volume"]
    VMALEN  = cfg["vol_ma_len"]
    VSMULT  = cfg["vol_sig_mult"]
    VFMULT  = cfg["vol_fb_mult"]
    URSIDIV = cfg["use_rsi_div"]
    RSILEN  = cfg["rsi_len"]
    RR_MIN  = cfg["rr_min"]
    ACC     = cfg["account"]
    RISKP   = cfg["risk_pct"]
    SLP     = cfg["sl_pct"]

    df = df_ltf.copy()

    # ── Candle helpers (dùng nến đã đóng shift(1)) ────────────
    df["atr14"]   = calc_atr(df, 14)
    df["body"]    = (df["close"] - df["open"]).abs()
    df["range"]   = df["high"] - df["low"]
    df["wick_up"] = df["high"] - df[["open","close"]].max(axis=1)
    df["wick_dn"] = df[["open","close"]].min(axis=1) - df["low"]
    df["is_bull"] = df["close"] > df["open"]
    df["is_bear"] = df["close"] < df["open"]
    rng = df["range"].replace(0, np.nan)
    df["big_body"] = (df["range"] > 0) & (df["body"] / rng >= BODY)
    df["lw_up"]    = (df["range"] > 0) & (df["wick_up"] / rng >= WICK)
    df["lw_dn"]    = (df["range"] > 0) & (df["wick_dn"] / rng >= WICK)
    df["sig_big"]  = df["range"] >= df["atr14"] * ATRM
    df["vol_ma"]   = df["volume"].rolling(VMALEN).mean()

    # v6: volume check trên nến đã đóng shift(1)
    df["vol_sig_ok"] = (~UVOL) | (df["volume"].shift(1) >= df["vol_ma"].shift(1) * VSMULT)
    df["vol_fb_ok"]  = (~UVOL) | (df["volume"].shift(1) <= df["vol_ma"].shift(1) * VFMULT)
    # backward compat cho bottom bar
    df["vol_ok"] = df["vol_sig_ok"]

    # ── DK1: Xu hướng ─────────────────────────────────────────
    htf = df_htf.copy()
    htf["ph"]    = find_pivot_high(htf["high"], SWING, SWING)
    htf["pl"]    = find_pivot_low(htf["low"],   SWING, SWING)
    htf["ema20"] = calc_ema(htf["close"], 20)
    htf["ema50"] = calc_ema(htf["close"], 50)

    ph_vals = htf["ph"].dropna().values
    pl_vals = htf["pl"].dropna().values
    last_ph = ph_vals[-1] if len(ph_vals) >= 1 else np.nan
    prev_ph = ph_vals[-2] if len(ph_vals) >= 2 else np.nan
    last_pl = pl_vals[-1] if len(pl_vals) >= 1 else np.nan
    prev_pl = pl_vals[-2] if len(pl_vals) >= 2 else np.nan

    def ok(x): return not (isinstance(x, float) and np.isnan(x))
    pa_up = all(ok(x) for x in [last_ph,prev_ph,last_pl,prev_pl]) and last_ph>prev_ph and last_pl>prev_pl
    pa_dn = all(ok(x) for x in [last_ph,prev_ph,last_pl,prev_pl]) and last_ph<prev_ph and last_pl<prev_pl

    htf_last = htf.iloc[-1]
    e20, e50, hc = htf_last["ema20"], htf_last["ema50"], htf_last["close"]
    ema_up = not any(np.isnan(x) for x in [e20,e50,hc]) and e20>e50 and hc>e20
    ema_dn = not any(np.isnan(x) for x in [e20,e50,hc]) and e20<e50 and hc<e20

    is_up  = pa_up or (not pa_up and not pa_dn and ema_up)
    is_dn  = pa_dn or (not pa_up and not pa_dn and ema_dn)
    is_sw  = not is_up and not is_dn
    trend_src = "PA" if (pa_up or pa_dn) else "EMA"

    df["is_uptrend"]   = is_up
    df["is_downtrend"] = is_dn
    df["is_sideways"]  = is_sw
    df["trend_src"]    = trend_src
    df["cond1_ok"]     = is_up or is_dn

    # ── DK2: Vùng KC/HT + R:R ─────────────────────────────────
    df["resist"]  = df["high"].shift(1).rolling(ZLEN).max()
    df["support"] = df["low"].shift(1).rolling(ZLEN).min()
    df["zone_w"]  = df["atr14"] * ZATR

    # v6: tiếp cận đúng chiều
    df["in_resist"]  = ((df["close"] >= df["resist"] - df["zone_w"]) &
                        (df["close"] <= df["resist"]))
    df["in_support"] = ((df["close"] <= df["support"] + df["zone_w"]) &
                        (df["close"] >= df["support"]))
    df["near_resist"] = df["in_resist"]
    df["near_support"] = df["in_support"]

    df["cond2_zone"] = ((is_up & df["in_support"]) |
                        (is_dn & df["in_resist"])  |
                        (is_sw & (df["in_support"] | df["in_resist"])))

    # v6: R:R dùng nến đã đóng [1] làm SL
    # LONG: SL=low[1], TP=resist
    sl_d_up = df["close"].shift(1) - df["low"].shift(1)
    tp_d_up = df["resist"] - df["close"].shift(1)
    rr_up   = (tp_d_up / sl_d_up.replace(0, np.nan)).clip(lower=0)

    # SHORT: SL=high[1], TP=support
    sl_d_dn = df["high"].shift(1) - df["close"].shift(1)
    tp_d_dn = df["close"].shift(1) - df["support"]
    rr_dn   = (tp_d_dn / sl_d_dn.replace(0, np.nan)).clip(lower=0)

    df["rr_value"] = np.where(is_up, rr_up,
                     np.where(is_dn, rr_dn,
                     pd.concat([rr_up, rr_dn], axis=1).max(axis=1)))
    df["cond2_rr"] = df["rr_value"] >= RR_MIN

    # DK2 = vùng đúng VÀ R:R đủ
    df["cond2_ok"] = df["cond2_zone"] & df["cond2_rr"]

    # ── DK3: Nến đảo chiều trên khung H4 (df_entry) ──────────
    # Nếu có df_entry (H4) → tính tín hiệu nến trên đó
    # Kết quả tín hiệu mới nhất gắn vào nến cuối của df chính
    ENTRY = df_entry if (df_entry is not None and not df_entry.empty and len(df_entry) > 10) else df
    de = ENTRY.copy()
    de["atr14_e"]   = calc_atr(de, 14)
    de["body_e"]    = (de["close"] - de["open"]).abs()
    de["range_e"]   = de["high"] - de["low"]
    de["wick_up_e"] = de["high"] - de[["open","close"]].max(axis=1)
    de["wick_dn_e"] = de[["open","close"]].min(axis=1) - de["low"]
    de["is_bull_e"] = de["close"] > de["open"]
    de["is_bear_e"] = de["close"] < de["open"]
    rng_e = de["range_e"].replace(0, np.nan)
    de["big_body_e"] = (de["range_e"] > 0) & (de["body_e"] / rng_e >= BODY)
    de["lw_up_e"]    = (de["range_e"] > 0) & (de["wick_up_e"] / rng_e >= WICK)
    de["lw_dn_e"]    = (de["range_e"] > 0) & (de["wick_dn_e"] / rng_e >= WICK)
    de["sig_big_e"]  = de["range_e"] >= de["atr14_e"] * ATRM
    de["vol_ma_e"]   = de["volume"].rolling(VMALEN).mean()
    de["vol_sig_ok_e"] = (~UVOL) | (de["volume"].shift(1) >= de["vol_ma_e"].shift(1) * VSMULT)
    de["vol_fb_ok_e"]  = (~UVOL) | (de["volume"].shift(1) <= de["vol_ma_e"].shift(1) * VFMULT)

    def se(col, i): return de[col].shift(i)
    vse  = de["vol_sig_ok_e"]
    vfbe = de["vol_fb_ok_e"]

    # Tín hiệu nến trên khung entry
    de["t1s_up_e"] = (se("is_bear_e",2) & se("big_body_e",2) & se("sig_big_e",2) &
                      se("is_bull_e",1) & se("big_body_e",1) & se("sig_big_e",1) &
                      (se("close",1) > se("open",2)) & vse)
    de["t1s_dn_e"] = (se("is_bull_e",2) & se("big_body_e",2) & se("sig_big_e",2) &
                      se("is_bear_e",1) & se("big_body_e",1) & se("sig_big_e",1) &
                      (se("close",1) < se("open",2)) & vse)
    de["t1_up_e"] = (se("is_bear_e",3) & se("big_body_e",3) & se("sig_big_e",3) &
                     se("is_bull_e",2) & se("lw_dn_e",2)   & se("sig_big_e",2) &
                     se("is_bull_e",1) & se("big_body_e",1) & se("sig_big_e",1) & vse)
    de["t1_dn_e"] = (se("is_bull_e",3) & se("big_body_e",3) & se("sig_big_e",3) &
                     se("is_bear_e",2) & se("lw_up_e",2)   & se("sig_big_e",2) &
                     se("is_bear_e",1) & se("big_body_e",1) & se("sig_big_e",1) & vse)
    de["t2_up_e"] = (se("is_bear_e",3) & se("sig_big_e",3) &
                     se("is_bear_e",2) & se("sig_big_e",2) & (se("body_e",2) < se("body_e",3)*0.85) &
                     se("sig_big_e",1) & (se("body_e",1) < se("body_e",2)*0.85) &
                     (se("close",3) < se("close",5)) &
                     (se("lw_dn_e",1) | se("is_bull_e",1)) & vse)
    de["t2_dn_e"] = (se("is_bull_e",3) & se("sig_big_e",3) &
                     se("is_bull_e",2) & se("sig_big_e",2) & (se("body_e",2) < se("body_e",3)*0.85) &
                     se("sig_big_e",1) & (se("body_e",1) < se("body_e",2)*0.85) &
                     (se("close",3) > se("close",5)) &
                     (se("lw_up_e",1) | se("is_bear_e",1)) & vse)
    de["t3_up_e"] = (se("lw_dn_e",2) & se("sig_big_e",2) &
                     se("lw_dn_e",1) & se("sig_big_e",1) &
                     (se("low",1) < se("low",2)) & (se("close",1) > se("close",2)) & vse)
    de["t3_dn_e"] = (se("lw_up_e",2) & se("sig_big_e",2) &
                     se("lw_up_e",1) & se("sig_big_e",1) &
                     (se("high",1) > se("high",2)) & (se("close",1) < se("close",2)) & vse)

    # Lấy tín hiệu của nến entry mới nhất
    last_e = de.iloc[-1]
    entry_bull = bool(last_e.get("t3_up_e",False) or last_e.get("t1_up_e",False) or
                      last_e.get("t1s_up_e",False) or last_e.get("t2_up_e",False))
    entry_bear = bool(last_e.get("t3_dn_e",False) or last_e.get("t1_dn_e",False) or
                      last_e.get("t1s_dn_e",False) or last_e.get("t2_dn_e",False))
    def entry_sig_name():
        if entry_bull:
            if last_e.get("t3_up_e"):  return "🎯 T3 SL-Hunter ↑"
            if last_e.get("t1_up_e"):  return "T1 Đảo Chiều ↑"
            if last_e.get("t1s_up_e"): return "T1S Đảo Chiều ↑"
            if last_e.get("t2_up_e"):  return "T2 Yếu Dần ↑"
        if entry_bear:
            if last_e.get("t3_dn_e"):  return "🎯 T3 SL-Hunter ↓"
            if last_e.get("t1_dn_e"):  return "T1 Đảo Chiều ↓"
            if last_e.get("t1s_dn_e"): return "T1S Đảo Chiều ↓"
            if last_e.get("t2_dn_e"):  return "T2 Yếu Dần ↓"
        return "—"
    entry_signal_name = entry_sig_name()

    # ── DK3 trên khung chính (cho chart lịch sử) ─────────────
    df["rsi"]    = calc_rsi(df["close"], RSILEN)
    df["rsi_ph"] = find_pivot_high(df["rsi"],   SWING, SWING)
    df["rsi_pl"] = find_pivot_low(df["rsi"],    SWING, SWING)
    df["prc_ph"] = find_pivot_high(df["close"], SWING, SWING)
    df["prc_pl"] = find_pivot_low(df["close"],  SWING, SWING)
    for col in ["rsi_ph","rsi_pl","prc_ph","prc_pl"]:
        df[f"r_{col}"] = df[col].ffill()
        df[f"p_{col}"] = df[col].shift(1).where(df[col].notna()).ffill()

    df["hidden_div_bull"] = (URSIDIV &
        df["r_prc_pl"].notna() & df["p_prc_pl"].notna() &
        df["r_rsi_pl"].notna() & df["p_rsi_pl"].notna() &
        (df["r_prc_pl"] > df["p_prc_pl"]) & (df["r_rsi_pl"] < df["p_rsi_pl"]))
    df["hidden_div_bear"] = (URSIDIV &
        df["r_prc_ph"].notna() & df["p_prc_ph"].notna() &
        df["r_rsi_ph"].notna() & df["p_rsi_ph"].notna() &
        (df["r_prc_ph"] < df["p_prc_ph"]) & (df["r_rsi_ph"] > df["p_rsi_ph"]))

    # Shorthand shift — dùng nến đã đóng (index >= 1)
    def s(col, i): return df[col].shift(i)
    vs  = df["vol_sig_ok"]
    vfb = df["vol_fb_ok"]

    # T1S: Đỏ dài[2] → Xanh dài[1] đóng trên open[2]
    df["t1s_up"] = (s("is_bear",2) & s("big_body",2) & s("sig_big",2) &
                    s("is_bull",1) & s("big_body",1) & s("sig_big",1) &
                    (s("close",1) > s("open",2)) & vs)
    df["t1s_dn"] = (s("is_bull",2) & s("big_body",2) & s("sig_big",2) &
                    s("is_bear",1) & s("big_body",1) & s("sig_big",1) &
                    (s("close",1) < s("open",2)) & vs)

    # T1: Đỏ dài[3] → Xanh râu dưới[2] → Xanh dài[1]
    df["t1_up"] = (s("is_bear",3) & s("big_body",3) & s("sig_big",3) &
                   s("is_bull",2) & s("lw_dn",2)   & s("sig_big",2) &
                   s("is_bull",1) & s("big_body",1) & s("sig_big",1) & vs)
    df["t1_dn"] = (s("is_bull",3) & s("big_body",3) & s("sig_big",3) &
                   s("is_bear",2) & s("lw_up",2)   & s("sig_big",2) &
                   s("is_bear",1) & s("big_body",1) & s("sig_big",1) & vs)

    # T2: 3 nến bé dần + context close[3]<close[5]
    df["t2_up"] = (s("is_bear",3) & s("sig_big",3) &
                   s("is_bear",2) & s("sig_big",2) & (s("body",2) < s("body",3)*0.85) &
                   s("sig_big",1) & (s("body",1) < s("body",2)*0.85) &
                   (s("close",3) < s("close",5)) &
                   (s("lw_dn",1) | s("is_bull",1)) & vs)
    df["t2_dn"] = (s("is_bull",3) & s("sig_big",3) &
                   s("is_bull",2) & s("sig_big",2) & (s("body",2) < s("body",3)*0.85) &
                   s("sig_big",1) & (s("body",1) < s("body",2)*0.85) &
                   (s("close",3) > s("close",5)) &
                   (s("lw_up",1) | s("is_bear",1)) & vs)

    # T3: 2 nến râu, low[1]<low[2], close[1]>close[2]
    df["t3_up"] = (s("lw_dn",2) & s("sig_big",2) &
                   s("lw_dn",1) & s("sig_big",1) &
                   (s("low",1) < s("low",2)) & (s("close",1) > s("close",2)) & vs)
    df["t3_dn"] = (s("lw_up",2) & s("sig_big",2) &
                   s("lw_up",1) & s("sig_big",1) &
                   (s("high",1) > s("high",2)) & (s("close",1) < s("close",2)) & vs)

    # False Breakout (dùng vol_fb)
    df["fb_resist"]  = ((s("high",1) > df["resist"]) & (s("close",1) < df["resist"]) &
                        s("lw_up",1) & s("sig_big",1) & vfb)
    df["fb_support"] = ((s("low",1) < df["support"])  & (s("close",1) > df["support"]) &
                        s("lw_dn",1) & s("sig_big",1) & vfb)

    # Hammer/Shooting Star chỉ tại KC/HT
    df["hammer_up"]   = (df["in_support"] & s("lw_dn",1) & s("sig_big",1) &
                         (s("wick_dn",1) >= s("body",1) * 2.0) & vs)
    df["shooting_dn"] = (df["in_resist"]  & s("lw_up",1) & s("sig_big",1) &
                         (s("wick_up",1) >= s("body",1) * 2.0) & vs)

    df["bull_signal"] = (df["t3_up"] | df["t1_up"] | df["t1s_up"] | df["t2_up"] |
                         df["fb_support"] | df["hidden_div_bull"] | df["hammer_up"])
    df["bear_signal"] = (df["t3_dn"] | df["t1_dn"] | df["t1s_dn"] | df["t2_dn"] |
                         df["fb_resist"] | df["hidden_div_bear"] | df["shooting_dn"])

    # cond3 trên khung chính (cho lịch sử chart)
    df["cond3_ok"] = ((is_up & df["bull_signal"]) |
                      (is_dn & df["bear_signal"]) |
                      (is_sw & (df["bull_signal"] | df["bear_signal"])))

    # cond3 THỰC TẾ cho điểm: dùng tín hiệu H4 (entry frame)
    cond3_entry = ((is_up and entry_bull) or (is_dn and entry_bear) or
                   (is_sw and (entry_bull or entry_bear)))
    # Gắn vào nến cuối
    df.iloc[-1, df.columns.get_loc("cond3_ok")] = cond3_entry
    df.attrs["entry_signal_name"] = entry_signal_name
    df.attrs["entry_bull"] = entry_bull
    df.attrs["entry_bear"] = entry_bear

    # ── Score /3 ──────────────────────────────────────────────
    df["score"]  = (df["cond1_ok"].astype(int) + df["cond2_ok"].astype(int) +
                    df["cond3_ok"].astype(int))
    df["all_ok"] = df["score"].astype(int) == 3

    # ── Quản lý vốn (tham khảo, không tính điểm) ─────────────
    sl_dist     = df["close"] * SLP / 100
    pos_size    = (ACC * RISKP / 100) / sl_dist.replace(0, np.nan)
    pos_lots    = (pos_size / 100).apply(lambda x: round(x)*100 if pd.notna(x) else 0)
    df["pos_size"] = pos_lots

    # ── Tên tín hiệu ──────────────────────────────────────────
    def sig_name(row):
        if row["bull_signal"]:
            for k,v in [("t3_up","🎯 T3 SL-Hunter ↑"),("fb_support","⚠️ False Breakout ↑"),
                        ("hidden_div_bull","📊 Phân Kỳ Ẩn ↑"),("t1_up","T1 Đảo Chiều ↑"),
                        ("t1s_up","T1S Đảo Chiều ↑"),("hammer_up","🔨 Hammer ↑"),
                        ("t2_up","T2 Yếu Dần ↑")]:
                if row[k]: return v
        if row["bear_signal"]:
            for k,v in [("t3_dn","🎯 T3 SL-Hunter ↓"),("fb_resist","⚠️ False Breakout ↓"),
                        ("hidden_div_bear","📊 Phân Kỳ Ẩn ↓"),("t1_dn","T1 Đảo Chiều ↓"),
                        ("t1s_dn","T1S Đảo Chiều ↓"),("shooting_dn","⭐ Shooting Star ↓"),
                        ("t2_dn","T2 Yếu Dần ↓")]:
                if row[k]: return v
        return "—"

    df["signal_name"] = df.apply(sig_name, axis=1)
    return df

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/login-page")
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    return send_file("login.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    ok, msg = do_login(data.get("username",""), data.get("password",""))
    return jsonify({"ok": ok, "message": msg})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    try:
        data   = request.json
        symbol = data.get("symbol","VNM").upper().strip()
        start  = data.get("start_date","2024-01-01")
        tf_ref = data.get("tf_ref","1W")
        tf_chart = data.get("tf_chart","1D")
        cfg = {
            "swing_len":     int(data.get("swing_len", 3)),
            "zone_len":      int(data.get("zone_len", 50)),
            "zone_atr_mult": float(data.get("zone_atr_mult", 1.5)),
            "body_ratio":    float(data.get("body_ratio", 0.50)),
            "wick_ratio":    float(data.get("wick_ratio", 0.40)),
            "atr_mult":      float(data.get("atr_mult", 0.3)),
            "use_volume":    bool(data.get("use_volume", True)),
            "vol_ma_len":    int(data.get("vol_ma_len", 20)),
            "vol_sig_mult":  float(data.get("vol_sig_mult", 1.2)),
            "vol_fb_mult":   float(data.get("vol_fb_mult", 1.5)),
            "use_rsi_div":   bool(data.get("use_rsi_div", True)),
            "rsi_len":       int(data.get("rsi_len", 14)),
            "rr_min":        float(data.get("rr_min", 2.0)),
            "account":       float(data.get("account", 100_000_000)),
            "risk_pct":      float(data.get("risk_pct", 2.0)),
            "monthly_loss":  float(data.get("monthly_loss", 6.0)),
            "sl_pct":        float(data.get("sl_pct", 5.0)),
        }

        df_main = get_data(symbol, start, timeframe=tf_chart)
        df_htf  = get_data(symbol, start, timeframe=tf_ref)
        df_h4, h4_src = get_intraday_h4(symbol, start)   # v7: DK3 dùng H4
        if df_main.empty:
            return jsonify({"error": f"Không có dữ liệu cho '{symbol}'"}), 400

        result   = run_checklist(df_main, df_htf, cfg, df_entry=df_h4)
        row      = result.iloc[-1]
        score    = int(row["score"])
        chart_df = result.tail(200).copy()
        chart_df["date_str"] = chart_df["date"].dt.strftime("%Y-%m-%d")

        def sl(col): return [safe(x) for x in chart_df[col]] if col in chart_df else []
        def sig_recs(mask, pcol):
            sub = chart_df[mask][["date_str",pcol]].rename(columns={"date_str":"date"})
            return [{k:safe(v) for k,v in r.items()} for r in sub.to_dict("records")]

        chart_data = {
            "dates":       chart_df["date_str"].tolist(),
            "open":        sl("open"), "high": sl("high"),
            "low":         sl("low"),  "close": sl("close"),
            "volume":      sl("volume"), "vol_ma": sl("vol_ma"),
            "resist_top":  [safe(a+b) for a,b in zip(chart_df["resist"], chart_df["zone_w"])],
            "resist_bot":  [safe(a) for a in chart_df["resist"]],
            "support_top": [safe(a) for a in chart_df["support"]],
            "support_bot": [safe(a-b) for a,b in zip(chart_df["support"], chart_df["zone_w"])],
            "bull_4": sig_recs((chart_df["score"].astype(int)==3) & chart_df["bull_signal"], "low"),
            "bear_4": sig_recs((chart_df["score"].astype(int)==3) & chart_df["bear_signal"], "high"),
            "bull_3": sig_recs((chart_df["score"].astype(int)==2) & chart_df["bull_signal"], "low"),
            "bear_3": sig_recs((chart_df["score"].astype(int)==2) & chart_df["bear_signal"], "high"),
        }

        trend_txt = (f"▲ UPTREND [{row['trend_src']}]"  if row["is_uptrend"]   else
                     f"▼ DOWNTREND [{row['trend_src']}]" if row["is_downtrend"] else
                     "↔ SIDEWAY")
        zone_txt  = (f"KC: {row['resist']:,.0f}" if row["in_resist"] else
                     f"HT: {row['support']:,.0f}" if row["in_support"] else
                     "Giữa vùng — chờ")
        rr_val    = safe(row.get("rr_value", np.nan))
        rr_txt    = f"{rr_val:.2f}:1" if rr_val else "N/A"
        zone2_txt = f"{zone_txt} | R:R {rr_txt}"
        vol_txt   = (" | Vol✅" if row["vol_sig_ok"] else " | Vol❌") if cfg["use_volume"] else ""
        max_risk  = cfg["account"] * cfg["risk_pct"] / 100
        pos       = safe(row["pos_size"]) or 0
        rsi_val   = safe(row["rsi"]) or 0

        summary = {
            "symbol":    symbol,
            "date":      str(row["date"])[:10],
            "close":     f"{row['close']:,.0f}",
            "score":     score,
            "score_max": 3,
            "action":    ["🔴 ĐỨNG NGOÀI","🔴 ĐỨNG NGOÀI",
                          "🟡 THEO DÕI SÁT","🟢 VÀO LỆNH"][score],
            "direction": ("↑ LONG"  if result.attrs.get("entry_bull") else
                          "↓ SHORT" if result.attrs.get("entry_bear")  else "— Chờ tín hiệu"),
            "cond1": bool(row["cond1_ok"]), "cond1_txt": trend_txt,
            "cond2": bool(row["cond2_ok"]),
            "cond2_txt": zone2_txt,
            "cond2a": bool(row["cond2_zone"]), "cond2a_txt": zone_txt,
            "cond2b": bool(row["cond2_rr"]),   "cond2b_txt": f"R:R {rr_txt}",
            "cond3": bool(row["cond3_ok"]),
            "cond3_txt": result.attrs.get("entry_signal_name","—") + f" [{h4_src}]" + vol_txt,
            "rsi":        f"{rsi_val:.1f}",
            "rsi_signal": ("📈 Phân kỳ ẩn tăng" if bool(row["hidden_div_bull"]) else
                           "📉 Phân kỳ ẩn giảm" if bool(row["hidden_div_bear"]) else "Bình thường"),
            "rr_value":   rr_txt,
            "monthly_loss_vnd": f"{cfg['account']*cfg['monthly_loss']/100:,.0f}",
            "pos_size":   f"{int(pos):,} cổ",
            "max_risk":   f"{max_risk:,.0f}₫",
            "trend_src":  row["trend_src"],
        }

        sig_df  = result[result["bull_signal"]|result["bear_signal"]].tail(5).iloc[::-1]
        signals = [{"date":  str(r["date"])[:10],
                    "close": f"{r['close']:,.0f}",
                    "score": int(r["score"]),
                    "signal":str(r["signal_name"]),
                    "action":["❌","❌","🟡","🟢"][int(r["score"])]}
                   for _,r in sig_df.iterrows()]

        return jsonify({"chart":chart_data, "summary":summary, "signals":signals})

    except Exception as e:
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": f"Lỗi '{data.get('symbol','?')}': {str(e)[:200]}"}), 500

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=PORT, debug=False)
