"""
UG Checklist v4 — Web App (Flask)
5 điều kiện:
  DK1: Xu hướng PA pivot + EMA backup (khung D)
  DK2: Vùng KC/HT (zone_len=50, atr_mult=1.5)
  DK3: Nến đảo chiều + volume tách 2 loại
  DK4: RS vs VNINDEX (outperform/underperform)
  DK5: EMA20/50 ngày + R:R >= 2.0
"""
import sys, os, warnings, logging
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from datetime import date, timedelta
import pandas as pd
import numpy as np
import os, secrets

app = Flask(__name__)

# ── Security ───────────────────────────────────────────────────
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
    except (TypeError, ValueError): pass
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)):
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
        window = arr[i-left: i+right+1]
        if arr[i] == window.max() and arr[i] > arr[i-1]:
            result.iloc[i] = arr[i]
    return result

def find_pivot_low(series, left, right):
    result = pd.Series(np.nan, index=series.index)
    arr = series.values
    for i in range(left, len(arr) - right):
        window = arr[i-left: i+right+1]
        if arr[i] == window.min() and arr[i] < arr[i-1]:
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
    interval = {"1D":"1D","1W":"1W","1H":"1H"}.get(timeframe,"1D")

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
        import yfinance as yf
        import socket; socket.setdefaulttimeout(20)
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

def get_vnindex(start, end=None):
    """Lấy data VNINDEX — thử nhiều nguồn."""
    end = end or date.today().strftime("%Y-%m-%d")

    # 1. yfinance trước — reliable nhất
    try:
        import yfinance as yf
        import socket; socket.setdefaulttimeout(20)
        df = _normalize(yf.download("^VNINDEX", start=start, end=end,
                                    auto_adjust=True, progress=False,
                                    timeout=15).reset_index())
        if not df.empty and len(df) > 5:
            log.info(f"VNINDEX yfinance OK: {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"VNINDEX yfinance: {e}")

    # 2. vnstock fallback
    for sym in ["VNINDEX", "VN-INDEX", "HOSE:VNINDEX"]:
        try:
            from vnstock import stock_historical_data
            import inspect
            sig = inspect.signature(stock_historical_data)
            df = (stock_historical_data(sym, start, end, resolution="1D", type="index")
                  if "resolution" in sig.parameters else
                  stock_historical_data(sym, start, end, interval="1D", asset_type="index"))
            df = _normalize(df)
            if not df.empty:
                log.info(f"VNINDEX vnstock OK ({sym}): {len(df)} nến")
                return df
        except: pass

    # 3. vnstock3 fallback
    try:
        from vnstock3 import Vnstock
        df = Vnstock().stock(symbol="VNINDEX", source="VCI").quote.history(
            start=start, end=end, interval="1D")
        df = _normalize(df)
        if not df.empty:
            log.info(f"VNINDEX vnstock3 OK: {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"VNINDEX vnstock3: {e}")

    log.warning("Không lấy được VNINDEX → DK4 N/A")
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
#  CHECKLIST v4
# ═══════════════════════════════════════════════════════════════
def run_checklist(df_ltf, df_htf, df_vni, cfg):
    SWING   = cfg["swing_len"]
    ZLEN    = cfg["zone_len"]
    ZATR    = cfg["zone_atr_mult"]
    BODY    = cfg["body_ratio"]
    WICK    = cfg["wick_ratio"]
    ATRM    = cfg["atr_mult"]
    UVOL    = cfg["use_volume"]
    VMALEN  = cfg["vol_ma_len"]
    VBMULT  = cfg["vol_bull_mult"]   # v4: volume xác nhận >= 1.2x
    VFBMULT = cfg["vol_fb_mult"]     # v4: volume FB <= 1.5x
    URSIDIV = cfg["use_rsi_div"]
    RSILEN  = cfg["rsi_len"]
    RS_LEN  = cfg["rs_len"]
    RS_THRESH = cfg["rs_thresh"]
    RR_MIN  = cfg["rr_min"]
    ACC     = cfg["account"]
    RISKP   = cfg["risk_pct"]
    SLP     = cfg["sl_pct"]

    df = df_ltf.copy()

    # ── Candle helpers ─────────────────────────────────────────
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

    # v4: volume tách 2 loại
    df["vol_confirm"] = (~UVOL) | (df["volume"] >= df["vol_ma"] * VBMULT)
    df["vol_fb_ok"]   = (~UVOL) | (df["volume"] <= df["vol_ma"] * VFBMULT)

    # ── DK1: Xu hướng (PA + EMA backup) ───────────────────────
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
    pa_up = (all(ok(x) for x in [last_ph,prev_ph,last_pl,prev_pl])
             and last_ph > prev_ph and last_pl > prev_pl)
    pa_dn = (all(ok(x) for x in [last_ph,prev_ph,last_pl,prev_pl])
             and last_ph < prev_ph and last_pl < prev_pl)

    htf_last   = htf.iloc[-1]
    ema20_last = htf_last["ema20"]
    ema50_last = htf_last["ema50"]
    htf_close  = htf_last["close"]
    ema_up = (not np.isnan(ema20_last) and not np.isnan(ema50_last)
              and ema20_last > ema50_last and htf_close > ema20_last)
    ema_dn = (not np.isnan(ema20_last) and not np.isnan(ema50_last)
              and ema20_last < ema50_last and htf_close < ema20_last)

    is_up  = pa_up or (not pa_up and not pa_dn and ema_up)
    is_dn  = pa_dn or (not pa_up and not pa_dn and ema_dn)
    is_sw  = not is_up and not is_dn
    trend_src = "PA" if (pa_up or pa_dn) else "EMA"

    df["is_uptrend"]   = is_up
    df["is_downtrend"] = is_dn
    df["is_sideways"]  = is_sw
    df["trend_src"]    = trend_src
    df["cond1_ok"]     = is_up or is_dn

    # ── DK2: Vùng KC/HT ───────────────────────────────────────
    df["resist"]      = df["high"].shift(1).rolling(ZLEN).max()
    df["support"]     = df["low"].shift(1).rolling(ZLEN).min()
    df["zone_w"]      = df["atr14"] * ZATR
    version = cfg.get("version","v4")
    if version == "v5":
        # v5 FIX: tiếp cận KC từ dưới lên, HT từ trên xuống
        df["in_resist"]  = ((df["close"] >= df["resist"] - df["zone_w"]) &
                            (df["close"] <= df["resist"]))
        df["in_support"] = ((df["close"] <= df["support"] + df["zone_w"]) &
                            (df["close"] >= df["support"]))
    else:
        df["in_resist"]   = ((df["close"] >= df["resist"] - df["zone_w"]) &
                             (df["close"] <= df["resist"] + df["zone_w"]))
        df["in_support"]  = ((df["close"] >= df["support"] - df["zone_w"]) &
                             (df["close"] <= df["support"] + df["zone_w"]))
    df["near_resist"] = df["in_resist"]
    df["near_support"]= df["in_support"]
    df["cond2_ok"]    = ((is_up & df["in_support"]) |
                         (is_dn & df["in_resist"])  |
                         (is_sw & (df["in_support"] | df["in_resist"])))

    # ── DK3: Nến đảo chiều ─────────────────────────────────────
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

    def s(col, i): return df[col].shift(i)
    vc  = df["vol_confirm"]
    vfb = df["vol_fb_ok"]

    df["t1s_up"] = (s("is_bear",1) & s("big_body",1) & s("sig_big",1) &
                    df["is_bull"] & df["big_body"] & df["sig_big"] &
                    (df["close"] > s("open",1)) & vc)
    df["t1s_dn"] = (s("is_bull",1) & s("big_body",1) & s("sig_big",1) &
                    df["is_bear"] & df["big_body"] & df["sig_big"] &
                    (df["close"] < s("open",1)) & vc)
    df["t1_up"]  = (s("is_bear",2) & s("big_body",2) & s("sig_big",2) &
                    s("sig_big",1) & s("lw_dn",1) &
                    df["is_bull"] & df["big_body"] & df["sig_big"] & vc)
    df["t1_dn"]  = (s("is_bull",2) & s("big_body",2) & s("sig_big",2) &
                    s("sig_big",1) & s("lw_up",1) &
                    df["is_bear"] & df["big_body"] & df["sig_big"] & vc)
    if version == "v5":
        # v5: T2 dùng 3 nến bé dần + close[3]<close[5]
        df["t2_up"] = (s("is_bear",3) & s("sig_big",3) &
                       s("is_bear",2) & s("sig_big",2) & (s("body",2) < s("body",3)*0.85) &
                       s("sig_big",1) & (df["body"] < s("body",2)*0.85) &
                       (s("close",3) < s("close",5)) &
                       (df["lw_dn"] | df["is_bull"]) & vc)
        df["t2_dn"] = (s("is_bull",3) & s("sig_big",3) &
                       s("is_bull",2) & s("sig_big",2) & (s("body",2) < s("body",3)*0.85) &
                       s("sig_big",1) & (df["body"] < s("body",2)*0.85) &
                       (s("close",3) > s("close",5)) &
                       (df["lw_up"] | df["is_bear"]) & vc)
    else:
        # v4: T2 đơn giản hơn
        df["t2_up"]  = (s("is_bear",1) & s("sig_big",1) &
                        df["sig_big"] & (df["body"] < s("body",1) * 0.85) &
                        (df["lw_dn"] | df["is_bull"]) &
                        (s("close",1) < s("close",3)) & vc)
        df["t2_dn"]  = (s("is_bull",1) & s("sig_big",1) &
                        df["sig_big"] & (df["body"] < s("body",1) * 0.85) &
                        (df["lw_up"] | df["is_bear"]) &
                        (s("close",1) > s("close",3)) & vc)
    df["t3_up"]  = (s("lw_dn",1) & s("sig_big",1) &
                    df["lw_dn"] & df["sig_big"] &
                    (df["low"] < s("low",1)) & (df["close"] > s("close",1)) & vc)
    df["t3_dn"]  = (s("lw_up",1) & s("sig_big",1) &
                    df["lw_up"] & df["sig_big"] &
                    (df["high"] > s("high",1)) & (df["close"] < s("close",1)) & vc)
    df["fb_resist"]  = ((df["high"] > df["resist"]) & (df["close"] < df["resist"]) &
                        df["lw_up"] & df["sig_big"] & vfb)
    df["fb_support"] = ((df["low"] < df["support"]) & (df["close"] > df["support"]) &
                        df["lw_dn"] & df["sig_big"] & vfb)
    if version == "v5":
        # v5: Hammer chỉ tại HT, Shooting Star chỉ tại KC
        df["hammer_up"]   = (df["in_support"] & df["lw_dn"] & df["sig_big"] &
                             (df["wick_dn"] >= df["body"] * 2.0) & vc)
        df["shooting_dn"] = (df["in_resist"]  & df["lw_up"] & df["sig_big"] &
                             (df["wick_up"] >= df["body"] * 2.0) & vc)
    else:
        df["hammer_up"]   = (df["lw_dn"] & df["sig_big"] &
                             (df["wick_dn"] >= df["body"] * 2.0) & vc)
        df["shooting_dn"] = (df["lw_up"] & df["sig_big"] &
                             (df["wick_up"] >= df["body"] * 2.0) & vc)

    df["bull_signal"] = (df["t1_up"] | df["t1s_up"] | df["t2_up"] | df["t3_up"] |
                         df["fb_support"] | df["hidden_div_bull"] | df["hammer_up"])
    df["bear_signal"] = (df["t1_dn"] | df["t1s_dn"] | df["t2_dn"] | df["t3_dn"] |
                         df["fb_resist"] | df["hidden_div_bear"] | df["shooting_dn"])
    df["cond3_ok"]    = ((is_up & df["bull_signal"]) |
                         (is_dn & df["bear_signal"]) |
                         (is_sw & (df["bull_signal"] | df["bear_signal"])))

    # ── DK4: RS vs VNINDEX ─────────────────────────────────────
    # Merge VNINDEX data vào df theo ngày
    has_vni = not df_vni.empty and len(df_vni) > RS_LEN

    if has_vni:
        vni = df_vni[["date","close"]].rename(columns={"close":"vni_close"})
        df  = df.merge(vni, on="date", how="left")
        df["vni_close"] = df["vni_close"].ffill()

        # % thay đổi trong RS_LEN nến
        df["pct_cp"]  = df["close"].pct_change(RS_LEN) * 100
        df["pct_vni"] = df["vni_close"].pct_change(RS_LEN) * 100
        df["rs_value"]= df["pct_cp"] - df["pct_vni"]

        df["rs_bull"]    = df["rs_value"] > RS_THRESH
        df["rs_bear"]    = df["rs_value"] < -RS_THRESH
        df["rs_neutral"] = ~df["rs_bull"] & ~df["rs_bear"] & df["rs_value"].notna()

        df["cond4_ok"] = ((is_up & df["rs_bull"])  |
                          (is_dn & df["rs_bear"])  |
                          (is_sw & df["rs_value"].notna()))
        df["cond4_na"] = False
    else:
        df["rs_value"]   = np.nan
        df["rs_bull"]    = False
        df["rs_bear"]    = False
        df["rs_neutral"] = False
        df["cond4_ok"]   = False
        df["cond4_na"]   = True   # không có data VNI

    # ── DK5: EMA ngày + R:R ────────────────────────────────────
    # 5a: EMA20/50 trên khung ngày (df_ltf là 1D)
    df["ema20_d"] = calc_ema(df["close"], 20)
    df["ema50_d"] = calc_ema(df["close"], 50)
    df["small_up"] = (df["ema20_d"] > df["ema50_d"]) & (df["close"] > df["ema20_d"])
    df["small_dn"] = (df["ema20_d"] < df["ema50_d"]) & (df["close"] < df["ema20_d"])
    df["small_align"] = ((is_up & df["small_up"]) |
                         (is_dn & df["small_dn"]) |
                         (is_sw & (df["small_up"] | df["small_dn"])))

    # 5b: R:R
    sl_dist_up = df["close"] - (df["support"] - df["zone_w"])
    tp_dist_up = df["resist"] - df["close"]
    rr_up      = tp_dist_up / sl_dist_up.replace(0, np.nan)

    sl_dist_dn = (df["resist"] + df["zone_w"]) - df["close"]
    tp_dist_dn = df["close"] - df["support"]
    rr_dn      = tp_dist_dn / sl_dist_dn.replace(0, np.nan)

    df["rr_value"] = np.where(is_up, rr_up,
                     np.where(is_dn, rr_dn,
                     pd.concat([rr_up, rr_dn], axis=1).max(axis=1)))
    df["rr_ok"]    = df["rr_value"] >= RR_MIN
    df["cond5_ok"] = df["small_align"] & df["rr_ok"]

    # ── Position size ──────────────────────────────────────────
    sl_dist     = df["close"] * SLP / 100
    pos_size    = (ACC * RISKP / 100) / sl_dist.replace(0, np.nan)
    pos_lots    = (pos_size / 100).apply(lambda x: round(x) * 100 if pd.notna(x) else 0)
    df["pos_size"] = pos_lots

    # ── Score /5 ───────────────────────────────────────────────
    df["score"] = (df["cond1_ok"].astype(int) + df["cond2_ok"].astype(int) +
                   df["cond3_ok"].astype(int) + df["cond4_ok"].astype(int) +
                   df["cond5_ok"].astype(int))
    df["all_ok"] = df["score"].astype(int) == 5

    # ── Tên tín hiệu ───────────────────────────────────────────
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
        tf_ref = data.get("tf_ref","1D")   # v4: default Daily
        cfg = {
            "swing_len":     int(data.get("swing_len", 3)),
            "zone_len":      int(data.get("zone_len", 50)),
            "zone_atr_mult": float(data.get("zone_atr_mult", 1.5)),
            "body_ratio":    float(data.get("body_ratio", 0.50)),
            "wick_ratio":    float(data.get("wick_ratio", 0.40)),
            "atr_mult":      float(data.get("atr_mult", 0.3)),
            "use_volume":    bool(data.get("use_volume", True)),
            "vol_ma_len":    int(data.get("vol_ma_len", 20)),
            "vol_bull_mult": float(data.get("vol_bull_mult", 1.2)),  # v4
            "vol_fb_mult":   float(data.get("vol_fb_mult", 1.5)),    # v4
            "use_rsi_div":   bool(data.get("use_rsi_div", True)),
            "rsi_len":       int(data.get("rsi_len", 14)),
            "rs_len":        int(data.get("rs_len", 20)),            # v4
            "rs_thresh":     float(data.get("rs_thresh", 1.0)),
            "version":       data.get("version", "v5"),
            "rr_min":        float(data.get("rr_min", 2.0)),         # v4
            "account":       float(data.get("account", 100_000_000)),
            "risk_pct":      float(data.get("risk_pct", 2.0)),
            "monthly_loss":  float(data.get("monthly_loss", 6.0)),
            "sl_pct":        float(data.get("sl_pct", 5.0)),
        }

        # Lấy data song song
        tf_chart = data.get("tf_chart", "1D")  # timeframe hiển thị chart
        df_main  = get_data(symbol, start, timeframe=tf_chart)
        df_htf   = get_data(symbol, start, timeframe=tf_ref)
        df_vni   = get_vnindex(start)   # DK4

        if df_main.empty:
            return jsonify({"error": f"Không có dữ liệu cho '{symbol}'"}), 400

        result   = run_checklist(df_main, df_htf, df_vni, cfg)
        row      = result.iloc[-1]
        score    = int(row["score"])
        chart_df = result.tail(200).copy()
        chart_df["date_str"] = chart_df["date"].dt.strftime("%Y-%m-%d")

        def sl(col):
            return [safe(x) for x in chart_df[col]] if col in chart_df else []
        def sig_recs(mask, pcol):
            sub = chart_df[mask][["date_str",pcol]].rename(columns={"date_str":"date"})
            return [{k:safe(v) for k,v in r.items()} for r in sub.to_dict("records")]

        chart_data = {
            "dates":       chart_df["date_str"].tolist(),
            "open":        sl("open"), "high": sl("high"),
            "low":         sl("low"),  "close": sl("close"),
            "volume":      sl("volume"), "vol_ma": sl("vol_ma"),
            "resist_top":  [safe(a+b) for a,b in zip(chart_df["resist"], chart_df["zone_w"])],
            "resist_bot":  [safe(a-b) for a,b in zip(chart_df["resist"], chart_df["zone_w"])],
            "support_top": [safe(a+b) for a,b in zip(chart_df["support"], chart_df["zone_w"])],
            "support_bot": [safe(a-b) for a,b in zip(chart_df["support"], chart_df["zone_w"])],
            "bull_4": sig_recs((chart_df["score"]>=4) & chart_df["bull_signal"],"low"),
            "bear_4": sig_recs((chart_df["score"]>=4) & chart_df["bear_signal"],"high"),
            "bull_3": sig_recs((chart_df["score"]==3) & chart_df["bull_signal"],"low"),
            "bear_3": sig_recs((chart_df["score"]==3) & chart_df["bear_signal"],"high"),
        }

        # RS info
        rs_val  = safe(row.get("rs_value", np.nan))
        rs_txt  = "N/A (không có data VNINDEX)"
        if rs_val is not None:
            if bool(row["rs_bull"]):   rs_txt = f"Mạnh hơn VNI {rs_val:+.2f}%"
            elif bool(row["rs_bear"]): rs_txt = f"Yếu hơn VNI {rs_val:+.2f}%"
            else:                      rs_txt = f"Tương đương VNI {rs_val:+.2f}%"

        # RR info
        rr_val = safe(row.get("rr_value", np.nan))
        rr_txt = f"R:R = {rr_val:.2f}:1" if rr_val else "Chưa tính được"

        trend_txt  = (f"▲ UPTREND [{row['trend_src']}]"  if row["is_uptrend"]   else
                      f"▼ DOWNTREND [{row['trend_src']}]" if row["is_downtrend"] else
                      "↔ SIDEWAY")
        zone_txt   = (f"KC: {row['resist']:,.0f}" if row["in_resist"] else
                      f"HT: {row['support']:,.0f}" if row["in_support"] else
                      "Giữa vùng — chờ")
        vol_txt    = (" | Vol✅" if row["vol_confirm"] else " | Vol❌") if cfg["use_volume"] else ""
        max_risk   = cfg["account"] * cfg["risk_pct"] / 100
        pos        = safe(row["pos_size"]) or 0
        rsi_val    = safe(row["rsi"]) or 0
        ema_align  = "↑ EMA align" if bool(row["small_up"]) else "↓ EMA align" if bool(row["small_dn"]) else "↔ EMA neutral"

        summary = {
            "symbol":    symbol,
            "date":      str(row["date"])[:10],
            "close":     f"{row['close']:,.0f}",
            "score":     score,
            "score_max": 5,
            "action":    ["🔴 ĐỨNG NGOÀI","🔴 ĐỨNG NGOÀI","🟠 CHỜ THÊM",
                          "🟡 THEO DÕI SÁT","🟡 THEO DÕI SÁT","🟢 VÀO LỆNH"][score],
            "direction": ("↑ LONG"  if bool(row["bull_signal"]) else
                          "↓ SHORT" if bool(row["bear_signal"])  else "— Chờ tín hiệu"),
            "cond1": bool(row["cond1_ok"]), "cond1_txt": trend_txt,
            "cond2": bool(row["cond2_ok"]), "cond2_txt": zone_txt,
            "cond3": bool(row["cond3_ok"]), "cond3_txt": str(row["signal_name"]) + vol_txt,
            "cond4": bool(row["cond4_ok"]), "cond4_txt": rs_txt,
            "cond4_na": bool(row.get("cond4_na", False)),
            "cond5": bool(row["cond5_ok"]), "cond5_txt": f"{ema_align} | {rr_txt}",
            "rsi":        f"{rsi_val:.1f}",
            "rsi_signal": ("📈 Phân kỳ ẩn tăng" if bool(row["hidden_div_bull"]) else
                           "📉 Phân kỳ ẩn giảm" if bool(row["hidden_div_bear"]) else "Bình thường"),
            "rs_value":   f"{rs_val:+.2f}%" if rs_val is not None else "N/A",
            "rr_value":   f"{rr_val:.2f}:1" if rr_val else "N/A",
            "monthly_loss_vnd": f"{cfg['account']*cfg['monthly_loss']/100:,.0f}",
            "trend_src":  row["trend_src"],
            "version":    cfg.get("version","v4"),
            "cond4_txt":  rs_txt,
            "cond5_txt":  f"{ema_align} | {rr_txt}",
            "tf_chart":   tf_chart,
        }

        sig_df  = result[result["bull_signal"]|result["bear_signal"]].tail(5).iloc[::-1]
        signals = [{"date":  str(r["date"])[:10],
                    "close": f"{r['close']:,.0f}",
                    "score": int(r["score"]),
                    "signal":str(r["signal_name"]),
                    "action":["❌","❌","🟠","🟡","🟡","🟢"][int(r["score"])]}
                   for _,r in sig_df.iterrows()]

        return jsonify({"chart":chart_data,"summary":summary,"signals":signals})

    except Exception as e:
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": f"Lỗi '{data.get('symbol','?')}': {str(e)[:200]}"}), 500


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=PORT, debug=False)
