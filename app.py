"""
UG Checklist v8 — Web App (Flask)
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
    # Một số nguồn trả MultiIndex columns (tuple) → flatten lấy level 0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
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

    raise RuntimeError(f"Không lấy được data cho '{symbol}' (chỉ dùng vnstock).")

# Cache H4 ở mức module: {symbol: (df, src, timestamp)}
_H4_CACHE = {}
_H4_TTL = 600  # 10 phút

def get_intraday_h4(symbol, start, end=None):
    """Lấy data khung H4 cho DK3 — CHỈ dùng vnstock 1H gộp thành H4.
    Gộp theo nửa phiên (sáng/chiều) để khớp giờ giao dịch VN, tránh lệch giờ nghỉ trưa.
    Có cache 10 phút."""
    import time as _time
    end = end or date.today().strftime("%Y-%m-%d")

    # Cache
    cached = _H4_CACHE.get(symbol)
    if cached and (_time.time() - cached[2] < _H4_TTL):
        log.info(f"H4 cache hit: {symbol} ({cached[1]})")
        return cached[0].copy(), cached[1]

    # vnstock 1H → gộp H4 theo nửa phiên
    try:
        from vnstock import stock_historical_data
        import inspect
        sig = inspect.signature(stock_historical_data)
        df = (stock_historical_data(symbol, start, end, resolution="1H", type="stock")
              if "resolution" in sig.parameters else
              stock_historical_data(symbol, start, end, interval="1H", asset_type="stock"))
        df = _normalize(df)
        if not df.empty and len(df) > 20:
            agg = _resample_h4_session(df)
            if not agg.empty:
                log.info(f"H4 vnstock(1H→4H) OK: {symbol} {len(agg)} nến")
                _H4_CACHE[symbol] = (agg, "H4(1H)", _time.time())
                return agg.copy(), "H4(1H)"
    except Exception as e:
        log.warning(f"H4 vnstock 1H: {e}")

    # vnstock3 1H fallback
    try:
        from vnstock3 import Vnstock
        df = Vnstock().stock(symbol=symbol, source="VCI").quote.history(
            start=start, end=end, interval="1H")
        df = _normalize(df)
        if not df.empty and len(df) > 20:
            agg = _resample_h4_session(df)
            if not agg.empty:
                log.info(f"H4 vnstock3(1H→4H) OK: {symbol} {len(agg)} nến")
                _H4_CACHE[symbol] = (agg, "H4(1H)", _time.time())
                return agg.copy(), "H4(1H)"
    except Exception as e:
        log.warning(f"H4 vnstock3 1H: {e}")

    # Không lấy được H4 → báo lỗi rõ ràng (KHÔNG fallback sang khung ngày)
    raise RuntimeError(f"Không lấy được data H4 (1H) cho '{symbol}'. vnstock/vnstock3 đều thất bại.")

def _resample_h4_session(df):
    """Gộp nến 1H thành H4 theo nửa phiên giao dịch VN.
    Phiên VN: sáng 9h-11h30, chiều 13h-15h (nghỉ trưa 11h30-13h).
    Gộp: nửa sáng (giờ < 12) = 1 nến H4, nửa chiều (giờ >= 12) = 1 nến H4.
    → Mỗi ngày đúng 2 nến H4 đều đặn, không lệch như resample('4h')."""
    d = df.copy()
    d = d.sort_values("date").reset_index(drop=True)
    d["_day"] = d["date"].dt.date
    d["_sess"] = (d["date"].dt.hour >= 12).map({False: 0, True: 1})  # 0=sáng, 1=chiều
    agg = (d.groupby(["_day", "_sess"])
             .agg(date=("date", "first"), open=("open", "first"),
                  high=("high", "max"), low=("low", "min"),
                  close=("close", "last"), volume=("volume", "sum"))
             .reset_index(drop=True)
             .sort_values("date").reset_index(drop=True))
    return agg

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
    VHMULT  = cfg.get("vol_hunter_mult", 1.5)
    USEFIBO = cfg.get("use_fibo", True)
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
    # v7_2: volume tăng dần cho T1/T2, volume đột biến cho T3
    df["vol_origin"] = (~UVOL) | (df["volume"].shift(3) >= df["vol_ma"].shift(3) * 1.0)
    df["vol_hunter_ok"] = (~UVOL) | (df["volume"].shift(1) >= df["vol_ma"].shift(1) * VHMULT)
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

    # v8: Fibo Retracement 38.2/50/61.8% từ swing (ZLEN nến)
    fibo_range = df["resist"] - df["support"]
    fibo_zone  = df["atr14"] * 0.8
    if USEFIBO:
        # Fibo HT (uptrend): các mức tính từ resist xuống
        f618 = df["resist"] - fibo_range * 0.618
        f500 = df["resist"] - fibo_range * 0.500
        f382 = df["resist"] - fibo_range * 0.382
        df["in_fibo_ht"] = ((df["close"] - f618).abs() <= fibo_zone) | \
                           ((df["close"] - f500).abs() <= fibo_zone) | \
                           ((df["close"] - f382).abs() <= fibo_zone)
        # Fibo KC (downtrend): các mức tính từ support lên
        g382 = df["support"] + fibo_range * 0.382
        g500 = df["support"] + fibo_range * 0.500
        g618 = df["support"] + fibo_range * 0.618
        df["in_fibo_kc"] = ((df["close"] - g382).abs() <= fibo_zone) | \
                           ((df["close"] - g500).abs() <= fibo_zone) | \
                           ((df["close"] - g618).abs() <= fibo_zone)
    else:
        df["in_fibo_ht"] = False
        df["in_fibo_kc"] = False

    # v8: vùng đầy đủ = KC/HT chính HOẶC Fibo (thuận chiều xu hướng)
    df["in_ht_full"] = df["in_support"] | (is_up & df["in_fibo_ht"])
    df["in_kc_full"] = df["in_resist"]  | (is_dn & df["in_fibo_kc"])

    df["cond2_zone"] = ((is_up & df["in_ht_full"]) |
                        (is_dn & df["in_kc_full"]) |
                        (is_sw & (df["in_resist"] | df["in_support"])))

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
    # v7_2: volume tăng dần qua chuỗi nến (nến gốc [3] cũng phải đủ) cho T1/T2
    # v7__1: tách v_origin (nến gốc [3] >= 1.0x) cho T1
    de["vol_origin_e"] = (~UVOL) | (de["volume"].shift(3) >= de["vol_ma_e"].shift(3) * 1.0)
    # v7__1: volume đột biến mạnh (>=1.5x) cho T3 SL-Hunter
    de["vol_hunter_ok_e"] = (~UVOL) | (de["volume"].shift(1) >= de["vol_ma_e"].shift(1) * VHMULT)

    def se(col, i): return de[col].shift(i)
    vse   = de["vol_sig_ok_e"]
    vfbe  = de["vol_fb_ok_e"]
    vorie = de["vol_origin_e"]
    vhune = de["vol_hunter_ok_e"]

    # Tín hiệu nến trên khung entry
    de["t1s_up_e"] = (se("is_bear_e",2) & se("big_body_e",2) & se("sig_big_e",2) &
                      se("is_bull_e",1) & se("big_body_e",1) & se("sig_big_e",1) &
                      (se("close",1) > se("open",2)) & vse)
    de["t1s_dn_e"] = (se("is_bull_e",2) & se("big_body_e",2) & se("sig_big_e",2) &
                      se("is_bear_e",1) & se("big_body_e",1) & se("sig_big_e",1) &
                      (se("close",1) < se("open",2)) & vse)
    de["t1_up_e"] = (se("is_bear_e",3) & se("big_body_e",3) & se("sig_big_e",3) &
                     se("is_bull_e",2) & se("lw_dn_e",2)   & se("sig_big_e",2) &
                     se("is_bull_e",1) & se("big_body_e",1) & se("sig_big_e",1) & vse & vorie)
    de["t1_dn_e"] = (se("is_bull_e",3) & se("big_body_e",3) & se("sig_big_e",3) &
                     se("is_bear_e",2) & se("lw_up_e",2)   & se("sig_big_e",2) &
                     se("is_bear_e",1) & se("big_body_e",1) & se("sig_big_e",1) & vse & vorie)
    de["t2_up_e"] = (se("is_bear_e",3) & se("sig_big_e",3) &
                     se("is_bear_e",2) & se("sig_big_e",2) & (se("body_e",2) < se("body_e",3)*0.85) &
                     se("sig_big_e",1) & (se("body_e",1) < se("body_e",2)*0.85) &
                     (se("lw_dn_e",1) | se("is_bull_e",1)) & vse)
    de["t2_dn_e"] = (se("is_bull_e",3) & se("sig_big_e",3) &
                     se("is_bull_e",2) & se("sig_big_e",2) & (se("body_e",2) < se("body_e",3)*0.85) &
                     se("sig_big_e",1) & (se("body_e",1) < se("body_e",2)*0.85) &
                     (se("lw_up_e",1) | se("is_bear_e",1)) & vse)
    de["t3_up_e"] = (se("lw_dn_e",2) & se("sig_big_e",2) &
                     se("lw_dn_e",1) & se("sig_big_e",1) &
                     (se("low",1) < se("low",2)) & (se("close",1) > se("close",2)) & vhune)
    de["t3_dn_e"] = (se("lw_up_e",2) & se("sig_big_e",2) &
                     se("lw_up_e",1) & se("sig_big_e",1) &
                     (se("high",1) > se("high",2)) & (se("close",1) < se("close",2)) & vhune)
    # v8: T3 Max (Fakey) — nến dài [4] → 2 nến nhỏ tích lũy [3][2] → nến cuối [1] quét SL
    def fakey_inner_e(i):
        return ((se("high",i) <= se("high",4)) & (se("low",i) >= se("low",4)) &
                (se("body_e",i) < se("body_e",4)*0.5))
    de["t3max_up_e"] = (se("sig_big_e",4) & se("is_bear_e",4) &
                        fakey_inner_e(3) & fakey_inner_e(2) &
                        se("lw_dn_e",1) & se("sig_big_e",1) &
                        (se("low",1) < se("low",4)) & (se("close",1) > se("open",4)) & vhune)
    de["t3max_dn_e"] = (se("sig_big_e",4) & se("is_bull_e",4) &
                        fakey_inner_e(3) & fakey_inner_e(2) &
                        se("lw_up_e",1) & se("sig_big_e",1) &
                        (se("high",1) > se("high",4)) & (se("close",1) < se("open",4)) & vhune)

    # False Breakout trên H4: dùng support/resist từ khung D (df_zone = df)
    # Lấy support/resist hiện tại (nến D cuối) áp cho nến H4
    cur_support = df["support"].iloc[-1] if "support" in df.columns and len(df) else np.nan
    cur_resist  = df["resist"].iloc[-1]  if "resist"  in df.columns and len(df) else np.nan
    cur_in_ht   = bool(df["in_ht_full"].iloc[-1]) if "in_ht_full" in df.columns and len(df) else False
    cur_in_kc   = bool(df["in_kc_full"].iloc[-1]) if "in_kc_full" in df.columns and len(df) else False

    if not np.isnan(cur_support):
        de["fb_up_e"] = ((se("low",1) < cur_support) & (se("close",1) > cur_support) &
                         se("lw_dn_e",1) & se("sig_big_e",1) & vfbe)
    else:
        de["fb_up_e"] = pd.Series(False, index=de.index)
    if not np.isnan(cur_resist):
        de["fb_dn_e"] = ((se("high",1) > cur_resist) & (se("close",1) < cur_resist) &
                         se("lw_up_e",1) & se("sig_big_e",1) & vfbe)
    else:
        de["fb_dn_e"] = pd.Series(False, index=de.index)
    # Hammer/Shooting Star tại vùng KC/HT (vùng từ D)
    de["hmr_e"] = (cur_in_ht & se("lw_dn_e",1) & se("sig_big_e",1) &
                   (se("wick_dn_e",1) >= se("body_e",1)*2.0) & vse)
    de["shs_e"] = (cur_in_kc & se("lw_up_e",1) & se("sig_big_e",1) &
                   (se("wick_up_e",1) >= se("body_e",1)*2.0) & vse)

    # v8: Cảnh báo 4+ nến râu dài liên tiếp = tín hiệu yếu đi (Bí Kíp Ti Nến Phần V)
    de["consec_wick_dn_e"] = (se("lw_dn_e",1) & se("lw_dn_e",2) &
                              se("lw_dn_e",3) & se("lw_dn_e",4))
    de["consec_wick_up_e"] = (se("lw_up_e",1) & se("lw_up_e",2) &
                              se("lw_up_e",3) & se("lw_up_e",4))
    de["signal_weak_e"] = de["consec_wick_dn_e"] | de["consec_wick_up_e"]

    # Lấy tín hiệu của nến entry mới nhất (đầy đủ như Pine: t3max,t3,fb,t1,t1s,hmr,t2)
    last_e = de.iloc[-1]
    entry_bull = bool(last_e.get("t3max_up_e",False) or last_e.get("t3_up_e",False) or
                      last_e.get("fb_up_e",False) or last_e.get("t1_up_e",False) or
                      last_e.get("t1s_up_e",False) or last_e.get("hmr_e",False) or
                      last_e.get("t2_up_e",False))
    entry_bear = bool(last_e.get("t3max_dn_e",False) or last_e.get("t3_dn_e",False) or
                      last_e.get("fb_dn_e",False) or last_e.get("t1_dn_e",False) or
                      last_e.get("t1s_dn_e",False) or last_e.get("shs_e",False) or
                      last_e.get("t2_dn_e",False))
    def entry_sig_name():
        if entry_bull:
            if last_e.get("t3max_up_e"): return "⚡ T3 Max (Fakey) ↑"
            if last_e.get("t3_up_e"):  return "🎯 T3 SL-Hunter ↑"
            if last_e.get("fb_up_e"):  return "False Breakout ↑"
            if last_e.get("t1_up_e"):  return "T1 Đảo Chiều ↑"
            if last_e.get("t1s_up_e"): return "T1S Đảo Chiều ↑"
            if last_e.get("hmr_e"):    return "Hammer ↑"
            if last_e.get("t2_up_e"):  return "T2 Yếu Dần ↑"
        if entry_bear:
            if last_e.get("t3max_dn_e"): return "⚡ T3 Max (Fakey) ↓"
            if last_e.get("t3_dn_e"):  return "🎯 T3 SL-Hunter ↓"
            if last_e.get("fb_dn_e"):  return "False Breakout ↓"
            if last_e.get("t1_dn_e"):  return "T1 Đảo Chiều ↓"
            if last_e.get("t1s_dn_e"): return "T1S Đảo Chiều ↓"
            if last_e.get("shs_e"):    return "Shooting Star ↓"
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
    vs   = df["vol_sig_ok"]
    vfb  = df["vol_fb_ok"]
    vori = df["vol_origin"]
    vhun = df["vol_hunter_ok"]

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
                   s("is_bull",1) & s("big_body",1) & s("sig_big",1) & vs & vori)
    df["t1_dn"] = (s("is_bull",3) & s("big_body",3) & s("sig_big",3) &
                   s("is_bear",2) & s("lw_up",2)   & s("sig_big",2) &
                   s("is_bear",1) & s("big_body",1) & s("sig_big",1) & vs & vori)

    # T2: 3 nến bé dần + context close[3]<close[5]
    df["t2_up"] = (s("is_bear",3) & s("sig_big",3) &
                   s("is_bear",2) & s("sig_big",2) & (s("body",2) < s("body",3)*0.85) &
                   s("sig_big",1) & (s("body",1) < s("body",2)*0.85) &
                   (s("lw_dn",1) | s("is_bull",1)) & vs)
    df["t2_dn"] = (s("is_bull",3) & s("sig_big",3) &
                   s("is_bull",2) & s("sig_big",2) & (s("body",2) < s("body",3)*0.85) &
                   s("sig_big",1) & (s("body",1) < s("body",2)*0.85) &
                   (s("lw_up",1) | s("is_bear",1)) & vs)

    # T3: 2 nến râu, low[1]<low[2], close[1]>close[2]
    df["t3_up"] = (s("lw_dn",2) & s("sig_big",2) &
                   s("lw_dn",1) & s("sig_big",1) &
                   (s("low",1) < s("low",2)) & (s("close",1) > s("close",2)) & vhun)
    df["t3_dn"] = (s("lw_up",2) & s("sig_big",2) &
                   s("lw_up",1) & s("sig_big",1) &
                   (s("high",1) > s("high",2)) & (s("close",1) < s("close",2)) & vhun)

    # v8: T3 Max (Fakey) khung chính
    def fakey_inner(i):
        return ((s("high",i) <= s("high",4)) & (s("low",i) >= s("low",4)) &
                (s("body",i) < s("body",4)*0.5))
    df["t3max_up"] = (s("sig_big",4) & s("is_bear",4) &
                      fakey_inner(3) & fakey_inner(2) &
                      s("lw_dn",1) & s("sig_big",1) &
                      (s("low",1) < s("low",4)) & (s("close",1) > s("open",4)) & vhun)
    df["t3max_dn"] = (s("sig_big",4) & s("is_bull",4) &
                      fakey_inner(3) & fakey_inner(2) &
                      s("lw_up",1) & s("sig_big",1) &
                      (s("high",1) > s("high",4)) & (s("close",1) < s("open",4)) & vhun)

    # False Breakout (dùng vol_fb)
    df["fb_resist"]  = ((s("high",1) > df["resist"]) & (s("close",1) < df["resist"]) &
                        s("lw_up",1) & s("sig_big",1) & vfb)
    df["fb_support"] = ((s("low",1) < df["support"])  & (s("close",1) > df["support"]) &
                        s("lw_dn",1) & s("sig_big",1) & vfb)

    # Hammer/Shooting Star chỉ tại KC/HT
    df["hammer_up"]   = (df["in_ht_full"] & s("lw_dn",1) & s("sig_big",1) &
                         (s("wick_dn",1) >= s("body",1) * 2.0) & vs)
    df["shooting_dn"] = (df["in_kc_full"] & s("lw_up",1) & s("sig_big",1) &
                         (s("wick_up",1) >= s("body",1) * 2.0) & vs)

    df["bull_signal"] = (df["t3max_up"] | df["t3_up"] | df["t1_up"] | df["t1s_up"] | df["t2_up"] |
                         df["fb_support"] | df["hidden_div_bull"] | df["hammer_up"])
    df["bear_signal"] = (df["t3max_dn"] | df["t3_dn"] | df["t1_dn"] | df["t1s_dn"] | df["t2_dn"] |
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
    df.attrs["signal_weak"] = bool(last_e.get("signal_weak_e", False))

    # Lưu 5 tín hiệu H4 gần nhất (đồng bộ với DK3)
    de["bull_sig_e"] = (de["t3max_up_e"] | de["t3_up_e"] | de["t1_up_e"] | de["t1s_up_e"] |
                        de["fb_up_e"] | de["hmr_e"] | de["t2_up_e"])
    de["bear_sig_e"] = (de["t3max_dn_e"] | de["t3_dn_e"] | de["t1_dn_e"] | de["t1s_dn_e"] |
                        de["fb_dn_e"] | de["shs_e"] | de["t2_dn_e"])
    def esig_nm(row):
        if row["bull_sig_e"]:
            if row.get("t3max_up_e"): return "⚡ T3 Max (Fakey) ↑"
            if row.get("t3_up_e"):  return "🎯 T3 SL-Hunter ↑"
            if row.get("fb_up_e"):  return "False Breakout ↑"
            if row.get("t1_up_e"):  return "T1 Đảo Chiều ↑"
            if row.get("t1s_up_e"): return "T1S Đảo Chiều ↑"
            if row.get("hmr_e"):    return "Hammer ↑"
            if row.get("t2_up_e"):  return "T2 Yếu Dần ↑"
        if row["bear_sig_e"]:
            if row.get("t3max_dn_e"): return "⚡ T3 Max (Fakey) ↓"
            if row.get("t3_dn_e"):  return "🎯 T3 SL-Hunter ↓"
            if row.get("fb_dn_e"):  return "False Breakout ↓"
            if row.get("t1_dn_e"):  return "T1 Đảo Chiều ↓"
            if row.get("t1s_dn_e"): return "T1S Đảo Chiều ↓"
            if row.get("shs_e"):    return "Shooting Star ↓"
            if row.get("t2_dn_e"):  return "T2 Yếu Dần ↓"
        return "—"
    de["esig_name"] = de.apply(esig_nm, axis=1)
    # Nếu ENTRY là df_entry (H4) VÀ df chính cũng là H4 (cùng data) → gắn cột tín hiệu vào df
    # để chart vẽ marker khớp. So sánh bằng độ dài + date cuối.
    same_as_main = (df_entry is not None and not df_entry.empty and
                    len(de) == len(df) and
                    de["date"].iloc[-1] == df["date"].iloc[-1])
    if same_as_main:
        df["h4_bull_mk"] = de["bull_sig_e"].values
        df["h4_bear_mk"] = de["bear_sig_e"].values
    all_entry = de[de["bull_sig_e"] | de["bear_sig_e"]]
    # 5 gần nhất cho bảng
    entry_sigs = all_entry.tail(5).iloc[::-1]
    df.attrs["entry_signals"] = [
        {"date": str(r["date"])[:10], "close": f"{r['close']:,.0f}",
         "signal": str(r["esig_name"]),
         "dir": "↑" if r["bull_sig_e"] else "↓"}
        for _, r in entry_sigs.iterrows()]
    # Tất cả (trong 200 nến cuối) cho chart markers
    df.attrs["entry_signals_all"] = [
        {"date": str(r["date"])[:10],
         "dir": "↑" if r["bull_sig_e"] else "↓"}
        for _, r in all_entry.tail(200).iterrows()]

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

@app.route("/test-h4")
@login_required
def test_h4_multi():
    """Test nhanh data H4 của nhiều mã. Dùng: /test-h4?syms=GEL,TVN,TNG"""
    syms = request.args.get("syms", "GEL,TVN,TNG,VNM,HVN,VGT,PET").split(",")
    start = "2024-01-01"
    out = ["=== TEST H4 NHIỀU MÃ ===\n"]
    out.append(f"{'Mã':<8}{'Nguồn':<12}{'Nến H4':<8}{'Nến mới nhất':<22}{'Tín hiệu cuối'}")
    out.append("─" * 75)

    for sym in syms:
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            df_h4, src = get_intraday_h4(sym, start)
            last_date = str(df_h4["date"].iloc[-1]) if not df_h4.empty else "—"
            # Tính tín hiệu nhanh
            df_zone = get_data(sym, start, timeframe="1D")
            df_htf = get_data(sym, start, timeframe="1W")
            cfg = {"swing_len":3,"zone_len":50,"zone_atr_mult":1.5,"body_ratio":0.50,
                   "wick_ratio":0.40,"atr_mult":0.3,"use_volume":True,"vol_ma_len":20,
                   "vol_sig_mult":1.2,"vol_fb_mult":1.5,"vol_hunter_mult":1.5,
                   "use_rsi_div":True,"rsi_len":14,"rr_min":2.0,"account":100_000_000,
                   "risk_pct":2.0,"monthly_loss":6.0,"sl_pct":5.0}
            result = run_checklist(df_zone, df_htf, cfg, df_entry=df_h4)
            sig = result.attrs.get("entry_signal_name", "—")
            n_sig = len(result.attrs.get("entry_signals_all", []))
            out.append(f"{sym:<8}{src:<12}{len(df_h4):<8}{last_date:<22}{sig} ({n_sig} tín hiệu)")
        except Exception as e:
            out.append(f"{sym:<8}LỖI: {str(e)[:50]}")

    return "<pre style='background:#0d1117;color:#0f6;padding:20px;font-size:13px;line-height:1.6'>" + "\n".join(out) + "</pre>"

@app.route("/test-sync")
@login_required
def test_sync():
    """Test đồng nhất data giữa chart / tín hiệu bảng / DK3 / H4."""
    sym = request.args.get("sym", "GEL")
    start = "2024-01-01"
    out = []
    out.append(f"=== TEST ĐỒNG NHẤT DATA cho {sym} ===\n")

    try:
        df_h4, h4_src = get_intraday_h4(sym, start)
        df_zone = get_data(sym, start, timeframe="1D")
        df_htf  = get_data(sym, start, timeframe="1W")

        out.append(f"Nguồn H4    : {h4_src}")
        out.append(f"Số nến H4   : {len(df_h4)}")
        out.append(f"Số nến D    : {len(df_zone)}")
        out.append(f"Số nến W    : {len(df_htf)}")
        out.append("")

        cfg = {"swing_len":3,"zone_len":50,"zone_atr_mult":1.5,"body_ratio":0.50,
               "wick_ratio":0.40,"atr_mult":0.3,"use_volume":True,"vol_ma_len":20,
               "vol_sig_mult":1.2,"vol_fb_mult":1.5,"vol_hunter_mult":1.5,
               "use_rsi_div":True,"rsi_len":14,"rr_min":2.0,"account":100_000_000,
               "risk_pct":2.0,"monthly_loss":6.0,"sl_pct":5.0}

        result = run_checklist(df_zone, df_htf, cfg, df_entry=df_h4)

        # 1. DK3 (nến H4 cuối)
        dk3_name = result.attrs.get("entry_signal_name", "—")
        dk3_bull = result.attrs.get("entry_bull", False)
        dk3_bear = result.attrs.get("entry_bear", False)
        out.append("─── DK3 (nến H4 mới nhất) ───")
        out.append(f"  Tín hiệu: {dk3_name}")
        out.append(f"  Hướng: {'LONG' if dk3_bull else 'SHORT' if dk3_bear else 'Chờ'}")
        out.append("")

        # 2. Bảng 5 tín hiệu (entry_signals)
        sigs = result.attrs.get("entry_signals", [])
        out.append("─── BẢNG 5 tín hiệu H4 ───")
        for s in sigs:
            out.append(f"  {s['date']} | {s['dir']} | {s['signal']}")
        out.append("")

        # 3. Chart markers H4 (entry_signals_all)
        all_sigs = result.attrs.get("entry_signals_all", [])
        out.append(f"─── CHART markers H4: {len(all_sigs)} tín hiệu ───")
        out.append(f"  5 gần nhất: {[s['date']+s['dir'] for s in all_sigs[-5:]]}")
        out.append("")

        # 4. KIỂM TRA ĐỒNG NHẤT
        out.append("═══ KIỂM TRA ═══")
        # Bảng phải nằm trong all_sigs
        table_dates = set((s['date'], s['dir']) for s in sigs)
        all_dates = set((s['date'], s['dir']) for s in all_sigs)
        match = table_dates.issubset(all_dates)
        out.append(f"  Bảng ⊆ Chart markers: {'✅ KHỚP' if match else '❌ LỆCH'}")
        # DK3 phải = tín hiệu mới nhất trong all_sigs
        if all_sigs:
            latest = all_sigs[-1]
            latest_dir = "LONG" if latest['dir']=="↑" else "SHORT"
            dk3_dir = "LONG" if dk3_bull else "SHORT" if dk3_bear else "Chờ"
            # DK3 chỉ tính nến H4 CUỐI CÙNG, có thể khác tín hiệu gần nhất nếu nến cuối ko phải tín hiệu
            out.append(f"  DK3 hướng: {dk3_dir} | Tín hiệu H4 mới nhất: {latest_dir} ({latest['date']})")
        out.append("")
        out.append(f"  → Nếu KHỚP: chart, bảng, DK3 dùng CHUNG data H4 ({h4_src})")

    except Exception as e:
        import traceback
        out.append("LỖI: " + traceback.format_exc())

    return "<pre style='background:#0d1117;color:#0f6;padding:20px;font-size:12px;line-height:1.6'>" + "\n".join(out) + "</pre>"

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
            "vol_hunter_mult": float(data.get("vol_hunter_mult", 1.5)),
            "use_fibo":        bool(data.get("use_fibo", True)),
            "use_rsi_div":   bool(data.get("use_rsi_div", True)),
            "rsi_len":       int(data.get("rsi_len", 14)),
            "rr_min":        float(data.get("rr_min", 2.0)),
            "account":       float(data.get("account", 100_000_000)),
            "risk_pct":      float(data.get("risk_pct", 2.0)),
            "monthly_loss":  float(data.get("monthly_loss", 6.0)),
            "sl_pct":        float(data.get("sl_pct", 5.0)),
        }

        df_h4, h4_src = get_intraday_h4(symbol, start)   # DK3: H4
        df_zone = get_data(symbol, start, timeframe="1D")  # DK2: LUÔN khung Ngày (D)
        df_htf  = get_data(symbol, start, timeframe=tf_ref) # DK1: khung Tuần (W)

        # Chart hiển thị: theo nút D/H4 người dùng chọn (chỉ để xem, không ảnh hưởng checklist)
        if tf_chart == "4H":
            df_chart = df_h4.copy() if not df_h4.empty else df_zone.copy()
        else:
            df_chart = df_zone.copy()
        if df_zone.empty:
            return jsonify({"error": f"Không có dữ liệu cho '{symbol}'"}), 400

        # Checklist LUÔN tính: DK1=W (df_htf), DK2=D (df_zone), DK3=H4 (df_h4)
        # df_zone là khung chính cho KC/HT + R:R + score
        result   = run_checklist(df_zone, df_htf, cfg, df_entry=df_h4)

        # Chart: nếu xem H4, tính checklist riêng trên H4 để vẽ zones/markers khớp chart H4
        # (chỉ phục vụ hiển thị, KHÔNG ảnh hưởng điểm số chính)
        if tf_chart == "4H" and not df_h4.empty:
            chart_result = run_checklist(df_chart, df_htf, cfg, df_entry=df_h4)
        else:
            chart_result = result
        row      = result.iloc[-1]
        score    = int(row["score"])
        chart_df = chart_result.tail(200).copy()
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

        # v7: Khi chart H4, vẽ marker trực tiếp từ cột tín hiệu H4 (khớp bảng + DK3)
        if tf_chart == "4H" and "h4_bull_mk" in chart_df.columns:
            chart_data["bull_4"] = sig_recs(chart_df["h4_bull_mk"].fillna(False), "low")
            chart_data["bear_4"] = sig_recs(chart_df["h4_bear_mk"].fillna(False), "high")
            chart_data["bull_3"] = []
            chart_data["bear_3"] = []

        trend_txt = (f"▲ UPTREND [{row['trend_src']}]"  if row["is_uptrend"]   else
                     f"▼ DOWNTREND [{row['trend_src']}]" if row["is_downtrend"] else
                     "↔ SIDEWAY")
        zone_txt  = (f"KC: {row['resist']:,.0f}" if row["in_resist"] else
                     f"HT: {row['support']:,.0f}" if row["in_support"] else
                     "Giữa vùng — chờ")
        rr_val    = safe(row.get("rr_value", np.nan))
        rr_txt    = f"{rr_val:.2f}:1" if rr_val else "N/A"
        zone2_txt = f"{zone_txt} | R:R {rr_txt}"
        # v7_2: hiển thị đúng loại volume theo tín hiệu H4 đang active
        if cfg["use_volume"]:
            eb = result.attrs.get("entry_bull", False)
            ebr = result.attrs.get("entry_bear", False)
            esn = result.attrs.get("entry_signal_name", "")
            if "T3" in esn or "Hunter" in esn:
                vol_txt = " | Vol✅(đột biến)" if (eb or ebr) else " | Vol❌"
            elif "Breakout" in esn:
                vol_txt = " | VolFB"
            elif "T1" in esn or "T2" in esn:
                vol_txt = " | Vol✅(chuỗi)" if (eb or ebr) else " | Vol❌"
            else:
                vol_txt = " | Vol✅" if row["vol_sig_ok"] else " | Vol❌"
        else:
            vol_txt = ""
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
            "chart_tf":   (h4_src if tf_chart == "4H" else tf_chart),
            "chart_bars": len(chart_df),
            "h4_src":     h4_src,
            "signal_weak": result.attrs.get("signal_weak", False),
        }

        # Dùng tín hiệu H4 (đồng bộ với DK3) thay vì khung chính
        entry_sigs = result.attrs.get("entry_signals", [])
        signals = [{"date": e["date"], "close": e["close"],
                    "signal": e["signal"], "dir": e["dir"]}
                   for e in entry_sigs]

        return jsonify({"chart":chart_data, "summary":summary, "signals":signals})

    except Exception as e:
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": f"Lỗi '{data.get('symbol','?')}': {str(e)[:200]}"}), 500

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=PORT, debug=False)
