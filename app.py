"""
UG Checklist v2 — Web App (Flask)
Fixes:
  - Dùng 127.0.0.1 thay localhost (fix Chrome 403)
  - Tránh port 5000 (macOS AirPlay chiếm)
  - requirements.txt có yfinance làm fallback
  - safe() xử lý đủ loại NA/inf/bool
  - Mở browser chờ server sẵn sàng hẳn mới mở
"""
import sys, os, threading, warnings, logging
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import date

app = Flask(__name__)

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
#  LOGIC
# ═══════════════════════════════════════════════════════════════
def calc_atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))

def find_pivot_high(series, left, right):
    result = pd.Series(np.nan, index=series.index)
    for i in range(left, len(series)-right):
        w = series.iloc[i-left:i+right+1]
        if series.iloc[i] == w.max() and series.iloc[i] > series.iloc[i-1]:
            result.iloc[i] = series.iloc[i]
    return result

def find_pivot_low(series, left, right):
    result = pd.Series(np.nan, index=series.index)
    for i in range(left, len(series)-right):
        w = series.iloc[i-left:i+right+1]
        if series.iloc[i] == w.min() and series.iloc[i] < series.iloc[i-1]:
            result.iloc[i] = series.iloc[i]
    return result

def _normalize(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    rename = {"time":"date","tradingdate":"date","open":"open",
              "high":"high","low":"low","close":"close","volume":"volume"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if "date" not in df.columns:
        df = df.reset_index(); df.columns = [c.lower() for c in df.columns]
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
            log.info(f"vnstock legacy OK: {symbol} {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"vnstock legacy: {e}")

    try:
        from vnstock3 import Vnstock
        df = Vnstock().stock(symbol=symbol, source="VCI").quote.history(start=start, end=end, interval=interval)
        df = _normalize(df)
        if not df.empty:
            log.info(f"vnstock3 OK: {symbol} {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"vnstock3: {e}")

    try:
        import yfinance as yf
        ticker = symbol+".VN" if not symbol.endswith(".VN") else symbol
        df = _normalize(yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False).reset_index())
        if not df.empty:
            log.info(f"yfinance OK: {ticker} {len(df)} nến")
            return df
    except Exception as e:
        log.warning(f"yfinance: {e}")

    raise RuntimeError(f"Không lấy được data cho '{symbol}'. Kiểm tra mã CK và kết nối internet.")

def run_checklist(df_ltf, df_htf, cfg):
    SWING  = cfg["swing_len"];  ZLEN   = cfg["zone_len"]
    ZATR   = cfg["zone_atr_mult"]; BODY = cfg["body_ratio"]
    WICK   = cfg["wick_ratio"]; ATRM   = cfg["atr_mult"]
    UVOL   = cfg["use_volume"]; VMALEN = cfg["vol_ma_len"]
    URSIDIV= cfg["use_rsi_div"]; RSILEN = cfg["rsi_len"]
    ACC    = cfg["account"];    RISKP  = cfg["risk_pct"]
    SLP    = cfg["sl_pct"]

    df = df_ltf.copy()
    df["atr14"]   = calc_atr(df, 14)
    df["body"]    = (df["close"]-df["open"]).abs()
    df["range"]   = df["high"]-df["low"]
    df["wick_up"] = df["high"]-df[["open","close"]].max(axis=1)
    df["wick_dn"] = df[["open","close"]].min(axis=1)-df["low"]
    df["is_bull"] = df["close"] > df["open"]
    df["is_bear"] = df["close"] < df["open"]
    rng = df["range"].replace(0, np.nan)
    df["big_body"]= (df["range"]>0)&(df["body"]/rng >= BODY)
    df["lw_up"]   = (df["range"]>0)&(df["wick_up"]/rng >= WICK)
    df["lw_dn"]   = (df["range"]>0)&(df["wick_dn"]/rng >= WICK)
    df["sig_big"] = df["range"] >= df["atr14"]*ATRM
    df["vol_ma"]  = df["volume"].rolling(VMALEN).mean()
    df["vol_ok"]  = (~UVOL)|(df["volume"] >= df["vol_ma"])

    htf = df_htf.copy()
    htf["ph"] = find_pivot_high(htf["high"], SWING, SWING)
    htf["pl"] = find_pivot_low(htf["low"],   SWING, SWING)
    ph_vals = htf["ph"].dropna().values
    pl_vals = htf["pl"].dropna().values
    last_ph = ph_vals[-1] if len(ph_vals)>=1 else np.nan
    prev_ph = ph_vals[-2] if len(ph_vals)>=2 else np.nan
    last_pl = pl_vals[-1] if len(pl_vals)>=1 else np.nan
    prev_pl = pl_vals[-2] if len(pl_vals)>=2 else np.nan

    def ok(x): return not (isinstance(x, float) and np.isnan(x))
    is_up  = all(ok(x) for x in [last_ph,prev_ph,last_pl,prev_pl]) and last_ph>prev_ph and last_pl>prev_pl
    is_dn  = all(ok(x) for x in [last_ph,prev_ph,last_pl,prev_pl]) and last_ph<prev_ph and last_pl<prev_pl
    is_sw  = not is_up and not is_dn

    df["is_uptrend"]   = is_up
    df["is_downtrend"] = is_dn
    df["is_sideways"]  = is_sw
    df["cond1_ok"]     = is_up or is_dn

    df["resist"]      = df["high"].shift(1).rolling(ZLEN).max()
    df["support"]     = df["low"].shift(1).rolling(ZLEN).min()
    df["zone_w"]      = df["atr14"]*ZATR
    df["near_resist"] = (df["close"]-df["resist"]).abs() <= df["zone_w"]
    df["near_support"]= (df["close"]-df["support"]).abs() <= df["zone_w"]
    df["cond2_ok"]    = ((is_up&df["near_support"])|(is_dn&df["near_resist"])|(is_sw&(df["near_support"]|df["near_resist"])))

    df["rsi"]    = calc_rsi(df["close"], RSILEN)
    df["rsi_ph"] = find_pivot_high(df["rsi"],   SWING, SWING)
    df["rsi_pl"] = find_pivot_low(df["rsi"],    SWING, SWING)
    df["prc_ph"] = find_pivot_high(df["close"], SWING, SWING)
    df["prc_pl"] = find_pivot_low(df["close"],  SWING, SWING)
    for col in ["rsi_ph","rsi_pl","prc_ph","prc_pl"]:
        df[f"r_{col}"] = df[col].ffill()
        df[f"p_{col}"] = df[col].shift(1).where(df[col].notna()).ffill()

    df["hidden_div_bull"] = (URSIDIV &
        df["r_prc_pl"].notna()&df["p_prc_pl"].notna()&
        df["r_rsi_pl"].notna()&df["p_rsi_pl"].notna()&
        (df["r_prc_pl"]>df["p_prc_pl"])&(df["r_rsi_pl"]<df["p_rsi_pl"]))
    df["hidden_div_bear"] = (URSIDIV &
        df["r_prc_ph"].notna()&df["p_prc_ph"].notna()&
        df["r_rsi_ph"].notna()&df["p_rsi_ph"].notna()&
        (df["r_prc_ph"]<df["p_prc_ph"])&(df["r_rsi_ph"]>df["p_rsi_ph"]))

    def s(col, i): return df[col].shift(i)
    vo = df["vol_ok"]

    df["t1s_up"]= s("is_bear",1)&s("big_body",1)&s("sig_big",1)&df["is_bull"]&df["big_body"]&df["sig_big"]&(df["close"]>s("open",1))&(df["open"]<s("close",1))&vo
    df["t1s_dn"]= s("is_bull",1)&s("big_body",1)&s("sig_big",1)&df["is_bear"]&df["big_body"]&df["sig_big"]&(df["close"]<s("open",1))&(df["open"]>s("close",1))&vo
    df["t1_up"] = s("is_bear",2)&s("big_body",2)&s("sig_big",2)&s("is_bull",1)&s("lw_dn",1)&s("sig_big",1)&df["is_bull"]&df["big_body"]&df["sig_big"]&(df["close"]>s("close",2))&vo
    df["t1_dn"] = s("is_bull",2)&s("big_body",2)&s("sig_big",2)&s("is_bear",1)&s("lw_up",1)&s("sig_big",1)&df["is_bear"]&df["big_body"]&df["sig_big"]&(df["close"]<s("close",2))&vo
    df["t2_up"] = s("is_bear",2)&s("sig_big",2)&s("is_bear",1)&s("sig_big",1)&(s("body",1)<s("body",2)*.75)&df["sig_big"]&(df["body"]<s("body",1)*.75)&(df["lw_dn"]|df["is_bull"])&vo
    df["t2_dn"] = s("is_bull",2)&s("sig_big",2)&s("is_bull",1)&s("sig_big",1)&(s("body",1)<s("body",2)*.75)&df["sig_big"]&(df["body"]<s("body",1)*.75)&(df["lw_up"]|df["is_bear"])&vo
    df["t2s_up"]= s("is_bear",2)&s("sig_big",2)&s("is_bear",1)&s("sig_big",1)&(s("body",1)<s("body",2)*.75)&df["is_bull"]&df["sig_big"]&(df["low"]<s("low",1))&vo
    df["t2s_dn"]= s("is_bull",2)&s("sig_big",2)&s("is_bull",1)&s("sig_big",1)&(s("body",1)<s("body",2)*.75)&df["is_bear"]&df["sig_big"]&(df["high"]>s("high",1))&vo
    df["t3_up"] = s("lw_dn",1)&s("sig_big",1)&df["lw_dn"]&df["sig_big"]&(df["low"]<s("low",1))&(df["close"]>s("close",1))&vo
    df["t3_dn"] = s("lw_up",1)&s("sig_big",1)&df["lw_up"]&df["sig_big"]&(df["high"]>s("high",1))&(df["close"]<s("close",1))&vo

    fb_vol_ok        = (~UVOL)|(df["volume"]<=df["vol_ma"]*1.5)
    df["fb_resist"]  = (df["high"]>df["resist"])&(df["close"]<df["resist"])&df["lw_up"]&df["sig_big"]&fb_vol_ok
    df["fb_support"] = (df["low"]<df["support"])&(df["close"]>df["support"])&df["lw_dn"]&df["sig_big"]&fb_vol_ok

    df["bull_signal"] = df["t1_up"]|df["t1s_up"]|df["t2_up"]|df["t2s_up"]|df["t3_up"]|df["fb_support"]|df["hidden_div_bull"]
    df["bear_signal"] = df["t1_dn"]|df["t1s_dn"]|df["t2_dn"]|df["t2s_dn"]|df["t3_dn"]|df["fb_resist"]|df["hidden_div_bear"]
    df["cond3_ok"]    = (is_up&df["bull_signal"])|(is_dn&df["bear_signal"])|(is_sw&(df["bull_signal"]|df["bear_signal"]))

    sl_dist        = df["close"]*SLP/100
    pos_size       = (ACC*RISKP/100)/sl_dist.replace(0, np.nan)
    df["pos_size"] = pos_size
    df["cond4_ok"] = bool((SLP>=0.3)and(SLP<=20.0)and(RISKP<=2.0))&(pos_size>0)

    df["score"]  = (df["cond1_ok"].astype(int)+df["cond2_ok"].astype(int)+
                    df["cond3_ok"].astype(int)+df["cond4_ok"].astype(int))
    df["all_ok"] = df["score"]==4

    def sig_name(row):
        if row["bull_signal"]:
            for k,v in [("t3_up","🎯 T3 SL-Hunter ↑"),("fb_support","⚠️ False Breakout ↑"),
                        ("hidden_div_bull","📊 Phân Kỳ Ẩn ↑"),("t1_up","T1 Đảo Chiều ↑"),
                        ("t1s_up","T1S Đảo Chiều ↑"),("t2s_up","T2S Đảo Chiều ↑"),("t2_up","T2 Yếu Dần ↑")]:
                if row[k]: return v
        if row["bear_signal"]:
            for k,v in [("t3_dn","🎯 T3 SL-Hunter ↓"),("fb_resist","⚠️ False Breakout ↓"),
                        ("hidden_div_bear","📊 Phân Kỳ Ẩn ↓"),("t1_dn","T1 Đảo Chiều ↓"),
                        ("t1s_dn","T1S Đảo Chiều ↓"),("t2s_dn","T2S Đảo Chiều ↓"),("t2_dn","T2 Yếu Dần ↓")]:
                if row[k]: return v
        return "—"

    df["signal_name"] = df.apply(sig_name, axis=1)
    return df

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data   = request.json
        symbol = data.get("symbol","VNM").upper().strip()
        start  = data.get("start_date","2024-01-01")
        tf_ref = data.get("tf_ref","1W")
        cfg = {
            "swing_len":     int(data.get("swing_len",5)),
            "zone_len":      int(data.get("zone_len",20)),
            "zone_atr_mult": float(data.get("zone_atr_mult",0.5)),
            "body_ratio":    float(data.get("body_ratio",0.60)),
            "wick_ratio":    float(data.get("wick_ratio",0.50)),
            "atr_mult":      float(data.get("atr_mult",0.6)),
            "use_volume":    bool(data.get("use_volume",True)),
            "vol_ma_len":    int(data.get("vol_ma_len",20)),
            "use_rsi_div":   bool(data.get("use_rsi_div",True)),
            "rsi_len":       int(data.get("rsi_len",14)),
            "account":       float(data.get("account",100_000_000)),
            "risk_pct":      float(data.get("risk_pct",2.0)),
            "monthly_loss":  float(data.get("monthly_loss",6.0)),
            "sl_pct":        float(data.get("sl_pct",2.0)),
        }

        df_main = get_data(symbol, start, timeframe="1D")
        df_htf  = get_data(symbol, start, timeframe=tf_ref)
        if df_main.empty:
            return jsonify({"error": f"Không có dữ liệu cho mã '{symbol}'"}), 400

        result   = run_checklist(df_main, df_htf, cfg)
        row      = result.iloc[-1]
        score    = int(row["score"])
        chart_df = result.tail(200).copy()
        chart_df["date_str"] = chart_df["date"].dt.strftime("%Y-%m-%d")

        def sl(col): return [safe(x) for x in chart_df[col]]
        def sig_recs(mask, pcol):
            sub = chart_df[mask][["date_str",pcol]].rename(columns={"date_str":"date"})
            return [{k:safe(v) for k,v in r.items()} for r in sub.to_dict("records")]

        chart_data = {
            "dates": chart_df["date_str"].tolist(),
            "open": sl("open"), "high": sl("high"), "low": sl("low"),
            "close": sl("close"), "volume": sl("volume"), "vol_ma": sl("vol_ma"),
            "resist_top": [safe(a+b) for a,b in zip(chart_df["resist"],chart_df["zone_w"])],
            "resist_bot": [safe(a-b) for a,b in zip(chart_df["resist"],chart_df["zone_w"])],
            "support_top":[safe(a+b) for a,b in zip(chart_df["support"],chart_df["zone_w"])],
            "support_bot":[safe(a-b) for a,b in zip(chart_df["support"],chart_df["zone_w"])],
            "bull_4": sig_recs(chart_df["all_ok"]&chart_df["bull_signal"],"low"),
            "bear_4": sig_recs(chart_df["all_ok"]&chart_df["bear_signal"],"high"),
            "bull_3": sig_recs((chart_df["score"]==3)&chart_df["bull_signal"],"low"),
            "bear_3": sig_recs((chart_df["score"]==3)&chart_df["bear_signal"],"high"),
        }

        trend_txt = "▲ UPTREND" if row["is_uptrend"] else "▼ DOWNTREND" if row["is_downtrend"] else "↔ SIDEWAY"
        zone_txt  = (f"KC: {row['resist']:,.0f}" if row["near_resist"] else
                     f"HT: {row['support']:,.0f}" if row["near_support"] else "Giữa vùng — chờ")
        vol_txt   = (" | Vol✅" if row["vol_ok"] else " | Vol❌") if cfg["use_volume"] else ""
        max_risk  = cfg["account"]*cfg["risk_pct"]/100
        pos       = safe(row["pos_size"]) or 0
        rsi_val   = safe(row["rsi"]) or 0

        summary = {
            "symbol": symbol, "date": str(row["date"])[:10],
            "close":  f"{row['close']:,.0f}", "score": score,
            "action": ["🔴 ĐỨNG NGOÀI","🔴 ĐỨNG NGOÀI","🟠 CHỜ THÊM","🟡 THEO DÕI SÁT","🟢 VÀO LỆNH"][score],
            "direction": "↑ LONG" if bool(row["bull_signal"]) else "↓ SHORT" if bool(row["bear_signal"]) else "— Chờ tín hiệu",
            "cond1": bool(row["cond1_ok"]), "cond1_txt": trend_txt,
            "cond2": bool(row["cond2_ok"]), "cond2_txt": zone_txt,
            "cond3": bool(row["cond3_ok"]), "cond3_txt": str(row["signal_name"])+vol_txt,
            "cond4": bool(row["cond4_ok"]), "cond4_txt": f"Risk {max_risk:,.0f}₫ | {pos:,.0f} cổ",
            "rsi": f"{rsi_val:.1f}",
            "rsi_signal": ("📈 Phân kỳ ẩn tăng" if bool(row["hidden_div_bull"]) else
                           "📉 Phân kỳ ẩn giảm" if bool(row["hidden_div_bear"]) else "Bình thường"),
            "monthly_loss_vnd": f"{cfg['account']*cfg['monthly_loss']/100:,.0f}",
        }

        sig_df  = result[result["bull_signal"]|result["bear_signal"]].tail(5).iloc[::-1]
        signals = [{"date":str(r["date"])[:10],"close":f"{r['close']:,.0f}",
                    "score":int(r["score"]),"signal":str(r["signal_name"]),
                    "action":["❌","❌","🟠","🟡","🟢"][int(r["score"])]}
                   for _,r in sig_df.iterrows()]

        return jsonify({"chart":chart_data,"summary":summary,"signals":signals})

    except Exception as e:
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=PORT, debug=False)
