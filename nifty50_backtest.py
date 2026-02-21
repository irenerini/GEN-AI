import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date as date_type

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[INFO] TensorFlow not found — LSTM model skipped.")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
TICKER          = "^NSEI"
INTERVAL        = "1d"
N_DAYS          = 5     # trading days ahead to predict at each test point
BACKTEST_MONTHS = 6     # how many months to walk forward over
STEP_DAYS       = 5     # re-train / predict every N trading days (~weekly)
LR_WINDOW       = 30    # look-back window for Linear Regression features
SEQ_LEN         = 40    # LSTM sequence length
# ════════════════════════════════════════════════════════════════════════════

FEAT_LR   = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "ATR", "Momentum5"]
FEAT_LSTM = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "ATR", "BB_width"]


# ── Indicator computation ────────────────────────────────────────────────────
def compute_indicators(d):
    d = d.copy()
    d["EMA9"]   = d["Close"].ewm(span=9,  adjust=False).mean()
    d["EMA21"]  = d["Close"].ewm(span=21, adjust=False).mean()
    d["SMA50"]  = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    bb_mid = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = bb_mid + 2 * bb_std
    d["BB_lower"] = bb_mid - 2 * bb_std
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / bb_mid.replace(0, np.nan)
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]      = ema12 - ema26
    d["MACDsig"]   = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACDsig"]
    hl  = d["High"] - d["Low"]
    hpc = (d["High"] - d["Close"].shift()).abs()
    lpc = (d["Low"]  - d["Close"].shift()).abs()
    d["ATR"]        = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    d["Momentum5"]  = d["Close"].pct_change(5)
    d["Momentum10"] = d["Close"].pct_change(10)
    d["Vol_ratio"]  = d["Volume"] / d["Volume"].rolling(20).mean()
    return d


# ── Helpers ──────────────────────────────────────────────────────────────────
def next_trading_day(d):
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def trading_days_between(start, end):
    """Return list of trading days (Mon–Fri) in [start, end)."""
    days = []
    cur = start
    while cur < end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ── LR helpers ───────────────────────────────────────────────────────────────
def train_lr(df):
    data = df[FEAT_LR].values
    X, y = [], []
    for i in range(LR_WINDOW, len(data) - 1):
        X.append(data[i - LR_WINDOW:i].flatten())
        y.append(data[i + 1, 3])
    X, y = np.array(X), np.array(y)
    sx = MinMaxScaler().fit(X)
    sy = MinMaxScaler().fit(y.reshape(-1, 1))
    m  = LinearRegression().fit(sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel())
    return m, sx, sy


def predict_lr(df, m, sx, sy):
    x = df[FEAT_LR].values[-LR_WINDOW:].flatten().reshape(1, -1)
    return float(sy.inverse_transform(m.predict(sx.transform(x)).reshape(-1, 1))[0, 0])


# ── LSTM helpers ─────────────────────────────────────────────────────────────
def train_lstm(df):
    if not LSTM_AVAILABLE:
        return None, None
    data_l = df[FEAT_LSTM].values
    sc     = MinMaxScaler().fit(data_l)
    ds     = sc.transform(data_l)
    X_l, y_l = [], []
    for i in range(SEQ_LEN, len(ds) - 1):
        X_l.append(ds[i - SEQ_LEN:i])
        y_l.append(ds[i + 1, 3])
    if len(X_l) < 10:
        return None, None
    X_l, y_l = np.array(X_l), np.array(y_l)
    sp = int(len(X_l) * 0.85)
    m  = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEAT_LSTM))),
        Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    m.fit(X_l[:sp], y_l[:sp], epochs=60, batch_size=16,
          validation_data=(X_l[sp:], y_l[sp:]),
          callbacks=[EarlyStopping(patience=6, restore_best_weights=True)], verbose=0)
    return m, sc


def predict_lstm(df, lstm_m, lstm_sc):
    if lstm_m is None:
        return None
    seq = lstm_sc.transform(df[FEAT_LSTM].values[-SEQ_LEN:]).reshape(1, SEQ_LEN, len(FEAT_LSTM))
    ps  = float(lstm_m.predict(seq, verbose=0)[0, 0])
    dummy = np.zeros((1, len(FEAT_LSTM)))
    dummy[0, 3] = ps
    return float(lstm_sc.inverse_transform(dummy)[0, 3])


def rule_score(row):
    s  = 1 if row["EMA9"]  > row["EMA21"]  else -1
    s += 1 if row["Close"] > row["SMA50"]  else -1
    s += 1 if row["Close"] > row["SMA200"] else -1
    if   row["RSI"] < 40: s += 1
    elif row["RSI"] > 65: s -= 1
    s += 1 if row["MACD_hist"] > 0 else -1
    bb_pos = (row["Close"] - row["BB_lower"]) / (row["BB_upper"] - row["BB_lower"] + 1e-9)
    if   bb_pos > 0.85: s -= 1
    elif bb_pos < 0.15: s += 1
    s += 1 if row["Momentum5"]  > 0 else -1
    s += 1 if row["Momentum10"] > 0 else -1
    return s


# ── Forecast N days from a trained model ─────────────────────────────────────
def forecast_n_days(base_df, lr_m, lr_sx, lr_sy, lstm_m, lstm_sc, n_days):
    """
    Roll forward n_days from the last row of base_df.
    Returns list of dicts with keys: step, pred_close, pred_open.
    """
    ohcv = base_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    current_date = base_df.index[-1].date()
    results = []

    for step in range(1, n_days + 1):
        pred_date = next_trading_day(current_date)
        df_ind = compute_indicators(ohcv).dropna()
        if df_ind.empty:
            break

        last_row = df_ind.iloc[-1]
        prev_close = float(last_row["Close"])
        atr        = float(last_row["ATR"])

        lr_c   = predict_lr(df_ind, lr_m, lr_sx, lr_sy)
        lstm_c = predict_lstm(df_ind, lstm_m, lstm_sc)
        preds  = [lr_c] + ([lstm_c] if lstm_c is not None else [])
        ens_c  = float(np.mean(preds))

        score  = rule_score(last_row)
        ens_c += score * atr * 0.04

        uncertainty = 1.0 + (step - 1) * 0.15
        p_open  = prev_close
        p_close = ens_c
        p_high  = max(p_open, p_close) + atr * 0.6 * uncertainty
        p_low   = min(p_open, p_close) - atr * 0.6 * uncertainty

        results.append({
            "step":       step,
            "pred_date":  pred_date,
            "pred_open":  p_open,
            "pred_close": p_close,
        })

        new_row = pd.DataFrame(
            {"Open": [p_open], "High": [p_high], "Low": [p_low],
             "Close": [p_close], "Volume": [float(ohcv["Volume"].iloc[-1])]},
            index=[pd.Timestamp(pred_date)]
        )
        ohcv = pd.concat([ohcv, new_row])
        current_date = pred_date

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ════════════════════════════════════════════════════════════════════════════
def main():
    today = datetime.today().date()

    # Backtest window — end at least N_DAYS*2 trading days before today so that
    # every test point has complete actual future data available for comparison.
    bt_end   = (datetime.combine(today, datetime.min.time())
                - timedelta(days=N_DAYS * 2 + 3)).date()
    bt_start = (datetime.combine(bt_end, datetime.min.time())
                - timedelta(days=int(BACKTEST_MONTHS * 30.5))).date()

    # Download enough history: backtest period + 2-year warm-up for indicators
    dl_start = (datetime.combine(bt_start, datetime.min.time())
                - timedelta(days=730)).strftime("%Y-%m-%d")
    dl_end   = (datetime.combine(bt_end, datetime.min.time())
                + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[INFO] Backtest period : {bt_start}  →  {bt_end}")
    print(f"[INFO] Downloading {TICKER} from {dl_start} ...")
    raw = yf.download(TICKER, start=dl_start, end=dl_end, interval=INTERVAL,
                      auto_adjust=True, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw.dropna(inplace=True)
    if raw.empty or len(raw) < 250:
        raise ValueError(f"Not enough data ({len(raw)} rows). Try a different date range.")
    print(f"    {len(raw)} candles downloaded  |  range: {raw.index[0].date()} → {raw.index[-1].date()}")

    # Identify trading days inside the backtest window
    all_bt_days = [d for d in raw.index.date if bt_start <= d <= bt_end]
    if not all_bt_days:
        raise ValueError("No trading days found in the backtest window.")

    # Choose test points every STEP_DAYS inside the backtest window
    test_dates = all_bt_days[::STEP_DAYS]
    print(f"    {len(test_dates)} test points (every {STEP_DAYS} trading days)\n")

    records = []  # one record per (test_date, step)

    for idx, test_date in enumerate(test_dates):
        # Training slice: all data strictly BEFORE test_date
        train_raw = raw[raw.index.date < test_date]
        if len(train_raw) < 250:
            continue

        train_df = compute_indicators(train_raw).dropna()
        if len(train_df) < LR_WINDOW + 10:
            continue

        # Train models
        try:
            lr_m, lr_sx, lr_sy = train_lr(train_df)
        except Exception as e:
            print(f"  [WARN] LR training failed at {test_date}: {e}")
            continue

        lstm_m, lstm_sc = None, None
        if LSTM_AVAILABLE:
            try:
                lstm_m, lstm_sc = train_lstm(train_df)
            except Exception as e:
                print(f"  [WARN] LSTM training failed at {test_date}: {e}")

        # Forecast N_DAYS ahead
        forecasts = forecast_n_days(train_df, lr_m, lr_sx, lr_sy, lstm_m, lstm_sc, N_DAYS)

        for fc in forecasts:
            actual_row = raw[raw.index.date == fc["pred_date"]]
            if actual_row.empty:
                continue
            actual_close = float(actual_row["Close"].iloc[0])

            # Previous actual close: last real candle strictly before pred_date
            prev_rows = raw[raw.index.date < fc["pred_date"]]
            if prev_rows.empty:
                continue
            prev_actual_close = float(prev_rows["Close"].iloc[-1])

            records.append({
                "test_date":        test_date,
                "step":             fc["step"],
                "pred_date":        fc["pred_date"],
                "actual_close":     actual_close,
                "pred_close":       fc["pred_close"],
                "prev_actual_close": prev_actual_close,
            })

        if (idx + 1) % 5 == 0 or idx == len(test_dates) - 1:
            print(f"  Processed {idx + 1}/{len(test_dates)} test points ...")

    if not records:
        raise ValueError("No backtest records produced. Check data availability.")

    results_df = pd.DataFrame(records)

    # ── Metrics ──────────────────────────────────────────────────────────────
    results_df["error"]      = results_df["pred_close"] - results_df["actual_close"]
    results_df["abs_error"]  = results_df["error"].abs()
    results_df["pct_error"]  = (results_df["abs_error"] / results_df["actual_close"]) * 100

    # Directional accuracy: both pred and actual measured against the same
    # reference — the previous actual close — so they are directly comparable.
    results_df["pred_dir"]   = (results_df["pred_close"] >= results_df["prev_actual_close"]).astype(int)
    results_df["actual_dir"] = (results_df["actual_close"] >= results_df["prev_actual_close"]).astype(int)
    results_df["dir_correct"] = (results_df["pred_dir"] == results_df["actual_dir"]).astype(int)

    # ── Summary table by step ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  6-Month Walk-Forward Backtest — {TICKER}  |  {len(test_dates)} test points")
    print("=" * 70)
    print(f"  {'Step':>4}  {'N':>5}  {'RMSE':>9}  {'MAE':>9}  {'MAPE%':>7}  {'DirAcc%':>8}")
    print("-" * 70)

    step_metrics = []
    for step in range(1, N_DAYS + 1):
        sub = results_df[results_df["step"] == step]
        if sub.empty:
            continue
        rmse     = float(np.sqrt((sub["error"] ** 2).mean()))
        mae      = float(sub["abs_error"].mean())
        mape     = float(sub["pct_error"].mean())
        dir_acc  = float(sub["dir_correct"].mean() * 100)
        n        = len(sub)
        step_metrics.append({"step": step, "n": n, "rmse": rmse,
                              "mae": mae, "mape": mape, "dir_acc": dir_acc})
        print(f"  Day {step:>1}  {n:>5}  {rmse:>9,.1f}  {mae:>9,.1f}  {mape:>7.2f}  {dir_acc:>8.1f}")

    print("=" * 70)

    overall_dir_acc  = float(results_df["dir_correct"].mean() * 100)
    overall_mape     = float(results_df["pct_error"].mean())
    overall_rmse     = float(np.sqrt((results_df["error"] ** 2).mean()))
    print(f"\n  Overall RMSE   : {overall_rmse:,.1f}")
    print(f"  Overall MAE    : {float(results_df['abs_error'].mean()):,.1f}")
    print(f"  Overall MAPE   : {overall_mape:.2f}%")
    print(f"  Overall DirAcc : {overall_dir_acc:.1f}%")

    # ── Chart ─────────────────────────────────────────────────────────────────
    BG = "#0d1117"
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.patch.set_facecolor(BG)

    def style_ax(ax):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors="white")
        ax.grid(color="#21262d", linewidth=0.4)

    for ax in axes:
        style_ax(ax)

    # Panel 1: Actual vs Predicted close for Day-1 forecasts
    day1 = results_df[results_df["step"] == 1].sort_values("pred_date")
    xs   = np.arange(len(day1))
    axes[0].plot(xs, day1["actual_close"].values,  color="#58a6ff", lw=1.4, label="Actual Close")
    axes[0].plot(xs, day1["pred_close"].values,    color="#f0a500", lw=1.0, ls="--", label="Predicted Close (Day 1)")
    axes[0].fill_between(xs,
                         day1["actual_close"].values,
                         day1["pred_close"].values,
                         alpha=0.15, color="#f0a500")
    axes[0].set_title(
        f"Nifty 50 — 6-Month Walk-Forward Backtest  |  Day-1 Actual vs Predicted",
        color="white", fontsize=12, pad=8)
    axes[0].set_ylabel("Price", color="white")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    tick_step = max(1, len(day1) // 8)
    axes[0].set_xticks(xs[::tick_step])
    axes[0].set_xticklabels(
        [str(d) for d in day1["pred_date"].values[::tick_step]],
        rotation=30, ha="right", color="white", fontsize=7)

    # Panel 2: Absolute % error per day-ahead
    step_list  = [m["step"]    for m in step_metrics]
    mape_list  = [m["mape"]    for m in step_metrics]
    # Colour MAPE bars by magnitude: green ≤ 1 %, yellow ≤ 2 %, red > 2 %
    colors_bar = ["#26a641" if m["mape"] <= 1.0 else
                  "#f0a500" if m["mape"] <= 2.0 else "#f85149"
                  for m in step_metrics]
    axes[1].bar(step_list, mape_list, color=colors_bar, width=0.5, zorder=3)
    for s, v in zip(step_list, mape_list):
        axes[1].text(s, v + 0.05, f"{v:.2f}%", color="white", fontsize=8, ha="center")
    axes[1].set_xlabel("Day Ahead", color="white")
    axes[1].set_ylabel("MAPE (%)", color="white")
    axes[1].set_title("Mean Absolute % Error by Day Ahead", color="white", fontsize=10, pad=6)
    axes[1].set_xticks(step_list)

    # Panel 3: Directional accuracy per day-ahead
    dir_list = [m["dir_acc"] for m in step_metrics]
    axes[2].bar(step_list, dir_list, color="#58a6ff", width=0.5, zorder=3)
    axes[2].axhline(50, color="#f85149", lw=0.8, ls="--", label="50% baseline")
    for s, v in zip(step_list, dir_list):
        axes[2].text(s, v + 0.5, f"{v:.1f}%", color="white", fontsize=8, ha="center")
    axes[2].set_ylim(0, 110)
    axes[2].set_xlabel("Day Ahead", color="white")
    axes[2].set_ylabel("Dir. Accuracy (%)", color="white")
    axes[2].set_title("Directional Accuracy by Day Ahead", color="white", fontsize=10, pad=6)
    axes[2].set_xticks(step_list)
    axes[2].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8)

    plt.tight_layout()
    out = "nifty50_backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[✓] Backtest chart saved → {out}")
    plt.show()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = "nifty50_backtest_results.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"[✓] Detailed results saved → {csv_out}")


if __name__ == "__main__":
    main()
