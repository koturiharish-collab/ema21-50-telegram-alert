"""Single-run EMA21/EMA50 crossover scan for all NSE EQ stocks (15m candles).
Designed to be run on a schedule (GitHub Actions). Sends Telegram alerts."""
import io
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
import yfinance as yf

IST = pytz.timezone("Asia/Kolkata")
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]

FAST, SLOW = 21, 50
CANDLE = timedelta(minutes=15)
BATCH = 100


def load_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"]
    return [s.strip() + ".NS" for s in df["SYMBOL"]]


def send(msg):
    print(msg)
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data={"chat_id": TG_CHAT_ID, "text": msg},
        timeout=15,
    )


def scan(symbols, now):
    data = yf.download(symbols, period="5d", interval="15m",
                       group_by="ticker", progress=False, threads=True)
    out = []
    for sym in symbols:
        try:
            close = (data[sym]["Close"] if len(symbols) > 1 else data["Close"]).dropna()
            # keep only fully closed candles
            close = close[[ts + CANDLE <= now for ts in close.index]]
            if len(close) < SLOW + 2:
                continue
            diff = close.ewm(span=FAST, adjust=False).mean() - close.ewm(span=SLOW, adjust=False).mean()
            prev, curr, ts = diff.iloc[-2], diff.iloc[-1], close.index[-1]
            # only alert on a cross that happened on the latest closed candle
            if now - (ts + CANDLE) >= CANDLE:
                continue
            name = sym[:-3]
            if prev <= 0 < curr:
                out.append(f"BULLISH {name}: EMA{FAST} crossed ABOVE EMA{SLOW} @ {close.iloc[-1]:.2f} ({ts:%H:%M})")
            elif prev >= 0 > curr:
                out.append(f"BEARISH {name}: EMA{FAST} crossed BELOW EMA{SLOW} @ {close.iloc[-1]:.2f} ({ts:%H:%M})")
        except Exception:
            continue
    return out


def main():
    now = datetime.now(IST)
    symbols = load_symbols()
    alerts = []
    for i in range(0, len(symbols), BATCH):
        alerts += scan(symbols[i:i + BATCH], now)
    print(f"{now:%d-%b %H:%M} IST: scanned {len(symbols)} stocks, {len(alerts)} crossovers")
    chunk = ""
    for a in alerts:
        if len(chunk) + len(a) > 3500:
            send(chunk)
            chunk = ""
        chunk += a + "\n"
    if chunk:
        send(chunk)


if __name__ == "__main__":
    sys.exit(main())
