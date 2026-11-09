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

# Set to False once you have confirmed Telegram works
TEST_PING = True

latest_candle_seen = None  # for diagnostics


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
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data={"chat_id": TG_CHAT_ID, "text": msg},
        timeout=15,
    )
    print("Telegram response:", r.status_code, r.text[:200])
    r.raise_for_status()  # makes the run FAIL (red) if Telegram rejects the message


def scan(symbols, now):
    global latest_candle_seen
    data = yf.download(symbols, period="5d", interval="15m",
                       group_by="ticker", progress=False, threads=True)
    out = []
    for sym in symbols:
        try:
            close = (data[sym]["Close"] if len(symbols) > 1 else data["Close"]).dropna()
            if len(close) == 0:
                continue
            # keep only fully closed candles
            close = close[[ts + CANDLE <= now for ts in close.index]]
            if len(close) < SLOW + 2:
                continue
            ts = close.index[-1]
            if latest_candle_seen is None or ts > latest_candle_seen:
                latest_candle_seen = ts
            diff = close.ewm(span=FAST, adjust=False).mean() - close.ewm(span=SLOW, adjust=False).mean()
            prev, curr = diff.iloc[-2], diff.iloc[-1]
            # only alert on a cross that happened on the latest closed candle
            if now - (ts + CANDLE) >= CANDLE:
                continue
            name = sym[:-3]
            if prev <= 0 < curr:
                out.append(f"BULLISH {name}: EMA{FAST} crossed ABOVE EMA{SLOW} @ {close.iloc[-1]:.2f} ({ts:%H:%M})")
            elif prev >= 0 > curr:
                out.append(f"BEARISH {name}: EMA{FAST} crossed BELOW EMA{SLOW} @ {close.iloc[-1]:.2f} ({ts:%H:%M})")
        except Exception as e:
            print(f"error on {sym}: {e}")
            continue
    return out


def main():
    now = datetime.now(IST)
    if TEST_PING:
        send(f"EMA scan started {now:%d-%b %H:%M} IST")
    symbols = load_symbols()
    alerts = []
    for i in range(0, len(symbols), BATCH):
        alerts += scan(symbols[i:i + BATCH], now)
    print(f"{now:%d-%b %H:%M} IST: scanned {len(symbols)} stocks, {len(alerts)} crossovers")
    print(f"Latest closed candle seen in data: {latest_candle_seen}")
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
