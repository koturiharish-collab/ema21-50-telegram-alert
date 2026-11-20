
"""NSE Daily EMA21/EMA50 Fresh Crossover Scanner.
Scans NSE EQ stocks using Yahoo Finance.
Designed for GitHub Actions and Telegram alerts.
"""

import io
import os
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
import yfinance as yf

IST = pytz.timezone("Asia/Kolkata")

TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]

FAST = 21
SLOW = 50
BATCH = 100

# Allow weekends and market holidays.
MAX_DATA_AGE_DAYS = 7


def send(msg):
    print(msg)

    response = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data={
            "chat_id": TG_CHAT_ID,
            "text": msg
        },
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")


def load_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df.columns = [c.strip() for c in df.columns]

    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    df = df[
        (df["SERIES"] == "EQ") &
        (df["SYMBOL"] != "")
    ]

    return sorted(set(df["SYMBOL"] + ".NS"))


def get_stock_data(data, symbol):

    if isinstance(data.columns, pd.MultiIndex):

        if symbol in data.columns.get_level_values(0):
            stock = data[symbol]

        elif symbol in data.columns.get_level_values(1):
            stock = data.xs(
                symbol,
                axis=1,
                level=1
            )

        else:
            return None

    else:
        stock = data

    if "Close" not in stock.columns:
        return None

    close = stock["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(
        close,
        errors="coerce"
    ).dropna()

    return close.sort_index()


def get_candle_date(timestamp):

    ts = pd.Timestamp(timestamp)

    # Convert timezone-aware timestamps to IST.
    if ts.tzinfo is not None:
        ts = ts.tz_convert(IST)

    return ts.date()


def scan_batch(symbols, now, stats):

    alerts = []

    try:

        data = yf.download(
            symbols,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=45
        )

    except Exception as error:

        print(f"Batch download failed: {error}")

        stats["errors"] += len(symbols)

        return alerts

    if data is None or data.empty:

        print("Yahoo Finance returned empty batch.")

        stats["no_data"] += len(symbols)

        return alerts

    for symbol in symbols:

        try:

            close = get_stock_data(data, symbol)

            if close is None or close.empty:

                stats["no_data"] += 1
                continue

            # Exclude today's incomplete daily candle.
            completed = close[
                [
                    get_candle_date(ts) < now.date()
                    for ts in close.index
                ]
            ]

            if len(completed) < SLOW + 2:

                stats["insufficient_data"] += 1
                continue

            latest_date = get_candle_date(
                completed.index[-1]
            )

            # Reject genuinely stale data.
            age_days = (
                now.date() - latest_date
            ).days

            if age_days > MAX_DATA_AGE_DAYS:

                stats["stale_data"] += 1
                continue

            # Calculate EMA difference.
            ema21 = completed.ewm(
                span=FAST,
                adjust=False
            ).mean()

            ema50 = completed.ewm(
                span=SLOW,
                adjust=False
            ).mean()

            diff = ema21 - ema50

            prev = diff.iloc[-2]
            curr = diff.iloc[-1]

            stats["processed"] += 1

            if (
                stats["latest_candle"] is None
                or latest_date > stats["latest_candle"]
            ):
                stats["latest_candle"] = latest_date

            name = symbol.replace(".NS", "")

            price = completed.iloc[-1]

            # Fresh bullish crossover.
            if prev <= 0 and curr > 0:

                alerts.append(
                    f"🟢 BULLISH {name}\n"
                    f"EMA21 crossed ABOVE EMA50\n"
                    f"Close: ₹{price:.2f}\n"
                    f"Date: {latest_date}"
                )

            # Fresh bearish crossover.
            elif prev >= 0 and curr < 0:

                alerts.append(
                    f"🔴 BEARISH {name}\n"
                    f"EMA21 crossed BELOW EMA50\n"
                    f"Close: ₹{price:.2f}\n"
                    f"Date: {latest_date}"
                )

        except Exception as error:

            stats["errors"] += 1

            print(
                f"Error processing {symbol}: {error}"
            )

    return alerts


def send_alerts(alerts):

    chunk = ""

    for alert in alerts:

        if len(chunk) + len(alert) + 2 > 3500:

            send(chunk)
            chunk = ""

        chunk += alert + "\n\n"

    if chunk:
        send(chunk)


def main():

    now = datetime.now(IST)

    print(
        f"Starting NSE EMA scanner: "
        f"{now:%d-%b-%Y %H:%M:%S} IST"
    )

    symbols = load_symbols()

    print(f"Symbols loaded: {len(symbols)}")

    stats = {
        "processed": 0,
        "no_data": 0,
        "insufficient_data": 0,
        "stale_data": 0,
        "errors": 0,
        "latest_candle": None
    }

    alerts = []

    # Download and scan in batches.
    for i in range(0, len(symbols), BATCH):

        batch = symbols[i:i + BATCH]

        print(
            f"Scanning batch "
            f"{i // BATCH + 1}: {len(batch)} stocks"
        )

        batch_alerts = scan_batch(
            batch,
            now,
            stats
        )

        alerts.extend(batch_alerts)

    bullish = sum(
        1 for a in alerts
        if a.startswith("🟢 BULLISH")
    )

    bearish = sum(
        1 for a in alerts
        if a.startswith("🔴 BEARISH")
    )

    # Determine actual scanner status.
    if stats["processed"] == 0:
        status = "FAILED"
    elif stats["errors"] > 0:
        status = "PARTIAL"
    else:
        status = "SUCCESS"

    # Send crossover alerts first.
    if alerts:
        send_alerts(alerts)

    else:
        print("No fresh EMA crossovers detected.")

    latest = stats["latest_candle"]

    latest_text = (
        str(latest)
        if latest is not None
        else "Not available"
    )

    report = (
        "📊 NSE DAILY EMA SCANNER\n\n"
        f"📅 Date: {now:%d-%b-%Y}\n"
        f"⏰ Time: {now:%H:%M} IST\n"
        "📈 Timeframe: DAILY (1D)\n\n"

        f"🔍 NSE EQ Symbols: {len(symbols)}\n"
        f"✅ Successfully Processed: {stats['processed']}\n"
        f"⚪ No Data: {stats['no_data']}\n"
        f"⚠️ Insufficient Data: {stats['insufficient_data']}\n"
        f"🕒 Stale Data: {stats['stale_data']}\n"
        f"❌ Processing Errors: {stats['errors']}\n\n"

        f"🟢 Bullish Crossovers: {bullish}\n"
        f"🔴 Bearish Crossovers: {bearish}\n\n"

        f"📆 Latest Completed Candle: {latest_text}\n"
        f"📡 Data Source: Yahoo Finance\n"
        f"📊 Indicators: EMA 21 / EMA 50\n"
        f"✅ Scanner Status: {status}"
    )

    send(report)

    print(report)


if __name__ == "__main__":
    main()
