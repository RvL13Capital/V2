"""
Dukascopy Raw Tick Data -> 5-Min OHLCV Bars
============================================
Downloads USATECHIDXUSD (Nasdaq 100 CFD) tick data directly from
Dukascopy's free datafeed, aggregates to 5-minute bars.

Output: NQ_CFD_5min_{start_year}_{end_year}.parquet  (parquet only, no CSV)
Index:  UTC timestamps, convert to US/Eastern before feeding kernel.

Instrument: USATECHIDXUSD = Dukascopy's Nasdaq 100 CFD
Price divisor: 1000 (raw int32 / 1000 = actual price)

Speed: parallel hourly fetches via ThreadPoolExecutor (8 workers).
       ~8-10x faster than sequential. Checkpoint every 7 days.
"""

import requests
import struct
import lzma
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import sys

# =====================================================================
# CONFIG
# =====================================================================
INSTRUMENT = "USATECHIDXUSD"
PRICE_DIVISOR = 1000.0  # Dukascopy stores prices as int * 1000
BASE_URL = "https://datafeed.dukascopy.com/datafeed"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

START_DATE = datetime(2017, 1, 1)
END_DATE = datetime(2026, 2, 22)

# Parallel workers — 8 concurrent hour-fetches per day
# Dukascopy allows this without rate-limiting at this scale
WORKERS = 8
RETRY_MAX = 3
RETRY_DELAY = 2.0

# Checkpoint every N days (more frequent = safer on interruption)
CHECKPOINT_INTERVAL_DAYS = 7


# =====================================================================
# TICK PARSER
# =====================================================================
def parse_bi5_ticks(raw_bytes, base_dt):
    """
    Parse Dukascopy .bi5 (LZMA-compressed) tick data.
    Each tick = 20 bytes: int32 time_offset_ms, int32 ask, int32 bid, float32 ask_vol, float32 bid_vol
    Returns list of (datetime, bid, ask, bid_vol, ask_vol)
    """
    if not raw_bytes or len(raw_bytes) == 0:
        return []

    try:
        data = lzma.decompress(raw_bytes)
    except Exception:
        return []

    n_ticks = len(data) // 20
    if n_ticks == 0:
        return []

    ticks = []
    for i in range(n_ticks):
        offset = i * 20
        t_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
            '>iiiff', data[offset:offset + 20]
        )
        tick_dt = base_dt + timedelta(milliseconds=t_ms)
        bid = bid_raw / PRICE_DIVISOR
        ask = ask_raw / PRICE_DIVISOR
        ticks.append((tick_dt, bid, ask, bid_vol, ask_vol))

    return ticks


# =====================================================================
# HOURLY FETCHER (thread-safe, no shared state)
# =====================================================================
def fetch_hour(dt_hour):
    """
    Fetch one hour of tick data from Dukascopy.
    dt_hour: datetime at the start of the hour (UTC).
    Returns (dt_hour, list_of_tick_tuples).
    """
    year = dt_hour.year
    month = dt_hour.month - 1  # Dukascopy uses 0-indexed months
    day = dt_hour.day
    hour = dt_hour.hour

    url = f"{BASE_URL}/{INSTRUMENT}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    for attempt in range(RETRY_MAX):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                return (dt_hour, parse_bi5_ticks(r.content, dt_hour))
            elif r.status_code == 404:
                return (dt_hour, [])
            else:
                return (dt_hour, [])
        except requests.exceptions.RequestException:
            if attempt < RETRY_MAX - 1:
                time.sleep(RETRY_DELAY)
            continue

    return (dt_hour, [])


# =====================================================================
# PARALLEL DAY FETCHER
# =====================================================================
def fetch_day_parallel(current_day, executor):
    """
    Fetch all 24 hours for a single day in parallel.
    Returns sorted flat list of all ticks for the day.
    """
    hours = [current_day.replace(hour=h, minute=0, second=0) for h in range(24)]
    futures = {executor.submit(fetch_hour, h): h for h in hours}

    day_ticks = []
    for future in as_completed(futures):
        _, ticks = future.result()
        if ticks:
            day_ticks.extend(ticks)

    # Sort by timestamp (parallel fetches may arrive out of order)
    day_ticks.sort(key=lambda x: x[0])
    return day_ticks


# =====================================================================
# TICK -> 5-MIN OHLCV AGGREGATOR
# =====================================================================
def ticks_to_5min_bars(ticks_list):
    """
    Convert raw tick list to 5-minute OHLCV bars using bid prices.
    """
    if not ticks_list:
        return pd.DataFrame()

    df = pd.DataFrame(ticks_list, columns=['timestamp', 'bid', 'ask', 'bid_vol', 'ask_vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')

    # Use bid price as the "price" (standard for CFD analysis)
    bars = df['bid'].resample('5min').agg(
        open='first',
        high='max',
        low='min',
        close='last'
    )

    # Tick count as volume proxy
    bars['volume'] = df['bid'].resample('5min').count()

    # Drop empty bars
    bars = bars.dropna(subset=['open'])
    bars = bars[bars['volume'] > 0]

    return bars


# =====================================================================
# MAIN DOWNLOAD LOOP
# =====================================================================
def download_range(start_dt, end_dt):
    """
    Download tick data day-by-day (parallel hours), aggregate to 5-min,
    save checkpoints every CHECKPOINT_INTERVAL_DAYS days.
    """
    output_base = os.path.join(OUTPUT_DIR, f"NQ_CFD_5min_{start_dt.year}_{end_dt.year}")
    checkpoint_file = output_base + "_checkpoint.parquet"

    # Resume from checkpoint if exists
    all_bars = []
    resume_from = start_dt

    if os.path.exists(checkpoint_file):
        existing = pd.read_parquet(checkpoint_file)
        if len(existing) > 0:
            last_ts = existing.index.max()
            resume_from = last_ts.to_pydatetime().replace(tzinfo=None) + timedelta(days=1)
            resume_from = resume_from.replace(hour=0, minute=0, second=0)
            all_bars.append(existing)
            print(f"Resuming from checkpoint: {resume_from.date()} ({len(existing)} bars loaded)")

    current_day = resume_from
    day_count = 0
    daily_bars_buffer = []

    # Progress denominator anchored to full start→end range
    total_days_full = max((end_dt - start_dt).days, 1)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        while current_day < end_dt:
            day_count += 1

            # Fetch all 24 hours in parallel
            day_ticks = fetch_day_parallel(current_day, executor)

            if day_ticks:
                bars = ticks_to_5min_bars(day_ticks)
                if len(bars) > 0:
                    daily_bars_buffer.append(bars)
                    tick_count = len(day_ticks)
                    bar_count = len(bars)
                else:
                    tick_count = len(day_ticks)
                    bar_count = 0
            else:
                tick_count = 0
                bar_count = 0

            elapsed_days = (current_day - start_dt).days + 1
            pct = (elapsed_days / total_days_full) * 100

            if bar_count > 0:
                print(f"  {current_day.date()} | {tick_count:>6} ticks -> {bar_count:>3} bars | {pct:.1f}%", flush=True)

            # Checkpoint save every N days
            if day_count % CHECKPOINT_INTERVAL_DAYS == 0 and daily_bars_buffer:
                chunk = pd.concat(daily_bars_buffer)
                all_bars.append(chunk)
                combined = pd.concat(all_bars)
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
                combined.to_parquet(checkpoint_file)
                print(f"  >>> Checkpoint saved: {len(combined)} total bars | {pct:.1f}%", flush=True)
                daily_bars_buffer = []

            current_day += timedelta(days=1)

    # Final save
    if daily_bars_buffer:
        all_bars.append(pd.concat(daily_bars_buffer))

    if not all_bars:
        print("ERROR: No data downloaded.")
        return None

    final = pd.concat(all_bars)
    final = final[~final.index.duplicated(keep='last')]
    final = final.sort_index()

    # Save parquet only (disk space conservation)
    final.to_parquet(output_base + ".parquet")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total bars: {len(final)}")
    print(f"Date range: {final.index[0]} to {final.index[-1]}")
    print(f"Saved to:   {output_base}.parquet")

    return final


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    print(f"{'='*70}")
    print(f"DUKASCOPY NQ CFD 5-MIN DOWNLOADER  [parallel mode, {WORKERS} workers]")
    print(f"{'='*70}")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Range:      {START_DATE.date()} to {END_DATE.date()}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Checkpoint: every {CHECKPOINT_INTERVAL_DAYS} days")
    print(f"{'='*70}")
    print()

    df = download_range(START_DATE, END_DATE)

    if df is not None:
        print(f"\nSample data:")
        print(df.head(20).to_string())
        print(f"\nStats:")
        print(df.describe().to_string())
