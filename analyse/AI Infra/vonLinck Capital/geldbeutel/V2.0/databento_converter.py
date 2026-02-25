"""
Databento DBN → NQ_CME_1min.parquet Converter
==============================================
Reads the Databento GLBX.MDP3 ohlcv-1m DBN file for NQ.FUT (parent symbol),
selects the front-month outright contract at each timestamp, detects roll
dates from instrument_id transitions, computes rolling median volume, and
saves the enriched parquet required by wfo_matrix_v2.py.

Usage:
    python databento_converter.py
    python databento_converter.py --input path/to/file.dbn.zst --output NQ_CME_1min.parquet

Input format:
    Databento DBN with zstd compression.
    Schema: ohlcv-1m
    Dataset: GLBX.MDP3
    Symbol: NQ.FUT (parent stype) → delivers all NQ contract expiries

Front-month selection:
    Near contract rolls, multiple instrument_ids coexist at the same timestamp.
    The front-month contract has the dominant volume. For each timestamp,
    the instrument_id with the highest volume is selected. Ties broken by
    lowest instrument_id (earliest registered contract).

Roll date detection:
    A roll date is any calendar day where the dominant instrument_id changes
    from the previous day's dominant instrument_id. The is_roll_date flag is
    set on the FIRST 1-min bar of that day.

Spread filtering:
    Calendar spreads (e.g., NQM4-NQU4) are excluded. Only outright contracts
    (symbol without '-') are retained.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import zstandard
import databento_dbn as dbn

# =====================================================================
# CONSTANTS
# =====================================================================
TICK              = 0.25
FIXED_PRICE_SCALE = 1_000_000_000  # Databento int64 price → float
RVOL_TRAILING_DAYS = 20            # Trading days for ToD trailing median
DEFAULT_INPUT     = (
    r"C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra"
    r"\vonLinck Capital\geldbeutel\V1.0"
    r"\GLBX-20260223-MA8BH8ABV3"
    r"\glbx-mdp3-20100606-20260222.ohlcv-1m.dbn.zst"
)
DEFAULT_OUTPUT    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "NQ_CME_1min.parquet"
)


# =====================================================================
# STEP 1 — DECODE
# =====================================================================

def decode_dbn(path: str) -> tuple:
    """
    Stream-decode the full DBN zstd file.
    Returns (metadata, all_ohlcv_records).
    """
    print(f"Decoding: {path}")
    print(f"File size: {os.path.getsize(path) / 1e6:.1f} MB (compressed)")

    decoder = dbn.DBNDecoder(has_metadata=True)
    dctx    = zstandard.ZstdDecompressor()

    total_bytes = 0
    with open(path, 'rb') as fh:
        reader = dctx.stream_reader(fh)
        while True:
            chunk = reader.read(8 * 1024 * 1024)  # 8 MB chunks
            if not chunk:
                break
            decoder.write(chunk)
            total_bytes += len(chunk)
            print(f"\r  Decompressed: {total_bytes / 1e6:.0f} MB", end='', flush=True)

    print(f"\r  Decompressed: {total_bytes / 1e6:.1f} MB total", flush=True)

    all_objects = decoder.decode()
    print(f"  Total decoded objects: {len(all_objects):,}")

    metadata = all_objects[0]
    records  = all_objects[1:]
    print(f"  Metadata + {len(records):,} OHLCVMsg records")
    return metadata, records


# =====================================================================
# STEP 2 — SYMBOL MAP
# =====================================================================

def build_instrument_map(metadata) -> dict:
    """
    Build instrument_id → symbol string mapping from metadata.
    Only returns outrights (no '-' in symbol key = not a calendar spread).

    metadata.mappings structure:
        { contract_symbol: [ {'start_date': date, 'end_date': date, 'symbol': str_iid}, ... ] }
    The 'symbol' value inside each interval dict is the instrument_id as a string.
    """
    iid_to_symbol = {}

    for symbol_str, intervals in metadata.mappings.items():
        # Skip calendar spreads
        if '-' in symbol_str:
            continue
        for interval in intervals:
            # instrument_id is stored under the 'symbol' key as a numeric string
            iid = int(interval['symbol'])
            if iid not in iid_to_symbol:
                iid_to_symbol[iid] = symbol_str
            elif len(symbol_str) < len(iid_to_symbol[iid]):
                iid_to_symbol[iid] = symbol_str

    print(f"  Outright contracts mapped: {len(iid_to_symbol)}")
    return iid_to_symbol


# =====================================================================
# STEP 3 — EXTRACT TO ARRAYS
# =====================================================================

def records_to_arrays(records, outright_iids: set) -> dict:
    """
    Convert list of OHLCVMsg objects to numpy arrays.
    Filters to outright instrument_ids only.
    Returns dict of arrays: ts, iid, open, high, low, close, volume.
    """
    print("Extracting records to arrays...")
    n = len(records)

    ts_arr  = np.empty(n, dtype=np.int64)
    iid_arr = np.empty(n, dtype=np.int64)
    o_arr   = np.empty(n, dtype=np.float64)
    h_arr   = np.empty(n, dtype=np.float64)
    l_arr   = np.empty(n, dtype=np.float64)
    c_arr   = np.empty(n, dtype=np.float64)
    v_arr   = np.empty(n, dtype=np.float64)

    scale = float(FIXED_PRICE_SCALE)
    kept  = 0

    for i, rec in enumerate(records):
        if i % 500_000 == 0:
            print(f"\r  {i:,} / {n:,}  kept: {kept:,}", end='', flush=True)
        iid = rec.instrument_id
        if iid not in outright_iids:
            continue
        ts_arr[kept]  = rec.ts_event
        iid_arr[kept] = iid
        o_arr[kept]   = rec.open  / scale
        h_arr[kept]   = rec.high  / scale
        l_arr[kept]   = rec.low   / scale
        c_arr[kept]   = rec.close / scale
        v_arr[kept]   = float(rec.volume)
        kept += 1

    print(f"\r  Extraction complete: {kept:,} outright records ({n - kept:,} spreads dropped)")

    return {
        'ts':     ts_arr[:kept],
        'iid':    iid_arr[:kept],
        'open':   o_arr[:kept],
        'high':   h_arr[:kept],
        'low':    l_arr[:kept],
        'close':  c_arr[:kept],
        'volume': v_arr[:kept],
    }


# =====================================================================
# STEP 4 — FRONT-MONTH SELECTION
# =====================================================================

def select_front_month(arrays: dict) -> pd.DataFrame:
    """
    For each timestamp, retain only the front-month contract.
    Selection rule: highest volume wins. Ties: lowest instrument_id.

    Returns DataFrame with one row per timestamp, sorted chronologically.
    """
    print("Building DataFrame...")
    df = pd.DataFrame({
        'ts':     arrays['ts'],
        'iid':    arrays['iid'],
        'open':   arrays['open'],
        'high':   arrays['high'],
        'low':    arrays['low'],
        'close':  arrays['close'],
        'volume': arrays['volume'],
    })

    # Convert ts (nanoseconds UTC) to DatetimeIndex
    df['ts'] = pd.to_datetime(df['ts'], unit='ns', utc=True)
    # Aggregate true exchange volume across all expiries for each timestamp.
    # During roll week, CME liquidity splits between expiring and new contracts.
    # Dropping the non-dominant contract destroys that volume. Sum first so the
    # selected front-month row carries total order-book depth.
    vol_agg = df.groupby('ts')['volume'].sum()

    # Front-month selection: highest per-contract volume wins (iid breaks ties).
    # This sort runs on original per-contract volumes, preserving correct selection.
    df = df.sort_values(['ts', 'volume', 'iid'], ascending=[True, False, True])

    total_before = len(df)
    print(f"  Rows before front-month selection: {total_before:,}")

    # Keep highest-volume row per timestamp (front-month price continuity)
    df = df.drop_duplicates(subset='ts', keep='first').reset_index(drop=True)
    print(f"  Rows after deduplication (front-month only): {len(df):,}")
    print(f"  Dropped {total_before - len(df):,} non-front-month rows")

    # Replace per-contract volume with true aggregate exchange volume
    df['volume'] = df['ts'].map(vol_agg)

    df = df.set_index('ts')
    df.index.name = None
    df.index = df.index.tz_convert('US/Eastern')

    return df


# =====================================================================
# STEP 5 — ROLL DATE DETECTION
# =====================================================================

def detect_rolls(df: pd.DataFrame) -> np.ndarray:
    """
    Detect contract roll dates from instrument_id (iid) transitions.
    For each calendar day, find the dominant iid (most common across bars).
    A roll date is any day where the day's dominant iid differs from
    the prior day's dominant iid.

    The is_roll_date flag is set on the FIRST bar of the roll day.
    Returns bool array aligned with df.index.
    """
    print("Detecting roll dates from instrument_id transitions...")
    is_roll = np.zeros(len(df), dtype=bool)

    dates = df.index.normalize()
    unique_dates = sorted(dates.unique())

    # Find dominant instrument_id per calendar day
    daily_dominant = {}
    for d in unique_dates:
        mask   = dates == d
        day_df = df[mask]
        # Mode of iid (most frequent instrument_id that day)
        dominant = day_df['iid'].mode().iloc[0]
        daily_dominant[d] = dominant

    # Find days where dominant iid changes
    roll_days = set()
    prev_iid  = None
    for d in unique_dates:
        curr_iid = daily_dominant[d]
        if prev_iid is not None and curr_iid != prev_iid:
            roll_days.add(d)
        prev_iid = curr_iid

    print(f"  Roll dates detected: {len(roll_days)}")
    if roll_days:
        for d in sorted(roll_days)[:5]:
            prev_d = [x for x in unique_dates if x < d]
            prev_sym = daily_dominant[prev_d[-1]] if prev_d else '?'
            print(f"    {d.date()}  iid {prev_sym} -> {daily_dominant[d]}")
        if len(roll_days) > 5:
            print(f"    ... and {len(roll_days) - 5} more")

    # Mark first bar of each roll day
    for d in roll_days:
        day_mask    = dates == d
        day_indices = np.where(day_mask)[0]
        if len(day_indices) > 0:
            is_roll[day_indices[0]] = True

    return is_roll


# =====================================================================
# STEP 6 — ENRICH AND SAVE
# =====================================================================

def enrich_and_save(df: pd.DataFrame, is_roll: np.ndarray, output_path: str):
    """
    Add is_roll_date, tod_baseline_vol, tick-quantise prices, validate, save.
    """
    print("Enriching dataset...")

    # Tick quantisation
    for col in ('open', 'high', 'low', 'close'):
        df[col] = (df[col] / TICK).round() * TICK

    # Roll date flag
    df['is_roll_date'] = is_roll

    # Time-of-Day RVOL baseline (diurnal-normalised structural volume)
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['tod_baseline_vol'] = (
        df.groupby('minute_of_day')['volume']
        .transform(lambda x: x.shift(1).rolling(window=20, min_periods=1).median())
    )
    df['tod_baseline_vol'] = df.groupby('minute_of_day')['tod_baseline_vol'].bfill()
    df['tod_baseline_vol'] = df['tod_baseline_vol'].fillna(1.0).clip(lower=1.0)
    df.drop(columns=['minute_of_day'], inplace=True)

    # Drop iid column (not needed by wfo_matrix_v2.py)
    df = df.drop(columns=['iid'])

    # Data quality checks
    bad_ohlc = (df['high'] < df['low']).sum()
    nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
    zero_close = (df['close'] <= 0).sum()

    print(f"\n=== Final dataset stats ===")
    print(f"  Bars:              {len(df):,}")
    print(f"  Date range:        {df.index[0]} to {df.index[-1]}")
    print(f"  Roll dates:        {int(df['is_roll_date'].sum())}")
    print(f"  Zero-vol bars:     {(df['volume'] == 0).sum():,} ({(df['volume'] == 0).mean()*100:.1f}%)")
    print(f"  RVOL baseline:     {df['tod_baseline_vol'].min():.0f} – {df['tod_baseline_vol'].max():.0f}")
    print(f"  Price range:       {df['close'].min():.2f} – {df['close'].max():.2f}")
    print(f"  NaN OHLC:          {nan_count}")
    print(f"  High < Low bars:   {bad_ohlc}")

    if nan_count > 0:
        print("  WARNING: NaN values present — check source data")
    if bad_ohlc > 0:
        print("  WARNING: Inverted OHLC bars — check source data")
    if zero_close > 0:
        print(f"  WARNING: {zero_close} bars with close <= 0")

    # Save
    df.to_parquet(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved -> {output_path}  ({size_mb:.1f} MB)")

    return df


# =====================================================================
# MAIN
# =====================================================================

def convert(input_path: str, output_path: str):
    import time
    t0 = time.time()

    # 1. Decode DBN
    metadata, records = decode_dbn(input_path)

    # 2. Build instrument map (outrights only)
    print("\nBuilding instrument map...")
    iid_to_symbol  = build_instrument_map(metadata)
    outright_iids  = set(iid_to_symbol.keys())
    print(f"  Spread/non-outright instrument_ids will be dropped")

    # 3. Extract to arrays
    arrays = records_to_arrays(records, outright_iids)

    # Free records memory
    del records

    # 4. Select front-month (deduplicate timestamps)
    df = select_front_month(arrays)
    del arrays

    # 4b. Drop bad-price bars (negative/zero close — DBN artefacts near rolls)
    n_before = len(df)
    df = df[(df['close'] > 0) & (df['open'] > 0)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} bad-price bars (close/open <= 0)")

    # 5. Detect roll dates
    is_roll = detect_rolls(df)

    # 6. Enrich and save
    df = enrich_and_save(df, is_roll, output_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return df


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Databento NQ.FUT ohlcv-1m DBN file to V2.0 parquet"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=DEFAULT_INPUT,
        help=f'Path to .dbn.zst file (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Output parquet path (default: {DEFAULT_OUTPUT})'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found:\n  {args.input}")
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == '__main__':
    main()
