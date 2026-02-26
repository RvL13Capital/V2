"""
CME NQ 1-Minute Data Loader — geldbeutel V2.0
==============================================
Prepares raw CME NQ futures 1-min OHLCV data for the swing engine.

Pipeline:
    1. Ingest CSV or Parquet with flexible column mapping
    2. Detect contract roll dates (gap-based + quarterly calendar)
    3. Compute rolling median volume (baseline structural volume)
    4. Tick-quantise prices to 0.25 MNQ grid
    5. Save enriched Parquet: NQ_CME_1min.parquet

Roll date detection strategy:
    Unadjusted CME NQ data has quarterly contract rolls (H/M/U/Z).
    Rolls produce price gaps that the kernel would misread as liquidity sweeps.
    Two complementary detection methods:
        A. Calendar: NQ rolls on the Thursday of the week before expiry
           (third Friday of March/June/September/December).
           This produces an approximate calendar list.
        B. Gap filter: consecutive close→open gaps > ROLL_GAP_THRESHOLD (0.30%)
           restricted to roll calendar months. Handles delayed rolls and data
           source discrepancies.
    The union of A and B is flagged as is_roll_date.
    Override path: provide a CSV at ROLL_DATES_OVERRIDE_PATH with a
    single column 'date' (YYYY-MM-DD) for manual roll date specification.

Output parquet schema:
    Index:               DatetimeIndex, tz-aware US/Eastern, 1-min frequency
    open, high, low, close: float64, tick-quantised to 0.25
    volume:              float64, CME exchange contracts per bar
    is_roll_date:        bool, True on first 1-min bar of new contract session
    tod_baseline_vol:    float64, time-of-day RVOL_TRAILING_DAYS-period trailing median volume

Usage:
    python data_loader_v2.py --input path/to/nq_1min.csv --output NQ_CME_1min.parquet
    python data_loader_v2.py --input path/to/nq_1min.parquet
    python data_loader_v2.py --test   # validate an existing NQ_CME_1min.parquet
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


# =====================================================================
# CONSTANTS
# =====================================================================
TICK = 0.25
ROLL_GAP_THRESHOLD   = 0.003   # 0.30% — gaps below this are normal; above = roll candidate
RVOL_TRAILING_DAYS   = 10      # trading days for ToD trailing median (2 weeks — adapts faster to NQ volume regime shifts)
ROLL_DATES_OVERRIDE_PATH = "roll_dates_override.csv"  # optional manual override

# CME NQ quarterly expiry months
EXPIRY_MONTHS = {3, 6, 9, 12}

# Column name aliases (maps various vendor naming conventions to standard names)
COLUMN_ALIASES = {
    'open':     ['open', 'Open', 'OPEN', 'o', 'price_open'],
    'high':     ['high', 'High', 'HIGH', 'h', 'price_high'],
    'low':      ['low',  'Low',  'LOW',  'l', 'price_low'],
    'close':    ['close', 'Close', 'CLOSE', 'c', 'price_close', 'last'],
    'volume':   ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'qty'],
}

TIMESTAMP_ALIASES = [
    'timestamp', 'Timestamp', 'datetime', 'DateTime', 'date', 'Date',
    'time', 'Time', 'ts', 'index',
]


# =====================================================================
# CME QUARTERLY ROLL CALENDAR GENERATOR
# =====================================================================

def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the third Friday of the given month."""
    first = pd.Timestamp(year=year, month=month, day=1)
    # Weekday: Monday=0 … Friday=4
    day_offset = (4 - first.weekday()) % 7
    first_friday = first + pd.Timedelta(days=day_offset)
    return first_friday + pd.Timedelta(weeks=2)


def generate_cme_roll_calendar(start_year: int, end_year: int) -> set:
    """
    Generate approximate CME NQ roll dates.
    Roll day = Thursday before expiry (third Friday of expiry month).
    Returns a set of date strings 'YYYY-MM-DD'.
    """
    roll_dates = set()
    for year in range(start_year, end_year + 1):
        for month in EXPIRY_MONTHS:
            expiry_friday = _third_friday(year, month)
            roll_thursday = expiry_friday - pd.Timedelta(days=1)
            # If Thursday is a market holiday (rough heuristic: skip weekends)
            while roll_thursday.weekday() > 4:
                roll_thursday -= pd.Timedelta(days=1)
            roll_dates.add(roll_thursday.strftime('%Y-%m-%d'))
    return roll_dates


# =====================================================================
# ROLL DATE DETECTION
# =====================================================================

def detect_roll_dates(df: pd.DataFrame,
                      start_year: int,
                      end_year: int) -> np.ndarray:
    """
    Build is_roll_date boolean array.

    Method A: Calendar-based approximate CME roll dates.
    Method B: Gap-based detection within roll calendar months.
    Method C: Manual override from ROLL_DATES_OVERRIDE_PATH if present.

    Returns bool array aligned with df.index.
    """
    is_roll = np.zeros(len(df), dtype=bool)

    # Method A: Calendar
    calendar_dates = generate_cme_roll_calendar(start_year, end_year)
    print(f"  Calendar roll dates generated: {len(calendar_dates)}")

    # Method B: Gap-based — compute close-to-open returns
    close_arr = df['close'].values
    open_arr  = df['open'].values
    # Daily: find first bar of each calendar date
    date_col = df.index.normalize()  # date component only
    # Compute overnight gaps: compare each bar's open to the PREVIOUS bar's close
    gap_pct = np.zeros(len(df))
    gap_pct[1:] = np.abs(open_arr[1:] - close_arr[:-1]) / np.where(
        close_arr[:-1] != 0, close_arr[:-1], 1.0
    )
    gap_dates = set(
        df.index[gap_pct > ROLL_GAP_THRESHOLD].normalize().strftime('%Y-%m-%d')
    )
    # Filter gap_dates to expiry months only (reduce false positives from crash gaps)
    gap_dates_filtered = {
        d for d in gap_dates
        if int(d[5:7]) in EXPIRY_MONTHS
    }
    print(f"  Gap-detected roll candidates (expiry months only): {len(gap_dates_filtered)}")

    # Method C: Manual override
    override_dates = set()
    if os.path.exists(ROLL_DATES_OVERRIDE_PATH):
        try:
            ov_df = pd.read_csv(ROLL_DATES_OVERRIDE_PATH)
            override_dates = set(pd.to_datetime(ov_df['date']).dt.strftime('%Y-%m-%d'))
            print(f"  Manual roll date override: {len(override_dates)} dates from {ROLL_DATES_OVERRIDE_PATH}")
        except Exception as e:
            print(f"  WARNING: Could not load roll dates override: {e}")

    all_roll_date_strs = calendar_dates | gap_dates_filtered | override_dates
    print(f"  Total roll dates (union): {len(all_roll_date_strs)}")

    # Mark is_roll on the FIRST bar of each roll date
    idx_start = df.index[0].normalize().tz_localize(None)
    idx_end   = df.index[-1].normalize().tz_localize(None)
    roll_ts = pd.DatetimeIndex([
        pd.Timestamp(d, tz=df.index.tz) for d in all_roll_date_strs
        if pd.Timestamp(d) >= idx_start and
           pd.Timestamp(d) <= idx_end
    ])

    for ts in roll_ts:
        # Find first bar on or after this date
        day_mask = (df.index.normalize() == ts.normalize())
        day_indices = np.where(day_mask)[0]
        if len(day_indices) > 0:
            is_roll[day_indices[0]] = True

    return is_roll


# =====================================================================
# COLUMN DETECTION & NORMALISATION
# =====================================================================

def _resolve_column(df: pd.DataFrame, standard_name: str) -> str:
    """Find the column in df matching standard_name via COLUMN_ALIASES."""
    for alias in COLUMN_ALIASES[standard_name]:
        if alias in df.columns:
            return alias
    raise KeyError(
        f"Cannot find column for '{standard_name}'. "
        f"Expected one of: {COLUMN_ALIASES[standard_name]}. "
        f"Available: {list(df.columns)}"
    )


def _resolve_timestamp_column(df: pd.DataFrame) -> str:
    """Find the timestamp column by trying known aliases."""
    for alias in TIMESTAMP_ALIASES:
        if alias in df.columns:
            return alias
    return None  # Caller will try using the index


# =====================================================================
# MAIN LOADER
# =====================================================================

def load_and_prepare(
        input_path: str,
        output_path: str = None,
        timezone: str = 'UTC',
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Load CME NQ 1-min OHLCV data, enrich with roll dates and median volume,
    and save to Parquet.

    Args:
        input_path  : Path to CSV or Parquet file.
        output_path : Destination Parquet path. Defaults to NQ_CME_1min.parquet
                      in the same directory as input_path.
        timezone    : Timezone of the input timestamps (default UTC).
        verbose     : Print progress.

    Returns:
        Prepared DataFrame ready for wfo_matrix_v2.py.
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(input_path))
        output_path = os.path.join(base_dir, "NQ_CME_1min.parquet")

    if verbose:
        print(f"Loading: {input_path}")

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.parquet':
        raw = pd.read_parquet(input_path)
    elif ext in ('.csv', '.txt', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        raw = pd.read_csv(input_path, sep=sep, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV or Parquet.")

    if verbose:
        print(f"  Raw shape: {raw.shape}")

    # ------------------------------------------------------------------
    # 2. Build DatetimeIndex
    # ------------------------------------------------------------------
    if isinstance(raw.index, pd.DatetimeIndex):
        df = raw.copy()
    else:
        ts_col = _resolve_timestamp_column(raw)
        if ts_col:
            df = raw.set_index(ts_col)
        else:
            raise ValueError(
                "Could not find a timestamp column. "
                f"Expected one of: {TIMESTAMP_ALIASES}. "
                "Ensure the DataFrame has a parseable datetime index or column."
            )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, infer_datetime_format=True)

    # Ensure timezone-aware in US/Eastern
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    df.index = df.index.tz_convert('US/Eastern')
    df = df.sort_index()

    if verbose:
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Bars: {len(df):,}")

    # ------------------------------------------------------------------
    # 3. Normalise column names
    # ------------------------------------------------------------------
    col_map = {}
    for std in ('open', 'high', 'low', 'close', 'volume'):
        src = _resolve_column(df, std)
        col_map[src] = std
    df = df.rename(columns=col_map)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df = df.apply(pd.to_numeric, errors='coerce')

    # ------------------------------------------------------------------
    # 4. Data quality checks
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    n_after = len(df)
    if verbose and n_before != n_after:
        print(f"  Dropped {n_before - n_after} bad rows (NaN / inverted OHLC).")

    # Fill missing volume with 0 (some overnight sessions have sparse volume)
    df['volume'] = df['volume'].fillna(0.0)

    # ------------------------------------------------------------------
    # 5. Tick quantisation
    # ------------------------------------------------------------------
    for col in ('open', 'high', 'low', 'close'):
        df[col] = (df[col] / TICK).round() * TICK

    if verbose:
        print(f"  Prices quantised to {TICK}-point MNQ grid.")

    # ------------------------------------------------------------------
    # 6. Roll date detection
    # ------------------------------------------------------------------
    if verbose:
        print("\nDetecting roll dates...")

    start_year = df.index[0].year
    end_year   = df.index[-1].year
    is_roll = detect_roll_dates(df, start_year, end_year)
    df['is_roll_date'] = is_roll

    if verbose:
        print(f"  Roll dates flagged: {int(is_roll.sum())}")

    # ------------------------------------------------------------------
    # 7. Time-of-Day RVOL baseline (diurnal-normalised structural volume)
    # ------------------------------------------------------------------
    if verbose:
        print("\nComputing Time-of-Day RVOL baseline...")

    # Group by minute-of-day, shift(1) to avoid T+0 data leakage,
    # then 20-period trailing median within each minute bucket
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['tod_baseline_vol'] = (
        df.groupby('minute_of_day')['volume']
        .transform(lambda x: x.shift(1).rolling(window=RVOL_TRAILING_DAYS, min_periods=1).median())
    )
    # Backfill early-data NaNs within each minute group
    df['tod_baseline_vol'] = df.groupby('minute_of_day')['tod_baseline_vol'].bfill()
    # Absolute floor: prevent division-by-zero in Numba
    df['tod_baseline_vol'] = df['tod_baseline_vol'].fillna(1.0).clip(lower=1.0)
    df.drop(columns=['minute_of_day'], inplace=True)

    if verbose:
        print(f"  RVOL baseline range: "
              f"{df['tod_baseline_vol'].min():.0f} – "
              f"{df['tod_baseline_vol'].max():.0f} contracts/bar")
        print(f"  Zero-volume bars: "
              f"{(df['volume'] == 0).sum():,} "
              f"({(df['volume'] == 0).mean()*100:.1f}%)")

    # ------------------------------------------------------------------
    # 8. Final validation
    # ------------------------------------------------------------------
    required_cols = {'open', 'high', 'low', 'close', 'volume',
                     'is_roll_date', 'tod_baseline_vol'}
    assert required_cols.issubset(set(df.columns)), "Missing required output columns."
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex."
    assert df.index.tz is not None, "Index must be timezone-aware."
    assert not df['close'].isna().any(), "NaN values remain in close column."

    if verbose:
        print(f"\nFinal dataset:")
        print(df[['open', 'high', 'low', 'close', 'volume',
                   'is_roll_date', 'tod_baseline_vol']].describe().round(3).to_string())

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    df.to_parquet(output_path)
    if verbose:
        print(f"\nSaved → {output_path}")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

    return df


# =====================================================================
# VALIDATION / TEST MODE
# =====================================================================

def validate_parquet(parquet_path: str):
    """Quick sanity-check on an existing prepared parquet."""
    print(f"Validating: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')

    required = {'open', 'high', 'low', 'close', 'volume',
                'is_roll_date', 'tod_baseline_vol'}
    missing = required - set(df.columns)
    if missing:
        print(f"  FAIL: Missing columns: {missing}")
        return False

    print(f"  Bars:           {len(df):,}")
    print(f"  Date range:     {df.index[0]} to {df.index[-1]}")
    print(f"  Roll dates:     {int(df['is_roll_date'].sum())}")
    print(f"  Vol coverage:   {(df['volume'] > 0).mean()*100:.1f}%")
    print(f"  RVOL baseline:  {df['tod_baseline_vol'].min():.0f} – "
          f"{df['tod_baseline_vol'].max():.0f}")
    print(f"  Price range:    {df['close'].min():.2f} – {df['close'].max():.2f}")
    nan_count = df[['open', 'high', 'low', 'close']].isna().sum().sum()
    print(f"  NaN OHLC:       {nan_count}")

    # Check OHLC integrity
    bad = (df['high'] < df['low']).sum()
    if bad:
        print(f"  WARNING: {bad} bars where high < low")

    bad_close = (df['close'] <= 0).sum()
    if bad_close:
        print(f"  WARNING: {bad_close} bars with close <= 0")

    print("  OK" if nan_count == 0 and bad == 0 else "  Validation issues found (see above)")
    return True


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CME NQ 1-min data for geldbeutel V2.0"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to raw CME NQ 1-min CSV or Parquet file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output Parquet path (default: NQ_CME_1min.parquet alongside input)'
    )
    parser.add_argument(
        '--timezone', '-tz',
        type=str,
        default='UTC',
        help='Timezone of input timestamps (default: UTC)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Validate an existing NQ_CME_1min.parquet without reprocessing'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.test:
        candidate = args.input or os.path.join(script_dir, "NQ_CME_1min.parquet")
        if not os.path.exists(candidate):
            print(f"ERROR: File not found: {candidate}")
            sys.exit(1)
        validate_parquet(candidate)
        return

    if args.input is None:
        print("ERROR: --input is required. Provide path to raw CME NQ 1-min data.")
        print("       Or use --test to validate an existing NQ_CME_1min.parquet.")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    output = args.output or os.path.join(script_dir, "NQ_CME_1min.parquet")

    load_and_prepare(
        input_path=args.input,
        output_path=output,
        timezone=args.timezone,
        verbose=True,
    )


if __name__ == '__main__':
    main()
