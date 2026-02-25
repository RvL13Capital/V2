"""
Real Consolidation Pattern Visualization
Shows actual consolidation patterns from historical data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import os
from google.cloud import storage
import warnings
warnings.filterwarnings('ignore')

# Set up GCS credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-key.json'

def load_best_patterns():
    """Load and identify the best consolidation patterns from historical data"""
    print("Loading historical patterns...")

    # Try to load from different sources
    pattern_files = [
        'historical_patterns.parquet',
        'output/test_patterns.parquet',
        'output/patterns_10_fixed.parquet'
    ]

    for file in pattern_files:
        if os.path.exists(file):
            print(f"Loading patterns from {file}")
            df = pd.read_parquet(file)

            # Filter for explosive patterns (K4 or 40%+ gains)
            if 'outcome_class' in df.columns:
                if 'K4_EXPLOSIVE' in df['outcome_class'].values:
                    explosive = df[df['outcome_class'] == 'K4_EXPLOSIVE']
                elif 'EXPLOSIVE (40%+)' in df['outcome_class'].values:
                    explosive = df[df['outcome_class'] == 'EXPLOSIVE (40%+)']
                else:
                    # Use outcome_max_gain if available
                    if 'outcome_max_gain' in df.columns:
                        explosive = df[df['outcome_max_gain'] > 40]
                    else:
                        explosive = df

                if len(explosive) > 0:
                    # Sort by gain and select best example
                    if 'outcome_max_gain' in explosive.columns:
                        best = explosive.nlargest(5, 'outcome_max_gain')
                    else:
                        best = explosive.head(5)

                    print(f"Found {len(explosive)} explosive patterns")
                    return best

            return df.head(5)

    raise FileNotFoundError("No pattern files found")

def fetch_ticker_data(ticker, start_date, end_date):
    """Fetch actual price data from GCS for a ticker"""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")

    try:
        # Initialize GCS client
        client = storage.Client(project='ignition-ki-csv-storage')
        bucket = client.bucket('ignition-ki-csv-data-2025-user123')

        # Try different blob paths
        blob_paths = [
            f'market_data/{ticker}.csv',
            f'tickers/{ticker}.csv',
            f'{ticker}.csv'
        ]

        for blob_path in blob_paths:
            blob = bucket.blob(blob_path)
            if blob.exists():
                print(f"Found data at {blob_path}")
                content = blob.download_as_text()

                # Parse CSV
                from io import StringIO
                df = pd.read_csv(StringIO(content))

                # Ensure date column is datetime
                date_col = 'Date' if 'Date' in df.columns else df.columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)

                # Filter date range (expand by 50 days before and 100 days after)
                start_dt = pd.to_datetime(start_date) - timedelta(days=50)
                end_dt = pd.to_datetime(end_date) + timedelta(days=100)

                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                df = df.sort_index()

                print(f"Loaded {len(df)} days of data")
                return df

        print(f"No data found for {ticker}")
        return None

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def visualize_real_pattern(pattern_row, price_data):
    """Create visualization of a real consolidation pattern"""

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ticker = pattern_row['ticker']

    # Get pattern dates
    pattern_start = pd.to_datetime(pattern_row['pattern_start_date'])
    pattern_end = pd.to_datetime(pattern_row['pattern_end_date'])

    if 'breakout_date' in pattern_row and pd.notna(pattern_row.get('breakout_date')):
        breakout_date = pd.to_datetime(pattern_row['breakout_date'])
    else:
        breakout_date = pattern_end

    # Title with pattern info
    gain = pattern_row.get('outcome_max_gain', 0)
    duration = (pattern_end - pattern_start).days

    fig.suptitle(f'{ticker} - Real Consolidation Pattern\n' +
                 f'Pattern: {pattern_start.strftime("%Y-%m-%d")} to {pattern_end.strftime("%Y-%m-%d")} ' +
                 f'({duration} days) | Max Gain: {gain:.1f}%',
                 fontsize=14, fontweight='bold')

    # Price chart
    ax1 = plt.subplot(3, 1, 1)

    # Plot price data
    ax1.plot(price_data.index, price_data['Close'], 'b-', linewidth=1.5, label='Close Price', alpha=0.8)

    # Calculate and plot Bollinger Bands
    window = 20
    rolling_mean = price_data['Close'].rolling(window).mean()
    rolling_std = price_data['Close'].rolling(window).std()
    upper_bb = rolling_mean + 2 * rolling_std
    lower_bb = rolling_mean - 2 * rolling_std

    ax1.plot(price_data.index, upper_bb, 'g--', alpha=0.3, label='Upper BB')
    ax1.plot(price_data.index, lower_bb, 'g--', alpha=0.3, label='Lower BB')
    ax1.fill_between(price_data.index, upper_bb, lower_bb, alpha=0.1, color='green')

    # Mark consolidation period
    pattern_mask = (price_data.index >= pattern_start) & (price_data.index <= pattern_end)
    pattern_prices = price_data.loc[pattern_mask, 'Close']

    if len(pattern_prices) > 0:
        # Calculate boundaries
        upper_boundary = pattern_prices.max()
        lower_boundary = pattern_prices.min()
        power_boundary = upper_boundary * 1.005

        # Highlight consolidation period
        ax1.axvspan(pattern_start, pattern_end, alpha=0.2, color='yellow', label='Consolidation Period')

        # Draw boundaries
        ax1.axhline(y=upper_boundary, color='red', linestyle='-', linewidth=2,
                   alpha=0.7, label=f'Upper: ${upper_boundary:.2f}')
        ax1.axhline(y=lower_boundary, color='blue', linestyle='-', linewidth=2,
                   alpha=0.7, label=f'Lower: ${lower_boundary:.2f}')
        ax1.axhline(y=power_boundary, color='purple', linestyle=':', linewidth=1.5,
                   alpha=0.7, label=f'Power: ${power_boundary:.2f}')

        # Mark breakout if it occurred
        if breakout_date and breakout_date != pattern_end:
            ax1.axvspan(breakout_date, breakout_date + timedelta(days=5),
                       alpha=0.3, color='red', label='Breakout')

            # Annotate breakout point
            breakout_price = price_data.loc[price_data.index >= breakout_date, 'Close'].iloc[0] if len(price_data.loc[price_data.index >= breakout_date]) > 0 else upper_boundary
            ax1.annotate('BREAKOUT', xy=(breakout_date, breakout_price),
                        xytext=(breakout_date + timedelta(days=10), breakout_price * 1.05),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold', color='red')

    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Price Action with Consolidation Pattern', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Volume chart
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)

    # Color volume bars
    colors = ['red' if date >= breakout_date and date <= breakout_date + timedelta(days=5)
              else 'yellow' if date >= pattern_start and date <= pattern_end
              else 'gray' for date in price_data.index]

    ax2.bar(price_data.index, price_data['Volume'], color=colors, alpha=0.6)

    # Add volume averages
    pre_pattern_volume = price_data.loc[price_data.index < pattern_start, 'Volume'].mean()
    pattern_volume = price_data.loc[pattern_mask, 'Volume'].mean()

    if pd.notna(pre_pattern_volume):
        ax2.axhline(y=pre_pattern_volume, color='blue', linestyle='--',
                   alpha=0.5, label=f'Pre-pattern avg: {pre_pattern_volume/1e6:.1f}M')
    if pd.notna(pattern_volume):
        ax2.axhline(y=pattern_volume, color='orange', linestyle='--',
                   alpha=0.5, label=f'Pattern avg: {pattern_volume/1e6:.1f}M')

    ax2.set_ylabel('Volume', fontsize=11)
    ax2.set_title('Volume Analysis', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Technical indicators
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)

    # Calculate and plot BBW
    bbw = ((upper_bb - lower_bb) / rolling_mean * 100)
    ax3.plot(price_data.index, bbw, 'purple', linewidth=1.5, label='BBW (%)')

    # Mark low BBW periods
    bbw_30th = np.nanpercentile(bbw.dropna(), 30)
    ax3.axhline(y=bbw_30th, color='red', linestyle='--', alpha=0.5,
               label=f'30th percentile: {bbw_30th:.1f}%')

    # Highlight pattern period
    ax3.axvspan(pattern_start, pattern_end, alpha=0.2, color='yellow')
    if breakout_date and breakout_date != pattern_end:
        ax3.axvspan(breakout_date, breakout_date + timedelta(days=5), alpha=0.3, color='red')

    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('BBW (%)', fontsize=11)
    ax3.set_title('Bollinger Band Width - Volatility Indicator', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add info boxes
    info_str = f'''Pattern Details:
    Ticker: {ticker}
    Duration: {duration} days
    Max Gain: {gain:.1f}%
    Pattern Quality: {pattern_row.get('pattern_quality', 'N/A')}
    Volume Ratio: {pattern_row.get('avg_volume_ratio', 0):.3f}
    BBW: {pattern_row.get('avg_bbw', 0):.3f}'''

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.98, info_str, fontsize=9, verticalalignment='top',
            bbox=props, transform=fig.transFigure)

    return fig

def main():
    """Generate visualization with real consolidation pattern data"""

    try:
        # Load best patterns
        patterns = load_best_patterns()
        print(f"\nTop patterns found:")

        for idx, row in patterns.iterrows():
            ticker = row['ticker']
            gain = row.get('outcome_max_gain', 0)
            print(f"  {ticker}: {gain:.1f}% gain")

        # Select best pattern with valid dates
        selected_pattern = None
        price_data = None

        for idx, row in patterns.iterrows():
            ticker = row['ticker']
            start = row['pattern_start_date']
            end = row['pattern_end_date'] if 'pattern_end_date' in row else row.get('breakout_date', start)

            # Try to fetch data
            data = fetch_ticker_data(ticker, start, end)
            if data is not None and len(data) > 20:
                selected_pattern = row
                price_data = data
                print(f"\nSelected {ticker} for visualization")
                break

        if selected_pattern is None or price_data is None:
            print("Could not load price data for any pattern. Creating example with synthetic data...")
            # Fall back to synthetic example
            import subprocess
            subprocess.run(['python', 'visualize_consolidation.py'])
            return

        # Create visualization
        print("Creating visualization...")
        fig = visualize_real_pattern(selected_pattern, price_data)

        # Save
        output_file = f'real_consolidation_{selected_pattern["ticker"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_file}")

        # Show
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to synthetic example...")
        import subprocess
        subprocess.run(['python', 'visualize_consolidation.py'])

if __name__ == "__main__":
    main()