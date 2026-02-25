import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_lens_score(ticker):
    try:
        df = yf.download(ticker, period='2y', interval='1d', progress=False, auto_adjust=True)
        if df.empty or len(df) < 252:
            return {'Ticker': ticker, 'Price': None, 'Score': None, 'Signal': None,
                    'Vol_Ratio': None, 'Stubborn_Pct': None, 'Dist_From_Low': None,
                    'Range_Contract': None, 'Liq_Pass': None, 'Error': 'Insufficient data'}

        df = df.reset_index()

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] for col in df.columns]

        # Metrics
        df['Range_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        high_low_diff = df['High'] - df['Low']
        df['Body_Pos'] = np.where(high_low_diff > 0, (df['Close'] - df['Low']) / high_low_diff, 0.5)
        df['Up_Vol'] = np.where(df['Close'] > df['Open'], df['Volume'], 0)
        df['Down_Vol'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)

        window = 20

        # Range Contraction (40%+ tighter)
        avg_range_now = df['Range_Pct'].rolling(10).mean()
        avg_range_prev = df['Range_Pct'].rolling(10).mean().shift(10)
        range_contract = avg_range_now < (avg_range_prev * 0.6)

        # Volume Asymmetry
        sum_up_vol = df['Up_Vol'].rolling(window).sum()
        sum_down_vol = df['Down_Vol'].rolling(window).sum()
        vol_ratio_series = pd.Series(
            np.where(sum_down_vol > 0, sum_up_vol / sum_down_vol, 10.0),
            index=df.index
        )

        # Stubborn Closes (>70% in top 40% of range)
        stubborn = df['Body_Pos'].rolling(window).apply(lambda x: (x > 0.6).mean(), raw=True)

        # Location
        min_252d = df['Low'].rolling(252).min()
        dist_from_low = (df['Close'] - min_252d) / min_252d * 100

        # Liquidity
        adv_val = df['Close'] * df['Volume'].rolling(20).mean()
        spread_pct = (df['High'] - df['Low']) / df['Low'] * 100

        # Get last values safely
        rc_val = bool(range_contract.iloc[-1]) if pd.notna(range_contract.iloc[-1]) else False
        vr_val = float(vol_ratio_series.iloc[-1]) if pd.notna(vol_ratio_series.iloc[-1]) else 0.0
        st_val = float(stubborn.iloc[-1]) if pd.notna(stubborn.iloc[-1]) else 0.0
        dl_val = float(dist_from_low.iloc[-1]) if pd.notna(dist_from_low.iloc[-1]) else 999.0
        av_val = float(adv_val.iloc[-1]) if pd.notna(adv_val.iloc[-1]) else 0.0
        sp_val = float(spread_pct.iloc[-1]) if pd.notna(spread_pct.iloc[-1]) else 999.0

        # Scoring
        score = 0
        if rc_val: score += 30
        if vr_val > 2.0: score += 25
        elif vr_val > 1.5: score += 15
        if st_val > 0.7: score += 25
        elif st_val > 0.6: score += 15
        if dl_val < 20: score += 20
        elif dl_val < 40: score += 10

        liq_pass = (av_val > 10_000_000) and (sp_val < 2.0)

        signal = (score >= 70) and liq_pass

        return {
            'Ticker': ticker,
            'Price': round(float(df['Close'].iloc[-1]), 4),
            'Score': int(score),
            'Signal': signal,
            'Vol_Ratio': round(vr_val, 2),
            'Stubborn_Pct': round(st_val * 100, 1),
            'Dist_From_Low': round(dl_val, 1),
            'Range_Contract': rc_val,
            'Liq_Pass': liq_pass,
            'Error': None
        }
    except Exception as e:
        return {'Ticker': ticker, 'Price': None, 'Score': None, 'Signal': None,
                'Vol_Ratio': None, 'Stubborn_Pct': None, 'Dist_From_Low': None,
                'Range_Contract': None, 'Liq_Pass': None, 'Error': str(e)}


if __name__ == '__main__':
    watchlist = [
        "IWM", "KLR", "WLDN", "BBBY", "ZM", "LCID", "HOOD", "PLTR", "NVDA",
        "RKLB", "SMR", "ASTS", "IONQ", "SERV", "SOUN", "BITF", "MARA", "RIOT",
        "HIMS", "UPST", "COIN", "MSTR", "SMCI", "ARM", "CRWD"
    ]

    results = []
    for t in watchlist:
        print(f"Processing {t}...", flush=True)
        res = get_lens_score(t)
        results.append(res)
        if res['Score'] is not None:
            print(f"  -> Score={res['Score']}, Signal={res['Signal']}, Price={res['Price']}", flush=True)
        else:
            print(f"  -> {res['Error']}", flush=True)

    df_results = pd.DataFrame(results)

    print()
    print("=" * 120)
    print("FULL RAW TABLE")
    print("=" * 120)
    print(df_results.to_string(index=False))

    # Scored tickers
    scored = df_results[df_results['Score'].notna()].copy()
    scored['Score'] = scored['Score'].astype(int)
    scored = scored.sort_values('Score', ascending=False)

    print()
    print("=" * 120)
    print("SIGNALS SORTED BY SCORE (descending)")
    print("=" * 120)
    cols = ['Ticker', 'Price', 'Score', 'Signal', 'Vol_Ratio', 'Stubborn_Pct', 'Dist_From_Low', 'Range_Contract']
    print(scored[cols].to_string(index=False))

    # Markdown table
    print()
    print("=" * 120)
    print("MARKDOWN TABLE")
    print("=" * 120)
    print("| Ticker | Price | Score | Signal | Vol_Ratio | Stubborn_Pct | Dist_From_Low | Range_Contract |")
    print("|--------|-------|-------|--------|-----------|--------------|---------------|----------------|")
    for _, row in scored.iterrows():
        print(f"| {row['Ticker']} | {row['Price']} | {int(row['Score'])} | {row['Signal']} | {row['Vol_Ratio']} | {row['Stubborn_Pct']}% | {row['Dist_From_Low']}% | {row['Range_Contract']} |")

    # Error tickers
    errors = df_results[df_results['Error'].notna()]
    if not errors.empty:
        print()
        print("=" * 120)
        print("ERROR / INSUFFICIENT DATA TICKERS")
        print("=" * 120)
        print(errors[['Ticker', 'Error']].to_string(index=False))

    # Summary
    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    if not scored.empty:
        signals_fired = scored[(scored['Score'] >= 70) & (scored['Liq_Pass'] == True)]
        print(f"Total tickers scanned: {len(watchlist)}")
        print(f"Tickers with valid data: {len(scored)}")
        print(f"Signals fired (Score>=70 AND Liq_Pass): {len(signals_fired)}")
        if not signals_fired.empty:
            for _, row in signals_fired.iterrows():
                print(f"  SIGNAL: {row['Ticker']} | Score={int(row['Score'])} | Price={row['Price']} | Vol_Ratio={row['Vol_Ratio']} | Stubborn={row['Stubborn_Pct']}% | Dist_Low={row['Dist_From_Low']}%")
        else:
            print("  (No signals fired)")

        print()
        top3 = scored.head(3)
        print("Top 3 by Score:")
        for _, row in top3.iterrows():
            print(f"  {row['Ticker']}: Score={int(row['Score'])}, Price={row['Price']}, Vol_Ratio={row['Vol_Ratio']}, Stubborn={row['Stubborn_Pct']}%, Dist_Low={row['Dist_From_Low']}%, Range_Contract={row['Range_Contract']}")

        print()
        print("False-positive warnings (Dist_From_Low < 20% but Vol_Ratio < 1.2 = weak buying pressure):")
        fp = scored[(scored['Dist_From_Low'] < 20) & (scored['Vol_Ratio'] < 1.2)]
        if not fp.empty:
            for _, row in fp.iterrows():
                print(f"  WARNING: {row['Ticker']} - Dist_From_Low={row['Dist_From_Low']}% but Vol_Ratio only {row['Vol_Ratio']} (weak buying, potential value trap / BBBY-style false positive)")
        else:
            print("  None detected.")
