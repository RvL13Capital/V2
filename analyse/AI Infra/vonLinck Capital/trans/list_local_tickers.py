
from pathlib import Path

def main():
    data_dir = Path("data/raw")
    tickers = []
    print(f"Scanning {data_dir.absolute()}...")
    
    for f in data_dir.glob("*.parquet"):
        tickers.append(f.stem)
    for f in data_dir.glob("*.csv"):
        tickers.append(f.stem)
        
    tickers = sorted(list(set(tickers)))
    
    print(f"Found {len(tickers)} tickers.")
    
    with open("all_downloaded_tickers.txt", "w") as f:
        f.write("\n".join(tickers))
        
    print("Saved to all_downloaded_tickers.txt")

if __name__ == "__main__":
    main()
