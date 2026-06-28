"""Download daily adjusted-close prices for the asset universe (auth-free, via yfinance)."""
import os
import yfinance as yf

TICKERS = ["SPY", "TLT", "GLD", "QQQ", "IWM", "EFA", "VNQ", "HYG"]
START, END = "2010-01-01", "2024-12-31"
DEST = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DEST, exist_ok=True)

if __name__ == "__main__":
    print(f"Downloading {len(TICKERS)} tickers from Yahoo Finance ...")
    df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    close = df["Close"][TICKERS].dropna()
    out = os.path.join(DEST, "prices.csv")
    close.to_csv(out)
    print(f"Saved {close.shape[0]} rows x {close.shape[1]} assets -> {out}")
