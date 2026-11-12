"""가격 데이터 다운로드와 수익률 계산. 매번 받지 않도록 data/에 CSV로 저장해 둔다."""
from pathlib import Path

import pandas as pd

from core import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def price_path(market: str) -> Path:
    return DATA_DIR / f"prices_{market}.csv"


def download_prices(market: str) -> pd.DataFrame:
    import yfinance as yf

    tks = config.tickers(market)
    raw = yf.download(tks, start=config.START, end=config.END, auto_adjust=True, progress=False)
    px = raw["Close"][tks]
    px.columns = config.names(market)
    # 공통 거래일만 사용: 모든 ETF 가격이 있는 첫날부터, 중간 결측은 직전 값으로 최대 5일 채움
    px = px.loc[px.notna().all(axis=1).idxmax():].ffill(limit=5).dropna()
    DATA_DIR.mkdir(exist_ok=True)
    px.to_csv(price_path(market))
    return px


def load_prices(market: str) -> pd.DataFrame:
    p = price_path(market)
    if not p.exists():
        return download_prices(market)
    return pd.read_csv(p, index_col=0, parse_dates=True)


def load_returns(market: str) -> pd.DataFrame:
    return load_prices(market).pct_change().dropna()
