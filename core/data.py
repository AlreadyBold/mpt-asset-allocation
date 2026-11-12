"""가격 데이터 다운로드와 수익률 계산. 매번 받지 않도록 data/에 CSV로 저장해 둔다."""
from pathlib import Path

import pandas as pd

from core import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def price_path(market: str) -> Path:
    return DATA_DIR / f"prices_{market}.csv"


def end_date() -> str:
    """yfinance end(미포함). 설정이 없으면 오늘 -> 어제 종가까지 받음."""
    return config.END or pd.Timestamp.today().strftime("%Y-%m-%d")


def download_prices(market: str, save: bool = True) -> pd.DataFrame:
    """save=False면 저장하지 않고 돌려만 줌 (build_data가 계산을 다 끝낸 뒤 한꺼번에 저장하려고)."""
    import yfinance as yf

    tks = config.tickers(market)
    raw = yf.download(tks, start=config.MARKETS[market].get("start", config.START), end=end_date(),
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"{market} 가격을 받지 못했습니다 (인터넷 연결이나 Yahoo Finance 상태를 확인하세요)")
    px = raw["Close"][tks]
    if config.MARKETS[market].get("usd_to_krw"):
        # 미국 종목(.KS가 아닌 티커) 가격에 원/달러 환율을 곱해 원화 기준으로 바꿈
        fx = yf.download("KRW=X", start=px.index[0], end=end_date(), auto_adjust=True, progress=False)["Close"].iloc[:, 0]
        fx = fx.reindex(px.index).ffill()
        us = [t for t in tks if not t.endswith(".KS")]
        px[us] = px[us].mul(fx, axis=0)
    px.columns = config.names(market)
    # 공통 거래일만 사용: 모든 종목 가격이 있는 첫날부터, 중간 결측은 직전 값으로 최대 5일 채움
    px = px.loc[px.notna().all(axis=1).idxmax():].ffill(limit=5).dropna()
    if save:
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


def last_date(market: str) -> pd.Timestamp | None:
    """저장된 가격의 마지막 날짜 (파일이 없으면 None). 파일 끝 줄만 읽어서 빠름."""
    p = price_path(market)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        f.seek(0, 2)
        f.seek(max(f.tell() - 4096, 0))
        tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()[-1]
    return pd.Timestamp(tail.split(",")[0])


def days_old(market: str) -> int | None:
    d = last_date(market)
    return None if d is None else (pd.Timestamp.today().normalize() - d.normalize()).days
