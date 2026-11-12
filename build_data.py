"""
앱이 쓰는 데이터를 미리 계산해 data/에 저장한다 (교수님 피드백: 슬라이더는 미리 계산한 근사값).

    python build_data.py            # 저장된 가격 CSV로 계산
    python build_data.py --download # 가격부터 다시 받기

저장 파일 (시장 = US, KR)
- prices_{시장}.csv          일별 수정종가
- frontier_{시장}_cap{상한}.csv  전체기간 프론티어 100개 지점 (수익률, 변동성, 샤프, 비중)
- stats_{시장}.csv           ETF별 연 기대수익률, 변동성
- corr_{시장}.csv            상관계수
- bt_nav_{시장}_lb{기간}.csv      모형별 / 성향 위치별 기준가 (기간 = 추정에 쓴 직전 거래일 수 252, 756)
- bt_weights_{시장}_lb{기간}.csv  리밸런싱 날짜별 비중 (long 형식)
"""
import sys
import time

import numpy as np
import pandas as pd

from core import backtest as bt
from core import config, data
from core.frontier import frontier_table

POSITIONS = np.round(np.linspace(0, 1, 21), 2)  # 성향 위치 0.00, 0.05, ..., 1.00


def pos_key(p: float, cap: float) -> str:
    return f"pos_{p:.2f}_cap{cap:.1f}"


def build(market: str, download: bool, log=print):
    """download=True면 최신 가격을 받아 다시 계산. 계산이 모두 끝난 뒤에만 파일을 쓴다 -
    중간에 인터넷이 끊기거나 에러가 나도 기존 data/가 반쯤 바뀐 상태로 남지 않게."""
    t0 = time.time()
    px = data.download_prices(market, save=False) if download else data.load_prices(market)
    r = px.pct_change().dropna()
    out = {}   # 파일 이름 -> (DataFrame, index 저장 여부)

    mu, cov = r.mean() * config.ANNUAL, r.cov() * config.ANNUAL
    for cap in config.WEIGHT_CAPS:
        out[f"frontier_{market}_cap{cap:.1f}.csv"] = (frontier_table(mu, cov, config.N_FRONTIER, cap), False)
    out[f"stats_{market}.csv"] = (pd.DataFrame({"Return": mu, "Volatility": np.sqrt(np.diag(cov))}), True)
    out[f"corr_{market}.csv"] = (r.corr(), True)

    st = None
    for lb in sorted(config.LOOKBACKS, reverse=True):   # 가장 긴 기간부터: 시작일을 그 기준으로 통일
        weights = bt.model_weights(r, lb)
        rebal = weights["MVP"].index
        for cap in config.WEIGHT_CAPS:
            for p, W in bt.position_weights(r, POSITIONS, rebal, cap, lb).items():
                weights[pos_key(p, cap)] = W
        # 추정 기간이 달라도 같은 날부터 비교해야 공정함 -> 가장 긴 기간의 첫 리밸런싱 날짜를 공통 시작일로
        st = st or bt.start_date(rebal)
        navs = pd.DataFrame({k: bt.run_nav(W, r, st) for k, W in weights.items()})
        out[f"bt_nav_{market}_lb{lb}.csv"] = (navs.round(4), True)
        out[f"bt_weights_{market}_lb{lb}.csv"] = (
            pd.concat({k: W.loc[st:] for k, W in weights.items()}, names=["model", "date"]), True)

    data.DATA_DIR.mkdir(exist_ok=True)
    if download:
        px.to_csv(data.price_path(market))
    for name, (df, idx) in out.items():
        df.to_csv(data.DATA_DIR / name, index=idx)
    log(f"[{market}] {px.index[0].date()}~{px.index[-1].date()}, 백테스트 {st.date()}~, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    for m in config.MARKETS:
        build(m, "--download" in sys.argv)
