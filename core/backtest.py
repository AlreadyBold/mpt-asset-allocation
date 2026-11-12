"""
M3 비교 모형 비중 + M4 성과 리포트 (연말 리밸런싱 백테스트).

- 비교 모형(MVP, 최대샤프, RP, MD, EW)은 수업 quant_model_modules 함수로 계산
- '내 포트폴리오'는 매년 말 직전 1년 데이터로 프론티어를 다시 그리고,
  성향 위치(0 = MVP ~ 1 = 최고 수익률)에 해당하는 지점을 보유
- NAV는 수업 cal_nav로 계산 (거래비용 반영)
- 리밸런싱 날짜 t의 비중은 t 종가까지의 데이터로 계산하고 t 종가에 매매 -> 미래 정보 사용 없음
"""
import numpy as np
import pandas as pd

from core import config
from core.frontier import weights_at_position
from course import quant_model_modules as qm
from course.cal_nav_module import cal_nav, get_summary_pf

MODEL_DESC = {
    "MVP": "변동성이 가장 낮은 조합 (기대수익률을 쓰지 않음)",
    "최대샤프": "위험 1단위당 수익이 가장 큰 조합 (참고용: 기대수익률 추정오차에 민감)",
    "RP": "자산마다 포트폴리오 위험을 똑같이 나눠 지는 조합",
    "MD": "분산비율(개별 변동성 합 ÷ 포트폴리오 변동성)이 가장 큰 조합",
    "EW": "모든 종목에 같은 비중 (벤치마크)",
}


def model_weights(returns: pd.DataFrame, lb: int = config.LOOKBACK) -> dict[str, pd.DataFrame]:
    """lb: 리밸런싱 때 쓰는 직전 거래일 수 (252 = 1년, 756 = 3년)."""
    rb, fq = config.REBAL, config.BT_FREQ
    return {
        "MVP": qm.MeanVarianceMinVolatility(rb, returns, lb, frequency=fq),
        "최대샤프": qm.MeanVarianceMaxSharpe(rb, returns, lb, frequency=fq),
        "RP": qm.RP(rb, returns, lb, frequency=fq),
        "MD": qm.MaximizeDiversification(rb, returns, lb, frequency=fq),
        "EW": qm.EW(returns, rb).loc[lambda d: d.index >= returns.index[lb]],
    }


def position_weights(returns: pd.DataFrame, positions, rebal_index, cap: float = 1.0,
                     lb: int = config.LOOKBACK) -> dict[float, pd.DataFrame]:
    """성향 위치별로 매년 말 프론티어 위 비중을 다시 계산."""
    out = {p: [] for p in positions}
    for t in rebal_index:
        ext = returns.loc[:t].iloc[-lb:]
        mu, cov = ext.mean().to_numpy() * config.ANNUAL, ext.cov().to_numpy() * config.ANNUAL
        for p in positions:
            out[p].append(weights_at_position(mu, cov, p, cap))
    return {p: pd.DataFrame(v, index=rebal_index, columns=returns.columns) for p, v in out.items()}


def start_date(rebal_index: pd.DatetimeIndex) -> pd.Timestamp:
    return rebal_index[rebal_index.year >= config.BT_START_YEAR][0]


def run_nav(W: pd.DataFrame, returns: pd.DataFrame, st) -> pd.Series:
    nav, _, _ = cal_nav(W, st, returns.index[-1], 100.0, config.TRADE_COST, returns)
    return nav


def summary(navs: pd.DataFrame) -> pd.DataFrame:
    return get_summary_pf(navs)


def annual_returns(navs: pd.DataFrame) -> pd.DataFrame:
    ye = navs.groupby(navs.index.year).last()
    first = navs.iloc[0]
    prev = ye.shift(1)
    prev.iloc[0] = first
    return ye / prev - 1


def drawdown(navs: pd.DataFrame) -> pd.DataFrame:
    return navs / navs.cummax() - 1
