"""
수업 자료 quant_model_modules_수정분.py (7주차) 중 이 프로젝트에서 쓰는 함수만 옮긴 것.

원본과 달라진 점
- SectorMomentumRotationalTopRank, RB는 쓰지 않아서 뺐다 (원본에서도 정의 안 된
  변수 pr, rets를 써서 실행되지 않음)
- RP 안의 쓰이지 않는 RC 함수(오타 weigt/wieght)를 뺐다
- 나머지 계산 로직은 원본 그대로
"""
import numpy as np
import pandas as pd
from pypfopt import expected_returns, objective_functions, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.optimize import minimize


def _rebal_index(returns: pd.DataFrame, start_pos: int, rebal_periods: str) -> pd.DatetimeIndex:
    """리밸런싱 날짜: 달력 기준 주기 날짜를 그 이전 마지막 거래일로 맞춘다 (원본 방식)."""
    idx = pd.date_range(returns.index[start_pos], returns.index[-1], freq=rebal_periods)
    return returns.index[returns.index.get_indexer(idx, method="ffill")]


def EW(returns: pd.DataFrame, rebal_periods: str) -> pd.DataFrame:
    """동일비중 포트폴리오."""
    n = len(returns.columns)
    rebal = _rebal_index(returns, 0, rebal_periods)
    return pd.DataFrame([[1 / n] * n for _ in range(len(rebal))], index=rebal, columns=returns.columns)


def clean_weights(weights, cutoff=1e-4, rounding=5):
    w = np.array(weights, dtype=float).copy()
    w[np.abs(w) < cutoff] = 0
    if rounding is not None:
        w = np.round(w, rounding)
    return w


def MaximizeDiversification(rebal_periods: str, returns: pd.DataFrame, lookback_periods: int,
                            bnd=None, long_only=True, frequency=252) -> pd.DataFrame:
    """최대분산(MD) 포트폴리오: 분산비율(개별 변동성 가중합 / 포트폴리오 변동성) 최대화."""

    def neg_dr(w, V):
        w_vol = np.dot(np.sqrt(np.diag(V)), w.T)
        port_vol = np.sqrt(np.dot(np.array(w).T, np.dot(V, w)))
        return -w_vol / port_vol

    def get_weights(w0, V, bnd, long_only):
        cons = ({"type": "eq", "fun": lambda x: x.sum() - 1.0},)
        if long_only and bnd is None:
            bnd = [(0, 1) for _ in range(len(w0))]
        res = minimize(neg_dr, w0, bounds=bnd, args=V, method="SLSQP", constraints=cons)
        return clean_weights(res.x)

    w0 = [1 / len(returns.columns)] * len(returns.columns)
    rebal = _rebal_index(returns, lookback_periods, rebal_periods)
    rows = []
    for idx in rebal:
        ext = returns.loc[:idx].iloc[-lookback_periods:]
        V = risk_models.sample_cov(ext, returns_data=True, frequency=frequency).values
        rows.append(get_weights(w0, V, bnd, long_only))
    return pd.DataFrame(rows, index=rebal, columns=returns.columns)


def MeanVarianceMaxSharpe(rebal_periods: str, returns: pd.DataFrame, lookback_periods: int,
                          frequency=252) -> pd.DataFrame:
    """평균-분산 최대샤프 포트폴리오 (PyPortfolioOpt, 무위험수익률 0)."""
    rebal = _rebal_index(returns, lookback_periods, rebal_periods)
    rows = []
    for idx in rebal:
        ext = returns.loc[:idx].iloc[-lookback_periods:]
        mu = expected_returns.mean_historical_return(ext, returns_data=True, compounding=True, frequency=frequency)
        S = risk_models.sample_cov(ext, returns_data=True, frequency=frequency)
        ef = EfficientFrontier(mu, S)
        res = ef.nonconvex_objective(objective_functions.sharpe_ratio,
                                     objective_args=(ef.expected_returns, ef.cov_matrix),
                                     weights_sum_to_one=True)
        rows.append([res[i] for i in ef.clean_weights()])
    return pd.DataFrame(rows, index=rebal, columns=returns.columns)


def MeanVarianceMinVolatility(rebal_periods: str, returns: pd.DataFrame, lookback_periods: int,
                              frequency=252) -> pd.DataFrame:
    """최소분산(MVP) 포트폴리오."""
    rebal = _rebal_index(returns, lookback_periods, rebal_periods)
    rows = []
    for idx in rebal:
        ext = returns.loc[:idx].iloc[-lookback_periods:]
        mu = expected_returns.mean_historical_return(ext, returns_data=True, compounding=True, frequency=frequency)
        S = risk_models.sample_cov(ext, returns_data=True, frequency=frequency)
        ef = EfficientFrontier(mu, S)
        res = ef.min_volatility()
        rows.append([res[i] for i in ef.clean_weights()])
    return pd.DataFrame(rows, index=rebal, columns=returns.columns)


def RP(rebal_periods: str, returns: pd.DataFrame, lookback_periods: int, frequency=252,
       cov_type="simple") -> pd.DataFrame:
    """리스크 패리티: 자산별 위험기여도가 같아지도록 비중 결정."""

    def get_weights(rets: pd.DataFrame):
        covmat = rets.cov().values

        def objective(x):
            sigma = (x.T @ covmat @ x) ** 0.5
            rc = x * (covmat @ x) / sigma
            a = rc.reshape(-1, 1)
            return np.sum(np.square(np.ravel(a - a.T)))

        x0 = np.repeat(1 / covmat.shape[1], covmat.shape[1])
        cons = ({"type": "eq", "fun": lambda x: x.sum() - 1.0},
                {"type": "ineq", "fun": lambda x: x})
        res = minimize(objective, x0, method="SLSQP", constraints=cons,
                       options={"ftol": 1e-20, "maxiter": 800})
        return res.x

    rebal = _rebal_index(returns, lookback_periods, rebal_periods)
    rows = [get_weights(returns.loc[:idx].iloc[-lookback_periods:]) for idx in rebal]
    return pd.DataFrame(rows, index=rebal, columns=returns.columns)
