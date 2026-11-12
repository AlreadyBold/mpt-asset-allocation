"""
M2 효율적 프론티어 엔진.

수업 efficient_frontier_plot.py의 get_efficient_frontier와 같은 문제(목표 수익률에서
변동성 최소화, 0 <= w <= 1, 비중 합 1)를 풀되, 다음을 고쳤다.
- 진짜 MVP를 먼저 구하고 MVP 수익률 ~ 최고 수익률 구간을 N등분 -> 요청한 N개가 그대로 남음
  (원본은 자산 최저 수익률부터 나눈 뒤 MVP 아래를 버려서 지점 수가 줄어듦)
- 최적화 실패(res.success == False) 지점은 버림
- 비중을 반올림하지 않음 (표시할 때만 반올림)
"""
import numpy as np
import pandas as pd
import scipy.optimize as sco


def portf_rtn(w, mu):
    return float(np.dot(w, mu))


def portf_vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def _solve(fun, n, extra_cons=(), x0=None, bounds=None):
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}, *extra_cons]
    bounds = bounds or tuple((0.0, 1.0) for _ in range(n))
    x0 = np.repeat(1 / n, n) if x0 is None else x0
    res = sco.minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, 1)
    return w / w.sum(), res.success


def min_variance(cov):
    n = cov.shape[0]
    w, _ = _solve(lambda x: x @ cov @ x, n)
    return w


def max_sharpe(mu, cov, rf=0.0):
    n = len(mu)
    w, _ = _solve(lambda x: -(x @ mu - rf) / np.sqrt(x @ cov @ x), n)
    return w


def risk_parity(cov) -> np.ndarray:
    """위험기여도가 모두 같은 비중 (수업 RP와 같은 목적함수, 한 시점용)."""
    n = cov.shape[0]

    def obj(x):
        rc = x * (cov @ x) / np.sqrt(x @ cov @ x)
        return np.sum((rc[:, None] - rc[None, :]) ** 2)

    w, _ = _solve(obj, n, bounds=tuple((1e-6, 1.0) for _ in range(n)))
    return w


def max_diversification(cov) -> np.ndarray:
    """분산비율 최대 비중 (수업 MaximizeDiversification과 같은 목적함수, 한 시점용)."""
    n = cov.shape[0]
    sd = np.sqrt(np.diag(cov))
    w, _ = _solve(lambda x: -(x @ sd) / np.sqrt(x @ cov @ x), n)
    return w


def frontier_table(mu: pd.Series, cov: pd.DataFrame, n_points: int = 100, cap: float = 1.0) -> pd.DataFrame:
    """MVP부터 최고 수익률 끝까지 n_points개 효율적 포트폴리오. 열: Return, Volatility, Sharpe, 자산별 비중.
    cap: 자산별 비중 상한."""
    m, c = mu.to_numpy(), cov.to_numpy()
    n = len(m)
    bounds = tuple((0.0, cap) for _ in range(n))
    w_mvp, _ = _solve(lambda x: x @ c @ x, n, bounds=bounds)
    w_top, _ = _solve(lambda x: -(x @ m) + 1e-6 * (x @ c @ x), n, bounds=bounds)
    targets = np.linspace(portf_rtn(w_mvp, m), portf_rtn(w_top, m), n_points)

    rows, x0 = [], w_mvp
    for i, tgt in enumerate(targets):
        if i == 0:
            w, ok = w_mvp, True
        else:
            w, ok = _solve(lambda x: x @ c @ x, n,
                           ({"type": "eq", "fun": lambda x, t=tgt: x @ m - t},), x0=x0, bounds=bounds)
        if not ok:
            continue
        x0 = w
        r, v = portf_rtn(w, m), portf_vol(w, c)
        rows.append([r, v, r / v, *w])
    return pd.DataFrame(rows, columns=["Return", "Volatility", "Sharpe", *mu.index])


def weights_at_position(mu: np.ndarray, cov: np.ndarray, pos: float, cap: float = 1.0) -> np.ndarray:
    """
    프론티어 위 '상대 위치'(0 = MVP, 1 = 최고 수익률 끝)에 해당하는 비중.
    위치는 변동성 기준: 목표 변동성 = MVP 변동성 + pos x (끝 변동성 - MVP 변동성).
    목표 변동성 이하에서 기대수익률이 최대인 포트폴리오를 찾는다.
    cap: 자산별 비중 상한 (1.0 = 제한 없음). 백테스트에서 매년 다시 계산할 때 쓴다.
    """
    n = len(mu)
    bounds = tuple((0.0, cap) for _ in range(n))
    w_mvp, _ = _solve(lambda x: x @ cov @ x, n, bounds=bounds)
    if pos <= 0:
        return w_mvp
    w_top, _ = _solve(lambda x: -(x @ mu) + 1e-6 * (x @ cov @ x), n, bounds=bounds)
    v_lo, v_hi = portf_vol(w_mvp, cov), portf_vol(w_top, cov)
    if pos >= 1 or v_hi <= v_lo:
        return w_top if pos >= 1 else w_mvp
    target = v_lo + pos * (v_hi - v_lo)
    w, ok = _solve(lambda x: -(x @ mu), n,
                   ({"type": "ineq", "fun": lambda x: target ** 2 - x @ cov @ x},), x0=w_mvp, bounds=bounds)
    return w if ok else w_mvp


def risk_contribution(w, cov) -> np.ndarray:
    """자산별 위험기여도 비율 (합 = 1): RC_i = w_i (Σw)_i / σ²."""
    w = np.asarray(w, dtype=float)
    var = w @ cov @ w
    return w * (cov @ w) / var
