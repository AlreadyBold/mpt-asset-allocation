"""
수업 자료 cal_nav_module.py (9주차) 중 이 프로젝트에서 쓰는 함수만 옮긴 것.

원본과 달라진 점
- get_summary_pf에 소르티노 비율 추가
- 시작일 검사의 빈 raise를 ValueError로 바꿈 (빈 raise는 그 자체로 에러가 남)
- pandas 3에서 object dtype 경고가 나지 않도록 패딩 DataFrame을 float으로 만듦
- pr_reverse, get_corr_ts는 쓰지 않아서 뺐다
"""
import numpy as np
import pandas as pd
import quantstats as qs


def cal_nav(W: pd.DataFrame, st_date: str, ed_date: str, init_value: float, bp: float, ret_df: pd.DataFrame):
    """
    W: 리밸런싱 날짜별 비중, ret_df: 일별 수익률, bp: 거래금액 대비 비용(0.001 = 10bp)
    장 마감 직전에 팔고 사면서 리밸런싱한다고 가정.
    """
    if pd.to_datetime(st_date) < W.index[0]:
        raise ValueError("시뮬레이션 시작 시점이 포트폴리오 비중 시작일보다 앞섭니다.")

    W = W.loc[st_date:ed_date]
    ret_df = ret_df.loc[st_date:ed_date]

    daily_dates = ret_df.index
    rebal_dates = set(W.index)

    vh = np.zeros((len(daily_dates), W.shape[1]))
    rets = ret_df[W.columns].to_numpy()
    for i, t in enumerate(daily_dates):
        if i == 0:
            vh[i] = W.iloc[0].to_numpy() * init_value
            continue
        vh[i] = vh[i - 1] * (1 + rets[i])
        if t in rebal_dates:
            cur = vh[i]
            new = W.loc[t].to_numpy() * cur.sum()
            fee = np.abs(new - cur).sum() * bp
            vh[i] = W.loc[t].to_numpy() * (cur.sum() - fee)

    Vh = pd.DataFrame(vh, index=daily_dates, columns=W.columns)
    Pw = Vh.div(Vh.sum(axis=1), axis=0)
    nav = Vh.sum(axis=1)
    return nav, Pw, Vh


def get_summary_pf(concat_nav: pd.DataFrame) -> pd.DataFrame:
    r = concat_nav.pct_change().fillna(0)
    cum_ret = (concat_nav.iloc[-1] / concat_nav.iloc[0]) - 1
    pf_ms = pd.concat([
        cum_ret,
        qs.stats.cagr(r),
        qs.stats.volatility(r),
        qs.stats.sharpe(r),
        qs.stats.sortino(r),
        qs.stats.max_drawdown(r),
    ], axis=1)
    pf_ms.columns = ["누적수익률", "CAGR", "변동성", "샤프", "소르티노", "MDD"]
    return pf_ms.apply(pd.to_numeric, errors="coerce")
