"""
MPT 포트폴리오 추천 (Streamlit)

    streamlit run app.py

데이터는 build_data.py로 미리 계산해 둔 data/*.csv를 읽는다.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import backtest as bt
from core import config, data, survey
from core.frontier import (max_diversification, max_sharpe, min_variance, portf_rtn, portf_vol,
                           risk_contribution, risk_parity)

st.set_page_config(page_title="MPT 포트폴리오 추천", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- 스타일 상수
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLASS_COLOR = {"주식": "#2a78d6", "채권": "#eb6834", "부동산": "#1baf7a", "실물": "#eda100"}
MODEL_COLOR = {"내 포트폴리오": "#2a78d6", "MVP": "#eb6834", "최대샤프": "#1baf7a",
               "RP": "#eda100", "MD": "#e87ba4", "EW": MUTED}
TYPE_FILL = {"안정추구형": "rgba(27,175,122,0.10)", "위험중립형": "rgba(237,161,0,0.12)",
             "공격추구형": "rgba(227,73,72,0.10)"}
MODELS = ["MVP", "최대샤프", "RP", "MD", "EW"]
STRESS = [("2018년 4분기 급락", "2018-09-20", "2018-12-24"),
          ("코로나 폭락 (2020)", "2020-02-19", "2020-03-23"),
          ("금리 급등 (2022)", "2022-01-03", "2022-10-14")]
GLOSSARY = {
    "기대수익률": "과거 일별수익률 평균 × 250. 1년에 평균적으로 기대할 수 있는 수익률",
    "변동성": "일별수익률 표준편차 × √250. 수익률이 얼마나 출렁이는지, 위험의 기본 척도",
    "샤프비율": "수익률 ÷ 변동성 (무위험수익률 0 가정). 위험 1단위당 수익",
    "소르티노비율": "수익률 ÷ 하락한 날의 변동성. 손실 위험만 벌점으로 봄",
    "MDD (최대낙폭)": "고점 대비 가장 많이 떨어졌던 폭. 겪을 수 있었던 최악의 손실",
    "CAGR": "여러 해의 수익을 1년 복리로 환산한 연평균 수익률",
    "효율적 프론티어": "같은 위험에서 기대수익률이 가장 높은 포트폴리오를 이은 곡선",
    "MVP (최소분산)": "프론티어의 왼쪽 끝. 변동성이 가장 낮은 조합",
    "최대샤프": "샤프비율이 가장 큰 조합. 기대수익률 추정에 민감해 참고용으로만 표시",
    "RP (리스크 패리티)": "자산마다 포트폴리오 위험을 똑같이 나눠 지게 만든 조합",
    "MD (최대분산)": "분산비율(개별 변동성 합 ÷ 포트폴리오 변동성)을 최대로 만든 조합",
    "EW (동일비중)": "10개 ETF에 10%씩. 최적화 모형이 넘어야 할 벤치마크",
    "위험기여도": "포트폴리오 전체 변동성 중 각 자산이 만들어 낸 몫",
    "리밸런싱": "정해진 주기(매년 말)마다 비중을 다시 계산해 맞추는 것",
}


def fig_layout(fig: go.Figure, height=380, **kw) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK2, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal"), hoverlabel=dict(font_size=13), **kw)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------- 데이터
@st.cache_data
def load(market: str):
    d = data.DATA_DIR
    r = data.load_returns(market)
    out = {
        "returns": r,
        "stats": pd.read_csv(d / f"stats_{market}.csv", index_col=0),
        "corr": pd.read_csv(d / f"corr_{market}.csv", index_col=0),
        "frontier": {c: pd.read_csv(d / f"frontier_{market}_cap{c:.1f}.csv") for c in config.WEIGHT_CAPS},
        "nav": pd.read_csv(d / f"bt_nav_{market}.csv", index_col=0, parse_dates=True),
        "weights": pd.read_csv(d / f"bt_weights_{market}.csv", index_col=[0, 1], parse_dates=["date"]),
        "mu": r.mean() * config.ANNUAL,
        "cov": r.cov() * config.ANNUAL,
    }
    return out


def pos_key(p: float, cap: float) -> str:
    return f"pos_{p:.2f}_cap{cap:.1f}"


# ================================================================ 1단계: 설문 화면
# 설문을 끝내고 '결과 보기'를 누르면 대시보드로 넘어간다. 응답은 session_state["answers"]에 보관
# (위젯 값은 화면에서 사라지면 Streamlit이 지우므로 따로 저장해야 '설문 다시 하기' 때 이전 응답을 채울 수 있음)
ss = st.session_state
ss.setdefault("page", "survey")
ss.setdefault("answers", [None] * len(survey.QUESTIONS))

if ss.page == "survey":
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.title("내 성향에 맞는 효율적 포트폴리오")
        st.write("7개 질문에 답하면 투자성향을 진단하고, 그 성향에 맞는 효율적 포트폴리오(ETF 투자비중)를 추천합니다.")
        answers = [ss.get(f"q{i}") for i in range(len(survey.QUESTIONS))]
        n_done = sum(a is not None for a in answers)
        st.progress(n_done / 7, text=f"{n_done} / 7 문항 응답")
        for i, (q, opts) in enumerate(survey.QUESTIONS):
            prev = ss.answers[i]
            st.radio(f"**Q{i + 1}. {q}**", opts, index=None if prev is None else prev, key=f"q{i}",
                     format_func=lambda o, opts=opts: f"{'①②③④⑤'[opts.index(o)]} {o}")
        answers = [ss.get(f"q{i}") for i in range(len(survey.QUESTIONS))]
        done = all(a is not None for a in answers)
        idx = [survey.QUESTIONS[i][1].index(a) if a is not None else None for i, a in enumerate(answers)]
        partial = sum(survey.option_score(j) for j in idx if j is not None)
        st.divider()
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            c1.metric("설문 점수" if done else "지금까지 점수", f"{partial}점",
                      help="① 5점, ② 4점, ③ 3점, ④ 2점, ⑤ 1점 (7~35점)")
            if done:
                t = survey.classify(partial)
                c2.subheader(t)
                c2.write(survey.TYPES[t]["desc"])
            else:
                c2.info(f"아직 {7 - n_done}문항이 남았습니다.")
        if st.button("결과 보기 →", type="primary", disabled=not done, width="stretch"):
            ss.answers = idx
            ss.page = "dashboard"
            st.rerun()
        with st.expander("점수와 투자성향 구간"):
            st.dataframe(pd.DataFrame([{"점수": f"{v['range'][0]}~{v['range'][1]}점", "투자성향": k, "특징": v["desc"]}
                                       for k, v in survey.TYPES.items()]), hide_index=True, width="stretch")
            st.caption("점수가 높을수록 효율적 프론티어의 오른쪽(높은 위험·높은 기대수익)에 위치합니다.")
    st.stop()

# ================================================================ 2단계: 대시보드
score = survey.total_score(ss.answers)
ptype = survey.classify(score)
done = True

with st.sidebar:
    st.header("내 투자성향")
    st.subheader(f"{ptype} · {score}점")
    if st.button("설문 다시 하기", width="stretch"):
        ss.page = "survey"
        st.rerun()
    st.divider()
    st.header("설정")
    market = st.radio("투자 대상", list(config.MARKETS), format_func=lambda m: config.MARKETS[m]["label"],
                      help="미국 상장 ETF는 해외주식 계좌로, 한국 상장 ETF는 국내 계좌·연금·ISA에서 살 수 있다")
    cap_on = st.toggle("자산별 비중 상한 40%", value=False,
                       help="한 자산에 비중이 몰리는 것(코너해)을 막는다. 추천 비중과 백테스트 모두에 적용")
    cap = 0.4 if cap_on else 1.0
    cur = config.MARKETS[market]["currency"]
    amount = st.number_input(f"투자금 ({'원' if cur == 'KRW' else '달러'})", min_value=0,
                             value=10_000_000 if cur == "KRW" else 10_000,
                             step=1_000_000 if cur == "KRW" else 1_000)
    st.divider()
    st.caption("데이터: Yahoo Finance 수정종가 · 매년 말 리밸런싱 · 거래비용 10bp · "
               "리밸런싱 때 직전 3년 데이터 사용")

D = load(market)
assets = config.asset_table(market)
ft = D["frontier"][cap]
names = list(assets.index)

# 성향 -> 프론티어 위치 (변동성 기준 0~1) -> 가장 가까운 프론티어 지점
v_lo, v_hi = ft["Volatility"].iloc[0], ft["Volatility"].iloc[-1]
ft = ft.assign(pos=(ft["Volatility"] - v_lo) / (v_hi - v_lo))


def nearest_row(pos: float) -> int:
    return int((ft["pos"] - pos).abs().idxmin())


def row_type(pos: float) -> str:
    for t in survey.TYPES:
        lo, hi = survey.type_band(t)
        if lo <= pos <= hi:
            return t
    return ptype


default_row = nearest_row(survey.score_to_position(score))

st.title("내 포트폴리오 대시보드")
st.caption(f"투자 대상: **{config.MARKETS[market]['label']}** · 바로가기: "
           "[내 포트폴리오](#portfolio) · [모형 비교](#compare) · [성과 리포트](#report) · [자산 정보](#assets)")

# ================================================================ 내 포트폴리오
with st.container():
    st.header("내 포트폴리오", anchor="portfolio", divider="gray")

    rets_labels = [f"{r * 100:.2f}%" for r in ft["Return"]]
    sel_label = st.select_slider(
        "기대수익률 조정 (효율적 프론티어 위 100개 지점 중 선택)", options=rets_labels,
        value=rets_labels[default_row], key=f"slider_{market}_{cap}_{score}",
        help="미리 계산한 100개 지점 중 가장 가까운 지점을 보여줍니다. 처음 위치는 설문 점수로 정해집니다.")
    row = rets_labels.index(sel_label)
    P = ft.iloc[row]
    w_my = P[names].astype(float)
    my_pos = float(P["pos"])
    bt_pos = round(round(my_pos / 0.05) * 0.05, 2)
    my_nav = D["nav"][pos_key(bt_pos, cap)]
    my_dd = float((my_nav / my_nav.cummax() - 1).min())
    at_type = row_type(my_pos)

    c1, c2 = st.columns([1, 3], gap="medium")
    with c1:
        with st.container(border=True):
            st.caption("내 투자성향")
            st.subheader(ptype)
            st.write(f"설문 점수 **{score}점**")
            st.caption(survey.TYPES[ptype]["desc"])
            if at_type != ptype:
                st.warning(f"지금 고른 지점은 **{at_type}** 구간입니다. 내 성향보다 "
                           f"{'위험이 큽니다' if list(survey.TYPES).index(at_type) > list(survey.TYPES).index(ptype) else '보수적입니다'}.")
    with c2:
        m = st.columns(5)
        m[0].metric("기대수익률 (연)", pct(P["Return"]), help=GLOSSARY["기대수익률"])
        m[1].metric("변동성 (연)", pct(P["Volatility"]), help=GLOSSARY["변동성"])
        m[2].metric("샤프비율", f"{P['Sharpe']:.2f}", help=GLOSSARY["샤프비율"])
        m[3].metric("과거 최대낙폭", pct(my_dd), help="같은 위치를 매년 말 다시 계산해 보유했을 때의 백테스트 MDD")
        cagr = (my_nav.iloc[-1] / my_nav.iloc[0]) ** (252 / len(my_nav)) - 1
        m[4].metric("과거 연평균 수익률", pct(cagr), help=f"백테스트 CAGR ({my_nav.index[0]:%Y.%m}~)")
        lo_ret, hi_ret = ft["Return"].iloc[0], ft["Return"].iloc[-1]
        st.caption(f"프론티어 범위: 기대수익률 {pct(lo_ret)} ~ {pct(hi_ret)} · 변동성 {pct(v_lo)} ~ {pct(v_hi)} · "
                   f"프론티어 위 위치 {my_pos:.2f} (0 = MVP, 1 = 끝)")

    g1, g2 = st.columns([3, 2], gap="large")
    with g1:
        st.markdown("**효율적 프론티어 위 내 위치**")
        fig = go.Figure()
        for t in survey.TYPES:
            lo, hi = survey.type_band(t)
            fig.add_vrect(x0=v_lo + lo * (v_hi - v_lo), x1=v_lo + hi * (v_hi - v_lo), fillcolor=TYPE_FILL[t],
                          line_width=0, layer="below", annotation_text=t,
                          annotation_position="bottom left" if t == "안정추구형" else "top left",
                          annotation_font=dict(color=INK2, size=12))
        st_ = D["stats"]
        fig.add_trace(go.Scatter(x=st_["Volatility"], y=st_["Return"], mode="markers+text", text=st_.index,
                                 textposition="top center", textfont=dict(size=11, color=MUTED),
                                 marker=dict(size=8, color=MUTED, line=dict(width=2, color="#fcfcfb")),
                                 name="개별 ETF", hovertemplate="%{text}<br>수익률 %{y:.1%} · 변동성 %{x:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=ft["Volatility"], y=ft["Return"], mode="lines", line=dict(color=INK, width=2),
                                 name="효율적 프론티어", customdata=ft["Sharpe"],
                                 hovertemplate="수익률 %{y:.2%} · 변동성 %{x:.2%} · 샤프 %{customdata:.2f}<extra></extra>"))
        w_ms = max_sharpe(D["mu"].to_numpy(), D["cov"].to_numpy())
        mu_, cov_ = D["mu"].to_numpy(), D["cov"].to_numpy()
        fig.add_trace(go.Scatter(x=[ft["Volatility"].iloc[0]], y=[ft["Return"].iloc[0]], mode="markers",
                                 marker=dict(size=11, color="#eb6834", line=dict(width=2, color="#fcfcfb")),
                                 name="MVP" if cap == 1 else "MVP (상한 적용)",
                                 hovertemplate="MVP<br>수익률 %{y:.2%} · 변동성 %{x:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[portf_vol(w_ms, cov_)], y=[portf_rtn(w_ms, mu_)], mode="markers",
                                 marker=dict(size=11, color="#fcfcfb", symbol="diamond", line=dict(width=2, color="#1baf7a")),
                                 name="최대샤프 (참고)",
                                 hovertemplate="최대샤프 (상한 없음)<br>수익률 %{y:.2%} · 변동성 %{x:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[P["Volatility"]], y=[P["Return"]], mode="markers",
                                 marker=dict(size=18, color="#2a78d6", line=dict(width=3, color="#fcfcfb")),
                                 name="내 포트폴리오",
                                 hovertemplate="내 포트폴리오<br>수익률 %{y:.2%} · 변동성 %{x:.2%}<extra></extra>"))
        fig.update_xaxes(title="변동성 (연)", tickformat=".0%")
        fig.update_yaxes(title="기대수익률 (연)", tickformat=".0%")
        st.plotly_chart(fig_layout(fig, 430), width="stretch")

    with g2:
        st.markdown("**추천 투자비중**")
        tbl = assets.assign(weight=w_my).sort_values("weight", ascending=False)
        shown = tbl[tbl["weight"] >= 0.0005]
        fig = go.Figure(go.Bar(
            x=shown["weight"], y=shown.index, orientation="h",
            marker=dict(color=[CLASS_COLOR[c] for c in shown["class"]], cornerradius=4),
            text=[pct(w) for w in shown["weight"]], textposition="outside", textfont=dict(color=INK2),
            customdata=shown[["desc", "class"]], cliponaxis=False, showlegend=False,
            hovertemplate="%{y} · %{customdata[1]}<br>%{customdata[0]}<br>비중 %{x:.1%}<extra></extra>"))
        for c, col in CLASS_COLOR.items():
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color=col, name=c, showlegend=True))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=".0%", range=[0, min(1.0, shown["weight"].max() * 1.25)])
        fig_layout(fig, 60 + 34 * len(shown), bargap=0.35, showlegend=True)
        st.plotly_chart(fig, width="stretch")
        rest = len(tbl) - len(shown)
        if rest:
            st.caption(f"나머지 {rest}종 0%")
        by_class = tbl.groupby("class", sort=False)["weight"].sum()
        st.caption(" · ".join(f"{c} {pct(by_class.get(c, 0))}" for c in CLASS_COLOR))

    with st.expander(f"투자금 {amount:,.0f}{'원' if cur == 'KRW' else '달러'} 기준 ETF별 매수 금액", expanded=True):
        buy = tbl[tbl["weight"] >= 0.0005].assign(금액=lambda d: d["weight"] * amount)
        st.dataframe(
            buy.reset_index()[["name", "ticker", "class", "desc", "weight", "금액"]]
            .rename(columns={"name": "ETF", "ticker": "티커", "class": "자산군", "desc": "설명", "weight": "비중"})
            .style.format({"비중": "{:.1%}", "금액": "{:,.0f}"}),
            hide_index=True, width="stretch")

    st.markdown("**기대수익률에 따른 자산군 비중 변화** (교수님 자료의 '기대수익률에 해당하는 자산별 비중')")
    cls_w = ft[names].T.groupby(assets["class"]).sum().T
    fig = go.Figure()
    for c in CLASS_COLOR:
        if c in cls_w:
            fig.add_trace(go.Scatter(x=ft["Return"], y=cls_w[c], stackgroup="one", name=c, mode="lines",
                                     line=dict(width=0.5, color=CLASS_COLOR[c]), fillcolor=CLASS_COLOR[c],
                                     hovertemplate=c + " %{y:.1%}<extra></extra>"))
    fig.add_vline(x=P["Return"], line=dict(color=INK, width=2, dash="dot"),
                  annotation_text="내 위치", annotation_position="top")
    fig.update_xaxes(title="기대수익률 (연)", tickformat=".1%")
    fig.update_yaxes(title="비중", tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_layout(fig, 320, hovermode="x unified"), width="stretch")

# ================================================================ 모형 비교
with st.container():
    st.header("모형 비교", anchor="compare", divider="gray")
    mu_, cov_ = D["mu"].to_numpy(), D["cov"].to_numpy()
    W = pd.DataFrame({
        "내 포트폴리오": w_my.to_numpy(),
        "MVP": min_variance(cov_),
        "최대샤프": max_sharpe(mu_, cov_),
        "RP": risk_parity(cov_),
        "MD": max_diversification(cov_),
        "EW": np.repeat(1 / len(names), len(names)),
    }, index=names)
    st.markdown("**모형별 특징** (전체기간 데이터 기준)")
    rows = []
    for k in W:
        w = W[k].to_numpy()
        r_, v_ = portf_rtn(w, mu_), portf_vol(w, cov_)
        rows.append({"모형": k, "기대수익률": r_, "변동성": v_, "샤프": r_ / v_,
                     "편입 자산 수": int((w >= 0.005).sum()), "가장 큰 비중": f"{names[int(w.argmax())]} {pct(w.max())}",
                     "설명": "설문으로 고른 프론티어 위 지점" if k == "내 포트폴리오" else bt.MODEL_DESC[k]})
    st.dataframe(pd.DataFrame(rows).style.format({"기대수익률": "{:.1%}", "변동성": "{:.1%}", "샤프": "{:.2f}"}),
                 hide_index=True, width="stretch")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**투자비중**")
        st.dataframe(W.style.format("{:.1%}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                     width="stretch", height=390)
    with b:
        st.markdown("**위험기여도** (전체 변동성 중 각 자산이 만든 몫)", help=GLOSSARY["위험기여도"])
        RC = W.apply(lambda c: pd.Series(risk_contribution(c.to_numpy(), cov_), index=names))
        st.dataframe(RC.style.format("{:.1%}").background_gradient(cmap="Oranges", vmin=0, vmax=1),
                     width="stretch", height=390)
    st.info(
        "**최대샤프를 추천에 쓰지 않는 이유**  \n"
        "최대샤프는 이론상 위험 대비 가장 효율적인 지점이지만, 입력하는 기대수익률의 작은 오차에도 비중이 "
        "한두 자산으로 쏠립니다(코너해, 7주차 '평균-분산 모형의 한계'). 직전 1년 데이터로 매년 다시 계산하면 "
        "IEF 100%(2019), DBC 100%(2023) 같은 해가 생겼고 백테스트 샤프가 비교 모형 중 가장 낮았습니다. "
        "또 무위험수익률을 0으로 두면 한국 버전처럼 단기채(국고채3년)가 대부분을 차지합니다. "
        "그래서 이 서비스는 설문 점수에 비례한 **목표 변동성 지점**을 추천하고, 최대샤프는 프론티어 위 기준점으로만 표시합니다. "
        "기대수익률 추정 기간을 3년으로 늘리면 최대샤프의 성과도 크게 나아지는데, 이 결과는 아래 성과 리포트에서 비교할 수 있습니다.")

# ================================================================ 성과 리포트
with st.container():
    st.header("성과 리포트", anchor="report", divider="gray")
    nav_all = D["nav"].rename(columns={pos_key(bt_pos, cap): "내 포트폴리오"})
    pick = st.multiselect("함께 그릴 비교 기준 (추천 비중에는 영향 없음)", ["내 포트폴리오", *MODELS],
                          default=["내 포트폴리오", *MODELS],
                          help="추천은 설문 점수로만 정해집니다. MVP·RP·MD·EW·최대샤프는 내 포트폴리오가 "
                               "잘 만들어졌는지 확인하기 위한 비교 기준입니다.")
    if not pick:
        st.warning("모형을 하나 이상 고르세요. 지금은 내 포트폴리오만 보여줍니다.")
        pick = ["내 포트폴리오"]
    navs = nav_all[pick]
    navs = navs / navs.iloc[0] * 100
    st.caption(f"{navs.index[0]:%Y.%m.%d} ~ {navs.index[-1]:%Y.%m.%d} · 매년 말 리밸런싱(직전 3년 데이터로 비중 계산) · "
               f"거래비용 {config.TRADE_COST * 1e4:.0f}bp · 내 포트폴리오 = 프론티어 위치 {bt_pos:.2f}"
               + (" · 비중 상한 40%" if cap < 1 else ""))

    summ = bt.summary(navs)
    k = st.columns(4)
    me = summ.loc["내 포트폴리오"] if "내 포트폴리오" in summ.index else summ.iloc[0]
    ew = summ.loc["EW"] if "EW" in summ.index else None
    # 델타 문자열은 부호로 시작해야 Streamlit이 색(좋음/나쁨)을 맞게 칠함. MDD는 덜 떨어질수록(+) 좋음
    k[0].metric("누적수익률", pct(me["누적수익률"]), None if ew is None else f"{(me['누적수익률'] - ew['누적수익률']) * 100:+.1f}%p (EW 대비)")
    k[1].metric("CAGR", pct(me["CAGR"], 2), None if ew is None else f"{(me['CAGR'] - ew['CAGR']) * 100:+.2f}%p (EW 대비)")
    k[2].metric("샤프", f"{me['샤프']:.2f}", None if ew is None else f"{me['샤프'] - ew['샤프']:+.2f} (EW 대비)")
    k[3].metric("MDD", pct(me["MDD"]), None if ew is None else f"{(me['MDD'] - ew['MDD']) * 100:+.1f}%p (EW 대비)")

    def line_fig(df, title, fmt, height=360):
        fig = go.Figure()
        for c in df:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[c], name=c, mode="lines",
                line=dict(color=MODEL_COLOR[c], width=3 if c == "내 포트폴리오" else 2,
                          dash="dot" if c == "EW" else "solid"),
                hovertemplate=f"{c} %{{y:{fmt}}}<extra></extra>"))
        fig.update_yaxes(title=title, tickformat=fmt)
        return fig_layout(fig, height, hovermode="x unified")

    st.markdown("**누적성과** (시작 = 100)")
    st.plotly_chart(line_fig(navs, "기준가", ".0f"), width="stretch")
    st.markdown("**낙폭** (고점 대비 하락률)")
    st.plotly_chart(line_fig(bt.drawdown(navs), "낙폭", ".0%", 260), width="stretch")

    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("**성과평가 지표**")
        st.dataframe(summ.style.format({"누적수익률": "{:.1%}", "CAGR": "{:.2%}", "변동성": "{:.1%}",
                                        "샤프": "{:.2f}", "소르티노": "{:.2f}", "MDD": "{:.1%}"})
                     .highlight_max(subset=["CAGR", "샤프", "소르티노", "MDD"], color="#cde2fb"),
                     width="stretch")
        st.markdown("**하락장 구간 성과**")
        rows = {}
        for name, s, e in STRESS:
            seg = navs.loc[s:e]
            if len(seg) > 1:
                rows[f"{name} {s[2:7].replace('-', '.')}~{e[2:7].replace('-', '.')}"] = seg.iloc[-1] / seg.iloc[0] - 1
        if rows:
            st.dataframe(pd.DataFrame(rows).T.style.format("{:.1%}"), width="stretch")
    with b:
        st.markdown("**연도별 수익률**")
        ann = bt.annual_returns(navs)
        if (navs.index.year == ann.index[0]).sum() < 5:   # 시작일이 연말이면 첫 해는 하루뿐 -> 제외
            ann = ann.iloc[1:]
        ann.index.name = "연도"
        lim = float(np.nanmax(np.abs(ann.to_numpy())))
        st.dataframe(ann.style.format("{:.1%}").background_gradient(cmap="RdBu", vmin=-lim, vmax=lim),
                     width="stretch", height=38 + 35 * len(ann))
        st.caption(f"{ann.index[0]}년은 {navs.index[0]:%m.%d}부터, {ann.index[-1]}년은 {navs.index[-1]:%m.%d}까지")

    st.markdown("**내 포트폴리오 연도별 비중** (매년 말 다시 계산한 비중을 그다음 해에 보유)")
    wl = D["weights"].loc[pos_key(bt_pos, cap)]
    wl = wl[wl.index <= navs.index[-1]]
    fig = go.Figure()
    for c in CLASS_COLOR:
        cols = [n for n in names if assets.loc[n, "class"] == c]
        detail = wl[cols].apply(lambda r: "<br>".join(f"{k} {v:.0%}" for k, v in r.items() if v >= 0.005), axis=1)
        fig.add_trace(go.Bar(x=[f"{d.year + 1}" for d in wl.index], y=wl[cols].sum(axis=1), name=c,
                             marker=dict(color=CLASS_COLOR[c], line=dict(color="#fcfcfb", width=2)),
                             customdata=detail, hovertemplate=c + " %{y:.0%}<br>%{customdata}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="보유 연도")
    st.plotly_chart(fig_layout(fig, 320, barmode="stack", bargap=0.3), width="stretch")
    with st.expander("ETF별 연도별 비중 표"):
        st.dataframe(wl.set_index(wl.index.year + 1).rename_axis("보유 연도").style.format("{:.0%}")
                     .background_gradient(cmap="Blues", vmin=0, vmax=1), width="stretch")

# ================================================================ 자산 정보
with st.container():
    st.header("자산 정보", anchor="assets", divider="gray")
    s = D["stats"]
    info = assets.assign(기대수익률=s["Return"], 변동성=s["Volatility"], 샤프=s["Return"] / s["Volatility"])
    r = D["returns"]
    st.markdown(f"**투자 대상 ETF 10종** ({r.index[0]:%Y.%m.%d} ~ {r.index[-1]:%Y.%m.%d} 전체기간, 연율화)")
    st.dataframe(info.reset_index().rename(columns={"name": "ETF", "ticker": "티커", "class": "자산군", "desc": "설명"})
                 .style.format({"기대수익률": "{:.1%}", "변동성": "{:.1%}", "샤프": "{:.2f}"}),
                 hide_index=True, width="stretch")
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown("**상관계수** (낮거나 음수일수록 분산효과가 크다)")
        st.dataframe(D["corr"].style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                     width="stretch", height=390)
    with b:
        st.markdown("**용어 설명**")
        st.dataframe(pd.DataFrame(GLOSSARY.items(), columns=["용어", "뜻"]), hide_index=True,
                     width="stretch", height=390)
    with st.expander("계산 방법과 가정"):
        st.markdown(f"""
- **데이터**: Yahoo Finance(yfinance) 수정종가(배당·분할 반영), 10개 ETF 모두 가격이 있는 공통 거래일만 사용
- **프론티어**: 일별수익률 평균·공분산 × {config.ANNUAL}으로 연율화, 비중 0 ≤ w ≤ 1(상한 옵션 시 0.4), 합 1, 공매도·레버리지 없음.
  MVP부터 최고 수익률까지 {config.N_FRONTIER}개 지점을 미리 계산해 두고 슬라이더는 가장 가까운 지점을 보여줌
- **성향 → 위치**: 설문 점수(7~35점)를 0~1로 바꿔 변동성 기준 프론티어 위치로 사용. 10.5점, 24.5점이 성향 경계
- **백테스트**: 매년 말 리밸런싱, 직전 {config.LOOKBACK}거래일(약 3년) 데이터로 비중 계산, 거래비용 {config.TRADE_COST * 1e4:.0f}bp,
  리밸런싱 날 종가까지의 정보만 사용(미래 정보 없음). 모형은 수업 quant_model_modules, 기준가는 cal_nav로 계산
- **샤프비율**: 무위험수익률 0 가정
- **한국 상장 ETF**: 대부분 환헤지(H)·선물형 상품. 10종 공통 데이터가 2014.05부터라 백테스트는 2017년 말부터
""")

st.divider()
st.caption("⚠️ 이 화면은 수업 프로젝트용 프로토타입입니다. 과거 성과는 미래 수익을 보장하지 않으며, 투자 권유가 아닙니다.")
