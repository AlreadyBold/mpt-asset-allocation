"""
MPT 포트폴리오 추천 (Streamlit)

    streamlit run app.py

데이터는 build_data.py로 미리 계산해 둔 data/*.csv를 읽는다.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core import backtest as bt
from core import config, data, survey
from core.frontier import (max_diversification, max_sharpe, min_variance, portf_rtn, portf_vol,
                           risk_contribution, risk_parity)

st.set_page_config(page_title="MPT 포트폴리오 추천", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- 스타일 상수
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# 자산군(ETF) / 업종(개별 종목) 색: 종목 목록에 나오는 순서대로 고정 순서 팔레트를 배정
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"내 포트폴리오": "#2a78d6", "MVP": "#eb6834", "최대샤프": "#1baf7a",
               "RP": "#eda100", "MD": "#e87ba4", "EW": MUTED}
TYPE_FILL = {"안정추구형": "rgba(27,175,122,0.10)", "위험중립형": "rgba(237,161,0,0.12)",
             "공격추구형": "rgba(227,73,72,0.10)"}
MODELS = ["MVP", "최대샤프", "RP", "MD", "EW"]
STRESS = [("2018 4분기 급락", "2018-09-20", "2018-12-24"),
          ("2020 코로나 폭락", "2020-02-19", "2020-03-23"),
          ("2022 금리 급등", "2022-01-03", "2022-10-14")]
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
    "EW (동일비중)": "모든 종목에 같은 비중(10종이면 10%씩). 최적화 모형이 넘어야 할 벤치마크",
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


def shade(hex_color: str, t: float) -> str:
    """색을 흰색 쪽으로 t만큼 섞음 (0 = 원래 색). 같은 계열 안에서 종목을 구분할 때 씀."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(v + (255 - v) * t):02x}" for v in rgb)


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
        "nav": {lb: pd.read_csv(d / f"bt_nav_{market}_lb{lb}.csv", index_col=0, parse_dates=True)
                for lb in config.LOOKBACKS},
        "weights": {lb: pd.read_csv(d / f"bt_weights_{market}_lb{lb}.csv", index_col=[0, 1], parse_dates=["date"])
                    for lb in config.LOOKBACKS},
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
        st.write("7개 질문에 답하면 투자성향을 진단하고, 그 성향에 맞는 효율적 포트폴리오(ETF·주식 투자비중)를 추천합니다.")
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
    # PDF 버튼 자리만 먼저 잡아 둠: 리포트에 필요한 값(과거 성과, 비교)은 화면 아래쪽에서 계산되므로
    # 맨 끝에서 이 자리에 버튼을 넣는다
    pdf_slot = st.container()
    st.divider()
    st.header("설정")
    market = st.radio("투자 대상", list(config.MARKETS), format_func=lambda m: config.MARKETS[m]["label"],
                      key="market", on_change=lambda: ss.update(scroll_top=ss.get("scroll_top", 0) + 1),
                      help="ETF 버전은 주식·채권·리츠·금 등 자산군을 고르게 담고, 대형주 버전은 한국·미국 개별 종목 20개(원화 기준)로 구성한다. 미국 상장은 해외주식 계좌, 한국 상장은 국내 계좌에서 살 수 있다")
    cap_on = st.toggle("자산별 비중 상한 40%", value=False,
                       help="한 자산에 비중이 몰리는 것(코너해)을 막는다. 추천 비중과 백테스트 모두에 적용")
    cap = 0.4 if cap_on else 1.0
    lb = st.segmented_control(
        "과거 성과 계산 때 쓰는 추정 기간", list(config.LOOKBACKS), default=config.LOOKBACK,
        format_func=lambda x: f"직전 {config.LOOKBACKS[x]}", key="lookback",
        help="과거 성과(백테스트)에서 매년 말 비중을 다시 계산할 때 직전 몇 년 데이터를 쓸지. "
             "1년이면 지난해 성과를 그대로 따라가 비중이 한두 자산에 쏠리기 쉽다(특히 최대샤프). "
             "추천 비중 자체는 전체기간 데이터로 계산해서 바뀌지 않는다") or config.LOOKBACK
    cur = config.MARKETS[market]["currency"]
    amount = st.number_input(f"투자금 ({'원' if cur == 'KRW' else '달러'})", min_value=0,
                             value=10_000_000 if cur == "KRW" else 10_000,
                             step=1_000_000 if cur == "KRW" else 1_000)
    st.divider()
    # ---- 데이터 업데이트: 누르면 최신 가격으로 다시 받고 모든 값을 다시 계산
    st.subheader("데이터")
    if st.button("최신 데이터로 업데이트", width="stretch",
                 help="세 투자 대상의 가격을 어제 종가까지 다시 받고, 프론티어와 과거 성과를 모두 다시 계산합니다 "
                      "(1~2분). 계산이 다 끝나야 저장되므로 중간에 실패해도 기존 데이터는 그대로입니다."):
        import build_data
        with st.status("데이터 업데이트 중…", expanded=True) as status:
            try:
                for m in config.MARKETS:
                    st.write(f"{config.MARKETS[m]['label']} 가격 받고 다시 계산하는 중…")
                    build_data.build(m, True, log=lambda msg: st.write("✅ " + msg))
                st.cache_data.clear()
                status.update(label="업데이트 완료", state="complete")
                ss.update(scroll_top=ss.get("scroll_top", 0) + 1)
                st.rerun()
            except Exception as e:  # 인터넷 끊김, Yahoo 응답 없음 등 -> 기존 데이터 유지
                status.update(label="업데이트 실패: 기존 데이터를 그대로 씁니다", state="error")
                st.error(f"{e}")

if ss.get("scroll_top"):
    # 투자 대상을 바꾸면 화면 맨 위로 이동. Streamlit은 다시 그려도 스크롤 위치를 유지해서,
    # 아래쪽을 보다가 바꾸면 무엇이 바뀌었는지 알기 어려움. 숫자(nonce)가 바뀌어야 스크립트가 다시 실행됨
    components.html(f"""<script>// {ss.scroll_top}
      const d = window.parent.document;
      for (const sel of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main']) {{
        const el = d.querySelector(sel); if (el) el.scrollTo({{top: 0, behavior: 'smooth'}});
      }}
      window.parent.scrollTo({{top: 0, behavior: 'smooth'}});
    </script>""", height=0)

D = load(market)
assets = config.asset_table(market)
ft = D["frontier"][cap]
names = list(assets.index)
IS_STOCK = market == "STOCK"
UNIT, CLASS_LABEL = ("종목", "시장") if IS_STOCK else ("ETF", "자산군")
CLASS_COLOR = {c: PALETTE[i] for i, c in enumerate(dict.fromkeys(assets["class"]))}

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

# 처음 보는 사람도 읽을 수 있게 모형 이름을 쉬운 말로
FRIENDLY = {"내 포트폴리오": "내 추천 포트폴리오", "MVP": "가장 안전하게 (MVP)", "최대샤프": "효율 최고 (최대샤프)",
            "RP": "위험을 똑같이 나누기 (RP)", "MD": "최대한 분산하기 (MD)", "EW": "모두 같은 금액씩 (EW)"}
UP, DOWN = "#2a78d6", "#e34948"


def money(x: float) -> str:
    if cur == "KRW":
        return f"{x / 1e8:,.2f}억원" if abs(x) >= 1e8 else f"{x / 1e4:,.0f}만원"
    return f"{x:,.0f}달러"   # "$"는 Streamlit 마크다운에서 수식 기호로 읽혀 문장이 깨짐


def money_obj(x: float) -> str:
    """금액 + 목적격 조사 (원 -> 을, 달러 -> 를)."""
    return money(x) + ("을" if cur == "KRW" else "를")


# ---------------------------------------------------------------- 상단: 위험 수준 조절
st.title(f"{ptype}에게 추천하는 포트폴리오")
st.caption(f"설문 점수 {score}점 · 투자 대상: **{config.MARKETS[market]['label']}** · 바로가기: "
           "[추천 비중](#recommend) · [과거 성과](#history) · [위험 조절 원리](#frontier) · "
           "[다른 방법과 비교](#compare) · [자세히 보기](#details)")

rets_labels = [f"{r * 100:.1f}%" for r in ft["Return"]]
rets_labels = [f"{l} ({i})" if rets_labels.count(l) > 1 else l for i, l in enumerate(rets_labels)]
sel_label = st.select_slider(
    "위험 수준 조절 · 왼쪽으로 갈수록 안정적, 오른쪽으로 갈수록 공격적 (표시 값 = 1년 기대 수익률)",
    options=rets_labels, value=rets_labels[default_row], key=f"slider_{market}_{cap}_{score}_{data.last_date(market):%Y%m%d}",
    help="처음 위치는 설문 점수로 정해집니다. 효율적 프론티어 위 100개 지점을 미리 계산해 두고 가장 가까운 지점을 보여줍니다.")
row = rets_labels.index(sel_label)
P = ft.iloc[row]
w_my = P[names].astype(float)
my_pos = float(P["pos"])
bt_pos = round(round(my_pos / 0.05) * 0.05, 2)
nav_all = D["nav"][lb].rename(columns={pos_key(bt_pos, cap): "내 포트폴리오"})
my_nav = nav_all["내 포트폴리오"]
my_dd = float((my_nav / my_nav.cummax() - 1).min())
ann_my = bt.annual_returns(my_nav.to_frame("r"))["r"]
if (my_nav.index.year == ann_my.index[0]).sum() < 5:   # 시작일이 연말이면 첫 해는 하루뿐 -> 제외
    ann_my = ann_my.iloc[1:]
at_type = row_type(my_pos)
tbl = assets.assign(weight=w_my).sort_values("weight", ascending=False)
held = tbl[tbl["weight"] >= 0.0005]

# ================================================================ 1. 추천 비중
st.header(f"{money_obj(amount)} 이렇게 나눠 담으세요",
          anchor="recommend", divider="gray")
c1, c2, c3 = st.columns([1.1, 1.4, 2.2], gap="large")
with c1:
    with st.container(border=True, height=360):
        st.caption("내 투자성향")
        st.subheader(ptype)
        st.write(f"설문 점수 **{score}점**")
        st.caption(survey.TYPES[ptype]["desc"])
        st.caption("지금 고른 위험 수준")
        if at_type == ptype:
            st.markdown(f"✅ **{at_type}**에 맞는 수준")
        else:
            riskier = list(survey.TYPES).index(at_type) > list(survey.TYPES).index(ptype)
            st.markdown(f"⚠️ **{at_type}** 수준  \n내 성향보다 {'위험이 큽니다' if riskier else '보수적입니다'}")
with c2:
    by_class = tbl.groupby("class", sort=False)["weight"].sum()
    by_class = by_class[by_class > 0.0005]
    fig = go.Figure(go.Pie(
        labels=by_class.index, values=by_class.values, hole=0.62, sort=False, direction="clockwise",
        marker=dict(colors=[CLASS_COLOR[c] for c in by_class.index], line=dict(color="#fcfcfb", width=3)),
        texttemplate="%{label}<br><b>%{percent:.0%}</b>", textposition="outside",
        hovertemplate="%{label} %{percent:.1%}<extra></extra>"))
    fig.add_annotation(text=f"{CLASS_LABEL}별<br><b>비중</b>", showarrow=False, font=dict(size=14, color=INK2))
    fig_layout(fig, 360, showlegend=False)
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c3:
    st.markdown(f"**{UNIT}별 투자 금액**")
    fig = go.Figure(go.Bar(
        x=held["weight"], y=[f"{n}  " for n in held.index], orientation="h",
        marker=dict(color=[CLASS_COLOR[c] for c in held["class"]], cornerradius=4),
        text=[f"{pct(w, 0)} · {money(w * amount)}" for w in held["weight"]], textposition="outside",
        textfont=dict(color=INK, size=13), cliponaxis=False, customdata=held[["desc"]],
        hovertemplate="%{y}<br>%{customdata[0]}<br>비중 %{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=13, color=INK))
    fig.update_xaxes(visible=False, range=[0, held["weight"].max() * 1.6])
    st.plotly_chart(fig_layout(fig, max(160, 50 + 40 * len(held)), bargap=0.35, showlegend=False),
                    width="stretch", config={"displayModeBar": False})
    st.caption(" · ".join(f"{n}: {d}" for n, d in held["desc"].items()))
    if len(tbl) > len(held):
        st.caption(f"나머지 {len(tbl) - len(held)}개 {UNIT}은 0% (담지 않음)")

worst_year = ann_my.idxmin()
k = st.columns(4)
k[0].metric("1년 기대 수익률", pct(P["Return"]),
            help="과거 평균으로 계산한 1년 기대 수익률. 실제 수익은 해마다 크게 다를 수 있습니다.")
k[1].metric("보통 1년 동안의 흔들림", f"±{pct(P['Volatility'], 0)}",
            help="변동성(표준편차). 1년 수익률이 대략 '기대 수익률 ± 이 값' 범위에서 움직이는 해가 많다는 뜻")
k[2].metric(f"과거 가장 나빴던 해 ({worst_year}년)", pct(ann_my.min()),
            help="같은 위험 수준으로 매년 말 비중을 다시 맞추며 투자했을 때(백테스트) 가장 나빴던 1년 수익률")
k[3].metric("과거 최대 하락", pct(my_dd),
            help="고점에서 가장 많이 떨어졌던 폭(MDD). 이 정도 하락을 견딜 수 있는지 생각해 보세요.")
summary_md = (f"{money_obj(amount)} 투자하면 1년 뒤 평균 **{money(amount * (1 + P['Return']))}** 정도를 기대할 수 있습니다. "
              f"다만 {worst_year}년처럼 나쁜 해에는 **{pct(ann_my.min(), 0)}**({money(amount * ann_my.min())}) 손실이 날 수 있고, "
              f"가장 크게 떨어졌을 때는 고점 대비 **{pct(my_dd, 0)}까지** 내려간 적이 있습니다.")
st.info(f"**한 줄 요약** · {summary_md}")

# ================================================================ 2. 과거 성과
st.header("과거에 이렇게 투자했다면", anchor="history", divider="gray")
years = st.segmented_control("언제 투자를 시작했다면?", [10, 5, 3, 1], default=10, key="period",
                             format_func=lambda y: f"{y}년 전") or 10
avail = (nav_all.index[-1] - nav_all.index[0]).days / 365.25
w_start = max(nav_all.index[-1] - pd.DateOffset(years=years), nav_all.index[0])
win = nav_all.loc[w_start:]          # 아래 '다른 방법과 비교'도 같은 기간을 씀
if years > avail + 0.05:
    st.caption(f"이 투자 대상은 과거 성과 데이터가 {avail:.1f}년치뿐이라 처음({nav_all.index[0]:%Y.%m})부터 보여줍니다.")
eq = pd.DataFrame({"내 포트폴리오": win["내 포트폴리오"], "EW": win["EW"]})
eq = eq / eq.iloc[0] * amount
final, final_ew = eq["내 포트폴리오"].iloc[-1], eq["EW"].iloc[-1]
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
cagr_txt = f"연평균 {pct((final / amount) ** (1 / yrs) - 1)}" if yrs >= 1.5 else f"수익률 {pct(final / amount - 1)}"
history_md = (f"**{eq.index[0]:%Y년 %m월}**에 {money_obj(amount)} 넣고 매년 말 비중을 다시 맞췄다면, "
              f"**{eq.index[-1]:%Y년 %m월}**에는 **{money(final)}{'이' if cur == 'KRW' else '가'}** 됐습니다 "
              f"({yrs:.1f}년, {cagr_txt}). 모두 같은 금액씩 나눠 담았다면 {money(final_ew)}입니다.")
st.markdown(history_md)
ann_w = bt.annual_returns(win[["내 포트폴리오"]])["내 포트폴리오"]
if (win.index.year == ann_w.index[0]).sum() < 5:
    ann_w = ann_w.iloc[1:]
h1, h2 = st.columns([3, 2], gap="large")
with h1:
    fig = go.Figure()
    for c, col, wdt, dash in [("EW", MUTED, 2, "dot"), ("내 포트폴리오", UP, 3, "solid")]:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[c], name=FRIENDLY[c], mode="lines",
                                 line=dict(color=col, width=wdt, dash=dash),
                                 hovertemplate=FRIENDLY[c] + " %{y:,.0f}<extra></extra>"))
        fig.add_annotation(x=eq.index[-1], y=eq[c].iloc[-1], text=money(eq[c].iloc[-1]), showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=col if c != "EW" else INK2, size=13))
    fig.add_hline(y=amount, line=dict(color=AXIS, width=1, dash="dash"))
    fig.update_yaxes(title="평가 금액", tickformat=",.0f")
    fig_layout(fig, 340, hovermode="x unified")
    fig.update_layout(margin=dict(l=10, r=90, t=30, b=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with h2:
    st.markdown("**해마다 얼마나 벌고 잃었나**")
    fig = go.Figure(go.Bar(x=[str(y) for y in ann_w.index], y=ann_w,
                           marker=dict(color=[UP if v >= 0 else DOWN for v in ann_w], cornerradius=3),
                           text=[pct(v, 0) for v in ann_w], textposition="outside", cliponaxis=False,
                           textfont=dict(size=11, color=INK2), hovertemplate="%{x}년 %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", zeroline=True, zerolinecolor=AXIS)
    st.plotly_chart(fig_layout(fig, 300, showlegend=False), width="stretch", config={"displayModeBar": False})
    st.caption(f"{len(ann_w)}개 연도 중 수익 {int((ann_w >= 0).sum())}번 · 손실 {int((ann_w < 0).sum())}번 "
               f"(첫해는 {win.index[0]:%m월}부터, {ann_w.index[-1]}년은 {win.index[-1]:%m월 %d일}까지)")

# ================================================================ 3. 위험 조절 원리 (효율적 프론티어)
st.header("위험을 올리면 무엇이 바뀌나", anchor="frontier", divider="gray")
st.markdown("곡선 위의 점 하나하나는 **같은 위험에서 수익이 가장 높은 조합**(효율적 프론티어)입니다. "
            "오른쪽으로 갈수록 기대 수익과 흔들림이 함께 커집니다. 파란 점이 지금 추천, 배경색이 성향 구간입니다.")
g1, g2 = st.columns([3, 2], gap="large")
with g1:
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
                             name=f"{UNIT} 하나만 샀을 때",
                             hovertemplate="%{text}<br>수익 %{y:.1%} · 흔들림 %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ft["Volatility"], y=ft["Return"], mode="lines", line=dict(color=INK, width=2),
                             name="효율적 프론티어 (나눠 담았을 때)",
                             hovertemplate="수익 %{y:.1%} · 흔들림 %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[P["Volatility"]], y=[P["Return"]], mode="markers",
                             marker=dict(size=18, color=UP, line=dict(width=3, color="#fcfcfb")),
                             name="지금 추천", hovertemplate="지금 추천<br>수익 %{y:.1%} · 흔들림 %{x:.1%}<extra></extra>"))
    fig.update_xaxes(title="흔들림 (변동성, 1년)", tickformat=".0%")
    fig.update_yaxes(title="기대 수익률 (1년)", tickformat=".0%")
    st.plotly_chart(fig_layout(fig, 420), width="stretch", config={"displayModeBar": False})
    st.caption(f"회색 점({UNIT} 하나만 샀을 때)보다 곡선이 왼쪽 위에 있다는 것 = 나눠 담으면 같은 수익을 더 적은 흔들림으로 얻는다는 뜻")
with g2:
    by_asset = st.segmented_control("위험 수준에 따른 비중", [CLASS_LABEL, UNIT], default=CLASS_LABEL,
                                    format_func=lambda x: f"{x}별", key="weight_view") == UNIT
    fig = go.Figure()
    if by_asset:
        # 종목별: 같은 자산군 종목끼리 같은 계열 색, 진하기만 다르게 (색이 8개를 넘지 않게)
        for c in CLASS_COLOR:
            members = [n for n in names if assets.loc[n, "class"] == c and ft[n].max() > 0.001]
            for k_, n in enumerate(members):
                col = shade(CLASS_COLOR[c], 0.6 * k_ / max(len(members), 2))
                fig.add_trace(go.Scatter(x=ft["Return"], y=ft[n], stackgroup="one", name=n, mode="lines",
                                         line=dict(width=0.5, color="#fcfcfb"), fillcolor=col,
                                         hovertemplate=n + " %{y:.0%}<extra></extra>"))
    else:
        cls_w = ft[names].T.groupby(assets["class"]).sum().T
        for c in CLASS_COLOR:
            if c in cls_w:
                fig.add_trace(go.Scatter(x=ft["Return"], y=cls_w[c], stackgroup="one", name=c, mode="lines",
                                         line=dict(width=0.5, color=CLASS_COLOR[c]), fillcolor=CLASS_COLOR[c],
                                         hovertemplate=c + " %{y:.0%}<extra></extra>"))
    fig.add_vline(x=P["Return"], line=dict(color=INK, width=2, dash="dot"),
                  annotation_text="지금 추천", annotation_position="bottom right",
                  annotation_font=dict(color="#fcfcfb", size=12))
    fig.update_xaxes(title="기대 수익률 (1년)", tickformat=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_layout(fig, 420, hovermode="x unified"), width="stretch", config={"displayModeBar": False})

# ================================================================ 4. 다른 방법과 비교
st.header("다른 투자 방법과 비교하면", anchor="compare", divider="gray")
st.markdown(f"위에서 고른 기간({win.index[0]:%Y.%m}~{win.index[-1]:%Y.%m}) 동안 {money_obj(amount)} 각 방법으로 투자했을 때입니다. "
            "**최종 금액은 클수록, 최대 하락은 작을수록** 좋습니다.")
order = ["내 포트폴리오", *MODELS]
fin = pd.Series({m: win[m].iloc[-1] / win[m].iloc[0] * amount for m in order})
mdd = pd.Series({m: float((win[m] / win[m].cummax() - 1).min()) for m in order})
labels = [FRIENDLY[m] for m in order]
colors = [UP if m == "내 포트폴리오" else "#b9b8b2" for m in order]
b1, b2 = st.columns(2, gap="large")
with b1:
    st.markdown("**지금 얼마가 됐을까** (최종 금액)")
    fig = go.Figure(go.Bar(x=fin, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[money(v) for v in fin], textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK), hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK))
    fig.update_xaxes(visible=False, range=[0, fin.max() * 1.3])
    st.plotly_chart(fig_layout(fig, 300, showlegend=False), width="stretch", config={"displayModeBar": False})
with b2:
    st.markdown("**가장 크게 떨어졌을 때** (최대 하락)")
    fig = go.Figure(go.Bar(x=-mdd, y=labels, orientation="h",
                           marker=dict(color=[DOWN if m == "내 포트폴리오" else "#b9b8b2" for m in order], cornerradius=4),
                           text=[pct(v, 0) for v in mdd], textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK), hovertemplate="%{y}<br>-%{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK))
    fig.update_xaxes(visible=False, range=[0, (-mdd).max() * 1.3])
    st.plotly_chart(fig_layout(fig, 300, showlegend=False), width="stretch", config={"displayModeBar": False})
rk_f = int(fin.rank(ascending=False)["최대샤프"])
rk_m = int((-mdd).rank()["최대샤프"])
st.caption(f"📌 사이드바 추정 기간 = 직전 {config.LOOKBACKS[lb]}: 효율 최고(최대샤프)는 {len(order)}개 방법 중 "
           f"최종 금액 {rk_f}위, 최대 하락 {rk_m}위 (하락이 작을수록 1위). "
           "추정 기간을 1년 ↔ 3년으로 바꿔 보면 최대샤프가 기대수익률 추정에 얼마나 민감한지 볼 수 있습니다.")
with st.expander("각 방법은 무엇인가요?"):
    st.markdown("\n".join(f"- **{FRIENDLY[m]}**: {bt.MODEL_DESC[m]}" for m in MODELS))
    st.markdown(
        "- **효율 최고(최대샤프)를 추천에 쓰지 않는 이유**: 이론상 위험 대비 가장 효율적인 지점이지만, 기대 수익률의 작은 "
        "오차에도 비중이 한두 자산으로 쏠립니다(직전 1년 데이터로 매년 다시 계산하면 IEF 100%(2019), DBC 100%(2023) 같은 해가 생김). "
        "그래서 이 서비스는 설문 점수에 맞는 위험 수준을 추천하고, 최대샤프는 비교용으로만 보여줍니다.")

# ================================================================ 5. 자세히 보기 (전문 지표는 접어 둠)
st.header("자세히 보기", anchor="details", divider="gray")
st.caption("수업에서 다룬 지표와 계산 근거입니다. 필요할 때 펼쳐 보세요.")

with st.expander("성과 지표 표 (CAGR · 변동성 · 샤프 · 소르티노 · MDD) · 연도별 수익률 · 하락장 구간"):
    navs = nav_all[order] / nav_all[order].iloc[0] * 100
    summ = bt.summary(navs)
    summ.index = [FRIENDLY[m] for m in summ.index]
    st.dataframe(summ.style.format({"누적수익률": "{:.1%}", "CAGR": "{:.2%}", "변동성": "{:.1%}",
                                    "샤프": "{:.2f}", "소르티노": "{:.2f}", "MDD": "{:.1%}"})
                 .highlight_max(subset=["CAGR", "샤프", "소르티노", "MDD"], color="#cde2fb"), width="stretch")
    fig = go.Figure()
    for c in order:
        fig.add_trace(go.Scatter(x=navs.index, y=navs[c], name=FRIENDLY[c], mode="lines",
                                 line=dict(color=MODEL_COLOR[c], width=3 if c == "내 포트폴리오" else 1.5,
                                           dash="dot" if c == "EW" else "solid"),
                                 hovertemplate=FRIENDLY[c] + " %{y:.0f}<extra></extra>"))
    fig.update_yaxes(title="누적성과 (시작 = 100)")
    st.plotly_chart(fig_layout(fig, 340, hovermode="x unified"), width="stretch")
    ann = bt.annual_returns(navs)
    if (navs.index.year == ann.index[0]).sum() < 5:
        ann = ann.iloc[1:]
    ann.index.name = "연도"
    ann.columns = [FRIENDLY[m] for m in ann.columns]
    lim = float(np.nanmax(np.abs(ann.to_numpy())))
    st.markdown("**연도별 수익률**")
    st.dataframe(ann.style.format("{:.1%}").background_gradient(cmap="RdBu", vmin=-lim, vmax=lim),
                 width="stretch", height=38 + 35 * len(ann))
    rows = {}
    for name, s, e in STRESS:
        seg = navs.loc[s:e]
        if len(seg) > 1:
            rows[name] = seg.iloc[-1] / seg.iloc[0] - 1
    if rows:
        st.markdown("**하락장 구간 성과**")
        st.dataframe(pd.DataFrame(rows).rename(index=FRIENDLY).style.format("{:.1%}"), width="stretch")
        st.caption(" · ".join(f"{n}: {a[2:].replace('-', '.')} – {b[2:].replace('-', '.')}" for n, a, b in STRESS))

with st.expander("방법별 투자 비중과 위험기여도"):
    mu_, cov_ = D["mu"].to_numpy(), D["cov"].to_numpy()
    W = pd.DataFrame({
        "내 포트폴리오": w_my.to_numpy(), "MVP": min_variance(cov_), "최대샤프": max_sharpe(mu_, cov_),
        "RP": risk_parity(cov_), "MD": max_diversification(cov_), "EW": np.repeat(1 / len(names), len(names)),
    }, index=names)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**투자 비중** (전체기간 데이터 기준)")
        st.dataframe(W.style.format("{:.1%}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                     width="stretch", height=38 + 35 * len(names))
    with b:
        st.markdown("**위험기여도** (전체 흔들림 중 각 자산이 만든 몫)", help=GLOSSARY["위험기여도"])
        RC = W.apply(lambda c: pd.Series(risk_contribution(c.to_numpy(), cov_), index=names))
        st.dataframe(RC.style.format("{:.1%}").background_gradient(cmap="Oranges", vmin=0, vmax=1),
                     width="stretch", height=38 + 35 * len(names))

with st.expander("내 포트폴리오의 연도별 비중 변화"):
    wl = D["weights"][lb].loc[pos_key(bt_pos, cap)]
    wl = wl[wl.index <= nav_all.index[-1]]
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
    st.caption(f"매년 말 직전 {config.LOOKBACKS[lb]} 데이터로 같은 위험 수준의 비중을 다시 계산해 그다음 해에 보유합니다.")

with st.expander(f"투자 대상 {UNIT} 정보와 상관계수"):
    s = D["stats"]
    info = assets.assign(기대수익률=s["Return"], 변동성=s["Volatility"], 샤프=s["Return"] / s["Volatility"])
    r = D["returns"]
    st.markdown(f"**{config.MARKETS[market]['label']}** ({r.index[0]:%Y.%m.%d} ~ {r.index[-1]:%Y.%m.%d} 전체기간, 연율화)")
    st.dataframe(info.reset_index().rename(columns={"name": UNIT, "ticker": "티커", "class": CLASS_LABEL, "desc": "설명"})
                 .style.format({"기대수익률": "{:.1%}", "변동성": "{:.1%}", "샤프": "{:.2f}"}),
                 hide_index=True, width="stretch", height=38 + 35 * len(info))
    st.markdown("**상관계수** (두 자산이 같이 움직이는 정도. 낮거나 음수일수록 섞었을 때 흔들림이 줄어든다)")
    corr = D["corr"]
    n_c = len(corr)
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, zmid=0,
        colorscale=[[0, "#c53030"], [0.5, "#f0efec"], [1, "#1c5cab"]], xgap=2, ygap=2,
        texttemplate="%{z:.2f}", textfont=dict(size=12 if n_c <= 10 else 9),
        hovertemplate="%{y} · %{x}<br>상관계수 %{z:.2f}<extra></extra>", colorbar=dict(thickness=10, len=0.8)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False, tickangle=-45 if n_c > 10 else 0)
    st.plotly_chart(fig_layout(fig, 160 + 34 * n_c), width="stretch", config={"displayModeBar": False})

with st.expander("용어 설명"):
    st.dataframe(pd.DataFrame(GLOSSARY.items(), columns=["용어", "뜻"]), hide_index=True,
                 width="stretch", height=38 + 35 * len(GLOSSARY))

with st.expander("계산 방법과 가정"):
    st.markdown(f"""
- **데이터**: Yahoo Finance(yfinance) 수정종가(배당·분할 반영), 모든 종목 가격이 있는 공통 거래일만 사용
- **프론티어**: 일별수익률 평균·공분산 × {config.ANNUAL}으로 연율화, 비중 0 ≤ w ≤ 1(상한 옵션 시 0.4), 합 1, 공매도·레버리지 없음.
  MVP부터 최고 수익률까지 {config.N_FRONTIER}개 지점을 미리 계산해 두고 슬라이더는 가장 가까운 지점을 보여줌
- **성향 → 위치**: 설문 점수(7~35점)를 0~1로 바꿔 변동성 기준 프론티어 위치로 사용. 10.5점, 24.5점이 성향 경계
- **백테스트**: 매년 말 리밸런싱, 직전 {config.LOOKBACK}거래일(약 3년) 데이터로 비중 계산, 거래비용 {config.TRADE_COST * 1e4:.0f}bp,
  리밸런싱 날 종가까지의 정보만 사용(미래 정보 없음). 모형은 수업 quant_model_modules, 기준가는 cal_nav로 계산
- **샤프비율**: 무위험수익률 0 가정
- **한국 상장 ETF**: 대부분 환헤지(H)·선물형 상품. 10종 공통 데이터가 2014.05부터라 백테스트는 2017년 말부터
- **대형주 20종목**: 한국 10 + 미국 10. 미국 주식은 원/달러 환율을 곱해 원화 기준으로 계산(환율 변동 포함).
  채권·금 같은 위험을 낮춰 줄 자산이 없어 MVP도 변동성이 높다. NAVER 분할(2013) 때 가격 데이터가 튀어서 2014년부터 사용,
  백테스트는 2016년 말부터. 종목 선정 자체가 현재 대형주라 과거 성과가 좋게 보이는 생존편향이 있다
""")

# ================================================================ PDF 리포트 (사이드바 버튼)
from core.report import ReportData, build_pdf  # noqa: E402

win_summ = bt.summary(win[order] / win[order].iloc[0] * 100)
report = ReportData(
    market_label=config.MARKETS[market]["label"], money=money, amount=amount,
    score=score, ptype=ptype, type_desc=survey.TYPES[ptype]["desc"], at_type=at_type,
    answers=[(f"Q{i + 1}. {q}", f"{a + 1}번. {opts[a]}", survey.option_score(a))  # PDF 글꼴에 ①② 글자가 없음
             for i, ((q, opts), a) in enumerate(zip(survey.QUESTIONS, ss.answers))],
    data_period=(D["returns"].index[0], D["returns"].index[-1]),
    holdings=held[["ticker", "class", "desc", "weight"]], class_weights=by_class, class_colors=CLASS_COLOR,
    class_label=CLASS_LABEL, unit=UNIT,
    exp_return=float(P["Return"]), volatility=float(P["Volatility"]),
    worst_year=int(worst_year), worst_year_ret=float(ann_my.min()), max_drawdown=my_dd,
    summary=summary_md.replace("**", ""), equity=eq, annual=ann_w, history_text=history_md.replace("**", ""),
    compare=pd.DataFrame({"최종 금액": fin, "CAGR": win_summ["CAGR"], "샤프": win_summ["샤프"], "MDD": mdd})
            .rename(index=FRIENDLY),
    assumptions=[
        "가격: Yahoo Finance 수정종가(배당·분할 반영). 모든 종목 가격이 있는 공통 거래일만 사용",
        f"추천 비중: 전체기간 일별 수익률 평균·공분산 × {config.ANNUAL}으로 연율화한 효율적 프론티어 위의 지점. "
        "비중 0~100%" + (", 자산별 상한 40%" if cap < 1 else "") + ", 공매도·레버리지 없음",
        "성향 → 위치: 설문 점수(7~35점)에 비례해 프론티어 위 위험 수준(변동성)을 정함",
        f"과거 성과: 매년 말 직전 {config.LOOKBACKS[lb]} 데이터로 비중을 다시 계산해 그다음 해에 보유, "
        f"거래비용 {config.TRADE_COST * 1e4:.0f}bp, 그날 종가까지의 정보만 사용",
        "샤프비율은 무위험수익률 0 가정. 세금·환전 수수료는 반영하지 않음",
        "과거 성과는 미래 수익을 보장하지 않습니다. 이 리포트는 수업 프로젝트용 프로토타입이며 투자 권유가 아닙니다.",
    ],
)
pdf_slot.download_button(
    "📄 PDF 리포트 받기", data=lambda: build_pdf(report), mime="application/pdf", width="stretch",
    file_name=f"MPT_리포트_{ptype}_{pd.Timestamp.today():%Y%m%d}.pdf",
    help="지금 화면의 투자 대상·위험 수준·시작 시점·투자금 그대로 리포트를 만듭니다 (누를 때 생성, 몇 초 걸림)")

st.divider()
st.caption("⚠️ 이 화면은 수업 프로젝트용 프로토타입입니다. 과거 성과는 미래 수익을 보장하지 않으며, 투자 권유가 아닙니다.")
