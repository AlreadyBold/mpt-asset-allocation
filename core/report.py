"""
PDF 리포트 생성. 화면(app.py)에서 계산한 값을 그대로 받아 A4 PDF(bytes)로 만든다.

- 한글은 저장소에 넣어 둔 나눔고딕(assets/fonts)을 씀 -> 어느 컴퓨터에서도 깨지지 않음
- 차트는 matplotlib으로 그려 PNG로 넣음 (plotly 이미지 저장은 크롬이 따로 필요해서)
- Streamlit 다운로드 버튼이 '누를 때만' 이 함수를 부르므로 화면 속도에는 영향 없음
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402
from fpdf.fonts import FontFace  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_REG, FONT_BOLD = FONT_DIR / "NanumGothic-Regular.ttf", FONT_DIR / "NanumGothic-Bold.ttf"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
UP, DOWN, NEUTRAL = "#2a78d6", "#e34948", "#b9b8b2"

font_manager.fontManager.addfont(str(FONT_REG))
font_manager.fontManager.addfont(str(FONT_BOLD))
_FONT_NAME = font_manager.FontProperties(fname=str(FONT_REG)).get_name()


@dataclass
class ReportData:
    """리포트에 들어갈 값. 모두 화면에서 이미 계산한 것."""
    market_label: str
    money: Callable[[float], str]           # 금액 표시 함수 (원/달러)
    amount: float
    score: int
    ptype: str
    type_desc: str
    at_type: str                            # 지금 고른 위험 수준이 속한 성향
    answers: list[tuple[str, str, int]]     # (문항, 고른 보기, 점수)
    data_period: tuple[pd.Timestamp, pd.Timestamp]
    holdings: pd.DataFrame                  # index=종목명, columns: ticker, class, desc, weight
    class_weights: pd.Series                # 자산군(또는 시장)별 비중
    class_colors: dict[str, str]
    class_label: str                        # "자산군" 또는 "시장"
    unit: str                               # "ETF" 또는 "종목"
    exp_return: float
    volatility: float
    worst_year: int
    worst_year_ret: float
    max_drawdown: float
    summary: str                            # 한 줄 요약 (마크다운 없는 평문)
    equity: pd.DataFrame                    # 과거 평가 금액, columns: 내 포트폴리오, EW
    annual: pd.Series                       # 연도별 수익률
    history_text: str
    compare: pd.DataFrame                   # index=방법 이름, columns: 최종 금액, CAGR, 샤프, MDD
    assumptions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 차트
def _style():
    plt.rcParams.update({
        "font.family": _FONT_NAME, "axes.unicode_minus": False, "font.size": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_allocation(d: ReportData) -> io.BytesIO:
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
    cw = d.class_weights[d.class_weights > 0.0005]
    a1.pie(cw.values, labels=[f"{c}\n{w:.0%}" for c, w in cw.items()],
           colors=[d.class_colors[c] for c in cw.index], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2), textprops=dict(color=INK, fontsize=9))
    a1.set_title(f"{d.class_label}별 비중", color=INK, fontsize=10)
    h = d.holdings.iloc[::-1]
    bars = a2.barh(h.index, h["weight"], color=[d.class_colors[c] for c in h["class"]], height=0.6)
    for b, w in zip(bars, h["weight"]):
        a2.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{w:.0%} · {d.money(w * d.amount)}", va="center", fontsize=8.5, color=INK)
    a2.set_xlim(0, h["weight"].max() * 1.7)
    a2.xaxis.set_visible(False)
    a2.grid(False)
    a2.spines["bottom"].set_visible(False)
    a2.set_title(f"{d.unit}별 투자 금액", color=INK, fontsize=10)
    fig.tight_layout()
    return _png(fig)


def _chart_history(d: ReportData) -> io.BytesIO:
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    eq = d.equity
    a1.plot(eq.index, eq["EW"], color=MUTED, lw=1.2, ls=":", label="모두 같은 금액씩 (EW)")
    span = (eq.index[-1] - eq.index[0]).days / 365.25   # 기간에 맞춰 연도 눈금 간격 조절 (겹침 방지)
    a1.xaxis.set_major_locator(matplotlib.dates.YearLocator(2 if span > 6 else 1))
    a1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    a1.plot(eq.index, eq["내 포트폴리오"], color=UP, lw=1.8, label="내 추천 포트폴리오")
    a1.axhline(d.amount, color=GRID, lw=0.8, ls="--")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: d.money(v)))
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a1.set_title("평가 금액", color=INK, fontsize=10)
    ann = d.annual
    a2.bar([str(y) for y in ann.index], ann.values, color=[UP if v >= 0 else DOWN for v in ann.values], width=0.7)
    for i, v in enumerate(ann.values):
        a2.text(i, v + (0.01 if v >= 0 else -0.01), f"{v:.0%}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.5, color=INK2)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.axhline(0, color=INK2, lw=0.6)
    a2.tick_params(axis="x", labelrotation=90, labelsize=7)
    a2.set_title("해마다 수익/손실", color=INK, fontsize=10)
    fig.tight_layout()
    return _png(fig)


def _chart_compare(d: ReportData) -> io.BytesIO:
    _style()
    c = d.compare.iloc[::-1]
    mine = [n.startswith("내 ") for n in c.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.4))
    b1 = a1.barh(c.index, c["최종 금액"], color=[UP if m else NEUTRAL for m in mine], height=0.6)
    for b, v in zip(b1, c["최종 금액"]):
        a1.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + d.money(v), va="center", fontsize=8)
    a1.set_xlim(0, c["최종 금액"].max() * 1.35)
    a1.set_title("최종 금액 (클수록 좋음)", color=INK, fontsize=10)
    b2 = a2.barh(c.index, -c["MDD"], color=[DOWN if m else NEUTRAL for m in mine], height=0.6)
    for b, v in zip(b2, c["MDD"]):
        a2.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.0%}", va="center", fontsize=8)
    a2.set_xlim(0, (-c["MDD"]).max() * 1.35)
    a2.set_yticklabels([])
    a2.set_title("최대 하락 (작을수록 좋음)", color=INK, fontsize=10)
    for a in (a1, a2):
        a.xaxis.set_visible(False)
        a.grid(False)
        a.spines["bottom"].set_visible(False)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------- PDF
class _PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Nanum", size=8)
            self.set_text_color(137, 135, 129)
            self.cell(0, 6, "MPT 포트폴리오 추천 리포트", align="R")
            self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Nanum", size=8)
        self.set_text_color(137, 135, 129)
        self.cell(0, 6, f"{self.page_no()} / {{nb}}  ·  과거 성과는 미래 수익을 보장하지 않으며 투자 권유가 아닙니다.",
                  align="C")


def _h(pdf: _PDF, text: str):
    pdf.ln(3)
    pdf.set_font("Nanum", "B", 13)
    pdf.set_text_color(11, 11, 11)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(195, 194, 183)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def _need(pdf: _PDF, height_mm: float):
    """남은 공간이 모자라면 새 페이지 - 제목과 차트가 다른 페이지로 갈라지지 않게."""
    if pdf.get_y() + height_mm > pdf.h - pdf.b_margin:
        pdf.add_page()


def _p(pdf: _PDF, text: str, size=10, color=(82, 81, 78)):
    pdf.set_font("Nanum", size=size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")


def _table(pdf: _PDF, rows: list[list[str]], widths: list[int], align: list[str]):
    pdf.set_font("Nanum", size=9)
    pdf.set_text_color(11, 11, 11)
    pdf.set_draw_color(225, 224, 217)
    head = FontFace(emphasis="BOLD", fill_color=(243, 242, 238))
    with pdf.table(col_widths=widths, text_align=align, headings_style=head, line_height=6.2,
                   borders_layout="HORIZONTAL_LINES") as t:
        for r in rows:
            t.row(r)


def build_pdf(d: ReportData) -> bytes:
    pdf = _PDF(format="A4")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("Nanum", "", str(FONT_REG))
    pdf.add_font("Nanum", "B", str(FONT_BOLD))
    pdf.alias_nb_pages()
    pdf.add_page()
    w = pdf.w - pdf.l_margin - pdf.r_margin

    # ---- 표지 줄
    pdf.set_font("Nanum", "B", 20)
    pdf.set_text_color(11, 11, 11)
    pdf.cell(0, 11, f"{d.ptype}에게 추천하는 포트폴리오", new_x="LMARGIN", new_y="NEXT")
    _p(pdf, f"작성일 {datetime.now():%Y.%m.%d} · 투자 대상 {d.market_label} · 투자금 {d.money(d.amount)} · "
            f"가격 데이터 {d.data_period[0]:%Y.%m.%d} ~ {d.data_period[1]:%Y.%m.%d}", size=9,
       color=(137, 135, 129))

    # ---- 1. 투자성향
    _h(pdf, "1. 투자성향 진단")
    _p(pdf, f"설문 점수 {d.score}점 (7~35점) → {d.ptype}: {d.type_desc}", size=10, color=(11, 11, 11))
    if d.at_type != d.ptype:
        _p(pdf, f"※ 위험 수준을 직접 조정해 {d.at_type} 구간의 포트폴리오를 골랐습니다.", size=9)
    pdf.ln(1)
    _table(pdf, [["문항", "고른 보기", "점수"]] + [[q, a, str(s)] for q, a, s in d.answers],
           [int(w * 0.5), int(w * 0.4), int(w * 0.1)], ["LEFT", "LEFT", "CENTER"])

    # ---- 2. 추천 비중
    amt = d.money(d.amount)
    _need(pdf, 95)
    _h(pdf, f"2. {amt}{'을' if amt.endswith('원') else '를'} 이렇게 나눠 담으세요")
    pdf.image(_chart_allocation(d), w=w)
    rows = [[d.unit, "설명", d.class_label, "비중", "금액"]]
    rows += [[n, r["desc"], r["class"], f"{r['weight']:.1%}", d.money(r["weight"] * d.amount)]
             for n, r in d.holdings.iterrows()]
    _table(pdf, rows, [int(w * 0.18), int(w * 0.42), int(w * 0.14), int(w * 0.1), int(w * 0.16)],
           ["LEFT", "LEFT", "LEFT", "RIGHT", "RIGHT"])

    # ---- 3. 핵심 지표
    _need(pdf, 45)
    _h(pdf, "3. 핵심 지표")
    _table(pdf, [["1년 기대 수익률", "보통 1년 동안의 흔들림", f"과거 가장 나빴던 해 ({d.worst_year}년)", "과거 최대 하락"],
                 [f"{d.exp_return:.1%}", f"±{d.volatility:.0%}", f"{d.worst_year_ret:.1%}", f"{d.max_drawdown:.1%}"]],
           [int(w / 4)] * 4, ["CENTER"] * 4)
    pdf.ln(2)
    _p(pdf, d.summary, size=10, color=(28, 92, 171))

    # ---- 4. 과거 성과
    _need(pdf, 95)
    _h(pdf, "4. 과거에 이렇게 투자했다면")
    _p(pdf, d.history_text, size=10, color=(11, 11, 11))
    pdf.image(_chart_history(d), w=w)

    # ---- 5. 비교
    _need(pdf, 120)
    _h(pdf, "5. 다른 투자 방법과 비교")
    pdf.image(_chart_compare(d), w=w)
    rows = [["방법", "최종 금액", "연평균(CAGR)", "샤프", "최대 하락"]]
    rows += [[n, d.money(r["최종 금액"]), f"{r['CAGR']:.1%}", f"{r['샤프']:.2f}", f"{r['MDD']:.1%}"]
             for n, r in d.compare.iterrows()]
    _table(pdf, rows, [int(w * 0.36), int(w * 0.18), int(w * 0.16), int(w * 0.12), int(w * 0.18)],
           ["LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])

    # ---- 6. 가정
    _h(pdf, "6. 계산 방법과 유의사항")
    for a in d.assumptions:
        _p(pdf, f"· {a}", size=9)
    return bytes(pdf.output())
