"""두 가지 투자 대상(미국 상장 ETF / 한국 상장 ETF)과 공통 설정."""

START = "2010-01-01"
END = "2026-10-30"          # yfinance end는 미포함 -> 2026-10-29까지
ANNUAL = 250                # 프론티어 연율화 (제안서와 동일: 일별 x 250)
BT_FREQ = 252               # 수업 모듈 백테스트 기본값
LOOKBACK = 756              # 리밸런싱 때 쓰는 직전 데이터 길이 (약 3년). 1년(252)보다
                            # 기대수익률 추정이 안정적이라 백테스트 샤프가 크게 개선됨
WEIGHT_CAPS = (1.0, 0.4)    # 내 포트폴리오 자산별 비중 상한: 제한 없음 / 40%
REBAL = "YE"                # 매년 말 리밸런싱
BT_START_YEAR = 2014        # 백테스트는 이 연말 리밸런싱부터 시작
TRADE_COST = 0.001          # 거래금액 대비 10bp
N_FRONTIER = 100            # 미리 계산해 둘 프론티어 지점 수

MARKETS = {
    "US": {
        "label": "미국 상장 ETF",
        "currency": "USD",
        "assets": [
            # (티커, 표시 이름, 자산군, 설명)
            ("SPY", "SPY", "주식", "미국 주식 (S&P 500)"),
            ("IEV", "IEV", "주식", "유럽 주식"),
            ("EWJ", "EWJ", "주식", "일본 주식"),
            ("EEM", "EEM", "주식", "이머징 주식"),
            ("TLT", "TLT", "채권", "미국 장기채 (20년+)"),
            ("IEF", "IEF", "채권", "미국 중기채 (7~10년)"),
            ("IYR", "IYR", "부동산", "미국 리츠"),
            ("RWX", "RWX", "부동산", "글로벌(미국 외) 리츠"),
            ("GLD", "GLD", "실물", "금"),
            ("DBC", "DBC", "실물", "원자재"),
        ],
    },
    "KR": {
        "label": "한국 상장 ETF",
        "currency": "KRW",
        "assets": [
            ("143850.KS", "S&P500(H)", "주식", "TIGER 미국S&P500선물(H) · 미국 주식"),
            ("195930.KS", "유로스탁스50(H)", "주식", "TIGER 유로스탁스50(합성 H) · 유럽 주식"),
            ("101280.KS", "일본TOPIX100", "주식", "KODEX 일본TOPIX100 · 일본 주식"),
            ("195980.KS", "신흥국(H)", "주식", "PLUS 신흥국MSCI(합성 H) · 이머징 주식"),
            ("069500.KS", "KOSPI200", "주식", "KODEX 200 · 한국 주식"),
            ("148070.KS", "국고채10년", "채권", "KIWOOM 국고채10년 · 한국 장기채"),
            ("114260.KS", "국고채3년", "채권", "KODEX 국고채3년 · 한국 중단기채"),
            ("182480.KS", "미국리츠(H)", "부동산", "TIGER 미국MSCI리츠(합성 H) · 미국 리츠"),
            ("132030.KS", "골드(H)", "실물", "KODEX 골드선물(H) · 금"),
            ("130680.KS", "원유(H)", "실물", "TIGER 원유선물Enhanced(H) · 원유"),
        ],
    },
}


def tickers(market: str) -> list[str]:
    return [a[0] for a in MARKETS[market]["assets"]]


def names(market: str) -> list[str]:
    return [a[1] for a in MARKETS[market]["assets"]]


def asset_table(market: str):
    import pandas as pd
    return pd.DataFrame(MARKETS[market]["assets"], columns=["ticker", "name", "class", "desc"]).set_index("name")
