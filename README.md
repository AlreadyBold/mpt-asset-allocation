# MPT Asset Allocation

투자성향 설문 → 효율적 프론티어 → 글로벌 ETF 맞춤 자산배분 추천 (Streamlit)

- 설문 7문항 점수로 성향(안정추구형 / 위험중립형 / 공격추구형)과 프론티어 위 위치를 정하고, 그 지점의 ETF 비중을 추천
- 투자 대상 세 가지: 미국 상장 ETF 10종 / 한국 상장 ETF 10종 / 한국·미국 대형주 20종목(원화 기준)
- 비교 기준: MVP, 최대샤프(참고), Risk Parity, Most Diversified, Equal Weight
- 성과: 매년 말 리밸런싱 백테스트 (CAGR, 변동성, 샤프, 소르티노, MDD, 연도별 수익률, 하락장 구간)
- 결과물: 지금 화면 설정 그대로 PDF 리포트 내려받기, 버튼 한 번으로 최신 가격 업데이트

## 실행

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

`data/`에 미리 계산한 결과가 들어 있어서 바로 실행된다. 다시 계산하려면:

```bash
.venv/bin/python build_data.py              # 저장된 가격으로 다시 계산
.venv/bin/python build_data.py --download   # 가격부터 다시 받기
```

## 구조

| 경로 | 내용 |
|---|---|
| `app.py` | Streamlit 화면 (설문 → 추천 대시보드) |
| `build_data.py` | 프론티어 100개 지점, 백테스트를 미리 계산해 `data/`에 저장 |
| `core/config.py` | 투자 대상 세 가지와 공통 설정 |
| `core/survey.py` | M1 투자성향 진단 |
| `core/frontier.py` | M2 효율적 프론티어, 성향 위치별 비중 |
| `core/backtest.py` | M3 비교 모형 + M4 백테스트 |
| `core/report.py` | PDF 리포트 생성 |
| `assets/fonts/` | PDF용 한글 글꼴 (나눔고딕, OFL) |
| `course/` | 수업 모듈(quant_model_modules, cal_nav_module) 중 쓰는 함수 |

## 주요 가정

- 데이터: Yahoo Finance 수정종가, 10종 공통 거래일만 사용
- 프론티어: 일별 평균·공분산 × 250, 0 ≤ w ≤ 1 (옵션: 자산별 상한 40%), 무위험수익률 0
- 성향 → 위치: (점수 − 7) ÷ 28을 변동성 기준 프론티어 위치(0 = MVP, 1 = 끝)로 사용
- 백테스트: 매년 말 리밸런싱, 직전 756거래일(약 3년) 데이터, 거래비용 10bp

> 과거 성과는 미래 수익을 보장하지 않습니다. 수업 프로젝트용 프로토타입이며 투자 권유가 아닙니다.

문서: [docs/](docs/README.md) (발표자 가이드, 구조와 모듈, 문제와 해결)
