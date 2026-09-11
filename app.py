"""
지영 & 경아의 보금자리 찾아 삼만리 — 매매 vs 전세 vs 월세 주거비용 종합 비교 및 의사결정 지원 웹앱
Streamlit + Pandas + NumPy + Plotly
"""

import concurrent.futures
import difflib
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="지영 & 경아의 보금자리 찾아 삼만리", page_icon="🏠", layout="wide")

# 기본 테마의 과도하게 큰 폰트 크기를 줄여 화면 밀도와 시인성을 개선
st.markdown(
    """
    <style>
    h1 { font-size: 1.6rem !important; line-height: 1.3 !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.05rem !important; }
    h4 { font-size: 0.95rem !important; }
    p, li, label, .stMarkdown, .stCaption { font-size: 0.9rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
    .stAlert p { font-size: 0.88rem !important; }
    [data-testid="stWidgetLabel"] p { font-size: 0.85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================================
# 접근 제어 — 허가된 사람만 로그인 가능하도록 비밀번호로 게이트
# 비밀번호는 st.secrets(.streamlit/secrets.toml, git에는 올라가지 않음)에만 저장한다.
# ======================================================================================
def get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return None


def check_password() -> bool:
    """비밀번호가 맞으면 True. secrets에 APP_PASSWORD가 설정되지 않았으면 접근을 막는다(fail-closed)."""
    if st.session_state.get("authenticated"):
        return True

    app_password = get_secret("APP_PASSWORD")

    st.title("🐶지영 & 🐯경아만 출입 가능")
    if not app_password:
        st.error(
            "APP_PASSWORD가 설정되지 않았습니다. 관리자는 `.streamlit/secrets.toml`에 "
            "`APP_PASSWORD = \"...\"` 값을 추가한 뒤 앱을 다시 시작하세요."
        )
        return False

    entered = st.text_input("비밀번호를 입력하세요", type="password", key="password_input")
    if st.button("Welcome back 👋"):
        if entered == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    return False


if not check_password():
    st.stop()


# ======================================================================================
# 유틸 함수
# ======================================================================================
def fmt_money(value: float) -> str:
    """만원 단위 숫자를 '억원/만원' 형태의 읽기 쉬운 문자열로 변환"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(value) >= 10000:
        return f"{value / 10000:,.2f}억원"
    return f"{value:,.0f}만원"


def money_hint(value: float) -> None:
    """number_input 아래에 콤마(천 단위 구분) 표기를 작은 캡션으로 보여준다."""
    try:
        st.caption(f"{float(value):,.0f} 만원")
    except (TypeError, ValueError):
        pass


def safe_div(a: float, b: float) -> float:
    """0으로 나누기 방지"""
    return a / b if b not in (0, None) else 0.0


# ======================================================================================
# 법정 비용 계산 로직
# ======================================================================================
def calc_acquisition_tax(price: float, tax_rates: dict):
    """
    법정 취득세(+지방교육세) 계산. price 단위: 만원
    구간/세율은 정부 고시에 따라 자주 바뀌므로 tax_rates(dict)로 외부에서 주입받는다.
    - threshold1 이하: rate_min
    - threshold1 초과 threshold2 이하: rate_min → rate_max 구간 선형 산출
    - threshold2 초과: rate_max
    - 지방교육세: 취득세율 × edu_tax_ratio(%) 가산 (전용 85㎡ 이하 농특세 면제 가정, 1주택자 기준 단순화)
    """
    if price <= 0:
        return 0.0, 0.0
    threshold1 = tax_rates["acq_threshold1"]
    threshold2 = tax_rates["acq_threshold2"]
    rate_min = tax_rates["acq_rate_min"]
    rate_max = tax_rates["acq_rate_max"]
    edu_tax_ratio = tax_rates["edu_tax_ratio"]

    if price <= threshold1:
        base_rate = rate_min
    elif price <= threshold2 and threshold2 > threshold1:
        base_rate = rate_min + (price - threshold1) / (threshold2 - threshold1) * (rate_max - rate_min)
    else:
        base_rate = rate_max
    edu_rate = base_rate * edu_tax_ratio / 100
    total_rate = base_rate + edu_rate
    tax = price * total_rate / 100
    return tax, total_rate


def calc_brokerage_fee(amount: float, tx_type: str):
    """
    중개보수 법정 상한 요율 자동 계산 (서울시 기준 예시 요율표). amount 단위: 만원
    tx_type: 'sale'(매매) 또는 'lease'(전월세)
    """
    if amount <= 0:
        return 0.0, 0.0
    amount_eok = amount / 10000
    if tx_type == "sale":
        table = [
            (0.5, 0.6, 25), (2, 0.5, 80), (9, 0.4, None),
            (12, 0.5, None), (15, 0.6, None), (float("inf"), 0.7, None),
        ]
    else:
        table = [
            (0.5, 0.5, 20), (1, 0.4, 30), (6, 0.3, None),
            (12, 0.4, None), (15, 0.5, None), (float("inf"), 0.6, None),
        ]
    for upper, rate, cap in table:
        if amount_eok < upper:
            fee = amount * rate / 100
            if cap is not None:
                fee = min(fee, cap)
            return fee, rate
    return amount * 0.007, 0.7  # 안전망 (도달하지 않음)


def wolse_conversion_amount(deposit: float, monthly_rent: float) -> float:
    """월세 중개보수 산정을 위한 환산보증금 = 보증금 + (월세 × 100), 5천만원 미만 시 ×70 환산"""
    conv = deposit + monthly_rent * 100
    if conv < 5000:
        conv = deposit + monthly_rent * 70
    return conv


def total_rent_with_escalation(monthly_rent: float, inflation_rate: float, period: float) -> float:
    """임대료 상승률을 연 단위 복리로 반영한 총 월세 지출액 (기간이 소수인 경우 마지막 해 일할 반영)"""
    if monthly_rent <= 0 or period <= 0:
        return 0.0
    total = 0.0
    full_years = int(period)
    remainder = period - full_years
    current = monthly_rent
    for _ in range(full_years):
        total += current * 12
        current *= (1 + inflation_rate / 100)
    if remainder > 0:
        total += current * 12 * remainder
    return total


# ======================================================================================
# 시나리오별 Total Net Cost 계산 엔진
# ======================================================================================
def calc_purchase(g: dict, p: dict):
    """매매 시나리오: g=공통 입력, p=매매 입력"""
    period = max(g["target_period"], 0.0001)
    tax, _ = calc_acquisition_tax(p["sale_price"], g)
    brokerage, _ = calc_brokerage_fee(p["sale_price"], "sale")
    initial_cost = tax + brokerage + g["moving_cost"] + g["cleaning_cost"] + p["renovation_cost"]
    total_needed = p["sale_price"] + initial_cost

    own_capital = g["own_capital"]
    loan_amount = max(0.0, total_needed - own_capital)
    capital_used = min(own_capital, total_needed)
    surplus = max(0.0, own_capital - total_needed)

    loan_interest = loan_amount * p["mortgage_rate"] / 100 * period
    holding_tax = p["sale_price"] * g["holding_tax_rate"] / 100 * period  # 보유세(재산세 등) 근사치, 실효세율은 사용자 입력값
    mgmt_fee = g["monthly_mgmt_fee"] * 12 * period
    opp_cost = capital_used * g["opportunity_rate"] / 100 * period
    surplus_gain = surplus * g["opportunity_rate"] / 100 * period
    capital_gain = p["sale_price"] * ((1 + p["price_growth_rate"] / 100) ** period - 1)

    net_cost = initial_cost + loan_interest + holding_tax + mgmt_fee + opp_cost - surplus_gain - capital_gain

    waterfall = [
        ("초기비용\n(취득세·중개보수·이사비 등)", initial_cost),
        ("대출이자", loan_interest),
        ("보유세(재산세 등)", holding_tax),
        ("관리비", mgmt_fee),
        ("자기자본 기회비용", opp_cost),
        ("여유자금 운용수익", -surplus_gain),
        ("주택 시세차익", -capital_gain),
    ]

    detail = {
        "옵션": "매매",
        "취득세 등": tax, "중개보수": brokerage,
        "이사/청소/수리비": g["moving_cost"] + g["cleaning_cost"] + p["renovation_cost"],
        "필요자금": total_needed, "대출필요액": loan_amount,
        "투입 자기자본": capital_used, "여유자금": surplus,
        "초기비용": initial_cost, "대출이자": loan_interest, "보유세": holding_tax,
        "관리비": mgmt_fee, "기회비용": opp_cost, "여유자금운용수익": surplus_gain,
        "시세차익": capital_gain, "순비용": net_cost,
    }
    return net_cost, waterfall, detail


def calc_jeonse(g: dict, j: dict):
    """전세 시나리오: g=공통 입력, j=전세 입력"""
    period = max(g["target_period"], 0.0001)
    brokerage, _ = calc_brokerage_fee(j["jeonse_deposit"], "lease")
    initial_cost = brokerage + g["moving_cost"] + g["cleaning_cost"]
    total_needed = j["jeonse_deposit"] + initial_cost

    own_capital = g["own_capital"]
    loan_amount = max(0.0, total_needed - own_capital)
    capital_used = min(own_capital, total_needed)
    surplus = max(0.0, own_capital - total_needed)

    loan_interest = loan_amount * j["jeonse_loan_rate"] / 100 * period
    mgmt_fee = g["monthly_mgmt_fee"] * 12 * period
    opp_cost = capital_used * g["opportunity_rate"] / 100 * period
    surplus_gain = surplus * g["opportunity_rate"] / 100 * period

    net_cost = initial_cost + loan_interest + mgmt_fee + opp_cost - surplus_gain

    waterfall = [
        ("초기비용\n(중개보수·이사비 등)", initial_cost),
        ("전세대출 이자", loan_interest),
        ("관리비", mgmt_fee),
        ("자기자본 기회비용", opp_cost),
        ("여유자금 운용수익", -surplus_gain),
    ]

    detail = {
        "옵션": "전세",
        "중개보수": brokerage, "이사/청소비": g["moving_cost"] + g["cleaning_cost"],
        "필요자금": total_needed, "대출필요액": loan_amount,
        "투입 자기자본": capital_used, "여유자금": surplus,
        "초기비용": initial_cost, "대출이자": loan_interest, "보유세": 0.0,
        "관리비": mgmt_fee, "기회비용": opp_cost, "여유자금운용수익": surplus_gain,
        "시세차익": 0.0, "순비용": net_cost,
    }
    return net_cost, waterfall, detail


def calc_wolse(g: dict, w: dict):
    """월세 시나리오: g=공통 입력, w=월세 입력"""
    period = max(g["target_period"], 0.0001)
    conv_amount = wolse_conversion_amount(w["wolse_deposit"], w["wolse_monthly"])
    brokerage, _ = calc_brokerage_fee(conv_amount, "lease")
    initial_cost = brokerage + g["moving_cost"] + g["cleaning_cost"]
    total_needed = w["wolse_deposit"] + initial_cost

    own_capital = g["own_capital"]
    loan_amount = max(0.0, total_needed - own_capital)
    capital_used = min(own_capital, total_needed)
    surplus = max(0.0, own_capital - total_needed)

    loan_interest = loan_amount * w["wolse_loan_rate"] / 100 * period
    mgmt_fee = g["monthly_mgmt_fee"] * 12 * period
    opp_cost = capital_used * g["opportunity_rate"] / 100 * period
    surplus_gain = surplus * g["opportunity_rate"] / 100 * period
    total_rent = total_rent_with_escalation(w["wolse_monthly"], g["inflation_rate"], period)

    net_cost = initial_cost + loan_interest + total_rent + mgmt_fee + opp_cost - surplus_gain

    waterfall = [
        ("초기비용\n(중개보수·이사비 등)", initial_cost),
        ("보증금대출 이자", loan_interest),
        ("월세 총액(상승분 반영)", total_rent),
        ("관리비", mgmt_fee),
        ("자기자본 기회비용", opp_cost),
        ("여유자금 운용수익", -surplus_gain),
    ]

    detail = {
        "옵션": "월세",
        "중개보수": brokerage, "이사/청소비": g["moving_cost"] + g["cleaning_cost"],
        "필요자금": total_needed, "대출필요액": loan_amount,
        "투입 자기자본": capital_used, "여유자금": surplus,
        "초기비용": initial_cost, "대출이자": loan_interest, "보유세": total_rent,
        "관리비": mgmt_fee, "기회비용": opp_cost, "여유자금운용수익": surplus_gain,
        "시세차익": 0.0, "순비용": net_cost,
    }
    return net_cost, waterfall, detail


# ======================================================================================
# 시각화 함수
# ======================================================================================
def make_waterfall(title: str, steps: list, net_cost: float) -> go.Figure:
    labels = [s[0] for s in steps] + ["순비용 합계"]
    values = [s[1] for s in steps]
    measures = ["relative"] * len(steps) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values + [net_cost],
            text=[fmt_money(v) for v in values] + [fmt_money(net_cost)],
            textposition="outside",
            connector={"line": {"color": "rgb(150,150,150)"}},
            increasing={"marker": {"color": "#EF553B"}},
            decreasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}},
        )
    )
    fig.update_layout(
        title=f"{title} — 초기 자금 투입부터 최종 순비용까지",
        showlegend=False, height=420,
        yaxis_title="금액(만원)", margin=dict(t=60, b=10),
    )
    return fig


def compute_sensitivity(g: dict, p: dict, j: dict):
    """주택가격 상승률(X) × 대출금리(Y) 변동에 따른 (매매 순비용 - 전세 순비용) 히트맵 데이터"""
    price_growth_range = np.arange(-3, 5.01, 1.0)
    loan_rate_range = np.arange(2, 7.01, 0.5)
    diff_matrix = np.zeros((len(loan_rate_range), len(price_growth_range)))

    for i, lr in enumerate(loan_rate_range):
        for k, pg in enumerate(price_growth_range):
            p2 = dict(p); p2["mortgage_rate"] = lr; p2["price_growth_rate"] = pg
            j2 = dict(j); j2["jeonse_loan_rate"] = lr
            purchase_net, _, _ = calc_purchase(g, p2)
            jeonse_net, _, _ = calc_jeonse(g, j2)
            diff_matrix[i, k] = purchase_net - jeonse_net

    return price_growth_range, loan_rate_range, diff_matrix


def compute_option_crossovers(g: dict, p: dict, j: dict, w: dict, max_years: float = 40.0, step: float = 0.5):
    """
    거주(보유) 기간을 늘려가며 매매/전세/월세 중 실질 순비용이 가장 낮은 옵션이 몇 년째에 바뀌는지 계산한다.
    전월세 계약은 보통 2~4년 단위라 사용자가 설정한 거주기간은 짧게 잡히기 쉬운데,
    매매는 장기로 갈수록 유리해지는 경우가 많아 "이대로 오래 살면 결과가 달라지는지" 보여주기 위함.
    반환: (timeline: list[(period, best_option)], transitions: list[(period, new_best_option)])
    """
    periods = np.arange(step, max_years + step / 2, step)
    timeline = []
    transitions = []
    prev_best = None
    for period in periods:
        g_p = dict(g, target_period=float(period))
        pn, _, _ = calc_purchase(g_p, p)
        jn, _, _ = calc_jeonse(g_p, j)
        wn, _, _ = calc_wolse(g_p, w)
        costs = {"매매": pn, "전세": jn, "월세": wn}
        best = min(costs, key=costs.get)
        timeline.append((float(period), best))
        if prev_best is not None and best != prev_best:
            transitions.append((float(period), best))
        prev_best = best
    return timeline, transitions


def format_crossover_message(transitions: list, current_best: str, max_years: float) -> str:
    """손익분기 전환 시점을 사람이 읽을 문장으로 변환"""
    if not transitions:
        return f"📐 보유기간을 최대 {max_years:.0f}년까지 늘려봐도 **{current_best}**가 계속 유리한 것으로 예상됩니다 (이 조건에서는 순위 역전이 없습니다)."
    parts = [f"약 **{period:.1f}년차부터 {new_best}**가 더 유리" for period, new_best in transitions]
    return "📐 보유기간이 길어지면 " + " → ".join(parts) + "해질 것으로 예상됩니다. (다른 조건은 현재 입력값 그대로 가정)"


def make_sensitivity_heatmap(price_growth_range, loan_rate_range, diff_matrix) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=diff_matrix / 10000,  # 억원 단위로 표시
            x=[f"{v:+.0f}%" for v in price_growth_range],
            y=[f"{v:.1f}%" for v in loan_rate_range],
            colorscale="RdYlGn_r",
            zmid=0,
            colorbar=dict(title="억원"),
            text=[[f"{v:+.2f}억" for v in row] for row in diff_matrix / 10000],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="주택가격상승률 %{x}<br>대출금리 %{y}<br>매매-전세 차이: %{z:.2f}억원<extra></extra>",
        )
    )
    fig.update_layout(
        title="민감도 분석: 주택가격 상승률 × 대출금리에 따른 (매매 − 전세) 순비용 차이",
        xaxis_title="예상 연간 주택가격 상승률", yaxis_title="대출 금리",
        height=440, margin=dict(t=60, b=10),
    )
    return fig


# ======================================================================================
# 실거래가 조회 — 국토교통부 공공데이터포털 API 연동
# secrets.toml에 DATA_GO_KR_SERVICE_KEY가 설정되어 있으면 실제 API를 호출하고,
# 없으면 목업(Mock) 데이터로 자동 대체되어 앱이 계속 동작한다.
# ======================================================================================
DATA_GO_KR_BASE = "https://apis.data.go.kr/1613000"
TRADE_ENDPOINT = f"{DATA_GO_KR_BASE}/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
RENT_ENDPOINT = f"{DATA_GO_KR_BASE}/RTMSDataSvcAptRent/getRTMSDataSvcAptRent"

# 전국 시/군/구 법정동코드(앞 5자리, 행정표준코드관리시스템 code.go.kr 기준).
# selectbox는 입력 시 자동으로 후보를 필터링해 보여주므로, 목록만 전국으로 채우면
# 사용자는 시/군/구 이름 일부만 타이핑해서 바로 찾을 수 있다.
_REGION_TREE = {
    "서울특별시": {
        "종로구": "11110", "중구": "11140", "용산구": "11170", "성동구": "11200",
        "광진구": "11215", "동대문구": "11230", "중랑구": "11260", "성북구": "11290",
        "강북구": "11305", "도봉구": "11320", "노원구": "11350", "은평구": "11380",
        "서대문구": "11410", "마포구": "11440", "양천구": "11470", "강서구": "11500",
        "구로구": "11530", "금천구": "11545", "영등포구": "11560", "동작구": "11590",
        "관악구": "11620", "서초구": "11650", "강남구": "11680", "송파구": "11710",
        "강동구": "11740",
    },
    "부산광역시": {
        "중구": "26110", "서구": "26140", "동구": "26170", "영도구": "26200",
        "부산진구": "26230", "동래구": "26260", "남구": "26290", "북구": "26320",
        "해운대구": "26350", "사하구": "26380", "금정구": "26410", "강서구": "26440",
        "연제구": "26470", "수영구": "26500", "사상구": "26530", "기장군": "26710",
    },
    "대구광역시": {
        "중구": "27110", "동구": "27140", "서구": "27170", "남구": "27200",
        "북구": "27230", "수성구": "27260", "달서구": "27290", "달성군": "27710",
    },
    "인천광역시": {
        "중구": "28110", "동구": "28140", "미추홀구": "28177", "연수구": "28185",
        "남동구": "28200", "부평구": "28237", "계양구": "28245", "서구": "28260",
        "강화군": "28710", "옹진군": "28720",
    },
    "광주광역시": {
        "동구": "29110", "서구": "29140", "남구": "29155", "북구": "29170", "광산구": "29200",
    },
    "대전광역시": {
        "동구": "30110", "중구": "30140", "서구": "30170", "유성구": "30200", "대덕구": "30230",
    },
    "울산광역시": {
        "중구": "31110", "남구": "31140", "동구": "31170", "북구": "31200", "울주군": "31710",
    },
    "세종특별자치시": {"": "36110"},
    "경기도": {
        "수원시 장안구": "41111", "수원시 권선구": "41113", "수원시 팔달구": "41115", "수원시 영통구": "41117",
        "성남시 수정구": "41131", "성남시 중원구": "41133", "성남시 분당구": "41135",
        "의정부시": "41150",
        "안양시 만안구": "41171", "안양시 동안구": "41173",
        "부천시": "41190", "광명시": "41210", "평택시": "41220", "동두천시": "41250",
        "안산시 상록구": "41271", "안산시 단원구": "41273",
        "고양시 덕양구": "41281", "고양시 일산동구": "41285", "고양시 일산서구": "41287",
        "과천시": "41290", "구리시": "41310", "남양주시": "41360", "오산시": "41370", "시흥시": "41390",
        "군포시": "41410", "의왕시": "41430", "하남시": "41450",
        "용인시 처인구": "41461", "용인시 기흥구": "41463", "용인시 수지구": "41465",
        "파주시": "41480", "이천시": "41500", "안성시": "41550", "김포시": "41570",
        "화성시": "41590", "광주시": "41610", "양주시": "41630", "포천시": "41650", "여주시": "41670",
        "연천군": "41800", "가평군": "41820", "양평군": "41830",
    },
    "강원도": {
        "춘천시": "42110", "원주시": "42130", "강릉시": "42150", "동해시": "42170", "태백시": "42190",
        "속초시": "42210", "삼척시": "42230",
        "홍천군": "42720", "횡성군": "42730", "영월군": "42750", "평창군": "42760", "정선군": "42770",
        "철원군": "42780", "화천군": "42790", "양구군": "42800", "인제군": "42810", "고성군": "42820", "양양군": "42830",
    },
    "충청북도": {
        "청주시 상당구": "43111", "청주시 서원구": "43112", "청주시 흥덕구": "43113", "청주시 청원구": "43114",
        "충주시": "43130", "제천시": "43150",
        "보은군": "43720", "옥천군": "43730", "영동군": "43740", "증평군": "43745", "진천군": "43750",
        "괴산군": "43760", "음성군": "43770", "단양군": "43800",
    },
    "충청남도": {
        "천안시 동남구": "44131", "천안시 서북구": "44133",
        "공주시": "44150", "보령시": "44180", "아산시": "44200", "서산시": "44210", "논산시": "44230",
        "계룡시": "44250", "당진시": "44270",
        "금산군": "44710", "부여군": "44760", "서천군": "44770", "청양군": "44790", "홍성군": "44800",
        "예산군": "44810", "태안군": "44825",
    },
    "전라북도": {
        "전주시 완산구": "45111", "전주시 덕진구": "45113",
        "군산시": "45130", "익산시": "45140", "정읍시": "45180", "남원시": "45190", "김제시": "45210",
        "완주군": "45710", "진안군": "45720", "무주군": "45730", "장수군": "45740", "임실군": "45750",
        "순창군": "45770", "고창군": "45790", "부안군": "45800",
    },
    "전라남도": {
        "목포시": "46110", "여수시": "46130", "순천시": "46150", "나주시": "46170", "광양시": "46230",
        "담양군": "46710", "곡성군": "46720", "구례군": "46730", "고흥군": "46770", "보성군": "46780",
        "화순군": "46790", "장흥군": "46800", "강진군": "46810", "해남군": "46820", "영암군": "46830",
        "무안군": "46840", "함평군": "46860", "영광군": "46870", "장성군": "46880", "완도군": "46890",
        "진도군": "46900", "신안군": "46910",
    },
    "경상북도": {
        "포항시 남구": "47111", "포항시 북구": "47113",
        "경주시": "47130", "김천시": "47150", "안동시": "47170", "구미시": "47190", "영주시": "47210",
        "영천시": "47230", "상주시": "47250", "문경시": "47280", "경산시": "47290",
        "군위군": "47720", "의성군": "47730", "청송군": "47750", "영양군": "47760", "영덕군": "47770",
        "청도군": "47820", "고령군": "47830", "성주군": "47840", "칠곡군": "47850", "예천군": "47900",
        "봉화군": "47920", "울진군": "47930", "울릉군": "47940",
    },
    "경상남도": {
        "창원시 의창구": "48121", "창원시 성산구": "48123", "창원시 마산합포구": "48125",
        "창원시 마산회원구": "48127", "창원시 진해구": "48129",
        "진주시": "48170", "통영시": "48220", "사천시": "48240", "김해시": "48250", "밀양시": "48270",
        "거제시": "48310", "양산시": "48330",
        "의령군": "48720", "함안군": "48730", "창녕군": "48740", "고성군": "48820",
        "남해군": "48840", "하동군": "48850", "산청군": "48860", "함양군": "48870", "거창군": "48880",
        "합천군": "48890",
    },
    "제주특별자치도": {"제주시": "50110", "서귀포시": "50130"},
}


def _flatten_regions(tree: dict) -> dict:
    flat = {}
    for sido, districts in tree.items():
        for district, code in districts.items():
            label = f"{sido} {district}" if district else sido
            flat[label] = code
    return flat


REGION_CODE_MAP = _flatten_regions(_REGION_TREE)
CUSTOM_REGION_LABEL = "직접 입력 (5자리 법정동코드)"
REGION_OPTIONS = sorted(REGION_CODE_MAP.keys()) + [CUSTOM_REGION_LABEL]

# API 연동 전까지 오프라인 대체용 목업 DB (서비스키/지역코드가 없을 때만 사용됨)
MOCK_APT_DB = [
    {"name": "래미안 강남", "base_sale": 135000, "base_jeonse": 95000, "base_wolse_rent": 250},
    {"name": "래미안 서초", "base_sale": 142000, "base_jeonse": 98000, "base_wolse_rent": 260},
    {"name": "e편한세상 목동", "base_sale": 98000, "base_jeonse": 68000, "base_wolse_rent": 180},
    {"name": "자이 판교", "base_sale": 118000, "base_jeonse": 78000, "base_wolse_rent": 200},
    {"name": "푸르지오 마포", "base_sale": 105000, "base_jeonse": 72000, "base_wolse_rent": 190},
    {"name": "힐스테이트 광교", "base_sale": 92000, "base_jeonse": 62000, "base_wolse_rent": 160},
    {"name": "아크로 리버파크", "base_sale": 210000, "base_jeonse": 140000, "base_wolse_rent": 350},
    {"name": "테스트아파트", "base_sale": 60000, "base_jeonse": 42000, "base_wolse_rent": 100},
]


def region_label_to_code(label: str, custom_code: str) -> str:
    """selectbox에서 고른 지역 라벨을 5자리 법정동코드로 변환"""
    if label == CUSTOM_REGION_LABEL:
        return (custom_code or "").strip()
    return REGION_CODE_MAP.get(label, "")


def _parse_amount(value):
    """'110,000' 같은 실거래가 API 금액 문자열을 숫자로 변환"""
    if value is None:
        return None
    value = value.replace(",", "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _recent_year_months(n: int):
    """오늘 기준 최근 n개월의 'YYYYMM' 문자열 목록 (과거 -> 최신 순)"""
    this_month = pd.Timestamp.today().replace(day=1)
    return [(this_month - pd.DateOffset(months=i)).strftime("%Y%m") for i in range(n - 1, -1, -1)]


REAL_ESTATE_DATA_START = (2006, 1)  # 아파트 매매/전월세 실거래가 공개 시작 시점(국토교통부)


def _all_year_months_since(year: int, month: int):
    """지정한 연-월부터 이번 달까지의 'YYYYMM' 문자열 목록 (과거 -> 최신 순)"""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = pd.Timestamp.today().replace(day=1)
    if start > end:
        return [end.strftime("%Y%m")]
    return [d.strftime("%Y%m") for d in pd.date_range(start=start, end=end, freq="MS")]


@st.cache_data(ttl=3600, show_spinner=False)
def _call_data_go_kr(endpoint: str, service_key: str, lawd_cd: str, deal_ymd: str) -> dict:
    """
    국토교통부 실거래가 API 원시 호출 + XML 파싱. 1시간 캐시로 동일 (지역, 월) 재조회 시
    API 호출량을 아낀다. 반환: {"error": str|None, "items": [dict, ...]}
    """
    url = (
        f"{endpoint}?serviceKey={service_key}&LAWD_CD={lawd_cd}&DEAL_YMD={deal_ymd}"
        f"&numOfRows=1000&pageNo=1"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"API 요청 실패: {e}", "items": []}

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        return {"error": f"응답 파싱 실패: {e}", "items": []}

    result_code_el = root.find(".//resultCode")
    result_code = result_code_el.text if result_code_el is not None else None
    if result_code not in ("000", "00"):
        result_msg_el = root.find(".//resultMsg")
        msg = result_msg_el.text if result_msg_el is not None else "알 수 없는 오류"
        return {"error": f"API 오류({result_code}): {msg}", "items": []}

    items = [
        {child.tag: (child.text or "").strip() for child in item_el}
        for item_el in root.findall(".//item")
    ]
    return {"error": None, "items": items}


def search_apartment_matches(query: str, lawd_cd: str = "", n: int = 5, cutoff: float = 0.35):
    """
    입력한 아파트 이름과 실거래가 등록명을 매칭한다.
    서비스키+지역코드가 있으면 최근 12개월 실거래(매매+전월세) 아파트명 목록에서, 없으면 목업 DB에서 매칭.
    (구축 아파트는 거래가 뜸할 수 있어 3개월보다 넉넉하게 잡는다.)
    반환: (matches: list[str], error: str|None)
    """
    query = (query or "").strip()
    if not query:
        return [], None

    service_key = get_secret("DATA_GO_KR_SERVICE_KEY")
    error = None
    if service_key and lawd_cd:
        names_set = set()
        for ym in _recent_year_months(12):
            for endpoint in (TRADE_ENDPOINT, RENT_ENDPOINT):
                result = _call_data_go_kr(endpoint, service_key, lawd_cd, ym)
                if result["error"] and not error:
                    error = result["error"]
                for item in result["items"]:
                    nm = item.get("aptNm", "").strip()
                    if nm:
                        names_set.add(nm)
        names = sorted(names_set)
    else:
        names = [d["name"] for d in MOCK_APT_DB]

    close = difflib.get_close_matches(query, names, n=n, cutoff=cutoff)
    substring_matches = [nm for nm in names if query in nm or nm in query]
    ordered = []
    for nm in close + substring_matches:
        if nm not in ordered:
            ordered.append(nm)
    return ordered[:n], error


def _fetch_mock_trend(matched_name: str, months: int) -> pd.DataFrame:
    apt_record = next((d for d in MOCK_APT_DB if d["name"] == matched_name), None)
    if apt_record is None:
        return pd.DataFrame(columns=["연월", "매매", "전세", "월세"])
    seed = abs(hash(matched_name)) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().replace(day=1), periods=months, freq="MS")
    sale = apt_record["base_sale"] * (1 + np.cumsum(rng.normal(0.004, 0.012, months)))
    jeonse = apt_record["base_jeonse"] * (1 + np.cumsum(rng.normal(0.003, 0.010, months)))
    wolse_rent = apt_record["base_wolse_rent"] * (1 + np.cumsum(rng.normal(0.002, 0.008, months)))
    return pd.DataFrame({"연월": dates, "매매": sale, "전세": jeonse, "월세": wolse_rent})


def fetch_real_trade_trend(matched_name: str, lawd_cd: str, year_months: list):
    """
    매칭된 아파트의 매매/전세/월세 월별 평균 추이를 year_months(조회할 'YYYYMM' 목록)에 대해 계산.
    국토교통부 API는 동시 요청 처리량 자체가 제한적이라 기간이 길수록(예: 2006년 전체) 오래 걸리므로,
    호출측(UI)이 조회 기간을 선택해 넘긴다. 서비스키+지역코드가 없으면 목업 데이터로 대체.
    반환: (trend_df: DataFrame[연월, 매매, 전세, 월세], error: str|None)
    """
    service_key = get_secret("DATA_GO_KR_SERVICE_KEY")
    if not (service_key and lawd_cd):
        return _fetch_mock_trend(matched_name, len(year_months) or 24), None

    error_holder = {"msg": None}

    def _fetch_one_month(ym: str) -> dict:
        trade_result = _call_data_go_kr(TRADE_ENDPOINT, service_key, lawd_cd, ym)
        rent_result = _call_data_go_kr(RENT_ENDPOINT, service_key, lawd_cd, ym)
        if not error_holder["msg"]:
            error_holder["msg"] = trade_result["error"] or rent_result["error"]

        sale_prices = [
            _parse_amount(item.get("dealAmount"))
            for item in trade_result["items"]
            if item.get("aptNm", "").strip() == matched_name
        ]
        sale_prices = [v for v in sale_prices if v is not None]

        jeonse_deposits, wolse_rents = [], []
        for item in rent_result["items"]:
            if item.get("aptNm", "").strip() != matched_name:
                continue
            deposit = _parse_amount(item.get("deposit"))
            monthly = _parse_amount(item.get("monthlyRent"))
            if monthly and monthly > 0:
                wolse_rents.append(monthly)
            elif deposit is not None:
                jeonse_deposits.append(deposit)

        return {
            "연월": pd.Timestamp(ym + "01"),
            "매매": np.mean(sale_prices) if sale_prices else np.nan,
            "전세": np.mean(jeonse_deposits) if jeonse_deposits else np.nan,
            "월세": np.mean(wolse_rents) if wolse_rents else np.nan,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        rows = list(executor.map(_fetch_one_month, year_months))

    trend_df = pd.DataFrame(rows)
    # 건물이 아직 없던(또는 거래가 전혀 없던) 앞쪽 구간은 잘라내고, 실제 거래가 시작된 시점부터 보여준다.
    has_data = trend_df[["매매", "전세", "월세"]].notna().any(axis=1)
    if has_data.any():
        trend_df = trend_df.loc[has_data.idxmax():].reset_index(drop=True)
    return trend_df, error_holder["msg"]


def make_trend_chart(trend_df: pd.DataFrame, apt_name: str, is_mock: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=trend_df["연월"], y=trend_df["매매"], mode="lines+markers", name="매매(만원)", connectgaps=True)
    fig.add_scatter(x=trend_df["연월"], y=trend_df["전세"], mode="lines+markers", name="전세(만원)", connectgaps=True)
    fig.add_scatter(
        x=trend_df["연월"], y=trend_df["월세"], mode="lines+markers", name="월세(만원, 우측축)",
        yaxis="y2", connectgaps=True,
    )
    if len(trend_df) > 0:
        period_label = f"{trend_df['연월'].min():%Y.%m} ~ {trend_df['연월'].max():%Y.%m}"
    else:
        period_label = ""
    fig.update_layout(
        title=f"{apt_name} 매매·전세·월세 실거래가 추이 ({period_label})" + (" (Mock 데이터)" if is_mock else ""),
        xaxis_title="계약년월", yaxis_title="매매·전세 보증금(만원)",
        yaxis2=dict(title="월세(만원)", overlaying="y", side="right"),
        height=460,
        margin=dict(t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


# ======================================================================================
# 물건(매물) 공유 저장소 — Google Sheets 연동
# secrets.toml에 [gcp_service_account]와 GSHEET_ID가 설정되어 있으면, 저장/삭제할 때마다
# Google Sheets에도 함께 기록해 컴퓨터·휴대폰 등 다른 기기에서도 같은 물건 목록을 볼 수 있다.
# 설정되어 있지 않으면 이 세션(브라우저 탭)에만 저장되는 기존 동작으로 자동 대체된다.
# ======================================================================================
GSHEET_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
PROPERTY_COLUMNS = [
    "pid", "name", "dong", "region_label", "custom_lawd_cd", "size_pyeong", "note",
    "monthly_mgmt_fee", "sale_price", "price_growth_rate", "mortgage_rate", "renovation_cost",
    "jeonse_deposit", "jeonse_loan_rate", "wolse_deposit", "wolse_monthly", "wolse_loan_rate",
    "saved_at",
]
# 저장할 때마다 한 줄씩 누적되는 이력 로그 (덮어쓰지 않음) — 쌓이면 나만의 가격 추이 그래프가 된다.
HISTORY_COLUMNS = [
    "saved_at", "pid", "name", "dong", "size_pyeong",
    "sale_price", "price_growth_rate", "mortgage_rate",
    "jeonse_deposit", "jeonse_loan_rate", "wolse_deposit", "wolse_monthly", "wolse_loan_rate",
]
_GSHEET_DEBUG = {"error": None}  # 마지막 연결 실패 사유를 UI에 노출해 진단을 돕는다 (민감정보 아님)


@st.cache_resource(show_spinner=False)
def _open_gsheet():
    """서비스 계정으로 스프레드시트 자체를 연다 (properties/history 워크시트가 여기서 파생됨)."""
    sa_info = get_secret("gcp_service_account")
    sheet_id = get_secret("GSHEET_ID")
    if not sa_info or not sheet_id:
        _GSHEET_DEBUG["error"] = "GSHEET_ID 또는 [gcp_service_account] 시크릿이 비어 있습니다."
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(dict(sa_info), scopes=GSHEET_SCOPES)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
        _GSHEET_DEBUG["error"] = None
        return sh
    except Exception as e:
        _GSHEET_DEBUG["error"] = f"{type(e).__name__}: {e}"
        return None


@st.cache_resource(show_spinner=False)
def _get_or_create_worksheet(sheet_name: str, columns: tuple):
    sh = _open_gsheet()
    if sh is None:
        return None
    import gspread

    try:
        return sh.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_name, rows=2000, cols=len(columns))
        ws.append_row(list(columns))
        return ws
    except Exception as e:
        _GSHEET_DEBUG["error"] = f"{type(e).__name__}: {e}"
        return None


def _get_properties_worksheet():
    return _get_or_create_worksheet("properties", tuple(PROPERTY_COLUMNS))


def _get_history_worksheet():
    return _get_or_create_worksheet("history", tuple(HISTORY_COLUMNS))


def gsheet_enabled() -> bool:
    return _open_gsheet() is not None


def load_properties_from_gsheet() -> dict:
    """Google Sheets의 모든 물건 행을 읽어 {pid: property_dict} 형태로 반환"""
    ws = _get_properties_worksheet()
    if ws is None:
        return {}
    try:
        records = ws.get_all_records()
    except Exception:
        return {}

    def _num(v, cast=float, default=0):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    props = {}
    for row in records:
        name = str(row.get("name", "")).strip()
        dong = str(row.get("dong", "")).strip()
        pid = str(row.get("pid", "")).strip() or build_property_key(name, dong)
        if not pid or not name:
            continue
        props[pid] = {
            "name": name,
            "dong": dong,
            "region_label": row.get("region_label") or DEFAULT_FORM_VALUES["f_region_label"],
            "custom_lawd_cd": str(row.get("custom_lawd_cd", "")),
            "size_pyeong": _num(row.get("size_pyeong"), float, 25.0),
            "note": str(row.get("note", "")),
            "monthly_mgmt_fee": _num(row.get("monthly_mgmt_fee"), int, 15),
            "sale_price": _num(row.get("sale_price"), int, 0),
            "price_growth_rate": _num(row.get("price_growth_rate"), float, 0.0),
            "mortgage_rate": _num(row.get("mortgage_rate"), float, 0.0),
            "renovation_cost": _num(row.get("renovation_cost"), int, 0),
            "jeonse_deposit": _num(row.get("jeonse_deposit"), int, 0),
            "jeonse_loan_rate": _num(row.get("jeonse_loan_rate"), float, 0.0),
            "wolse_deposit": _num(row.get("wolse_deposit"), int, 0),
            "wolse_monthly": _num(row.get("wolse_monthly"), int, 0),
            "wolse_loan_rate": _num(row.get("wolse_loan_rate"), float, 0.0),
            "saved_at": str(row.get("saved_at", "")),
        }
    return props


def save_property_to_gsheet(pid: str, prop: dict) -> None:
    """물건 하나를 Google Sheets에 upsert(있으면 갱신, 없으면 추가)"""
    ws = _get_properties_worksheet()
    if ws is None:
        return
    import gspread

    row_values = [pid] + [str(prop.get(col, "")) for col in PROPERTY_COLUMNS[1:]]
    try:
        cell = ws.find(pid, in_column=1)
    except gspread.exceptions.CellNotFound:
        cell = None
    except Exception:
        return
    try:
        if cell:
            end_a1 = gspread.utils.rowcol_to_a1(cell.row, len(PROPERTY_COLUMNS))
            ws.update(f"A{cell.row}:{end_a1}", [row_values])
        else:
            ws.append_row(row_values)
    except Exception:
        pass


def delete_property_from_gsheet(pid: str) -> None:
    ws = _get_properties_worksheet()
    if ws is None:
        return
    import gspread

    try:
        cell = ws.find(pid, in_column=1)
    except gspread.exceptions.CellNotFound:
        return
    except Exception:
        return
    try:
        ws.delete_rows(cell.row)
    except Exception:
        pass


def append_history_to_gsheet(pid: str, prop: dict, saved_at: str) -> None:
    """저장할 때마다 덮어쓰지 않고 한 줄씩 쌓는 이력 로그. 누적되면 내 입력값의 시세 추이가 된다."""
    ws = _get_history_worksheet()
    if ws is None:
        return
    row_values = [saved_at, pid] + [str(prop.get(col, "")) for col in HISTORY_COLUMNS[2:]]
    try:
        ws.append_row(row_values)
    except Exception:
        pass


def load_history_from_gsheet(pid: str) -> pd.DataFrame:
    """특정 물건(pid)의 누적 입력 이력을 시간순 DataFrame으로 반환"""
    ws = _get_history_worksheet()
    empty = pd.DataFrame(columns=HISTORY_COLUMNS)
    if ws is None:
        return empty
    try:
        records = ws.get_all_records()
    except Exception:
        return empty
    rows = [r for r in records if str(r.get("pid", "")) == pid]
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df["saved_at"] = pd.to_datetime(df["saved_at"], errors="coerce")
    for col in ["size_pyeong", "sale_price", "price_growth_rate", "mortgage_rate",
                "jeonse_deposit", "jeonse_loan_rate", "wolse_deposit", "wolse_monthly", "wolse_loan_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["saved_at"]).sort_values("saved_at").reset_index(drop=True)


def make_my_history_chart(history_df: pd.DataFrame, apt_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=history_df["saved_at"], y=history_df["sale_price"], mode="lines+markers", name="매매가(내 입력, 만원)")
    fig.add_scatter(x=history_df["saved_at"], y=history_df["jeonse_deposit"], mode="lines+markers", name="전세보증금(내 입력, 만원)")
    fig.add_scatter(
        x=history_df["saved_at"], y=history_df["wolse_monthly"], mode="lines+markers",
        name="월세액(내 입력, 만원, 우측축)", yaxis="y2",
    )
    fig.update_layout(
        title=f"{apt_name} — 내가 저장할 때마다 기록된 입력값 변화 (누적 {len(history_df)}회)",
        xaxis_title="저장 시각", yaxis_title="매매가·전세보증금(만원)",
        yaxis2=dict(title="월세(만원)", overlaying="y", side="right"),
        height=420, margin=dict(t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


# ======================================================================================
# 물건(매물) 저장소 — 세션 상태 초기화
# ======================================================================================
NEW_PROPERTY_ID = "__NEW__"  # '새 물건 추가' 상태를 나타내는 센티널 (None 사용 시 selectbox가 플레이스홀더를 잘못 표시함)

DEFAULT_FORM_VALUES = {
    "f_apt_name": "",
    "f_dong": "",
    "f_region_label": "서울특별시 강남구",
    "f_custom_lawd_cd": "",
    "f_size_pyeong": 25.0,
    "f_note": "",
    "f_monthly_mgmt_fee": 15,
    "f_sale_price": 60000,
    "f_price_growth_rate": 3.0,
    "f_mortgage_rate": 4.0,
    "f_renovation_cost": 1000,
    "f_jeonse_deposit": 45000,
    "f_jeonse_loan_rate": 3.5,
    "f_wolse_deposit": 5000,
    "f_wolse_monthly": 100,
    "f_wolse_loan_rate": 4.5,
}


def build_property_key(name: str, dong: str) -> str:
    """아파트 이름 + 동을 물건 저장소의 고유 키로 결합 (같은 단지의 다른 동을 구분하기 위함)"""
    name = (name or "").strip()
    dong = (dong or "").strip()
    return f"{name} {dong}" if dong else name


def load_property_into_form(pid: str):
    """저장된 물건 데이터를 입력 폼(session_state)에 채워 넣는다."""
    prop = st.session_state.properties[pid]
    st.session_state.f_apt_name = prop["name"]
    st.session_state.f_dong = prop.get("dong", "")
    st.session_state.f_region_label = prop.get("region_label", DEFAULT_FORM_VALUES["f_region_label"])
    st.session_state.f_custom_lawd_cd = prop.get("custom_lawd_cd", "")
    st.session_state.f_size_pyeong = prop["size_pyeong"]
    st.session_state.f_note = prop["note"]
    st.session_state.f_monthly_mgmt_fee = prop.get("monthly_mgmt_fee", DEFAULT_FORM_VALUES["f_monthly_mgmt_fee"])
    st.session_state.f_sale_price = prop["sale_price"]
    st.session_state.f_price_growth_rate = prop["price_growth_rate"]
    st.session_state.f_mortgage_rate = prop["mortgage_rate"]
    st.session_state.f_renovation_cost = prop["renovation_cost"]
    st.session_state.f_jeonse_deposit = prop["jeonse_deposit"]
    st.session_state.f_jeonse_loan_rate = prop["jeonse_loan_rate"]
    st.session_state.f_wolse_deposit = prop["wolse_deposit"]
    st.session_state.f_wolse_monthly = prop["wolse_monthly"]
    st.session_state.f_wolse_loan_rate = prop["wolse_loan_rate"]


if "properties" not in st.session_state:
    st.session_state.properties = load_properties_from_gsheet() if gsheet_enabled() else {}
if "selected_property_id" not in st.session_state:
    st.session_state.selected_property_id = NEW_PROPERTY_ID
for _k, _v in DEFAULT_FORM_VALUES.items():
    st.session_state.setdefault(_k, _v)

# 저장/삭제 버튼에서 예약해 둔 물건 선택 변경을 위젯 생성 전에 반영
if "pending_select_id" in st.session_state:
    _new_pid = st.session_state.pop("pending_select_id")
    st.session_state.selected_property_id = _new_pid
    if _new_pid == NEW_PROPERTY_ID:
        st.session_state.update(DEFAULT_FORM_VALUES)
    else:
        load_property_into_form(_new_pid)


def handle_property_switch():
    pid = st.session_state.selected_property_id
    if pid == NEW_PROPERTY_ID:
        st.session_state.update(DEFAULT_FORM_VALUES)
    else:
        load_property_into_form(pid)


# ======================================================================================
# 사이드바 — 공통 입력 (매수자 기준, 모든 물건에 공통 적용)
# ======================================================================================
st.title("🏠지영 & 경아의 보금자리 찾아 삼만리")
st.caption("물건별로 매매·전세·월세 조건을 한 화면에서 동시에 비교하고, 여러 물건을 저장해 종합 비교해 보세요.")

with st.sidebar:
    st.header("⚙️ 공통 입력값")
    st.caption("보유 자금 등 매수자 기준 값으로, 모든 물건에 동일하게 적용됩니다.")
    own_capital = st.number_input("보유 자금 (만원)", min_value=0, value=30000, step=1000)
    money_hint(own_capital)
    target_period = st.number_input("거주 예정 기간 (년)", min_value=0.5, value=4.0, step=0.5, format="%.1f")
    opportunity_rate = st.number_input("자기자본 기회비용 연수익률 (%)", min_value=0.0, value=4.0, step=0.1, format="%.1f")
    inflation_rate = st.number_input("물가상승률 / 임대료 상승률 (%)", min_value=0.0, value=3.0, step=0.1, format="%.1f")

    st.subheader("공통 기타비용")
    st.caption("월 관리비는 물건마다 달라 '물건 분석' 탭의 물건 정보에서 입력합니다.")
    moving_cost = st.number_input("이사비 (만원)", min_value=0, value=150, step=10)
    money_hint(moving_cost)
    cleaning_cost = st.number_input("청소비 (만원)", min_value=0, value=30, step=5)
    money_hint(cleaning_cost)

    with st.expander("🏛️ 세율 설정 (정책 변경 시 최신 고시 값으로 수정)"):
        st.caption(
            "취득세 구간·세율은 지방세법상 실제 법정 수치이며, 정부 정책에 따라 바뀔 수 있으니 "
            "최신 고시 내용으로 직접 수정하세요."
        )
        st.markdown("**취득세(지방세법) 구간·세율**")
        acq_threshold1 = st.number_input("취득세 1구간 기준금액 (만원, 예: 6억=60000)", min_value=0, value=60000, step=1000)
        money_hint(acq_threshold1)
        acq_threshold2 = st.number_input("취득세 2구간 기준금액 (만원, 예: 9억=90000)", min_value=0, value=90000, step=1000)
        money_hint(acq_threshold2)
        acq_rate_min = st.number_input("최저 취득세율 (%, 1구간 이하)", min_value=0.0, value=1.0, step=0.1, format="%.1f")
        acq_rate_max = st.number_input("최고 취득세율 (%, 2구간 초과)", min_value=0.0, value=3.0, step=0.1, format="%.1f")
        edu_tax_ratio = st.number_input("지방교육세율 (취득세율 대비 %)", min_value=0.0, value=10.0, step=1.0, format="%.1f")

        st.markdown("**보유세(재산세·종합부동산세 등) 근사 실효세율**")
        st.caption(
            "재산세·종부세는 공시가격·공정시장가액비율·누진세율이 매년 고시되어 정확한 계산이 복잡하므로, "
            "매매가 대비 연간 실효세율(%) 하나로 근사합니다. 최신 고시 공정시장가액비율·세율을 반영해 조정하세요."
        )
        holding_tax_rate = st.number_input(
            "보유세 실효세율 (연, % of 매매가)", min_value=0.0, value=0.15, step=0.01, format="%.2f",
        )

g_inputs = dict(
    own_capital=own_capital, target_period=target_period,
    opportunity_rate=opportunity_rate, inflation_rate=inflation_rate,
    moving_cost=moving_cost, cleaning_cost=cleaning_cost,
    acq_threshold1=acq_threshold1, acq_threshold2=acq_threshold2,
    acq_rate_min=acq_rate_min, acq_rate_max=acq_rate_max,
    edu_tax_ratio=edu_tax_ratio, holding_tax_rate=holding_tax_rate,
)

# ======================================================================================
# 메인 탭 — 물건 분석 / 물건 비교
# ======================================================================================
tab_analyze, tab_compare = st.tabs(["🏢 물건 분석", "📊 물건 비교"])

# --------------------------------------------------------------------------------------
# 탭 1. 물건 분석 — 하나의 물건에 대해 매매·전세·월세를 한 화면에서 동시 비교
# --------------------------------------------------------------------------------------
with tab_analyze:
    sel_head_col1, sel_head_col2 = st.columns([5, 1.3])
    with sel_head_col1:
        st.subheader("🔎 분석할 물건 선택")
    with sel_head_col2:
        if gsheet_enabled():
            if st.button("🔄 새로고침", use_container_width=True, help="다른 기기에서 저장한 최신 물건 목록을 다시 불러옵니다"):
                st.session_state.properties = load_properties_from_gsheet()
                st.session_state.pending_select_id = NEW_PROPERTY_ID
                st.rerun()
    if gsheet_enabled():
        st.caption("🔗 공유 저장소(Google Sheets) 연동됨 — 다른 기기에서 저장한 물건도 새로고침하면 보입니다.")
    else:
        st.caption("⚠️ 공유 저장소가 연동되지 않아 이 브라우저에만 저장됩니다. (secrets.toml에 GSHEET 설정 필요)")
        if _GSHEET_DEBUG.get("error"):
            st.caption(f"🔍 진단: {_GSHEET_DEBUG['error']}")

    pid_options = [NEW_PROPERTY_ID] + list(st.session_state.properties.keys())
    st.selectbox(
        "저장된 물건을 불러오거나 새 물건을 추가하세요",
        options=pid_options,
        format_func=lambda pid: (
            "➕ 새 물건 추가" if pid == NEW_PROPERTY_ID
            else (
                f'{st.session_state.properties[pid]["name"]}'
                f'{" " + st.session_state.properties[pid]["dong"] if st.session_state.properties[pid].get("dong") else ""}'
                f' ({st.session_state.properties[pid]["size_pyeong"]:.0f}평)'
            )
        ),
        key="selected_property_id",
        on_change=handle_property_switch,
    )

    st.subheader("🏷️ 물건 정보")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1.2])
    with c1:
        st.text_input("아파트 이름", key="f_apt_name", placeholder="예: OO아파트")
    with c2:
        st.text_input("동", key="f_dong", placeholder="예: 101동")
    with c3:
        st.number_input("평수", min_value=0.0, step=0.5, format="%.1f", key="f_size_pyeong")
    with c4:
        st.number_input("월 관리비 (만원)", min_value=0, step=1, key="f_monthly_mgmt_fee")
        money_hint(st.session_state.f_monthly_mgmt_fee)
    st.text_input("비고", key="f_note", placeholder="예: 역세권, 로열층, 남향 등")

    _current_prop = st.session_state.properties.get(st.session_state.selected_property_id)
    if _current_prop and _current_prop.get("saved_at"):
        st.caption(f"🕒 마지막 저장 일시: {_current_prop['saved_at']}")

    if gsheet_enabled() and st.session_state.selected_property_id != NEW_PROPERTY_ID:
        _history_df = load_history_from_gsheet(st.session_state.selected_property_id)
        if len(_history_df) >= 1:
            st.divider()
            st.subheader("📈 내가 기록한 시세 변화 (누적 이력)")
            st.caption("저장할 때마다 값을 덮어쓰지 않고 한 줄씩 쌓입니다 — 같은 물건을 주기적으로 다시 저장하면 나만의 시세 추이가 됩니다.")
            if len(_history_df) == 1:
                st.info("아직 저장 기록이 1건뿐이라 추이를 그리기엔 이릅니다. 나중에 다시 저장하면 그래프가 나타납니다.")
            else:
                st.plotly_chart(
                    make_my_history_chart(_history_df, _current_prop["name"] if _current_prop else st.session_state.selected_property_id),
                    use_container_width=True,
                )

    st.divider()
    st.subheader("🔍 실거래가 매칭 & 최근 시세 추이")

    service_key_configured = bool(get_secret("DATA_GO_KR_SERVICE_KEY"))
    if not service_key_configured:
        st.info(
            "⚠️ 아직 공공데이터 API 키가 연동되지 않아 임시 목업(Mock) 데이터로 동작합니다. "
            "`secrets.toml`에 DATA_GO_KR_SERVICE_KEY를 설정하면 국토교통부 실거래가로 자동 교체됩니다."
        )

    rcol1, rcol2 = st.columns([2, 1.2])
    with rcol1:
        st.selectbox("지역 (시/군/구)", options=REGION_OPTIONS, key="f_region_label")
    with rcol2:
        if st.session_state.f_region_label == CUSTOM_REGION_LABEL:
            st.text_input(
                "법정동코드 (5자리)", key="f_custom_lawd_cd", max_chars=5, placeholder="예: 41135",
            )
        else:
            st.caption("법정동코드")
            st.caption(f"`{region_label_to_code(st.session_state.f_region_label, '')}`")

    lawd_cd = region_label_to_code(st.session_state.f_region_label, st.session_state.f_custom_lawd_cd)
    using_real_data = service_key_configured and bool(lawd_cd)

    apt_query = st.session_state.f_apt_name.strip()
    if not apt_query:
        st.caption("💡 위 '아파트 이름'을 입력하면 실거래가 DB에서 유사한 물건을 자동으로 찾아드립니다.")
    elif service_key_configured and not lawd_cd:
        st.warning("지역(시/군/구)을 선택하거나 법정동코드를 입력해야 실거래가를 조회할 수 있습니다.")
    else:
        matches, match_error = search_apartment_matches(apt_query, lawd_cd)
        if match_error:
            st.warning(f"실거래가 API 조회 중 문제가 발생했습니다: {match_error}")
        if not matches:
            st.warning(f"'{apt_query}'와(과) 일치하는 실거래가 등록 아파트를 찾지 못했습니다. 이름을 다르게 입력해 보세요.")
            st.session_state.pop("matched_apt_name", None)
        else:
            if st.session_state.get("matched_apt_name") not in matches:
                st.session_state.matched_apt_name = matches[0]
            matched_name = st.selectbox(
                "실거래가 DB에서 매칭된 아파트를 선택하세요", options=matches, key="matched_apt_name",
            )
            st.caption(f"📍 매칭된 물건: {matched_name} ({st.session_state.f_region_label})")

            range_choice = st.radio(
                "조회 기간", ["최근 5년 (빠름)", "최근 10년", "전체 (2006년~, 오래된 아파트용·느림)"],
                horizontal=True, key="trend_range_choice",
            )
            if using_real_data and range_choice.startswith("전체"):
                st.caption("⏳ 정부 API 자체의 동시 요청 제한 때문에 전체 기간 조회는 지역당 최초 1회 약 2~3분 걸릴 수 있습니다. (같은 지역은 이후 1시간 동안 캐시되어 빨라집니다)")
            if range_choice.startswith("최근 5"):
                year_months = _recent_year_months(60)
            elif range_choice.startswith("최근 10"):
                year_months = _recent_year_months(120)
            else:
                year_months = _all_year_months_since(*REAL_ESTATE_DATA_START)

            with st.spinner(f"'{matched_name}' 실거래가 조회 중... ({range_choice})"):
                trend_df, trend_error = fetch_real_trade_trend(matched_name, lawd_cd, year_months)
            if trend_error:
                st.warning(f"실거래가 추이 조회 중 문제가 발생했습니다: {trend_error}")

            if trend_df.empty:
                st.info("표시할 실거래가 데이터가 없습니다.")
            else:
                st.plotly_chart(
                    make_trend_chart(trend_df, matched_name, is_mock=not using_real_data),
                    use_container_width=True,
                )

                def _last_valid(series):
                    s = series.dropna()
                    return s.iloc[-1] if not s.empty else None

                latest_sale = _last_valid(trend_df["매매"])
                latest_jeonse = _last_valid(trend_df["전세"])
                latest_wolse = _last_valid(trend_df["월세"])

                st.markdown("**📊 내 시뮬레이션 입력값 vs 실거래가 최근값 비교**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(
                    "매매가 (내 입력)", fmt_money(st.session_state.f_sale_price),
                    delta=None if latest_sale is None else fmt_money(st.session_state.f_sale_price - latest_sale),
                )
                mc2.metric(
                    "전세보증금 (내 입력)", fmt_money(st.session_state.f_jeonse_deposit),
                    delta=None if latest_jeonse is None else fmt_money(st.session_state.f_jeonse_deposit - latest_jeonse),
                )
                mc3.metric(
                    "월세액 (내 입력)", fmt_money(st.session_state.f_wolse_monthly),
                    delta=None if latest_wolse is None else fmt_money(st.session_state.f_wolse_monthly - latest_wolse),
                )
                st.caption("델타(▲/▼)는 '내가 입력한 시뮬레이션 값 − 실거래가 최근값' 기준입니다. 실거래가 없는 달은 빈 값으로 표시됩니다.")

    st.divider()

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        save_clicked = st.button("💾 이 물건 저장", type="primary", use_container_width=True)
    with btn_col2:
        delete_clicked = st.button(
            "🗑️ 이 물건 삭제", use_container_width=True,
            disabled=st.session_state.selected_property_id == NEW_PROPERTY_ID,
        )

    if save_clicked:
        name = st.session_state.f_apt_name.strip()
        dong = st.session_state.f_dong.strip()
        if not name:
            st.error("아파트 이름을 입력해야 저장할 수 있습니다.")
        else:
            new_pid = build_property_key(name, dong)
            prev_id = st.session_state.selected_property_id
            if prev_id != NEW_PROPERTY_ID and prev_id != new_pid and prev_id in st.session_state.properties:
                del st.session_state.properties[prev_id]  # 이름/동 변경 시 기존 항목 정리
                if gsheet_enabled():
                    delete_property_from_gsheet(prev_id)
            saved_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.properties[new_pid] = {
                "name": name,
                "dong": dong,
                "region_label": st.session_state.f_region_label,
                "custom_lawd_cd": st.session_state.f_custom_lawd_cd,
                "size_pyeong": st.session_state.f_size_pyeong,
                "note": st.session_state.f_note,
                "monthly_mgmt_fee": st.session_state.f_monthly_mgmt_fee,
                "sale_price": st.session_state.f_sale_price,
                "price_growth_rate": st.session_state.f_price_growth_rate,
                "mortgage_rate": st.session_state.f_mortgage_rate,
                "renovation_cost": st.session_state.f_renovation_cost,
                "jeonse_deposit": st.session_state.f_jeonse_deposit,
                "jeonse_loan_rate": st.session_state.f_jeonse_loan_rate,
                "wolse_deposit": st.session_state.f_wolse_deposit,
                "wolse_monthly": st.session_state.f_wolse_monthly,
                "wolse_loan_rate": st.session_state.f_wolse_loan_rate,
                "saved_at": saved_at,
            }
            st.session_state.pending_select_id = new_pid
            if gsheet_enabled():
                save_property_to_gsheet(new_pid, st.session_state.properties[new_pid])
                append_history_to_gsheet(new_pid, st.session_state.properties[new_pid], saved_at)
                st.success(
                    f"'{new_pid}' 물건이 저장되고 공유 저장소에 동기화되었습니다 ({saved_at}). "
                    "누적 입력 이력은 아래 '📈 내가 기록한 시세 변화'에서 확인할 수 있습니다."
                )
            else:
                st.success(f"'{new_pid}' 물건이 저장되었습니다 ({saved_at}). (공유 저장소 미연동 — 이 브라우저에서만 보이고 이력도 쌓이지 않습니다)")
            st.rerun()

    if delete_clicked and st.session_state.selected_property_id != NEW_PROPERTY_ID:
        deleted_name = st.session_state.selected_property_id
        del st.session_state.properties[deleted_name]
        if gsheet_enabled():
            delete_property_from_gsheet(deleted_name)
        st.session_state.pending_select_id = NEW_PROPERTY_ID
        st.info(f"'{deleted_name}' 물건을 삭제했습니다.")
        st.rerun()

    st.divider()
    st.caption("💡 아래에서 이 물건의 매매·전세·월세 조건을 한 화면에서 동시에 입력하고 비교할 수 있습니다.")

    col_sale, col_jeonse, col_wolse = st.columns(3)
    with col_sale:
        with st.container(border=True):
            st.markdown("#### 🏠 매매")
            st.number_input("매매가 (만원)", min_value=0, step=1000, key="f_sale_price")
            money_hint(st.session_state.f_sale_price)
            st.number_input("예상 연간 주택가격 상승률 (%)", step=0.1, format="%.1f", key="f_price_growth_rate")
            st.number_input("주택담보대출 금리 (%)", min_value=0.0, step=0.1, format="%.1f", key="f_mortgage_rate")
            st.number_input("수리/인테리어비 (만원)", min_value=0, step=100, key="f_renovation_cost")
            money_hint(st.session_state.f_renovation_cost)

    with col_jeonse:
        with st.container(border=True):
            st.markdown("#### 🏢 전세")
            st.number_input("전세 보증금 (만원)", min_value=0, step=1000, key="f_jeonse_deposit")
            money_hint(st.session_state.f_jeonse_deposit)
            st.number_input("전세자금대출 금리 (%)", min_value=0.0, step=0.1, format="%.1f", key="f_jeonse_loan_rate")

            jeonse_ratio = safe_div(st.session_state.f_jeonse_deposit, st.session_state.f_sale_price)
            if jeonse_ratio > 0.8:
                st.warning(
                    f"⚠️ 전세가율 {jeonse_ratio*100:.1f}% (80% 초과)\n\n"
                    "역전세 및 보증금 미반환 리스크가 높으니 계약 전 반드시 확인하세요."
                )

    with col_wolse:
        with st.container(border=True):
            st.markdown("#### 🔑 월세")
            st.number_input("월세 보증금 (만원)", min_value=0, step=500, key="f_wolse_deposit")
            money_hint(st.session_state.f_wolse_deposit)
            st.number_input("월세액 (만원)", min_value=0, step=5, key="f_wolse_monthly")
            money_hint(st.session_state.f_wolse_monthly)
            st.number_input("보증금 대출금리 (%)", min_value=0.0, step=0.1, format="%.1f", key="f_wolse_loan_rate")

    # ----- 현재 입력값 기준 계산 -----
    p_inputs = dict(
        sale_price=st.session_state.f_sale_price,
        price_growth_rate=st.session_state.f_price_growth_rate,
        mortgage_rate=st.session_state.f_mortgage_rate,
        renovation_cost=st.session_state.f_renovation_cost,
    )
    j_inputs = dict(
        jeonse_deposit=st.session_state.f_jeonse_deposit,
        jeonse_loan_rate=st.session_state.f_jeonse_loan_rate,
    )
    w_inputs = dict(
        wolse_deposit=st.session_state.f_wolse_deposit,
        wolse_monthly=st.session_state.f_wolse_monthly,
        wolse_loan_rate=st.session_state.f_wolse_loan_rate,
    )

    g_calc = dict(g_inputs, monthly_mgmt_fee=st.session_state.f_monthly_mgmt_fee)
    purchase_net, purchase_wf, purchase_detail = calc_purchase(g_calc, p_inputs)
    jeonse_net, jeonse_wf, jeonse_detail = calc_jeonse(g_calc, j_inputs)
    wolse_net, wolse_wf, wolse_detail = calc_wolse(g_calc, w_inputs)

    results = {"매매": purchase_net, "전세": jeonse_net, "월세": wolse_net}
    best_option = min(results, key=results.get)
    second_best_cost = sorted(results.values())[1]
    saving = second_best_cost - results[best_option]

    st.divider()
    st.header("📊 비교 분석 결과")

    m1, m2, m3 = st.columns(3)
    m1.metric("매매 순비용", fmt_money(purchase_net))
    m2.metric("전세 순비용", fmt_money(jeonse_net))
    m3.metric("월세 순비용", fmt_money(wolse_net))

    apt_label = build_property_key(st.session_state.f_apt_name, st.session_state.f_dong) or "현재 물건"
    st.success(
        f"✅ [{apt_label}] 최적 선택: {best_option} — 거주 예정기간 {target_period:.1f}년 기준 실질 순비용이 가장 낮습니다. "
        f"(차선 대비 약 {fmt_money(saving)} 절감)"
    )

    _crossover_max_years = 40.0
    _timeline, _transitions = compute_option_crossovers(g_calc, p_inputs, j_inputs, w_inputs, max_years=_crossover_max_years)
    st.info(format_crossover_message(_transitions, best_option, _crossover_max_years))
    st.caption(
        "💡 전월세는 보통 2~4년 단위 계약이라 거주기간을 짧게 잡기 쉬운데, "
        "실제로 그 집(또는 그 동네)에 얼마나 오래 살 것 같은지를 기준으로 위 전환 시점과 비교해보세요."
    )

    st.subheader("💧 자금 흐름 워터폴 차트")
    wf_tab1, wf_tab2, wf_tab3 = st.tabs(["매매", "전세", "월세"])
    with wf_tab1:
        st.plotly_chart(make_waterfall("매매", purchase_wf, purchase_net), use_container_width=True)
    with wf_tab2:
        st.plotly_chart(make_waterfall("전세", jeonse_wf, jeonse_net), use_container_width=True)
    with wf_tab3:
        st.plotly_chart(make_waterfall("월세", wolse_wf, wolse_net), use_container_width=True)

    st.subheader("🌡️ 민감도 분석: 매매 vs 전세")
    st.caption("주택가격 상승률과 대출 금리가 함께 변할 때, 매매가 전세보다 유리한지(녹색) 불리한지(빨강)를 보여줍니다.")
    pg_range, lr_range, diff_matrix = compute_sensitivity(g_calc, p_inputs, j_inputs)
    st.plotly_chart(make_sensitivity_heatmap(pg_range, lr_range, diff_matrix), use_container_width=True)

    st.subheader("📋 세부 계산 내역")
    summary_rows = [
        {
            "옵션": "매매", "초기비용": purchase_detail["초기비용"], "대출이자": purchase_detail["대출이자"],
            "보유세/월세": purchase_detail["보유세"], "관리비": purchase_detail["관리비"],
            "기회비용(순)": purchase_detail["기회비용"] - purchase_detail["여유자금운용수익"],
            "시세차익": -purchase_detail["시세차익"], "순비용": purchase_detail["순비용"],
        },
        {
            "옵션": "전세", "초기비용": jeonse_detail["초기비용"], "대출이자": jeonse_detail["대출이자"],
            "보유세/월세": jeonse_detail["보유세"], "관리비": jeonse_detail["관리비"],
            "기회비용(순)": jeonse_detail["기회비용"] - jeonse_detail["여유자금운용수익"],
            "시세차익": jeonse_detail["시세차익"], "순비용": jeonse_detail["순비용"],
        },
        {
            "옵션": "월세", "초기비용": wolse_detail["초기비용"], "대출이자": wolse_detail["대출이자"],
            "보유세/월세": wolse_detail["보유세"], "관리비": wolse_detail["관리비"],
            "기회비용(순)": wolse_detail["기회비용"] - wolse_detail["여유자금운용수익"],
            "시세차익": wolse_detail["시세차익"], "순비용": wolse_detail["순비용"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    display_df = summary_df.copy()
    for col in ["초기비용", "대출이자", "보유세/월세", "관리비", "기회비용(순)", "시세차익", "순비용"]:
        display_df[col] = display_df[col].round(0).map(lambda v: f"{v:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 현재 물건 비교 결과 CSV 다운로드", data=csv_bytes,
        file_name=f"{apt_label}_비교결과.csv", mime="text/csv",
    )

    with st.expander("ℹ️ 계산 가정 및 참고사항"):
        st.markdown(
            f"""
            - **취득세**: 사이드바 '🏛️ 세율 설정'에서 입력한 구간·세율을 그대로 적용합니다.
              현재 값 → {acq_threshold1/10000:.1f}억 이하 {acq_rate_min:.1f}%, {acq_threshold1/10000:.1f}억~{acq_threshold2/10000:.1f}억 구간 선형 산출,
              {acq_threshold2/10000:.1f}억 초과 {acq_rate_max:.1f}% + 지방교육세(취득세율의 {edu_tax_ratio:.1f}%) 가산 (1주택자·전용 85㎡ 이하 단순화 기준).
              정책이 바뀌면 사이드바에서 직접 최신 고시 값으로 수정하세요.
            - **중개보수**: 서울시 공인중개사 법정 상한 요율표 기준 자동 산출
            - **대출이자**: 부족 자금(필요자금 − 보유자금)에 대해 거주기간 동안 단리로 계산
            - **보유세**: 재산세·종부세 등을 매매가 대비 연 실효세율({holding_tax_rate:.2f}%)로 근사한 값입니다.
              실제로는 공시가격·공정시장가액비율·누진세율이 매년 고시되어 더 복잡하니, 사이드바에서 최신 고시 기준 실효세율로 조정해 사용하세요.
            - **월 관리비**: 아파트마다 다르므로 물건별로 입력하며, '물건 정보'에 입력한 값이 그대로 사용됩니다.
            - **월세 상승**: 물가상승률을 매년 복리로 반영하여 총 월세 지출액 산출
            - **기회비용**: 각 옵션에 실제 투입된 자기자본에 대해서만 산정하며, 여유자금은 별도 운용수익으로 차감
            - **공통 입력값**(보유자금, 거주기간, 기회비용률, 세율 등)은 모든 물건에 동일하게 적용되며, 사이드바에서 값을 바꾸면 저장된 모든 물건의 비교 결과도 함께 갱신됩니다.
            """
        )

# --------------------------------------------------------------------------------------
# 탭 2. 물건 비교 — 저장된 여러 물건의 매매/전세/월세 순비용 종합 비교
# --------------------------------------------------------------------------------------
with tab_compare:
    compare_head_col1, compare_head_col2 = st.columns([5, 1.3])
    with compare_head_col1:
        st.header("📊 저장된 물건 종합 비교")
    with compare_head_col2:
        if gsheet_enabled():
            if st.button(
                "🔄 새로고침", use_container_width=True, key="btn_refresh_compare",
                help="다른 기기에서 저장한 최신 물건 목록을 다시 불러옵니다",
            ):
                st.session_state.properties = load_properties_from_gsheet()
                st.rerun()

    props = st.session_state.properties
    if not props:
        st.info("아직 저장된 물건이 없습니다. '🏢 물건 분석' 탭에서 물건 정보를 입력한 뒤 '💾 이 물건 저장' 버튼을 눌러주세요.")
    else:
        rows = []
        for pid, prop in props.items():
            p_i = dict(
                sale_price=prop["sale_price"], price_growth_rate=prop["price_growth_rate"],
                mortgage_rate=prop["mortgage_rate"], renovation_cost=prop["renovation_cost"],
            )
            j_i = dict(jeonse_deposit=prop["jeonse_deposit"], jeonse_loan_rate=prop["jeonse_loan_rate"])
            w_i = dict(
                wolse_deposit=prop["wolse_deposit"], wolse_monthly=prop["wolse_monthly"],
                wolse_loan_rate=prop["wolse_loan_rate"],
            )
            g_calc = dict(g_inputs, monthly_mgmt_fee=prop.get("monthly_mgmt_fee", 15))
            pn, _, _ = calc_purchase(g_calc, p_i)
            jn, _, _ = calc_jeonse(g_calc, j_i)
            wn, _, _ = calc_wolse(g_calc, w_i)

            opt_costs = {"매매": pn, "전세": jn, "월세": wn}
            best = min(opt_costs, key=opt_costs.get)
            jr = safe_div(prop["jeonse_deposit"], prop["sale_price"])

            rows.append({
                "물건명": prop["name"], "동": prop.get("dong", ""), "평수": prop["size_pyeong"],
                "월관리비": prop.get("monthly_mgmt_fee", 15), "비고": prop["note"],
                "최근입력일": prop.get("saved_at", ""),
                "매매 순비용": pn, "전세 순비용": jn, "월세 순비용": wn,
                "최적옵션": best, "최적 순비용": opt_costs[best], "전세가율(%)": jr * 100,
            })

        comp_df = pd.DataFrame(rows).sort_values("최적 순비용").reset_index(drop=True)

        best_row = comp_df.iloc[0]
        st.success(
            f"🏆 전체 물건 중 최적 조합: **{best_row['물건명']} ({best_row['최적옵션']})** — "
            f"순비용 {fmt_money(best_row['최적 순비용'])}"
        )

        risky = comp_df[comp_df["전세가율(%)"] > 80]
        if not risky.empty:
            names = ", ".join(risky["물건명"].tolist())
            st.warning(f"⚠️ 전세가율 80% 초과 물건: {names} — 역전세/보증금 미반환 리스크를 확인하세요.")

        display_comp = comp_df.copy()
        display_comp["평수"] = display_comp["평수"].map(lambda v: f"{v:.1f}")
        display_comp["전세가율(%)"] = display_comp["전세가율(%)"].map(lambda v: f"{v:.1f}%")
        for c in ["매매 순비용", "전세 순비용", "월세 순비용", "최적 순비용"]:
            display_comp[c] = display_comp[c].round(0).map(lambda v: f"{v:,.0f}")
        st.dataframe(display_comp, use_container_width=True, hide_index=True)

        st.subheader("📊 물건별 매매·전세·월세 순비용 비교")
        fig = go.Figure()
        fig.add_bar(name="매매", x=comp_df["물건명"], y=comp_df["매매 순비용"])
        fig.add_bar(name="전세", x=comp_df["물건명"], y=comp_df["전세 순비용"])
        fig.add_bar(name="월세", x=comp_df["물건명"], y=comp_df["월세 순비용"])
        fig.update_layout(
            barmode="group", title="물건별 순비용(Net Cost) 비교",
            yaxis_title="순비용(만원)", xaxis_title="물건명", height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

        csv_bytes = comp_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 물건 비교 결과 CSV 다운로드", data=csv_bytes,
            file_name="물건_종합비교결과.csv", mime="text/csv",
        )

        with st.expander("🗑️ 저장된 물건 삭제"):
            to_delete = st.multiselect(
                "삭제할 물건을 선택하세요",
                options=list(props.keys()),
                format_func=lambda pid: props[pid]["name"],
            )
            if st.button("선택한 물건 삭제", disabled=not to_delete):
                for pid in to_delete:
                    if pid in st.session_state.properties:
                        del st.session_state.properties[pid]
                        if gsheet_enabled():
                            delete_property_from_gsheet(pid)
                if st.session_state.selected_property_id in to_delete:
                    st.session_state.pending_select_id = NEW_PROPERTY_ID
                st.rerun()
