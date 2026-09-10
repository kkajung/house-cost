"""
지영 & 경아의 보금자리 찾아 삼만리 — 매매 vs 전세 vs 월세 주거비용 종합 비교 및 의사결정 지원 웹앱
Streamlit + Pandas + NumPy + Plotly
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="지영 & 경아의 보금자리 찾아 삼만리", page_icon="🏠", layout="wide")


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
# 물건(매물) 저장소 — 세션 상태 초기화
# ======================================================================================
NEW_PROPERTY_ID = "__NEW__"  # '새 물건 추가' 상태를 나타내는 센티널 (None 사용 시 selectbox가 플레이스홀더를 잘못 표시함)

DEFAULT_FORM_VALUES = {
    "f_apt_name": "",
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


def load_property_into_form(pid: str):
    """저장된 물건 데이터를 입력 폼(session_state)에 채워 넣는다."""
    prop = st.session_state.properties[pid]
    st.session_state.f_apt_name = prop["name"]
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
    st.session_state.properties = {}
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
    target_period = st.number_input("거주 예정 기간 (년)", min_value=0.5, value=4.0, step=0.5, format="%.1f")
    opportunity_rate = st.number_input("자기자본 기회비용 연수익률 (%)", min_value=0.0, value=4.0, step=0.1, format="%.1f")
    inflation_rate = st.number_input("물가상승률 / 임대료 상승률 (%)", min_value=0.0, value=3.0, step=0.1, format="%.1f")

    st.subheader("공통 기타비용")
    st.caption("월 관리비는 물건마다 달라 '물건 분석' 탭의 물건 정보에서 입력합니다.")
    moving_cost = st.number_input("이사비 (만원)", min_value=0, value=150, step=10)
    cleaning_cost = st.number_input("청소비 (만원)", min_value=0, value=30, step=5)

    with st.expander("🏛️ 세율 설정 (정책 변경 시 최신 고시 값으로 수정)"):
        st.caption(
            "취득세 구간·세율은 지방세법상 실제 법정 수치이며, 정부 정책에 따라 바뀔 수 있으니 "
            "최신 고시 내용으로 직접 수정하세요."
        )
        st.markdown("**취득세(지방세법) 구간·세율**")
        acq_threshold1 = st.number_input("취득세 1구간 기준금액 (만원, 예: 6억=60000)", min_value=0, value=60000, step=1000)
        acq_threshold2 = st.number_input("취득세 2구간 기준금액 (만원, 예: 9억=90000)", min_value=0, value=90000, step=1000)
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
    st.subheader("🔎 분석할 물건 선택")
    pid_options = [NEW_PROPERTY_ID] + list(st.session_state.properties.keys())
    st.selectbox(
        "저장된 물건을 불러오거나 새 물건을 추가하세요",
        options=pid_options,
        format_func=lambda pid: (
            "➕ 새 물건 추가" if pid == NEW_PROPERTY_ID
            else f'{st.session_state.properties[pid]["name"]} ({st.session_state.properties[pid]["size_pyeong"]:.0f}평)'
        ),
        key="selected_property_id",
        on_change=handle_property_switch,
    )

    st.subheader("🏷️ 물건 정보")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c1:
        st.text_input("아파트 이름", key="f_apt_name", placeholder="예: OO아파트 101동")
    with c2:
        st.number_input("평수", min_value=0.0, step=0.5, format="%.1f", key="f_size_pyeong")
    with c3:
        st.number_input("월 관리비 (만원)", min_value=0, step=1, key="f_monthly_mgmt_fee")
    with c4:
        st.text_input("비고", key="f_note", placeholder="예: 역세권, 로열층, 남향 등")

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
        if not name:
            st.error("아파트 이름을 입력해야 저장할 수 있습니다.")
        else:
            prev_id = st.session_state.selected_property_id
            if prev_id != NEW_PROPERTY_ID and prev_id != name and prev_id in st.session_state.properties:
                del st.session_state.properties[prev_id]  # 이름 변경 시 기존 항목 정리
            st.session_state.properties[name] = {
                "name": name,
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
            }
            st.session_state.pending_select_id = name
            st.success(f"'{name}' 물건이 저장되었습니다. '📊 물건 비교' 탭에서 다른 물건과 비교할 수 있습니다.")
            st.rerun()

    if delete_clicked and st.session_state.selected_property_id != NEW_PROPERTY_ID:
        deleted_name = st.session_state.selected_property_id
        del st.session_state.properties[deleted_name]
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
            st.number_input("예상 연간 주택가격 상승률 (%)", step=0.1, format="%.1f", key="f_price_growth_rate")
            st.number_input("주택담보대출 금리 (%)", min_value=0.0, step=0.1, format="%.1f", key="f_mortgage_rate")
            st.number_input("수리/인테리어비 (만원)", min_value=0, step=100, key="f_renovation_cost")

    with col_jeonse:
        with st.container(border=True):
            st.markdown("#### 🏢 전세")
            st.number_input("전세 보증금 (만원)", min_value=0, step=1000, key="f_jeonse_deposit")
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
            st.number_input("월세액 (만원)", min_value=0, step=5, key="f_wolse_monthly")
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

    apt_label = st.session_state.f_apt_name.strip() or "현재 물건"
    st.success(
        f"✅ [{apt_label}] 최적 선택: {best_option} — 거주 예정기간 {target_period:.1f}년 기준 실질 순비용이 가장 낮습니다. "
        f"(차선 대비 약 {fmt_money(saving)} 절감)"
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
    st.header("📊 저장된 물건 종합 비교")

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
                "물건명": prop["name"], "평수": prop["size_pyeong"],
                "월관리비": prop.get("monthly_mgmt_fee", 15), "비고": prop["note"],
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
                if st.session_state.selected_property_id in to_delete:
                    st.session_state.pending_select_id = NEW_PROPERTY_ID
                st.rerun()
