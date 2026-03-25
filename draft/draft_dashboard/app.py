import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="KBO 신인 드래프트 대시보드",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── 커스텀 CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Bebas+Neue&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #e8b84b;
    --accent2: #4b9fe8;
    --text: #e8eaf0;
    --text-muted: #7a8499;
    --border: #1e2d45;
    --pitcher: #e84b4b;
    --batter: #4be8a0;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* 메인 헤더 */
.main-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1525 100%);
    border-bottom: 2px solid var(--accent);
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    color: var(--accent);
    letter-spacing: 4px;
    line-height: 1;
    margin: 0;
}
.main-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: var(--text-muted) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 8px 20px !important;
    border: none !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.5rem 0 !important;
}

/* 카드 */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.card-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

/* 섹션 타이틀 */
.section-title {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* 필터 영역 */
.filter-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}

/* 선택 박스, 슬라이더 공통 */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}

/* 메트릭 */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.8rem !important;
    letter-spacing: 2px !important;
}

/* 데이터프레임 */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* 배지 */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-pitcher { background: rgba(232,75,75,0.15); color: var(--pitcher); border: 1px solid rgba(232,75,75,0.3); }
.badge-batter  { background: rgba(75,232,160,0.15); color: var(--batter);  border: 1px solid rgba(75,232,160,0.3); }

/* 툴 점수 칩 */
.tool-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 3px;
    font-size: 0.82rem;
}
.tool-score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 1px;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* 다운로드 버튼 */
.stDownloadButton > button {
    background: var(--accent) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* 버튼 */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* 넘버 인풋 */
.stNumberInput > div > div > input {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ── 메인 헤더 ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <p class="main-title">⚾ KBO DRAFT BOARD</p>
        <p class="main-subtitle">2027 신인 드래프트 스카우팅 대시보드</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────


from data import tracking_pitcher, tracking_batter_discipline, tracking_batter_hitrack
from mock_draft_tab import render_mock_draft_tab
from utils import load_raw

p_tools_df, b_tools_df = load_raw()

pitchers      = tracking_pitcher()
batters_dis   = tracking_batter_discipline()
batter_hitrack = tracking_batter_hitrack()

# ── 탭 ─────────────────────────────────────────────────────────────────────
tab_pitcher, tab_batter, tab_player_pitcher, tab_player_batter, tab_tools, tab_ptype, tab_draft = st.tabs([
    "🔴 투수 비교", "🟢 타자 비교", "⚪ 투수 트래킹", "🔵 타자 트래킹", "📊 툴 평가", "⚾︎ 구종 평가", "📥모의 드래프트"
])

# ══════════════════════════════════════════════════════
# TAB 1 : 투수
# ══════════════════════════════════════════════════════
with tab_pitcher:
    from tabs.tab_pitcher import render
    render(pitchers)

# ══════════════════════════════════════════════════════
# TAB 2 : 타자
# ══════════════════════════════════════════════════════
with tab_batter:
    from tabs.tab_batter import render
    render(batters_dis)
    
with tab_batter:
    from tabs.tab_batter import render2
    render2(batter_hitrack)

# ══════════════════════════════════════════════════════
# TAB 3 : 선수
# ══════════════════════════════════════════════════════
with tab_player_pitcher:
    from tabs.tab_player_pit import render as render_player_pit
    render_player_pit()

with tab_player_batter:
    from tabs.tab_player_bat import render as render_player_bat
    render_player_bat()
    
# ══════════════════════════════════════════════════════
# TAB 4 : 툴 평가
# ══════════════════════════════════════════════════════
with tab_tools:
    # 투수 툴
    from tabs.tab_tools_pitcher import render as render_pitcher_tools
    render_pitcher_tools(master, p_tools_df, b_tools_df)

    st.divider()

    # 타자 툴
    from tabs.tab_tools_batter import render as render_batter_tools
    render_batter_tools(master, p_tools_df, b_tools_df)

# ══════════════════════════════════════════════════════
# TAB 5 : 구종 평가
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
# TAB 6 : 모의 드래프트
# ══════════════════════════════════════════════════════
with tab_draft:
    render_mock_draft_tab()
