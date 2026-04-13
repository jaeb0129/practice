import streamlit as st
import pandas as pd

def render(data, profile):
    st.markdown('<p class="section-title">타자 데이터</p>', unsafe_allow_html=True)
    
    b_data = pd.merge(data, profile.loc[:,['PLER_TRKNG_ID','PLER_NAME', 'BKNO', 'TEAM_NM'] ], left_on='BatterId', right_on='PLER_TRKNG_ID', how='left')
    b_data = b_data[b_data["TEAM_NM"].notna()]
    
    # ── 필터 ──
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        school_opts = ["전체"] + sorted(b_data["TEAM_NM"].unique().tolist())
        school_sel = st.selectbox("학교 필터", school_opts, key="b_school")
    with col2:
        min_pa = 0
        max_pa = 200
        pa_min = st.slider("최소 타석수", min_pa, max_pa, min_pa, key="b_pa")
    with col3:
        year_opts = ["전체"] + sorted(b_data["year"].unique().tolist())
        year_sel = st.selectbox("연도 필터", year_opts, key="b_year")
    st.markdown('</div>', unsafe_allow_html=True)

    df = b_data.copy()
    if school_sel != "전체":
        df = df[(df["TEAM_NM"] == school_sel) & (df["year"] == year__sel)]
    df = df[(df["타석"] >= pa_min)]

    # ── 메인 테이블 ──
    st.markdown('<p class="section-title">타자 타석 접근법</p>', unsafe_allow_html=True)
    
    display_cols = ["PLER_NAME", "TEAM_NM", "BatterSide", "투구수", "타석", "타석당투구수", "BB%", "K%", "초구반응%", "반응%", "헛스윙%", "컨택%", "컨택%(2S)", "컨택%(145이상)", "존반응%", "존밖반응%", "존컨택%", "존밖컨택%"]
    show_df = df[display_cols].copy()
    col_map = {
    "PLER_NAME":   "선수명",
    "TEAM_NM":  "학교",
    "BatterSide":   "타석방향",
    "투구수":         "투구수",
    "타석":           "타석",
    "타석당투구수":    "타석당투구수",
    "BB%":           "BB%",
    "K%":            "K%",
    "초구반응%":      "초구반응%",
    "반응%":          "반응%",
    "헛스윙%":        "헛스윙%",
    "컨택%":          "컨택%",
    "컨택%(2S)":      "컨택%(2S)",
    "컨택%(145이상)": "컨택%(145이상)",
    "존반응%":        "존반응%",
    "존밖반응%":      "존밖반응%",
    "존컨택%":        "존컨택%",
    "존밖컨택%":      "존밖컨택%",
    }
    cols    = [c for c in col_map if c in df.columns]
    show_df = df[cols].rename(columns=col_map).reset_index(drop=True)
    show_df.index += 1

    st.dataframe(
        show_df.style
        .format({
            "타석당투구수": "{:.2f}", "BB%": "{:.1f}", "K%": "{:.1f}", "초구반응%": "{:.1f}", "반응%": "{:.1f}", "헛스윙%": "{:.1f}",
            "컨택%": "{:.1f}", "컨택%(2S)": "{:.1f}", "컨택%(145이상)": "{:.1f}",
            "존반응%": "{:.1f}", "존밖반응%": "{:.1f}", "존컨택%": "{:.1f}",
            "존밖컨택%": "{:.1f}"
        })
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#111827"),
                ("color", "#7a8499"),
                ("font-size", "0.72rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
                ("border-bottom", "2px solid #1e2d45"),
            ]},
            {"selector": "tbody tr", "props": [
                ("background-color", "#111827"),
                ("border-bottom", "1px solid #1e2d45"),
            ]},
            {"selector": "tbody tr:hover", "props": [
                ("background-color", "#1a2235"),
            ]},
            {"selector": "td", "props": [("padding", "8px 12px"), ("font-size", "0.85rem")]},
        ]),
        use_container_width=True,
        height=520
    )

def render2(data, profile):
    st.markdown('<p class="section-title">타자 데이터</p>', unsafe_allow_html=True)
    
    b_data = pd.merge(data, profile.loc[:,['PLER_TRKNG_ID','PLER_NAME', 'BKNO', 'TEAM_NM']], left_on='BatterId', right_on='PLER_TRKNG_ID', how='left')
    b_data = b_data[b_data["TEAM_NM"].notna()]
    
    # ── 필터 ──
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        school_opts = ["전체"] + sorted(b_data["TEAM_NM"].unique().tolist())
        school_sel = st.selectbox("학교 필터", school_opts, key="b_school2")
    with col2:
        min_pa = 0
        max_pa = 200
        pa_min = st.slider("최소 타석수", min_pa, max_pa, min_pa, key="b_pa2")
    with col3:
        year_opts = ["전체"] + sorted(b_data["year"].unique().tolist())
        year_sel = st.selectbox("연도 필터", year_opts, key="b_year2")
    st.markdown('</div>', unsafe_allow_html=True)

    df = b_data.copy()
    if school_sel != "전체":
        df = df[df["TEAM_NM"] == school_sel]
    df = df[(df["타석"] >= pa_min)]

    st.markdown('<p class="section-title">타자 타구 트래킹</p>', unsafe_allow_html=True)

    col_map = {
        "PLER_NAME":  "선수명",
        "TEAM_NM": "학교",
        "BatterSide":   "타석방향",
        "투구수":        "투구수",
        "타석":          "타석",
        "인플레이":      "인플레이",
        "최고타구속도":   "최고타구속도",
        "평균타구속도":   "평균타구속도",
        "평균발사각도":   "평균발사각도",
        "최고비거리":     "최고비거리",
        "하드힛%":       "하드힛%",
        "ICR%":         "정타%",
        "BARREL%":      "배럴%",
        "스윗스팟%":      "스윗스팟%",
        "땅볼%":         "땅볼%",
        "뜬공%":         "뜬공%",
        "라인드라이브%":  "라인드라이브%",
        "번트%":         "번트%",
    }
    cols    = [c for c in col_map if c in df.columns]
    show_df = df[cols].rename(columns=col_map).reset_index(drop=True)
    show_df.index += 1

    st.dataframe(
        show_df.style
        .format({
            "하드힛%": "{:.1f}", "정타%": "{:.1f}", "배럴%": "{:.1f}", "스윗스팟%": "{:.1f}",
            "땅볼%": "{:.1f}", "뜬공%": "{:.1f}", "라인드라이브%": "{:.1f}", "번트%": "{:.1f}",
            "평균발사각도": "{:.1f}", "평균타구속도": "{:.1f}", "최고타구속도": "{:.1f}", "최고비거리": "{:.1f}"
        })
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#111827"),
                ("color", "#7a8499"),
                ("font-size", "0.72rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
                ("border-bottom", "2px solid #1e2d45"),
            ]},
            {"selector": "tbody tr", "props": [
                ("background-color", "#111827"),
                ("border-bottom", "1px solid #1e2d45"),
            ]},
            {"selector": "tbody tr:hover", "props": [
                ("background-color", "#1a2235"),
            ]},
            {"selector": "td", "props": [("padding", "8px 12px"), ("font-size", "0.85rem")]},
        ]),
        use_container_width=True,
        height=520
    )
