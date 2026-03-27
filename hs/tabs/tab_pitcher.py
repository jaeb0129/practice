import streamlit as st
import pandas as pd

def render(data, profile):
    st.markdown('<p class="section-title">투수 데이터</p>', unsafe_allow_html=True)
    
    p_data = pd.merge(data, profile.loc[:,['PLER_ID','PLER_NAME_KOR', 'BKNO', 'TEAM_NM'] ], left_on='PitcherId', right_on='PLER_ID', how='left')

    # ── 필터 ──
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        school_opts = ["전체"] + sorted(profile["TEAM_NM"].unique().tolist())
        school_sel = st.selectbox("학교 필터", school_opts, key="p_school")
    with col2:
        min_p = int(p_data["투구수"].min())
        max_p = int(p_data["투구수"].max())
        pitch_range = st.slider("최소 투구수", min_p, max_p, min_p, key="p_pitches")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    df = p_data.copy()
    if school_sel != "전체":
        df = df[df["TEAM_NM"] == school_sel]
    df = df[df["투구수"] >= pitch_range]

    # ── 메인 테이블 ──
    st.markdown('<p class="section-title">투수 타석 접근법</p>', unsafe_allow_html=True)
    col_map = {
    "PLER_NAME_KOR":  "선수명",
    "TEAM_NM": "학교",
    "PitcherThrows": "투구손",
    "투구수":        "투구수",
    "타석":          "타석",
    "타석당투구수":   "타석당투구수",
    "Zone%":        "Zone%",
    "초구S%":        "초구S%",
    "S%":           "S%",
    "CSW%":         "CSW%",
    "BB%":          "BB%",
    "K%":           "K%",
    "2S삼진결정%":   "2S삼진결정%",
    "루킹삼진%":     "루킹삼진%",
    "헛스윙%":       "헛스윙%",
    "반응%":         "반응%",
    "존반응%":       "존반응%",
    "존밖반응%":     "존밖반응%",
    "직구최고구속":   "직구최고구속",
    "직구평균구속":   "직구평균구속",
    }
    cols    = [c for c in col_map if c in df.columns]
    show_df = df[cols].rename(columns=col_map).reset_index(drop=True)
    show_df = show_df.sort_values("직구평균구속", ascending=False).reset_index(drop=True)
    show_df.index += 1

    styled = (
        show_df.style
        .format({
            "투구수": "{:.0f}",
            "타석": "{:.0f}",
            "타석당투구수": "{:.2f}",
            "Zone%": "{.1f}",
            "초구S%": "{.1f}",
            "S%": "{.1f}",
            "CSW%": "{.1f}",
            "BB%": "{.1f}",
            "K%": "{.1f}",
            "2S삼진결정%": "{.1f}",
            "루킹삼진%": "{.1f}",
            "헛스윙%": "{:.1f}",
            "반응%": "{:.1f}",
            "존반응%": "{:.1f}",
            "존밖반응%": "{:.1f}"
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
        ])
    )
    st.dataframe(show_df, use_container_width=True, height=520)
