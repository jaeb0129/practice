import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 스케일 함수
# ─────────────────────────────────────────────────────────────────────────────
 
def _scale_20_80(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return 20 + (series - mn) / (mx - mn) * 60
 
 
def _reverse_scale_20_80(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return 80 - (series - mn) / (mx - mn) * 60
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Raw → 집계 → Grade
# ─────────────────────────────────────────────────────────────────────────────
 
def _calc_flags(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["Plate_x"] = df["PlateLocSide"] * 100
    df["Plate_z"] = df["PlateLocHeight"] * 100
 
    play_hit = ["Single", "Double", "Triple", "HomeRun",
                "Error", "FieldersChoice", "Out", "Sacrifice"]
    play_ab  = ["Single", "Double", "Triple", "HomeRun",
                "Error", "FieldersChoice", "Out", "Strikeout"]
 
    df["PA_CHECK"] = (
        df["PlayResult"].isin(play_hit) |
        ((df["PlayResult"] == "Undefined") & df["KorBB"].isin(["Strikeout", "Walk", "IBB"]))
    ).astype(int)
 
    df["AB_CHECK"] = (
        df["PlayResult"].isin(play_ab) |
        ((df["PlayResult"] == "Undefined") & (df["KorBB"] == "Strikeout"))
    ).astype(int)
 
    es          = pd.to_numeric(df["ExitSpeed"], errors="coerce")
    exit_valid  = es.notna()
 
    df["INPLAY_CHECK"]  = ((df["PitchCall"] == "InPlay") & exit_valid).astype(int)
    df["HIT_TRY_CHECK"] = df["PitchCall"].isin(["FoulBall", "StrikeSwinging", "InPlay"]).astype(int)
    df["WHIFF_CHECK"]   = (df["PitchCall"] == "StrikeSwinging").astype(int)
    df["SO_CHECK"]      = (df["KorBB"] == "Strikeout").astype(int)
    df["BB_CHECK"]      = df["KorBB"].isin(["Walk", "IBB"]).astype(int)
 
    in_zone = (
        (df["Plate_z"] >= 43) & (df["Plate_z"] <= 107) &
        (df["Plate_x"] >= -30) & (df["Plate_x"] <= 30)
    )
    swing   = df["PitchCall"].isin(["FoulBall", "StrikeSwinging", "InPlay"])
    contact = df["PitchCall"].isin(["InPlay", "FoulBall"])
 
    df["LOC_IN_CHECK"]  = (in_zone).astype(int)
    df["LOC_IN_SWING_CHECK"]  = (swing & in_zone).astype(int)
    df["ZONE_CONTACT_CHECK"]  = (contact & in_zone).astype(int)
    df["LOC_OUT_CHECK"]       = (~in_zone).astype(int)
    df["CHASE_CHECK"]         = (swing & ~in_zone).astype(int)
    df["CHASE_CONTACT_CHECK"] = (swing & contact & ~in_zone).astype(int)
 
    df["HardHit_CHECK"] = (
        (es >= 152.88768) & exit_valid & (df["PitchCall"] == "InPlay")
    ).astype(int)
 
    ev  = es / 1.609344          # km/h → mph
    ang = pd.to_numeric(df["Angle"], errors="coerce")
    ip  = (df["PitchCall"] == "InPlay") & exit_valid
 
    cond6  = ip & (ev*1.5 - ang >= 117) & (ev + ang >= 124) & (ev >= 98) & ang.between(4, 50)
    cond5  = ip & (ev*1.5 - ang >= 111) & (ev + ang >= 119) & (ev >= 95) & ang.between(0, 52)
    cond1  = ip & (ev <= 59)
    cond4a = ip & (ev*2 - ang >= 87) & (ang <= 41) & (ev*2 + ang <= 175) & (ev + ang*1.3 >= 89) & ev.between(59, 72)
    cond4b = ip & (ev + ang*1.3 <= 112) & (ev + ang*1.55 >= 92) & (ev >= 72) & (ev >= 86)
    cond4c = ip & (ang <= 20) & (ev + ang*2.4 >= 98) & ev.between(86, 95)
    cond4d = ip & (ev - ang >= 76) & (ev + ang*2.4 >= 98) & (ev >= 95) & (ang <= 30)
    cond3  = ip & (ev + ang*2 >= 116)
    cond2  = ip & (ev + ang*2 <  116)
 
    sa = pd.Series(0, index=df.index)
    sa[cond2]  = 2; sa[cond3] = 3
    sa[cond4a | cond4b | cond4c | cond4d] = 4
    sa[cond5]  = 5; sa[cond6] = 6; sa[cond1] = 1
    df["SpeedAngle"] = sa
    df["ICR"]        = np.where(sa.isin([4,5,6]), 1, np.where(sa.isin([1,2,3]), 0, np.nan))
 
    barrel = (
        ((es >= 156)    & ang.between(23.5, 32.9))  |
        ((es >= 159.25) & ang.between(21.15, 37.6)) |
        ((es >= 162.5)  & ang.between(21.15, 39.95))|
        ((es >= 165.75) & ang.between(18.8, 47.0))  |
        ((es >= 169)    & (ang >= 18.8))  |
        ((es >= 172.25) & (ang >= 16.45)) |
        ((es >= 175.5)  & (ang >= 14.1))  |
        ((es >= 178.75) & (ang >= 11.75)) |
        ((es >= 182)    & (ang >= 9.4))
    )
    df["BARREL"] = barrel.fillna(False).astype(int)
    return df
 
 
@st.cache_data(show_spinner=False)
def _build_grade(raw: pd.DataFrame) -> pd.DataFrame:
    """raw  → 선수별 Grade DataFrame 반환"""
    df = _calc_flags(raw)
 
    g      = df.groupby(["year","BatterId", "Batter"])
    pa     = g["PA_CHECK"].sum()
    swing  = g["HIT_TRY_CHECK"].sum()
    inplay = g["INPLAY_CHECK"].sum()
 
    res = pd.DataFrame({
        "PA":             pa,
        "INPLAY":         inplay,
        "HardHit%":       (g["HardHit_CHECK"].sum() / inplay * 100).round(1),
        "ICR%":           (g["ICR"].sum()            / inplay * 100).round(1),
        "BARREL%":        (g["BARREL"].sum()          / inplay * 100).round(1),
        "WHIFF%":         (g["WHIFF_CHECK"].sum()     / swing  * 100).round(1),
        "K%":             (g["SO_CHECK"].sum()         / pa     * 100).round(1),
        "BB%":            (g["BB_CHECK"].sum()         / pa     * 100).round(1),
        "ZONE_SWING%":  (g["LOC_IN_SWING_CHECK"].sum()  / g["LOC_IN_CHECK"].sum() * 100).round(1),
        "ZONE_CONTACT%":  (g["ZONE_CONTACT_CHECK"].sum()  / g["LOC_IN_SWING_CHECK"].sum() * 100).round(1),
        "CHASE%":         (g["CHASE_CHECK"].sum()         / g["LOC_OUT_CHECK"].sum()      * 100).round(1),
        "CHASE_CONTACT%": (g["CHASE_CONTACT_CHECK"].sum() / g["CHASE_CHECK"].sum()        * 100).round(1),
    }).reset_index()
 
    res = res[res["INPLAY"] >= 0].copy()
 
    # 20-80 스케일
    res["scale_hardhit"]       = _scale_20_80(res["HardHit%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_icr"]           = _scale_20_80(res["ICR%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_barrel"]        = _scale_20_80(res["BARREL%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_WHIFF%"]        = _reverse_scale_20_80(res["WHIFF%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_K%"]            = _reverse_scale_20_80(res["K%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_BB%"]           = _scale_20_80(res["BB%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_ZONE_SWING%"]   = _scale_20_80(res["ZONE_SWING%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_ZONE_CONTACT%"] = _scale_20_80(res["ZONE_CONTACT%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_CHASE%"]        = _reverse_scale_20_80(res["CHASE%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
    res["scale_CHASE_CONTACT%"]= _scale_20_80(res["CHASE_CONTACT%"]).round(0).replace([np.inf, -np.inf], np.nan).fillna(50).astype(int)
 
    # Grade
    res["Grade_INPLAY"]           = ((res["scale_hardhit"] + res["scale_icr"]) / 2).round(0).astype(int)
    
    grade = ((res["scale_WHIFF%"] + res["scale_CHASE%"] + res["ZONE_SWING%"]) / 3).round(0)
    grade = grade.replace([np.inf, -np.inf], np.nan)    # inf를 NaN으로 치환
    grade = grade.fillna(0)                            # NaN을 0으로 대체 (또는 다른 값)
    res["Grade_PLATE_DISCIPLINE"] = grade.astype(int)
    
    #res["Grade_PLATE_DISCIPLINE"] = ((res["scale_WHIFF%"]  + res["scale_CHASE%"] + res["ZONE_SWING%"]) / 3).round(0).astype(int)
    res["Grade"]                  = (res["Grade_INPLAY"] * 0.5 + res["Grade_PLATE_DISCIPLINE"] * 0.5).round(1)
    
    return res
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 레이더 차트
# ─────────────────────────────────────────────────────────────────────────────
 
# 레이더에 표시할 툴 7개 (레이블 → 컬럼명)
RADAR_COLS = {
    "하드힛\n(HardHit)":  "scale_hardhit",
    "정타\n(ICR)":   "scale_icr",
    "배럴\n(Barrel)":     "scale_barrel",
    "컨택\n(헛스윙↓)":     "scale_WHIFF%",
    "존 컨택":             "scale_ZONE_CONTACT%",
    "존 스윙":             "scale_ZONE_SWING%",
    "존 밖\n반응":        "scale_CHASE%",
}
 
_COLORS = ["#38bdf8", "#818cf8", "#34d399", "#fb923c", "#f472b6", "#00ff00", "#facc15"]
 
 
def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
 
 
def _draw_radar(grade_df: pd.DataFrame, player_ids: list) -> go.Figure:
    cats   = list(RADAR_COLS.keys())
    angles = cats + [cats[0]]
    fig    = go.Figure()
 
    for i, pid in enumerate(player_ids):
        row   = grade_df[grade_df["BatterId"] == pid].iloc[0]
        vals  = [int(row[RADAR_COLS[c]]) for c in cats]
        color = _COLORS[i % len(_COLORS)]
        r, g, b = _hex_to_rgb(color)
 
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=angles,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.13)",
            line=dict(color=color, width=2.5),
            name=row["Batter"],
            hovertemplate="<b>%{theta}</b><br>%{r:.0f}<extra></extra>",
        ))
 
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 85],
                tickvals=[20, 40, 60, 80],
                tickfont=dict(size=10, color="#6b7280"),
                gridcolor="#1e2d45",
                linecolor="#1e2d45",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#cbd5e1",
                              family="'Noto Sans KR', sans-serif"),
                gridcolor="#1e2d45",
                linecolor="#1e2d45",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(color="#cbd5e1", size=12),
            bgcolor="rgba(17,24,39,0.85)",
            bordercolor="#1e2d45",
            borderwidth=1,
        ),
        margin=dict(l=70, r=70, t=40, b=40),
        height=460,
    )
    return fig
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 섹션별 렌더 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
 
def _sec(title: str):
    st.markdown(
        f'<p style="font-size:1rem;letter-spacing:.18em;text-transform:uppercase;'
        f'color:#38bdf8;font-weight:700;margin:1.4rem 0 .5rem;'
        f'border-left:3px solid #38bdf8;padding-left:.6rem">{title}</p>',
        unsafe_allow_html=True,
    )
 
 
def _grade_color(val: float) -> str:
    if val >= 65: return "#f87171"   # 빨강 (최상위)
    if val >= 50: return "#34d399"   # 초록
    if val >= 35: return "#FFFFFF"   # 검정(기본)
    return "#38bdf8"                 # 파랑 (최하위)


def _style_grade(val):
    try:
        v = float(val)
        if v >= 65: return "color:#f87171;font-weight:700"
        if v >= 50: return "color:#34d399;font-weight:700"
        if v >= 35: return "color:#1f1e33"
        return "color:#38bdf8"
    except:
        return ""
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 서브탭 1 : 레이더 차트
# ─────────────────────────────────────────────────────────────────────────────
 
def _render_radar(grade_df: pd.DataFrame, display_df: pd.DataFrame):
    _sec("선수 툴 스카우팅 레이더 (20-80 스케일) / 인플레이 10개 이상 대상 / 배럴%, 존 컨택%는 최종 점수 계산에서 제외")
    
    _sec("타구 점수 = (하드힛 (HardHit) + 정타 (ICR)) / 2")
    
    _sec("선구 점수 = (컨택 (헛스윙↓) + 존 밖 반응 + 존반응) / 3")
    
    _sec("계산 불가시 50점 부여")

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        year_opts = ["전체"] + sorted(display_df["year"].unique().tolist())
        year_sel = st.selectbox("연도 필터", year_opts, key="tools_r_year")
        
    filtered = display_df.copy()
    if year_sel != "전체":
        filtered = filtered[(filtered["year"] == year_sel)]
        
    with c2:
        school_opts = ["전체"] + sorted(display_df["TEAM_NM"].dropna().unique().tolist())
        school_sel  = st.selectbox("학교 필터", school_opts, key="tools_r_school")
    
    filtered = display_df.copy()
    if school_sel != "전체":
        filtered = filtered[(filtered["TEAM_NM"] == school_sel)]
        
    with c3:
        player_map = {
            row["BatterId"]: f"{row['PLER_NAME']}  ({row.get('TEAM_NM', '—')})"
            for _, row in filtered.iterrows()
        }
        # 이름 기준으로 오름차순 정렬
        sorted_keys = sorted(player_map.keys(), key=lambda x: player_map[x])
        
        selected = st.multiselect(
            "선수 선택 (최대 6명 동시 비교)",
            options=sorted_keys,
            format_func=lambda x: player_map[x],
            default=sorted_keys[:min(1, len(sorted_keys))],
            max_selections=6,
            key="tools_r_players",
        )
 
    if not selected:
        st.info("선수를 1명 이상 선택하세요.")
        return
 
    # 레이더
    fig = _draw_radar(grade_df, selected)
    st.plotly_chart(fig, use_container_width=True)
 
    # 선수 카드
    _sec("선택 선수 상세 지표")
    for i, pid in enumerate(selected):
        row  = grade_df[grade_df["BatterId"] == pid].iloc[0]
        prow = display_df[display_df["BatterId"] == pid].iloc[0]
        name  = prow.get("PLER_NAME", row["Batter"])
        team  = prow.get("TEAM_NM", "—")
        color = _COLORS[i % len(_COLORS)]

        # 타격 Tool(스탯) + 타석 접근법 Plate Approach(스탯) 란으로 구분
        tool_html = "".join([
                f'<div style="text-align:center">'
                f'  <div style="font-family:\'JetBrains Mono\',monospace;font-size:1.5rem;'
                f'font-weight:700;color:#fb923c">{int(row[RADAR_COLS[c]])}</div>'   # 괄호 수정
                f'  <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;'
                f'letter-spacing:.07em;margin-top:.1rem">{c.replace(chr(10)," ")}</div>'
        '</div>'
        for c in list(RADAR_COLS.keys())[:3]])

        approach_html = "".join([
                f'<div style="text-align:center">'
                f'  <div style="font-family:\'JetBrains Mono\',monospace;font-size:1.5rem;'
                f'font-weight:700;color:#38bdf8">{int(row[RADAR_COLS[c]])}</div>'   # 괄호 수정
                f'  <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;'
                f'letter-spacing:.07em;margin-top:.1rem">{c.replace(chr(10)," ")}</div>'
                f'</div>' for c in list(RADAR_COLS.keys())[3:]])

        gi_color  = _grade_color(row["Grade_INPLAY"])
        gpd_color = _grade_color(row["Grade_PLATE_DISCIPLINE"])
        g_color   = _grade_color(row["Grade"])

        hardhit = f'{row["HardHit%"]:.1f}' if pd.notna(row["HardHit%"]) else "—"
        icr   = f'{row["ICR%"]:.1f}' if pd.notna(row["ICR%"]) else "—"
        hard_pct     = f'{row["BARREL%"]:.1f}'        if pd.notna(row["BARREL%"]) else "—"
        whiff_pct  = f'{row["WHIFF%"]:.1f}'     if pd.notna(row["WHIFF%"]) else "—"
        zone_swing       = f'{row["ZONE_SWING%"]:.1f}'          if pd.notna(row["ZONE_SWING%"]) else "—"
        chase_pct      = f'{row["CHASE%"]:.1f}'           if pd.notna(row["CHASE%"]) else "—"

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:.7rem">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:.9rem">
            <div>
              <span style="font-weight:700;font-size:1rem;color:{color}">{name}</span>
              <span style="margin-left:.6rem;font-size:.72rem;color:#6b7280">{team}</span>
            </div>
            <span style="font-size:.72rem;color:#6b7280">{int(row['PA'])} 타석</span>
          </div>
          <div style="font-size:.6rem;color:#fb923c;text-transform:uppercase;
                      letter-spacing:.15em;margin-bottom:.4rem">▸ 타격 Tool</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;
                      margin-bottom:.3rem">{tool_html}</div>
          <div style="font-size:.65rem;color:#4b5563;text-align:center;margin-bottom:.8rem">
            하드힛 {hardhit} &nbsp;·&nbsp; 정타 {icr} &nbsp;·&nbsp;
          </div>
          <div style="font-size:.6rem;color:#38bdf8;text-transform:uppercase;
                      letter-spacing:.15em;margin-bottom:.4rem">▸ 타석 접근법 Plate Approach</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;
                      margin-bottom:.3rem">{approach_html}</div>
          <div style="font-size:.65rem;color:#4b5563;text-align:center;margin-bottom:.8rem">
            헛스윙% {whiff_pct}% &nbsp;·&nbsp; 존 반응% {zone_swing}% &nbsp;·&nbsp; 존 밖 반응% {chase_pct}% &nbsp;
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;
                      border-top:1px solid #1e2d45;padding-top:.7rem;text-align:center">
            <div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                          font-weight:700;color:{gi_color}">{int(row['Grade_INPLAY'])}</div>
              <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;
                          letter-spacing:.07em">타구 점수</div>
            </div>
            <div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                          font-weight:700;color:{gpd_color}">{int(row['Grade_PLATE_DISCIPLINE'])}</div>
              <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;
                          letter-spacing:.07em">선구 점수</div>
            </div>
            <div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;
                          font-weight:700;color:{g_color}">{row['Grade']:.1f}</div>
              <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;
                          letter-spacing:.07em">종합점수</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 서브탭 2 : 전체 성적표
# ─────────────────────────────────────────────────────────────────────────────
 
def _render_table(grade_df: pd.DataFrame, display_df: pd.DataFrame):
    _sec("전체 선수 종합 성적표  ·  인플레이 10회 이상")
 
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        opts       = ["전체"] + sorted(display_df["TEAM_NM"].dropna().unique().tolist())
        school_sel = st.selectbox("학교 필터", opts, key="tools_t_school")
    with c2:
        min_pa = int(display_df["PA"].min())
        max_pa = int(display_df["PA"].max())
        pa_min = st.slider("최소 타석수", 10, max_pa, 10, key="tools_t_pa")
        
    with c3:
        min_inplay = int(display_df["INPLAY"].min())
        max_inplay = int(display_df["INPLAY"].max())
        pa_min = st.slider("최소 인플레이", 10, max_inplay, 10, key="tools_t_inplay")
 
    df = display_df.copy()
    if school_sel != "전체":
        df = df[df["TEAM_NM"] == school_sel]
    df = df[df["INPLAY"] >= 10].sort_values("Grade", ascending=False)
 
    col_map = {
        "player_name":            "선수명",
        "TEAM_NM":           "학교",
        "PA":                     "타석",
        "INPLAY":                 "인플레이",
        "HardHit%":               "HardHit%",
        "ICR%":                   "ICR%",
        "BARREL%":                "BARREL%",
        "WHIFF%":                 "WHIFF%",
        "ZONE_SWING%":            "ZONE_SWING%",
        "CHASE%":                 "CHASE%",
        "Grade_INPLAY":           "타구점수",
        "Grade_PLATE_DISCIPLINE": "선구점수",
        "Grade":                  "종합점수",
    }
    cols    = [c for c in col_map if c in df.columns]
    show_df = df[cols].rename(columns=col_map).reset_index(drop=True)
    show_df.index += 1
 
    pct_fmt = {c: "{:.1f}" for c in ["HardHit%","ICR%","BARREL%","WHIFF%","ZONE_SWING%","CHASE%"] if c in show_df.columns}
    pct_fmt["종합점수"] = "{:.1f}"
 
    styled = (
        show_df.style
        .format(pct_fmt)
        .applymap(_style_grade, subset=["종합점수"])
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color","#111827"), ("color","#7a8499"),
                ("font-size","0.7rem"), ("text-transform","uppercase"),
                ("letter-spacing","1px"), ("border-bottom","2px solid #1e2d45"),
            ]},
            {"selector": "tbody tr",
             "props": [("background-color","#111827"),("border-bottom","1px solid #1e2d45")]},
            {"selector": "tbody tr:hover",
             "props": [("background-color","#1a2235")]},
            {"selector": "td",
             "props": [("padding","7px 10px"),("font-size","0.82rem")]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=540)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 공개 진입점
# ─────────────────────────────────────────────────────────────────────────────
 
def render(players_df: pd.DataFrame, p_tools_df, b_tools_df: pd.DataFrame):
 
    # ── 집계 ──────────────────────────────────────────────────────────────────
    with st.spinner("툴 데이터 계산 중…"):
        # b_tools_df 가 이미 flags 계산된 경우를 대비해 컬럼 체크
        if "PA_CHECK" not in b_tools_df.columns:
            b_tools_df = _calc_flags(b_tools_df)
        grade_df = _build_grade(b_tools_df)
 
    # ── 프로필 병합 ────────────────────────────────────────────────────────────
    profile_cols = [c for c in ["PLER_TRKNG_ID","PLER_NAME","TEAM_NM"]
                    if c in players_df.columns]
    display_df = pd.merge(
        grade_df,
        players_df[profile_cols],
        left_on="BatterId", right_on="PLER_TRKNG_ID", how="left"
    )
    # fallback
    if "PLER_NAME"  not in display_df.columns: display_df["PLER_NAME"]  = display_df["Batter"]
    if "TEAM_NM" not in display_df.columns: display_df["TEAM_NM"] = "—"
 
    # ── 서브탭 ────────────────────────────────────────────────────────────────
    sub1, sub2 = st.tabs(["📡  레이더 차트", "📋  전체 성적표"])
    with sub1:
        _render_radar(grade_df, display_df)
    with sub2:
        _render_table(grade_df, display_df)# tabs/tab_tools.py
