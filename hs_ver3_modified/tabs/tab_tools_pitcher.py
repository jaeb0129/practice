# tabs/tab_tools_pitcher.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

MIN_PITCHES = 50

# ─────────────────────────────────────────────────────────────────────────────
# 20-80 스케일
# ─────────────────────────────────────────────────────────────────────────────

def _scale(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return 20 + (series - mn) / (mx - mn) * 60


def _rscale(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return 80 - (series - mn) / (mx - mn) * 60


# ─────────────────────────────────────────────────────────────────────────────
# 플래그 & 집계
# ─────────────────────────────────────────────────────────────────────────────

_FB_TYPES = {"Fastball", "FourSeamFastBall", "TwoSeamFastBall",
             "Sinker", "Cutter", "OneSeamFastBall"}

_ZONE_X_LO, _ZONE_X_HI = -30, 30
_ZONE_Z_LO, _ZONE_Z_HI =  43, 107


def _calc_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Plate_x"]    = pd.to_numeric(df["PlateLocSide"],   errors="coerce") * 100
    df["Plate_z"]    = pd.to_numeric(df["PlateLocHeight"],  errors="coerce") * 100
    df["RelSpeed_n"] = pd.to_numeric(df["RelSpeed"],  errors="coerce")
    df["SpinRate_n"] = pd.to_numeric(df["SpinRate"],  errors="coerce")

    in_zone  = (df["Plate_x"].between(_ZONE_X_LO, _ZONE_X_HI) &
                df["Plate_z"].between(_ZONE_Z_LO, _ZONE_Z_HI))
    swing    = df["PitchCall"].isin(["FoulBall", "StrikeSwinging", "InPlay"])
    whiff    = df["PitchCall"] == "StrikeSwinging"
    out_zone = ~in_zone

    df["_in_zone"]  = in_zone.astype(int)
    df["_out_zone"] = out_zone.astype(int)
    df["_swing"]    = swing.astype(int)
    df["_whiff"]    = whiff.astype(int)
    df["_chase"]    = (swing & out_zone).astype(int)
    
    def calc_shadow(row):
        
        Plate_x = row['Plate_x']
        Plate_z = row['Plate_z']
        
        # 조건 1
        cond1 = (13.3*2.54 >= Plate_x) and (-13.3*2.54 <= Plate_x) and (46*2.54 >= Plate_z) and (38*2.54 <= Plate_z)
        # 조건 2
        cond2 = (13.3*2.54 >= Plate_x) and (6.7*2.54 <= Plate_x) and (22*2.54 <= Plate_z) and (38*2.54 >= Plate_z)
        # 조건 3
        cond3 = (-6.7*2.54 >= Plate_x) and (-13.3*2.54 <= Plate_x) and (22*2.54 <= Plate_z) and (38*2.54 >= Plate_z)
        # 조건 4
        cond4 = (13.3*2.54 >= Plate_x) and (-13.3*2.54 <= Plate_x) and (14*2.54 <= Plate_z) and (22*2.54 >= Plate_z)

        # shadow: 4가지 중 하나라도 맞으면 1, 아니면 0
        return int(cond1 or cond2 or cond3 or cond4)

    # 데이터프레임에 shadow 컬럼 추가
    df['shadow'] = df.apply(calc_shadow, axis=1)

    pa_mask = (
        df["PlayResult"].isin(["Single","Double","Triple","HomeRun",
                               "Error","FieldersChoice","Out","Sacrifice"]) |
        ((df["PlayResult"] == "Undefined") & df["KorBB"].isin(["Strikeout","Walk","IBB"]))
    )
    df["_pa"] = pa_mask.astype(int)
    df["_bb"] = df["KorBB"].isin(["Walk", "IBB"]).astype(int)

    fb_col       = df.get("TaggedPitchType", df.get("AutoPitchType", pd.Series("", index=df.index)))
    df["_is_fb"] = fb_col.isin(_FB_TYPES)

    g       = df.groupby(['year',"PitcherId", "Pitcher"])
    pitches = g["RelSpeed_n"].count()
    pa      = g["_pa"].sum()
    swing_n = g["_swing"].sum()
    out_n   = g["_out_zone"].sum()

    fb_df = df[df["_is_fb"]]
    if len(fb_df) > 0:
        velo_mean = fb_df.groupby(['year',"PitcherId", "Pitcher"])["RelSpeed_n"].mean()
        spin_mean = fb_df.groupby(['year', "PitcherId", "Pitcher"])["SpinRate_n"].mean()
    else:
        velo_mean = g["RelSpeed_n"].mean()
        spin_mean = g["SpinRate_n"].mean()

    res = pd.DataFrame({
        "Pitches": pitches,
        "AvgVelo": velo_mean.round(1),
        "AvgSpin": spin_mean.round(0),
        "Zone%":   (g["_in_zone"].sum() / pitches * 100).round(1),
        "BB%":     (g["_bb"].sum()      / pa.replace(0, np.nan) * 100).round(1),
        "Whiff%":  (g["_whiff"].sum()   / swing_n.replace(0, np.nan) * 100).round(1),
        "Chase%":  (g["_chase"].sum()   / out_n.replace(0, np.nan)   * 100).round(1),
        "Edge%":    (g["shadow"].sum()   / pitches * 100).round(1)
    }).reset_index()

    res = res[res["Pitches"] >= MIN_PITCHES].copy()

    # ── NaN/inf 전처리 + 대체 발생 컬럼 추적 ────────────────────────────────
    _LABEL_MAP = {
        "AvgVelo": "평균구속", "AvgSpin": "평균회전",
        "Whiff%": "Whiff%", "Zone%": "Zone%",
        "BB%": "BB%", "Chase%": "Chase%",
        "Edge%": "Edge%"
    }
    imputed: dict = {}

    def _safe_fill(s: pd.Series, col: str) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        nan_idx = s[s.isna()].index
        median  = s.median()
        filled  = s.fillna(median if pd.notna(median) else 50.0)
        label   = _LABEL_MAP.get(col, col)
        for pid in nan_idx:
            imputed.setdefault(pid, []).append(label)
        return filled

    res = res.set_index("PitcherId")
    res["scale_velo"]  = _scale(_safe_fill(res["AvgVelo"], "AvgVelo")).round(0).astype(int)
    res["scale_spin"]  = _scale(_safe_fill(res["AvgSpin"], "AvgSpin")).round(0).astype(int)
    res["scale_whiff"] = _scale(_safe_fill(res["Whiff%"],  "Whiff%")).round(0).astype(int)
    res["scale_zone"]  = _scale(_safe_fill(res["Zone%"],   "Zone%")).round(0).astype(int)
    res["scale_bb"]    = _rscale(_safe_fill(res["BB%"],    "BB%")).round(0).astype(int)
    res["scale_chase"] = _scale(_safe_fill(res["Chase%"],  "Chase%")).round(0).astype(int)
    res["scale_edge"] = _scale(_safe_fill(res["Edge%"],  "Edge%")).round(0).astype(int)
    res = res.reset_index()

    res["imputed_cols"] = res["PitcherId"].map(
        lambda pid: ", ".join(imputed[pid]) if pid in imputed else ""
    )

    res["Grade_STUFF"]   = ((res["scale_velo"] + res["scale_spin"] + res["scale_whiff"]) / 3).round(0).astype(int)
    res["Grade_COMMAND"] = ((res["scale_zone"] + res["scale_bb"]   + res["scale_chase"] + res["scale_edge"]) / 4).round(0).astype(int)
    res["Grade"]         = (res["Grade_STUFF"] * 0.5 + res["Grade_COMMAND"] * 0.5).round(1)

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 레이더 차트
# ─────────────────────────────────────────────────────────────────────────────

RADAR_COLS = {
    "구속\n(Velo)":     "scale_velo",
    "회전수\n(Spin)":   "scale_spin",
    "헛스윙\n(Whiff)":  "scale_whiff",
    "존 투구\n(Zone)":   "scale_zone",
    "볼넷 억제\n(BB↓)": "scale_bb",
    "존 밖\n반응(Chase)":      "scale_chase",
    "엣지\n투구(Edge)":      "scale_edge",
}

_STUFF_CATS    = ["구속\n(Velo)", "회전수\n(Spin)", "헛스윙\n(Whiff)"]
_COMMAND_CATS  = ["존 투구\n(Zone)", "볼넷 억제\n(BB↓)", "존 밖\n반응(Chase)", "엣지\n투구(Edge)"]
_PLAYER_COLORS = ["#38bdf8", "#fb923c", "#34d399", "#818cf8", "#f472b6", "#facc15"]


def _hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_radar(grade_df: pd.DataFrame, player_ids: list) -> go.Figure:
    cats   = list(RADAR_COLS.keys())
    angles = cats + [cats[0]]
    fig    = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[85] * (len(_STUFF_CATS) + 1), theta=_STUFF_CATS + [_STUFF_CATS[0]],
        fill="toself", fillcolor="rgba(251,146,60,0.06)",
        line=dict(color="rgba(251,146,60,0.25)", width=1, dash="dot"),
        name="구위 영역", hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatterpolar(
        r=[85] * (len(_COMMAND_CATS) + 1), theta=_COMMAND_CATS + [_COMMAND_CATS[0]],
        fill="toself", fillcolor="rgba(56,189,248,0.06)",
        line=dict(color="rgba(56,189,248,0.25)", width=1, dash="dot"),
        name="커맨드 영역", hoverinfo="skip", showlegend=True,
    ))

    for i, pid in enumerate(player_ids):
        row   = grade_df[grade_df["PitcherId"] == pid].iloc[0]
        vals  = [int(row[RADAR_COLS[c]]) for c in cats]
        color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]
        r, g, b = _hex_rgb(color)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=angles,
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.14)",
            line=dict(color=color, width=2.5),
            name=row["Pitcher"],
            hovertemplate="<b>%{theta}</b><br>%{r:.0f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0,85], tickvals=[20,40,60,80],
                            tickfont=dict(size=10, color="#6b7280"),
                            gridcolor="#1e2d45", linecolor="#1e2d45"),
            angularaxis=dict(tickfont=dict(size=11, color="#cbd5e1",
                             family="'Noto Sans KR', sans-serif"),
                             gridcolor="#1e2d45", linecolor="#1e2d45"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#cbd5e1", size=11),
                    bgcolor="rgba(17,24,39,0.85)",
                    bordercolor="#1e2d45", borderwidth=1),
        margin=dict(l=80, r=80, t=40, b=40), height=480,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _sec(title: str):
    st.markdown(
        f'<p style="font-size:1rem;letter-spacing:.18em;text-transform:uppercase;'
        f'color:#fb923c;font-weight:700;margin:1.4rem 0 .5rem;'
        f'border-left:3px solid #fb923c;padding-left:.6rem">{title}</p>',
        unsafe_allow_html=True,
    )


def _grade_color(val: float) -> str:
    if val >= 65: return "#f87171"
    if val >= 50: return "#34d399"
    if val >= 35: return "#FFFFFF"
    return "#38bdf8"


def _style_grade(val):
    try:
        v = float(str(val).replace("*", ""))
        if v >= 65: return "color:#f87171;font-weight:700"
        if v >= 50: return "color:#34d399;font-weight:700"
        if v >= 35: return "color:#e2e8f0"
        return "color:#38bdf8"
    except:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 서브탭 1 : 레이더 차트
# ─────────────────────────────────────────────────────────────────────────────

def _render_radar(grade_df: pd.DataFrame, display_df: pd.DataFrame):
    _sec(f"투수 툴 스카우팅 레이더 (20-80 스케일 · 최소 {MIN_PITCHES}구 이상) / 존 밖 반응 점수 계산에서 제외")
    
    _sec("구위 점수 = (포심 구속 + 포심 회전수 + 헛스윙%)/ 3")
    
    _sec("커맨드 점수 = (존 투구 + 볼넷 억제 + 엣지 투구) / 3")
    
    _sec("계산 불가시 50점 부여")

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        year_opts = ["전체"] + sorted(display_df["year"].dropna().unique().tolist())
        year_sel  = st.selectbox("연도 필터", year_opts, key="pt_r_year")

    filtered = display_df.copy()
    if year_sel != "전체":
        filtered = filtered[filtered["year"] == year_sel]
    
    with c2:
        school_opts = ["전체"] + sorted(display_df["TEAM_NM"].dropna().unique().tolist())
        school_sel  = st.selectbox("학교 필터", school_opts, key="pt_r_school")

    filtered = display_df.copy()
    if school_sel != "전체":
        filtered = filtered[filtered["TEAM_NM"] == school_sel]

    with c2:
        player_map  = {
            row["PitcherId"]: f"{row['PLER_NAME']}  ({row.get('TEAM_NM','—')})"
            for _, row in filtered.iterrows()
        }
        sorted_keys = sorted(player_map.keys(), key=lambda x: player_map[x])
        selected = st.multiselect(
            "선수 선택 (최대 6명 동시 비교)",
            options=sorted_keys,
            format_func=lambda x: player_map[x],
            default=sorted_keys[:min(1, len(sorted_keys))],
            max_selections=6, key="pt_r_players",
        )

    fig = _draw_radar(grade_df, selected)
    st.plotly_chart(fig, use_container_width=True)

    lc1, lc2 = st.columns(2)
    lc1.markdown(
        '<div style="background:rgba(251,146,60,.08);border:1px solid rgba(251,146,60,.3);'
        'border-radius:8px;padding:.5rem .9rem;font-size:.75rem;color:#fb923c">'
        '🟠 <b>구위 (Stuff)</b> — 구속 · 회전수 · 헛스윙 유도</div>', unsafe_allow_html=True)
    lc2.markdown(
        '<div style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.3);'
        'border-radius:8px;padding:.5rem .9rem;font-size:.75rem;color:#38bdf8">'
        '🔵 <b>커맨드 (Command)</b> — 존 투구 · 볼넷 억제 · 존 밖 반응 · 엣지 투구</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    _sec("선택 선수 상세 지표")
    for i, pid in enumerate(selected):
        row  = grade_df[grade_df["PitcherId"] == pid].iloc[0]
        prow = display_df[display_df["PitcherId"] == pid].iloc[0]
        name  = prow.get("PLER_NAME", row["Pitcher"])
        team  = prow.get("TEAM_NM", "—")
        color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]

        stuff_html = "".join([
            f'<div style="text-align:center">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.5rem;'
            f'font-weight:700;color:#fb923c">{int(row[RADAR_COLS[c]])}</div>'
            f'<div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;'
            f'letter-spacing:.07em;margin-top:.1rem">{c.replace(chr(10)," ")}</div>'
            f'</div>'
            for c in list(RADAR_COLS.keys())[:3]
        ])
        cmd_html = "".join([
            f'<div style="text-align:center">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.5rem;'
            f'font-weight:700;color:#38bdf8">{int(row[RADAR_COLS[c]])}</div>'
            f'<div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;'
            f'letter-spacing:.07em;margin-top:.1rem">{c.replace(chr(10)," ")}</div>'
            f'</div>'
            for c in list(RADAR_COLS.keys())[3:]
        ])

        gs_color = _grade_color(row["Grade_STUFF"])
        gc_color = _grade_color(row["Grade_COMMAND"])
        g_color  = _grade_color(row["Grade"])

        avg_velo = f'{row["AvgVelo"]:.1f}' if pd.notna(row["AvgVelo"]) else "—"
        avg_spin = f'{int(row["AvgSpin"])}' if pd.notna(row["AvgSpin"]) else "—"
        zone_pct = f'{row["Zone%"]:.1f}'   if pd.notna(row["Zone%"])   else "—"
        bb_pct   = f'{row["BB%"]:.1f}'     if pd.notna(row["BB%"])     else "—"
        whiff    = f'{row["Whiff%"]:.1f}'  if pd.notna(row["Whiff%"])  else "—"
        chase    = f'{row["Chase%"]:.1f}'  if pd.notna(row["Chase%"])  else "—"
        edge    = f'{row["Edge%"]:.1f}'  if pd.notna(row["Edge%"])  else "—"

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:.7rem">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:.9rem">
            <div>
              <span style="font-weight:700;font-size:1rem;color:{color}">{name}</span>
              <span style="margin-left:.6rem;font-size:.72rem;color:#6b7280">{team}</span>
            </div>
            <span style="font-size:.72rem;color:#6b7280">{int(row['Pitches'])}구 투구</span>
          </div>
          <div style="font-size:.6rem;color:#fb923c;text-transform:uppercase;
                      letter-spacing:.15em;margin-bottom:.4rem">▸ 구위 Stuff</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;
                      margin-bottom:.3rem">{stuff_html}</div>
          <div style="font-size:.65rem;color:#4b5563;text-align:center;margin-bottom:.8rem">
            평균구속(FB) {avg_velo} km/h &nbsp;·&nbsp; 평균회전(FB) {avg_spin} rpm
            &nbsp;·&nbsp; Whiff {whiff}%
          </div>
          <div style="font-size:.6rem;color:#38bdf8;text-transform:uppercase;
                      letter-spacing:.15em;margin-bottom:.4rem">▸ 커맨드 Command</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;
                      margin-bottom:.3rem">{cmd_html}</div>
          <div style="font-size:.65rem;color:#4b5563;text-align:center;margin-bottom:.8rem">
            Zone {zone_pct}% &nbsp;·&nbsp; BB {bb_pct}% &nbsp;·&nbsp; Chase {chase}% &nbsp;·&nbsp; Edge {edge}%
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;
                      border-top:1px solid #1e2d45;padding-top:.7rem;text-align:center">
            <div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                          font-weight:700;color:{gs_color}">{int(row['Grade_STUFF'])}</div>
              <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;
                          letter-spacing:.07em">구위 점수</div>
            </div>
            <div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                          font-weight:700;color:{gc_color}">{int(row['Grade_COMMAND'])}</div>
              <div style="font-size:.62rem;color:#6b7280;text-transform:uppercase;
                          letter-spacing:.07em">커맨드 점수</div>
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

_IMPUTED_TO_COL = {
    "평균구속": "평균구속(FB)",
    "평균회전": "평균회전(FB)",
    "Whiff%":   "Whiff%",
    "Zone%":    "Zone%",
    "BB%":      "BB%",
    "Chase%":   "Chase%",
    "Edge%":    "Edge%"
}


def _render_table(grade_df: pd.DataFrame, display_df: pd.DataFrame):
    _sec(f"전체 투수 종합 성적표  ·  {MIN_PITCHES}구 이상")

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        year_opts = ["전체"] + sorted(display_df["year"].dropna().unique().tolist())
        year_sel = st.selectbox("연도 필터", year_opts, key="pt_t_year")
    with c2:
        school_opts = ["전체"] + sorted(display_df["TEAM_NM"].dropna().unique().tolist())
        school_sel = st.selectbox("학교 필터", school_opts, key="pt_t_school")
    with c4:
        min_p = int(display_df["Pitches"].min())
        max_p = int(display_df["Pitches"].max())
        p_min = st.slider(f"최소 투구수 (기준 {MIN_PITCHES}구)",
                          min_p, max_p, min_p, key="pt_t_pitches")
    with c4:
        # PitcherThrows 는 우투/좌투 순서 고정
        if "PitcherThrows" in display_df.columns:
            hand_opts = ["전체", "우투", "좌투"]
        else:
            hand_opts = ["전체"]
        hand_sel = st.selectbox("투구 손", hand_opts, key="pt_t_hand")

    df = display_df.copy()
    if year_sel != "전체" and "year" in df.columns:
        df = df[df["year"] == year_sel]    
    if school_sel != "전체":
        df = df[df["TEAM_NM"] == school_sel]
    if hand_sel != "전체" and "PitcherThrows" in df.columns:
        df = df[df["PitcherThrows"] == hand_sel]   
    
    df = df[df["Pitches"] >= p_min].sort_values("Grade", ascending=False)

    col_map = {
        "PLER_NAME":   "선수명",
        "TEAM_NM":  "학교",
        "PitcherThrows": "투구손",
        "Pitches":       "투구수",
        "AvgVelo":       "평균구속(FB)",
        "AvgSpin":       "평균회전(FB)",
        "Zone%":         "Zone%",
        "BB%":           "BB%",
        "Whiff%":        "Whiff%",
        "Chase%":        "Chase%",
        "Edge%":        "Edge%",
        "Grade_STUFF":   "구위점수",
        "Grade_COMMAND": "커맨드점수",
        "Grade":         "종합점수",
        "imputed_cols":  "_imputed",
    }
    cols    = [c for c in col_map if c in df.columns]
    show_df = df[cols].rename(columns=col_map).reset_index(drop=True)

    # ── 수치 포맷 문자열 변환 ──────────────────────────────────────────────────
    for col, fmt in [("평균구속(FB)", "{:.1f}"), ("평균회전(FB)", "{:.0f}"),
                     ("Zone%", "{:.1f}"), ("BB%", "{:.1f}"),
                     ("Whiff%", "{:.1f}"), ("Chase%", "{:.1f}"), ("Edge%", "{:.1f}"),
                     ("종합점수", "{:.1f}")]:
        if col in show_df.columns:
            show_df[col] = show_df[col].apply(
                lambda v: fmt.format(v) if pd.notna(v) else "—"
            )

    # ── 대체된 셀에 * 추가 ───────────────────────────────────────────────────
    if "_imputed" in show_df.columns:
        for idx, imp_str in show_df["_imputed"].items():
            if imp_str and len(str(imp_str)) > 0:
                for imp_label, disp_col in _IMPUTED_TO_COL.items():
                    if imp_label in str(imp_str) and disp_col in show_df.columns:
                        show_df.at[idx, disp_col] = str(show_df.at[idx, disp_col]) + "*"

    show_df.index += 1
    display_cols = [c for c in show_df.columns if c != "_imputed"]

    styled = (
        show_df[display_cols].style
        .map(_style_grade, subset=["종합점수"])
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

    # ── 하단 주석 박스 ────────────────────────────────────────────────────────
    if "_imputed" in show_df.columns:
        imp_src = df[["PLER_NAME", "imputed_cols"]].reset_index(drop=True)
        imp_src = imp_src[imp_src["imputed_cols"].str.len() > 0]
        if not imp_src.empty:
            items = "".join([
                f'<div style="margin:.15rem 0;font-size:.78rem">'
                f'<span style="color:#fbbf24;font-weight:600">{row["PLER_NAME"]}</span>'
                f'<span style="color:#6b7280"> — </span>'
                f'<span style="color:#9ca3af">'
                + ", ".join(f'{lbl}*' for lbl in row["imputed_cols"].split(", ")) +
                f'</span>'
                f'<span style="color:#4b5563;font-size:.72rem"> → FB 기록 없음, 전체 중앙값 대체</span>'
                f'</div>'
                for _, row in imp_src.iterrows()
            ])
            st.markdown(
                f'<div style="background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.25);'
                f'border-radius:8px;padding:.65rem 1rem;margin-top:.5rem">'
                f'<div style="font-size:.62rem;color:#fbbf24;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.12em;margin-bottom:.35rem">'
                f'* 중앙값 대체 항목</div>'
                f'{items}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 공개 진입점
# ─────────────────────────────────────────────────────────────────────────────

def render(players_df: pd.DataFrame, p_tools_df: pd.DataFrame, b_tools_df):
    with st.spinner("투수 툴 데이터 계산 중…"):
        grade_df = _calc_pitcher_stats(p_tools_df)

    if grade_df.empty:
        st.warning(f"조건을 만족하는 투수가 없습니다. (최소 {MIN_PITCHES}구)")
        return

    # ── 프로필 병합 ────────────────────────────────────────────────────────────
    prof_cols = [c for c in ["PLER_TRKNG_ID",'PLER_NAME',"TEAM_NM"]
                 if c in players_df.columns]

    grade_df['PitcherId'] = grade_df['PitcherId'].astype(float).astype(int)
    players_df['PLER_TRKNG_ID'] = pd.to_numeric(players_df['PLER_TRKNG_ID'], errors='coerce').astype('Int64')
    
    display_df = pd.merge(
        grade_df,
        players_df[prof_cols],
        left_on="PitcherId", right_on="PLER_TRKNG_ID", how="left",
    )

    # ── PitcherThrows: Right/Left → 우투/좌투, 순서 고정 ─────────────────────
    if "PitcherThrows" in p_tools_df.columns:
        hand = (
            p_tools_df.groupby("PitcherId")["PitcherThrows"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            .reset_index()
        )
        hand["PitcherThrows"] = hand["PitcherThrows"].map({"Right": "우투", "Left": "좌투"})
        hand["PitcherThrows"] = pd.Categorical(
            hand["PitcherThrows"], categories=["우투", "좌투"], ordered=True
        )

        hand['PitcherId'] = hand['PitcherId'].astype(float).astype(int)
        display_df = pd.merge(display_df, hand, on="PitcherId", how="left")

    # fallback
    if "PLER_NAME"  not in display_df.columns: display_df["PLER_NAME"]  = display_df["Pitcher"]
    if "TEAM_NM" not in display_df.columns: display_df["TEAM_NM"] = "—"

    # ── 서브탭 ────────────────────────────────────────────────────────────────
    sub1, sub2 = st.tabs(["📡  레이더 차트", "📋  전체 성적표"])
    with sub1:
        _render_radar(grade_df, display_df)
    with sub2:
        _render_table(grade_df, display_df)
