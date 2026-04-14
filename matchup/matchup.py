import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
from io import BytesIO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from scipy.interpolate import griddata
import base64

import plotly.io as pio
import db_func as dbf

def clean_numeric_column(series):
    """Convert a series to numeric, replacing non-numeric values with NaN"""
    return pd.to_numeric(series, errors='coerce')

def get_sklearn_components():
    """Import sklearn components only when needed"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics.pairwise import euclidean_distances
    from kneed import KneeLocator
    return StandardScaler, GaussianMixture, euclidean_distances, KneeLocator


st.subheader('파일 업로드(.csv)')

uploaded_file = st.file_uploader("파일 선택", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['투수명_ID'] = df['투수명'] + '_' + df['PCER_ID'].astype(str)
    df['타자명_ID'] = df['타자명'] + '_' + df['BTER_ID'].astype(str)
else:
    df = None
    st.warning("CSV 파일을 업로드해야 데이터가 표시됩니다.")

distance_threshold = 0.6
swing_calls = ["헛스윙", "파울", "타격"]

woba_weights = {
    '볼넷': 0.692,
    '고의4구': 0.692,
    '몸에맞는공': 0.723,
    '1루타': 0.885,
    '2루타': 1.257,
    '3루타': 1.593,
    '홈런': 2.053
}

color_dict = {
    "직구": "red",
    "싱커": "#FF33FF",
    "커터": "#336633",
    "슬라이더": "#009933",
    "스위퍼": "#999900",
    "체인지업": "blue",
    "스플리터": "#B266FF",
    "커브": "orange"
}


def run_silent_mac_analysis(pitcher_name, target_hitters):
    """Silent MAC analysis - same process, without text and just spin loader"""
    import db_func as dbf
    #STEP 1: Get Data + Filter by Handedness
    df = pd.read_csv(uploaded_file)
    #df = data
    
    df['투수명_ID'] = df['투수명'] +  '_' + df['PCER_ID'].astype(str)
    df['타자명_ID'] = df['타자명'] +  '_' + df['BTER_ID'].astype(str)
    
    df = df.assign(EXIT_VELOCITY_CHECK=np.where(df['PICH_JUDG'] == '타격',
                                                df['HTNG_SPD'], np.nan))
    
    # mph 변환
    df['HTNG_SPD_MI'] = pd.to_numeric(df['EXIT_VELOCITY_CHECK']) / 1.609344

    # SpeedAngle, ICR, Barrel 계산
    df['SpeedAngle'] = df.apply(dbf.calc_speed_angle, axis=1)
    df['ICR'] = np.where(df['SpeedAngle'].isin([4, 5, 6]), 1, 0)
    df['BARREL'] = df.apply(dbf.barrel, axis=1)
    
    if df.empty:
        return None, None, None

    # Filter by pitcher handedness right here (silent)
    if 'PCER_LNR' in df.columns:
        # Get the input pitcher's handedness
        pitcher_data = df[df["투수명_ID"] == pitcher_name]
        if not pitcher_data.empty and not pitcher_data['PCER_LNR'].isna().all():
            pitcher_throws = pitcher_data['PCER_LNR'].mode().iloc[0]  # Most common value
            
            # Filter entire dataset to only include same handedness (silent)
            df = df[df['PCER_LNR'] == pitcher_throws].copy()

    # Filter for pitcher's data only for clustering 
    pitcher_pitches = df[df["투수명_ID"] == pitcher_name].copy()
    if pitcher_pitches.empty:
        return None, None, None 
    
    
    #STEP 2: Clean Numeric Columns
    numeric_columns = [
            'RLSE_SPD', 'INDU_VTCL_MOVE', 'HORZ_MOVE', 'SPIN_CNT', 'RLSE_HGHT', 'RLSE_SIDE',
            'RUN_VALUE', 'HIT_SCORE', 'OUT_RSLT_CNT', 'HTNG_SPD', 'HTNG_ANGL', 'UPDO_LOC', 'LNR_LOC']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Check for required columns
    required_cols = ['RLSE_SPD', 'INDU_VTCL_MOVE', 'HORZ_MOVE', 'SPIN_CNT', 'RLSE_HGHT', 'RLSE_SIDE', 'AUTO_PTYPE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None, None, None
    
    #STEP 3: League Environment (have to redefine once again for silent iteration)
    LEAGUE_R_OUT = 0.193
    
    #STEP 5: Feature sets
    scanning_features = ['RLSE_SPD', 'INDU_VTCL_MOVE', 'HORZ_MOVE', 'SPIN_CNT', 'RLSE_HGHT', 'RLSE_SIDE']
    clustering_features = ['RLSE_SPD', 'INDU_VTCL_MOVE', 'HORZ_MOVE', 'SPIN_CNT', 'SPIN_AXIS']
    
    df = df.dropna(subset=scanning_features + ["투수명_ID", "타자명_ID"])
    pitcher_pitches = pitcher_pitches.dropna(subset=scanning_features + ["투수명_ID", "타자명_ID"])
    
    #STEP 6: Scale features and cluster pitcher's arsenal
    StandardScaler, GaussianMixture, euclidean_distances, KneeLocator = get_sklearn_components()
    
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(pitcher_pitches[clustering_features])
    
    # BIC loop to find optimal number of clusters
    bic_scores = []
    ks = range(4, 10)
    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_cluster)
        bic_scores.append(gmm.bic(X_cluster))
    
    # Find the "elbow" (knee point)
    knee = KneeLocator(ks, bic_scores, curve='convex', direction='decreasing')
    optimal_k = knee.elbow or 4
    
    # Fit final GMM and assign cluster labels
    best_gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    pitcher_pitches['PitchCluster'] = best_gmm.fit_predict(X_cluster)
    
    #STEP 7: Assign PitchGroup using pitch_name majority
    autopitchtype_to_group = {
            '직구': '직구',
            '싱커': '싱커',
            '커터': '커터',
            '슬라이더': '슬라이더',
            '스위퍼': '스위퍼',
            '체인지업': '체인지업',
            '스플리터': '스플리터',
            '커브': '커브'}
    
    pitcher_pitches = pitcher_pitches.dropna(subset=["AUTO_PTYPE"])
    
    cluster_to_type = {}
    for cluster in pitcher_pitches['PitchCluster'].unique():
        cluster_data = pitcher_pitches[pitcher_pitches['PitchCluster'] == cluster]
        type_counts = cluster_data['AUTO_PTYPE'].value_counts()
        
        if type_counts.empty:
            cluster_to_type[cluster] = 'Unknown'
            continue
        
        most_common_type = type_counts.idxmax()
        pitch_group = autopitchtype_to_group.get(most_common_type, 'Unknown')
        cluster_to_type[cluster] = pitch_group
    
    pitcher_pitches['PitchGroup'] = pitcher_pitches['PitchCluster'].map(cluster_to_type)
    pitch_group_usage = pitcher_pitches['PitchGroup'].value_counts(normalize=True).to_dict()

    #STEP 8: Tag FULL dataset with MinDistToPitcher
    scaler_all = StandardScaler()
    df_scaled = scaler_all.fit_transform(df[scanning_features])
    X_pitcher_full = scaler_all.transform(pitcher_pitches[scanning_features])
    distances = euclidean_distances(df_scaled, X_pitcher_full) # 전체 투구와 해당 투수 전체 공과의 거리
    df['MinDistToPitcher'] = distances.min(axis=1)
    
    # Assign PitchGroup to entire dataset using cluster model
    df_subset_scaled = scaler.transform(df[clustering_features])
    df['PitchCluster'] = best_gmm.predict(df_subset_scaled)
    df['PitchGroup'] = df['PitchCluster'].map(cluster_to_type)

    #STEP 9: Matchup scoring
    results = []
    group_breakdown = []

    for hitter in target_hitters:
        hitter_result = {"타자명_ID": hitter}
        weighted_stats = []
        total_weight = 0

        # Initialize accumulators for summary
        
        total_pitches_seen = 0
        total_swings_seen = 0
        total_whiffs_seen = 0
        total_abs_seen = 0
        
        total_ev_sum = 0
        total_la_sum = 0
        total_hard_hits = 0
        total_icr_hits = 0
        total_barrel_hits = 0
        
        total_gbs = 0
        total_fbs = 0
        total_lds = 0
        
        total_bips = 0
        total_bips_hittype = 0
        total_hits = 0
        total_hrs = 0
        total_slgs = 0
        total_outs = 0
        total_woba_num = 0
        total_woba_den = 0
        total_icr = 0
        total_barrel = 0

        for group, usage in pitch_group_usage.items():
            group_pitches = df[
                (df["타자명_ID"] == hitter) &
                (df["PitchGroup"] == group) &
                (df["MinDistToPitcher"] <= distance_threshold)
            ].copy()

            if group_pitches.empty:
                continue

            group_pitches["InZone"] = (
                 (group_pitches["LOC_CODE"].isin(['S11', 'S12', 'S13',
                                                  'S21', 'S22', 'S23',
                                                  'S31', 'S32', 'S33']) )
            )
            group_pitches["AB"] = group_pitches["AB_YN"] == "Y"
            group_pitches["Swung"] = group_pitches["PICH_JUDG"].isin(swing_calls)
            group_pitches["Whiff"] = group_pitches["PICH_JUDG"] == "헛스윙"
            group_pitches["Ishit_into_play"] = group_pitches['PICH_JUDG'].isin(["타격"])

            total_pitches = len(group_pitches)
            total_abs = group_pitches["AB"].sum()
            total_swings = group_pitches["Swung"].sum()
            total_whiffs = group_pitches["Whiff"].sum()
            total_run_value = group_pitches["RUN_VALUE"].sum() if 'RUN_VALUE' in group_pitches.columns else 0

            # Clean exit speed and angle columns
            group_pitches['HTNG_SPD'] = clean_numeric_column(group_pitches['HTNG_SPD'])
            group_pitches['HTNG_ANGL'] = clean_numeric_column(group_pitches['HTNG_ANGL'])

            balls_in_play = group_pitches[(group_pitches["Ishit_into_play"]) &
                                          (group_pitches["HTNG_SPD"].notna()) &
                                          (group_pitches["HTNG_ANGL"].notna()) &
                                          (group_pitches["PICH_JUDG"] == "타격")]
            hittype_in_play = group_pitches[(group_pitches["HIT_TYPE"].isin(['GroundBall', 'Popup', 'LineDrive', 'FlyBall']))]
            balls_with_ev = balls_in_play[balls_in_play["HTNG_SPD"].notna()]
            exit_velo = (balls_with_ev[group_pitches["PICH_JUDG"] == "타격"]["HTNG_SPD"]).mean() if len(balls_with_ev) > 0 else np.nan
            launch_angle = group_pitches[group_pitches["PICH_JUDG"] == "타격"]["HTNG_ANGL"].mean()

            num_ground_balls = (balls_in_play["HIT_TYPE"] == 'GroundBall').sum()
            num_fly_balls = (balls_in_play["HIT_TYPE"].isin(['Popup', 'FlyBall'])).sum()
            num_ld_balls = (balls_in_play["HIT_TYPE"] == 'LineDrive').sum()
            GB_percent = round(100 * num_ground_balls / len(hittype_in_play), 1) if len(hittype_in_play) > 0 else np.nan
            FB_percent = round(100 * num_fly_balls / len(hittype_in_play), 1) if len(hittype_in_play) > 0 else np.nan
            LD_percent = round(100 * num_ld_balls / len(hittype_in_play), 1) if len(hittype_in_play) > 0 else np.nan

            num_hard_hits = (balls_with_ev["HTNG_SPD"]*0.621371 >= 95).sum()
            num_icr_hits = (balls_with_ev['ICR'] == 1).sum()
            num_barrel_hits = (balls_with_ev["BARREL"] == 1).sum()
            
            hh_percent = round(100 * num_hard_hits / len(balls_with_ev), 1) if len(balls_with_ev) > 0 else np.nan
            icr_percent = round(100 * num_icr_hits / len(balls_with_ev), 1) if len(balls_with_ev) > 0 else np.nan
            barrel_percent = round(100 * num_barrel_hits / len(balls_with_ev), 1) if len(balls_with_ev) > 0 else np.nan

            rv_per_100 = 100 * total_run_value / total_pitches if total_pitches > 0 else 0
            weighted_stats.append(usage * rv_per_100)
            total_weight += usage

            # Calculate AVG for this group
            hit_mask = (
                 (group_pitches["PICH_JUDG"] == "타격") &
                 (group_pitches["EVENT"].isin(["1루타", "2루타", "3루타", "홈런"])))
            hits = hit_mask.sum()
            
            slg_mask1 = (
                 (group_pitches["PICH_JUDG"] == "타격") &
                 (group_pitches["EVENT"].isin(["1루타"])))
            slg_mask2 = (
                 (group_pitches["PICH_JUDG"] == "타격") &
                 (group_pitches["EVENT"].isin(["2루타"])))
            slg_mask3 = (
                 (group_pitches["PICH_JUDG"] == "타격") &
                 (group_pitches["EVENT"].isin(["3루타"])))
            slg_mask4 = (
                 (group_pitches["PICH_JUDG"] == "타격") &
                 (group_pitches["EVENT"].isin(["홈런"])))
            slgs = slg_mask1.sum()+slg_mask2.sum()*2+slg_mask3.sum()*3+slg_mask4.sum()*4
            
            hrs = slg_mask4.sum()

            out_mask = (
                 (group_pitches["EVENT"].isin(["삼진"])) |
                 ((group_pitches["PICH_JUDG"] == "타격") &
                  (group_pitches["EVENT"].isin(["아웃"]))) &
                 (group_pitches["EVENT"] != "희생번트") &
                 (group_pitches["EVENT"] != "희생플라이")
             )
            outs = out_mask.sum()

            avg = round(hits / (total_abs), 3) if (total_abs) > 0 else np.nan
            
            slg = round(slgs / (total_abs), 3) if (total_abs) > 0 else np.nan

            # Accumulate full pitch data for summary
            total_pitches_seen += total_pitches
            total_swings_seen += total_swings
            total_whiffs_seen += total_whiffs
            total_abs_seen += total_abs
            total_bips += len(balls_in_play)
            total_bips_hittype += len(hittype_in_play)
            total_hits += hits
            total_hrs += hrs
            total_slgs += slgs
            total_outs += outs
            if not np.isnan(exit_velo) and len(balls_with_ev) > 0:
                total_ev_sum += exit_velo * len(balls_with_ev)
            if not np.isnan(launch_angle):
                total_la_sum += launch_angle * len(balls_in_play)
            if not np.isnan(num_hard_hits):
                total_hard_hits += num_hard_hits
            if not np.isnan(num_hard_hits):
                total_icr_hits += num_icr_hits
            if not np.isnan(num_hard_hits):
                total_barrel_hits += num_barrel_hits 
            if not np.isnan(num_ground_balls):
                total_gbs += num_ground_balls
            if not np.isnan(num_fly_balls):
                total_fbs += num_fly_balls
            if not np.isnan(num_ld_balls):
                total_lds += num_ld_balls
            
            # 구종o
            group_breakdown.append({
                 "투수명_ID": pitcher_pitches['투수명'].unique()[0],
                 "타자명_ID": hitter,
                 "PitchGroup": group,
                 "투구수": total_pitches,
                 "구사율": round(usage, 2),
                 "타수": total_abs,
                 "인플레이": len(balls_in_play),
                 "안타": hits,
                 "홈런": hrs,
                 "타율": avg,
                 "장타율": slg,
                 "반응%": round(100 * total_swings / total_pitches, 1) if total_pitches > 0 else np.nan,
                 "컨택%": 100 - round(100 * total_whiffs / total_swings, 1) if total_swings > 0 else np.nan,
                 "헛스윙%": round(100 * total_whiffs / total_swings, 1) if total_swings > 0 else np.nan,
                 "하드힛%": hh_percent,
                 "정타%": icr_percent,
                 "배럴%": barrel_percent,
                 "타구속도": round(exit_velo, 1) if not np.isnan(exit_velo) else np.nan,
                 "발사각도": round(launch_angle, 1) if not np.isnan(launch_angle) else np.nan,
                 "GB%": GB_percent,
                 "FB%": FB_percent,
                 "LD%": LD_percent,
                 #"wOBA": group_woba,
                 "RV/100": round(rv_per_100, 2)
             })

        # Summary calculations
        weighted_rv = sum(weighted_stats) / total_weight if total_weight > 0 else np.nan
        hitter_result["투구수"] = total_pitches_seen
        hitter_result["타수"] = total_abs_seen
        hitter_result["인플레이"] = total_bips
        hitter_result["안타"] = total_hits,
        hitter_result["홈런"] = total_hrs,
        hitter_result["타율"] = round(total_hits / (total_abs_seen), 3) if (total_abs_seen) > 0 else np.nan
        hitter_result["장타율"] = round(total_slgs / (total_abs_seen), 3) if (total_abs_seen) > 0 else np.nan
        hitter_result["반응%"] = round(100 * total_swings_seen / total_pitches_seen, 1) if total_pitches_seen > 0 else np.nan
        hitter_result["컨택%"] = 100 - round(100 * total_whiffs_seen / total_swings_seen, 1) if total_swings_seen > 0 else np.nan
        hitter_result["헛스윙%"] = round(100 * total_whiffs_seen / total_swings_seen, 1) if total_swings_seen > 0 else np.nan
        hitter_result["하드힛%"] = round(100 * total_hard_hits / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["정타%"] = round(100 * total_icr_hits / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["배럴%"] = round(100 * total_barrel_hits / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["타구속도"] = round(total_ev_sum / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["발사각도"] = round(total_la_sum / total_bips, 1) if total_bips > 0 else np.nan
        hitter_result["GB%"] = round(100 * total_gbs / total_bips_hittype, 1) if total_bips > 0 else np.nan
        hitter_result["FB%"] = round(100 * total_fbs / total_bips_hittype, 1) if total_bips > 0 else np.nan
        hitter_result["LD%"] = round(100 * total_lds / total_bips_hittype, 1) if total_bips > 0 else np.nan
        #hitter_result["wOBA"] = round(total_woba_num / total_woba_den, 3) if total_woba_den > 0 else np.nan
        hitter_result["RV/100"] = round(weighted_rv, 2)
        
        results.append(hitter_result)
        
    return pd.DataFrame(results), pd.DataFrame(group_breakdown), df

def run_silent_mac_analysis_multiple_pitchers(pitcher_names, target_hitters):
    import db_func as dbf
    df = pd.read_csv(uploaded_file)
    #df = data
    results_all = []
    group_breakdown_all = []
    df_all = []

    for pitcher_name in pitcher_names:
        # 기존 run_silent_mac_analysis 함수의 pitcher_name 부분을 사용
        result_df, group_breakdown_df, df_pitcher = run_silent_mac_analysis(pitcher_name, target_hitters)

        if result_df is not None:
            result_df['투수명_ID'] = pitcher_name
            results_all.append(result_df)
        if group_breakdown_df is not None:
            group_breakdown_df['투수명_ID'] = pitcher_name
            group_breakdown_all.append(group_breakdown_df)
        if df_pitcher is not None:
            df_pitcher['투수명_ID'] = pitcher_name
            df_all.append(df_pitcher)

    # 결과 합치기
    results_concat = pd.concat(results_all, ignore_index=True) if results_all else None
    group_breakdown_concat = pd.concat(group_breakdown_all, ignore_index=True) if group_breakdown_all else None
    df_concat = pd.concat(df_all, ignore_index=True) if df_all else None
    
    cols = results_concat.columns.tolist()
    cols = ['투수명_ID'] + [col for col in cols if col != '투수명_ID']
    results_concat = results_concat[cols]

    return results_concat, group_breakdown_concat, df_concat


def main():
    st.title("⚾ 투타 맞대결 분석")

    # Selection interface
    col1, col2 = st.columns([1, 1])

    if df is not None:
        available_pitchers = sorted(df['투수명_ID'].unique())
        available_batters = sorted(df['타자명_ID'].unique())
    else:
        available_pitchers = []
        available_batters = []

    with col1:
        st.subheader("투수 선택")
        selected_pitchers = st.multiselect(
            "투수를 선택하세요:",
            available_pitchers
        )

    with col2:
        st.subheader("타자 선택")
        selected_hitters = st.multiselect(
            "타자를 선택하세요:",
            available_batters
        )
        
    # Analysis button - runs analysis and stores data
    if st.button("맞대결 분석 시작", type="primary", use_container_width=True):
        if not selected_pitchers or not selected_hitters:
            st.warning("Please select both a pitcher and at least one hitter.")
        else:
            st.markdown("---")
            
            try:
                summary_df, breakdown_df, full_df = run_silent_mac_analysis_multiple_pitchers(
                    selected_pitchers, selected_hitters)
                
                if summary_df is not None and not summary_df.empty:
                    # Store results in session state for persistence
                    st.session_state.summary_df = summary_df
                    st.session_state.breakdown_df = breakdown_df
                    st.session_state.selected_pitchers = selected_pitchers
                    st.session_state.selected_hitters = selected_hitters
                    
                    # Filter movement data for charts and store in session state
                    movement_df = full_df[
                        (full_df["타자명_ID"].isin(selected_hitters)) &
                        (full_df["MinDistToPitcher"] <= distance_threshold)
                    ].copy()
                    st.session_state.movement_df = movement_df
                    

                else:
                    st.warning("No sufficient data found for this matchup.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Display persistent results - OUTSIDE button block - making sure that different aspects of the app don't rely on each other
    if 'summary_df' in st.session_state and 'breakdown_df' in st.session_state:
        st.markdown("---")
        st.header("결과")
        
        # Data tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("전체 성적")
            st.dataframe(st.session_state.summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("구종별 성적")
            st.dataframe(st.session_state.breakdown_df, use_container_width=True, hide_index=True)
        
if __name__ == "__main__":
    main()
