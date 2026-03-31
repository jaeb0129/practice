import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from data_loader import get_raw_df, get_master_df

def render():
    rslt = get_raw_df()
    master = get_master_df()
    
    rslt['BatterId'] = rslt['BatterId'].astype(str)
    master['PLER_TRKNG_ID'] = master['PLER_TRKNG_ID'].astype(str)

    # ── 공통 전처리 (한 번만) ──────────────────────────────────────────────────
    base = pd.merge(rslt, master.loc[:,['PLER_TRKNG_ID','PLER_NAME', 'BKNO', 'TEAM_NM']],
                    left_on='BatterId', right_on='PLER_TRKNG_ID', how='left')
    base = base[base["TEAM_NM"].notna()]

    base["BKNO"] = base["BKNO"].apply(
    lambda x: str(int(float(x))) if pd.notnull(x) else "")
    
    base["name_bk"] = base["PLER_NAME"] + "_" + base["BKNO"].astype(str)
    base['year'] = pd.to_datetime(base['Date'], errors='coerce').dt.year

    st.sidebar.title('타자')
    select_year = st.sidebar.selectbox('확인하고 싶은 연도를 선택하세요',
                                         sorted(base.year.dropna().unique()),
                                         key="player_bat_year")
    select_school = st.sidebar.selectbox('확인하고 싶은 학교를 선택하세요',
                                         sorted(base[(base.year == select_year)].TEAM_NM.dropna().unique()),
                                         key="player_bat_school")
    select_batter = st.sidebar.selectbox('확인하고 싶은 타자를 선택하세요',
                                         sorted(base[(base.year == select_year) & (base.TEAM_NM == select_school)].name_bk.unique()),
                                         key="player_bat_batter")
    select_date   = st.sidebar.multiselect('확인하고 싶은 날짜를 선택하세요',
                                           base[(base.year == select_year) & (base.TEAM_NM == select_school) & (base.name_bk == select_batter)].Date.unique(),
                                           key="player_bat_date")
    

    st.title(f'{select_batter} 타격 대시보드')

    # ── 선수/날짜 필터 ────────────────────────────────────────────────────────
    filtered = base.loc[(base.name_bk == select_batter) &
                        (base.Date.isin(select_date)) &
                        (base.TEAM_NM == select_school) &
                        (base.year == select_year)].copy()

    # ── 내부 함수: 필터된 df 받아서 집계만 수행 ──────────────────────────────
    def tracking_batter_discipline(df):
        df = df.copy()
        df['Zone'] = ((df["PlateLocHeight"] <= 1.07) & (df["PlateLocHeight"] >= 0.43) &
                      (df["PlateLocSide"]   <= 0.3)  & (df["PlateLocSide"]   >= -0.3)).astype(int)

        df["is_pa"]      = (df["PlayResult"].isin(['Single','Double','Triple','HomeRun','Error',
                             'FieldersChoice','Out','Sacrifice']) | df["KorBB"].isin(['Walk','Strikeout'])).astype(int)
        df["is_swing"]   = df["PitchCall"].isin(["StrikeSwinging","InPlay","FoulBall"]).astype(int)
        df["is_whiff"]   = (df["PitchCall"] == "StrikeSwinging").astype(int)
        df["is_bb"]      = (df["KorBB"] == "Walk").astype(int)
        df["is_so"]      = (df["KorBB"] == "Strikeout").astype(int)
        df["is_zone"]    = (df["Zone"] == 1).astype(int)
        df["is_outzone"] = (df["Zone"] == 0).astype(int)
        df["is_z_swing"] = ((df["is_zone"] == 1) & (df["is_swing"] == 1)).astype(int)
        df["is_o_swing"] = ((df["is_outzone"] == 1) & (df["is_swing"] == 1)).astype(int)
        df["is_contact"] = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
        df["is_z_contact"] = ((df["is_zone"] == 1)    & (df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
        df["is_o_contact"] = ((df["is_zone"] == 0)    & (df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
        df["is_2s_swing"]    = ((df["Strikes"] == 2)  & (df["is_swing"] == 1)).astype(int)
        df["is_2s_contact"]  = ((df["Strikes"] == 2)  & (df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
        df["is_145_swing"]   = ((df["RelSpeed"] >= 145) & (df["is_swing"] == 1)).astype(int)
        df["is_145_contact"] = ((df["RelSpeed"] >= 145) & (df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
        df["is_first_pitch"]       = ((df["Strikes"] == 0) & (df["Balls"] == 0)).astype(int)
        df["is_first_pitch_swing"] = ((df["Strikes"] == 0) & (df["Balls"] == 0) & (df["is_swing"] == 1)).astype(int)

        result = (df.groupby(["BatterId",'PLER_NAME','TEAM_NM',"BatterSide"])
                  .agg(투구수=("PitchCall","count"), 타석=("is_pa","sum"),
                       반응=("is_swing","sum"), 초구반응=("is_first_pitch_swing","sum"),
                       초구=("is_first_pitch","sum"), whiffs=("is_whiff","sum"),
                       contacts=("is_contact","sum"), 볼넷=("is_bb","sum"), 삼진=("is_so","sum"),
                       zone_pitches=("is_zone","sum"), z_swings=("is_z_swing","sum"),
                       z_contacts=("is_z_contact","sum"), ozone_pitches=("is_outzone","sum"),
                       o_swings=("is_o_swing","sum"), o_contacts=("is_o_contact","sum"),
                       swings_2s=("is_2s_swing","sum"), contacts_2s=("is_2s_contact","sum"),
                       swings_145=("is_145_swing","sum"), contacts_145=("is_145_contact","sum"))
                  .reset_index())

        result["타석당투구수"]    = (result["투구수"] / result["타석"]).round(2)
        result["BB%"]            = (result["볼넷"]    / result["타석"] * 100).round(1)
        result["K%"]             = (result["삼진"]    / result["타석"] * 100).round(1)
        result["반응%"]          = (result["반응"]    / result["투구수"] * 100).round(1)
        result["초구반응%"]      = (result["초구반응"] / result["초구"] * 100).round(1)
        result["헛스윙%"]        = (result["whiffs"]  / result["반응"] * 100).round(1)
        result["컨택%"]          = (result["contacts"] / result["반응"] * 100).round(1)
        result["컨택%(2S)"]      = (result["contacts_2s"]  / result["swings_2s"]  * 100).round(1)
        result["컨택%(145이상)"] = (result["contacts_145"] / result["swings_145"] * 100).round(1)
        result["존반응%"]        = (result["z_swings"]   / result["zone_pitches"]  * 100).round(1)
        result["존밖반응%"]      = (result["o_swings"]   / result["ozone_pitches"] * 100).round(1)
        result["존컨택%"]        = (result["z_contacts"] / result["z_swings"] * 100).round(1)
        result["존밖컨택%"]      = (result["o_contacts"] / result["o_swings"] * 100).round(1)

        stats = result[['PLER_NAME','TEAM_NM',"BatterSide","투구수","타석",
                         "타석당투구수","BB%","K%","초구반응%","반응%","헛스윙%","컨택%",
                         "컨택%(2S)","컨택%(145이상)","존반응%","존밖반응%","존컨택%","존밖컨택%"]]
        stats = stats.rename(columns={'PLER_NAME':"선수명",'TEAM_NM':"학교",
                                       "BatterSide":"타석방향"})
        stats["타석방향"] = stats["타석방향"].map({"Right":"우타","Left":"좌타"})
        stats["타석방향"] = pd.Categorical(stats["타석방향"], categories=["우타","좌타"], ordered=True)
        return stats

    def tracking_batter_hitrack(df):
        df = df.copy()
        df["is_pa"]     = (df["PlayResult"].isin(['Single','Double','Triple','HomeRun','Error',
                            'FieldersChoice','Out','Sacrifice']) | df["KorBB"].isin(['Walk','Strikeout'])).astype(int)
        df["is_inplay"] = (df["PlayResult"].isin(['Single','Double','Triple','HomeRun','Error',
                            'FieldersChoice','Out']) & df["ExitSpeed"].notna()).astype(int)

        df["is_exitspeed"] = df.loc[df['ExitSpeed'].notna() & df['Angle'].notna() & df["is_inplay"].astype(bool), "ExitSpeed"]
        df["is_angle"]     = df.loc[df['ExitSpeed'].notna() & df['Angle'].notna() & df["is_inplay"].astype(bool), "Angle"]
        df["is_distance"]     = df.loc[df['ExitSpeed'].notna() & df['Angle'].notna() & df["is_inplay"].astype(bool), "Distance"]
        df["is_hardhit"]   = ((df['ExitSpeed'] >= 152.88768) & (df["is_inplay"] == 1)).astype(int)
        df["is_sweetspot"]   = ((df['Angle'] >= 8) & (df['Angle'] <= 32) & (df["is_inplay"] == 1)).astype(int)
        
        df['is_gb']    = (df['TaggedHitType'] == 'GroundBall').astype(int)
        df['is_fb']    = (df['TaggedHitType'].isin(['FlyBall','Popup'])).astype(int)
        df['is_ld']    = (df['TaggedHitType'] == 'LineDrive').astype(int)
        df['is_bunt']  = (df['TaggedHitType'] == 'Bunt').astype(int)
        df['is_hitype']= (df['TaggedHitType'].notna() & df["PlayResult"].isin(['Single','Double','Triple',
                           'HomeRun','Error','FieldersChoice','Out','Sacrifice'])).astype(int)

        es  = pd.to_numeric(df["ExitSpeed"], errors="coerce")
        ang = pd.to_numeric(df["Angle"],     errors="coerce")
        ev  = es / 1.609344
        ip  = df["is_inplay"] == 1

        cond6  = ip & (ev*1.5-ang>=117) & (ev+ang>=124) & (ev>=98) & ang.between(4,50)
        cond5  = ip & (ev*1.5-ang>=111) & (ev+ang>=119) & (ev>=95) & ang.between(0,52)
        cond1  = ip & (ev<=59)
        cond4a = ip & (ev*2-ang>=87)    & (ang<=41) & (ev*2+ang<=175) & (ev+ang*1.3>=89) & ev.between(59,72)
        cond4b = ip & (ev+ang*1.3<=112) & (ev+ang*1.55>=92) & (ev>=72) & (ev>=86)
        cond4c = ip & (ang<=20)         & (ev+ang*2.4>=98)  & ev.between(86,95)
        cond4d = ip & (ev-ang>=76)      & (ev+ang*2.4>=98)  & (ev>=95) & (ang<=30)
        cond3  = ip & (ev+ang*2>=116)
        cond2  = ip & (ev+ang*2<116)

        sa = pd.Series(0, index=df.index)
        sa[cond2]=2; sa[cond3]=3; sa[cond4a|cond4b|cond4c|cond4d]=4; sa[cond5]=5; sa[cond6]=6; sa[cond1]=1
        df["is_icr"]    = np.where(sa.isin([4,5,6]),1, np.where(sa.isin([1,2,3]),0,np.nan))
        df["is_barrel"] = (((es>=156)&ang.between(23.5,32.9))|((es>=159.25)&ang.between(21.15,37.6))|
                           ((es>=162.5)&ang.between(21.15,39.95))|((es>=165.75)&ang.between(18.8,47.0))|
                           ((es>=169)&(ang>=18.8))|((es>=172.25)&(ang>=16.45))|((es>=175.5)&(ang>=14.1))|
                           ((es>=178.75)&(ang>=11.75))|((es>=182)&(ang>=9.4))).fillna(False).astype(int)

        result = (df.groupby(["BatterId",'PLER_NAME','TEAM_NM',"BatterSide"])
                  .agg(투구수=("PitchCall","count"), 타석=("is_pa","sum"),
                       인플레이=("is_inplay","sum"), 평균타구속도=("is_exitspeed","mean"),
                       평균발사각도=("is_angle","mean"), 최고비거리=("is_distance","max"), 하드힛=("is_hardhit","sum"),
                       icr=("is_icr","sum"), barrel=("is_barrel","sum"),
                       스윗스팟=("is_sweetspot", 'sum'),
                       땅볼=("is_gb","sum"), 뜬공=("is_fb","sum"),
                       라인드라이브=("is_ld","sum"), 타구유형=("is_hitype","sum"), 번트=("is_bunt","sum"))
                  .reset_index())

        result["평균타구속도"]   = result["평균타구속도"].round(1)
        result["평균발사각도"]   = result["평균발사각도"].round(1)
        result["최고비거리"]   = result["최고비거리"].round(1)
        result["하드힛%"]       = (result["하드힛"]       / result["인플레이"] * 100).round(1)
        result["정타%"]         = (result["icr"]          / result["인플레이"] * 100).round(1)
        result["배럴%"]         = (result["barrel"]       / result["인플레이"] * 100).round(1)
        result["스윗스팟%"]         = (result["스윗스팟"]       / result["인플레이"] * 100).round(1)
        result["땅볼%"]         = (result["땅볼"]         / result["타구유형"] * 100).round(1)
        result["뜬공%"]         = (result["뜬공"]         / result["타구유형"] * 100).round(1)
        result["라인드라이브%"] = (result["라인드라이브"] / result["타구유형"] * 100).round(1)
        result["번트%"]         = (result["번트"]         / result["타구유형"] * 100).round(1)

        stats = result[['PLER_NAME','TEAM_NM',"BatterSide","투구수","타석","인플레이",
                         "평균타구속도","평균발사각도","최고비거리","하드힛%","정타%","배럴%","스윗스팟%","땅볼%","뜬공%","라인드라이브%","번트%"]]
        stats = stats.rename(columns={"PLER_NAME":"선수명","TEAM_NM":"학교",
                                       "BatterSide":"타석방향"})
        stats["타석방향"] = stats["타석방향"].map({"Right":"우타","Left":"좌타"})
        stats["타석방향"] = pd.Categorical(stats["타석방향"], categories=["우타","좌타"], ordered=True)
        return stats
    
    def hitchart(df):
        import plotly.express as px
        import plotly.graph_objects as go
        
        filtered = df.copy()
        
        filtered['hc_x'] = filtered['Distance'] * np.sin(np.radians(filtered['Bearing']))
        filtered['hc_y'] = filtered['Distance'] * np.cos(np.radians(filtered['Bearing']))
        
        filtered = filtered.loc[filtered['PlayResult'] != 'Undefined',:]
        
        color_dict = {
        'Single': 'green',
        'Double': 'orange',
        'Triple': 'pink',
        'Out': 'gray',
        'HomeRun': 'red',
        'Error': 'gray',
        'Sacrifice': 'gray'

        }
        
        # Plotly scatter plot
        fig = px.scatter(
            filtered,
            x='hc_x',
            y='hc_y',
            color='PlayResult',
            color_discrete_map=color_dict,
            labels={
                'hc_x': '수평 낙하지점',
                'hc_y': '수직 낙하지점',
            }
        )
        # 외야 펜스 선 추가
        fig.add_trace(go.Scatter(
            x=[0, -370],
            y=[0, 335],
            mode='lines',
            line=dict(color='green', width=2),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, 370],
            y=[0, 335],
            mode='lines',
            line=dict(color='green', width=2),
            showlegend=False
        ))
        # 축 범위와 레이아웃
        fig.update_xaxes(range=[-100, 100], title='수평 낙하지점')
        fig.update_yaxes(range=[-50, 200], title='수직 낙하지점')
        fig.update_layout(template='simple_white', height=700, width=900, title=f"타구 분포", legend_title_text='타구 결과')
    
        return fig
    
    def hitchart2(df):
        
        import plotly.express as px
        import plotly.graph_objects as go
        
        filtered = df.copy()
        
        filtered = filtered.loc[filtered['PlayResult'] != 'Undefined',:]
        
        color_dict = {
        'Single': 'green',
        'Double': 'orange',
        'Triple': 'pink',
        'Out': 'gray',
        'HomeRun': 'red',
        'Error': 'gray',
        'Sacrifice': 'gray'

        }

        fig = px.scatter(data_frame=filtered , 
                     x='ExitSpeed', 
                     y='Angle',
                     size = 4,
                     color='PlayResult',
                     color_discrete_map=color_dict)

        fig.update_layout(xaxis=dict(range=[40, 200], dtick=10, autorange=False, tickfont=dict(size=10)),
                          yaxis=dict(range=[-80, 80], dtick=10, tickfont=dict(size=10)),
                          plot_bgcolor='whitesmoke',
                          width=700,
                          height=700,
                         title=f"타구 조합", title_x=0)

        #fig.add_trace(go.Scatter(x=[40, 0], y=[80, -80], mode='lines', line=dict(color='red'), showlegend=False))
        #fig.add_trace(go.Scatter(x=[80, -80], y=[0, 0], mode='lines', line=dict(color='red'), showlegend=False))

        return fig
    

    result_table  = tracking_batter_discipline(filtered)
    result_table2 = tracking_batter_hitrack(filtered)
    result_hit_chart = hitchart(filtered)
    result_hit_chart2 = hitchart2(filtered)



    st.dataframe(result_table,  width=1500, hide_index=True)
    st.dataframe(result_table2, width=1500, hide_index=True)
    st.plotly_chart(result_hit_chart, use_container_width=True)
    st.plotly_chart(result_hit_chart2, use_container_width=True)
