# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 02:58:26 2025

@author: jaebeom.soon
"""

import streamlit as st
#import os
import pandas as pd
import numpy as np
import plotly.express as px
#import plotly.graph_objects as go

st.title('2026시즌 프로젝션')
#st.subheader('파일 업로드(.xlsx)')
# uploaded_file = st.file_uploader(label = "파일 선택", type=["xlsx"])
uploaded_file = pd.read_excel('https://raw.githubusercontent.com/jaeb0129/practice/refs/heads/master/projection/2026_team_projection.xlsx')

#df = pd.read_excel(r'D:\jaebeom.soon\Desktop\WAR\2026\2026 팀순위 프로젝션.xlsx', sheet_name='신인')
#df

# 탭 생성
tab_titles = ['LG', '한화', 'SSG', '삼성', 'NC', 'KT', '롯데', 'KIA', '두산', '키움', '국내 전체', '외국인 선수']
tabs = st.tabs(tab_titles)
 
# 각 탭에 콘텐츠 추가
with tabs[0]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='LG')
    
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter)    
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="LG 트윈스 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)    
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="LG 트윈스 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == 'LG')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == 'LG') | (df2['현 소속팀'] == 'LG')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == 'LG 트윈스')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
 
with tabs[1]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='한화')
    
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter)    
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="한화 이글스 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="한화 이글스 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == '한화')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == '한화') | (df2['현 소속팀'] == '한화')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == '한화 이글스')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
 
with tabs[2]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='SSG')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter)  
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="SSG 랜더스 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="SSG 랜더스 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == 'SSG')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == 'SSG') | (df2['현 소속팀'] == 'SSG')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == 'SSG 랜더스')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[3]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='삼성')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter)  
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="삼성 라이온즈 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="삼성 라이온즈 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == '삼성')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == '삼성') | (df2['현 소속팀'] == '삼성')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == '삼성 라이온즈')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[4]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='NC')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter) 
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="NC 다이노스 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="NC 다이노스 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == 'NC')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == 'NC') | (df2['현 소속팀'] == 'NC')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == 'NC 다이노스')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[5]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='KT')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter) 
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="KT 위즈 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="KT 위즈 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == 'KT')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == 'KT') | (df2['현 소속팀'] == 'KT')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == 'KT 위즈')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[6]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='롯데')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter) 
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="롯데 자이언츠 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="롯데 자이언츠 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == '롯데')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == '롯데') | (df2['현 소속팀'] == '롯데')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == '롯데 자이언츠')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[7]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='KIA')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter) 
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="KIA 타이거즈 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="KIA 타이거즈 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == 'KIA')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == 'KIA') | (df2['현 소속팀'] == 'KIA')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == 'KIA 타이거즈')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[8]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='두산')
        
    st.subheader('야수')
    
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter) 
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="두산 베어스 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="두산 베어스 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == '두산')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == '두산') | (df2['현 소속팀'] == '두산')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == '두산 베어스')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
with tabs[9]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='키움')
        
    st.subheader('야수')
        
    # 타자
    batter = df
    batter.columns = batter.loc[2,]
    batter = df.loc[3:17,].reset_index(drop = True)
    batter = batter.set_index('순')
    
    st.dataframe(batter) 
    
    bat_chart = px.bar(batter, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 야수 레이아웃 설정
    bat_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="키움 히어로즈 야수 2026 WAR (타석수 상위 15인)",
        legend_title="선수명",
    )
    
    st.plotly_chart(bat_chart)
    
    st.divider() # --------------------
    
    st.subheader('투수')
    
    # 투수
    pitcher = df
    pitcher.columns = pitcher.loc[21,]
    pitcher = df.loc[22:36,].reset_index(drop = True)
    pitcher = pitcher.iloc[:,0:9]
    pitcher = pitcher.set_index('순')
    
    st.dataframe(pitcher)
    
    pit_chart = px.bar(pitcher, '선수명', '최종_회귀적용 WAR', color = '선수명')
    
    # 투수 레이아웃 설정
    pit_chart.update_layout(
            yaxis=dict(range=[-1, 6], dtick = 1),
        title="키움 히어로즈 투수 2026 WAR (이닝 상위 15인)",
        legend_title="선수명",
    )

    st.plotly_chart(pit_chart)
    
    st.divider() # --------------------
    
    df1 = pd.read_excel(uploaded_file, sheet_name='군대')
    df1 = df1.loc[(df1.소속팀 == '키움')].reset_index(drop=True)
    df1 = df1.set_index('순')
        
    st.subheader('2025-26 군복무 현황')
    st.dataframe(df1) 
    
    st.divider() # --------------------
    
    df2 = pd.read_excel(uploaded_file, sheet_name='이적 현황')
    df2 = df2.loc[(df2['전 소속팀'] == '키움') | (df2['현 소속팀'] == '키움')].reset_index(drop=True)
    df2 = df2.set_index('순')
        
    st.subheader('2025-26 이적 현황')
    st.dataframe(df2) 
    
    st.divider() # --------------------
    
    df3 = pd.read_excel(uploaded_file, sheet_name='신인')
    df3 = df3.loc[(df3.지명팀 == '키움 히어로즈')].reset_index(drop=True)
    df3 = df3.set_index('순')
        
    st.subheader('2026 신인')
    st.dataframe(df3) 
    
    
with tabs[10]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='국내 투타 전체')
        
    st.subheader('국내 투타 전체')
    
    # 종합
    all_df = df
    all_df.columns = all_df.loc[2]
    all_df = df.loc[3:,].reset_index(drop = True)
    all_df = all_df.set_index('순')
    
    st.dataframe(all_df) 
    
with tabs[11]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='외국인 선수_ver.4')
        
    st.subheader('외국인 선수 포함')
    
    # 종합
    foriegn_df = df.iloc[27:,10:]
    foriegn_df.columns = foriegn_df.iloc[0]
    foriegn_df = foriegn_df.reset_index(drop = True).loc[1:]
    foriegn_df = foriegn_df.set_index('순')
    
    st.dataframe(foriegn_df) 
    
    st.divider() # --------------------
    
    st.subheader('2026 팀별 신규 외국인 선수')
    st.markdown('* 교체 외국인 선수 -> 신규 외국인 선수로 설정 ex) LG 톨허스트')
    st.markdown('* 알칸타라(키움)의 경우 신규 외국인 선수 아닌 24, 25 시즌만 신규 외국인 선수로 가정해 WAR 계산')
    
    # 현황
    foriegn_contract = df.iloc[1:12,10:]
    foriegn_contract.columns = foriegn_contract.iloc[0]
    foriegn_contract = foriegn_contract.reset_index(drop = True).loc[1:]
    
    st.dataframe(foriegn_contract) 
    
    st.divider() # --------------------

    st.subheader('2026 팀별 외국인 선수 재계약')
    
    # 타자
    st.text('재계약 외국인 타자')
    foriegn_contract_bat = df.iloc[15:21,:7]
    foriegn_contract_bat.columns = foriegn_contract_bat.iloc[0]
    foriegn_contract_bat = foriegn_contract_bat.reset_index(drop = True).loc[1:]
    
    st.dataframe(foriegn_contract_bat) 
    
    # 투수
    st.text('재계약 외국인 투수')
    foriegn_contract_pit = df.iloc[24:33,:7]
    foriegn_contract_pit.columns = foriegn_contract_pit.iloc[0]
    foriegn_contract_pit = foriegn_contract_pit.reset_index(drop = True).loc[1:]
    
    st.dataframe(foriegn_contract_pit) 
    
    st.divider() # --------------------
    
    # 투수
    if uploaded_file is not None:
        df1 = pd.read_excel(uploaded_file, sheet_name='신규 외국인 선수')
        df1 = df1.set_index('순')
    
    st.subheader('신규 외국인 선수 프로필')
    st.dataframe(df1)
