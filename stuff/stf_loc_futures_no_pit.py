# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:54:04 2026

@author: jaebeom.soon
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
import polars as pl

st.title('2025시즌 퓨처스 Stuff+, Location+')
st.subheader('파일 업로드(.xlsx)')
uploaded_file = st.file_uploader(label = "파일 선택", type=["xlsx"])

#df = pd.read_excel(r'D:\jaebeom.soon\Desktop\stf25.xlsx', sheet_name='stf25')
#df

# Dictionary to map pitch types to their corresponding colors and names
pitch_colours = {
    '직구': {'colour': 'red', 'name': '직구'},
    '싱커': {'colour': 'FF33FF', 'name': '싱커'},
    
    '커터': {'colour': '#336633', 'name': '커터'},
    '슬라이더': {'colour': '#009933', 'name': '슬라이더'},
    '스위퍼': {'colour': '999900', 'name': '스위퍼'},

    '체인지업': {'colour': 'blue', 'name': '체인지업'},
    '스플리터': {'colour': '#B266FF', 'name': '스플리터'},
    '커브': {'colour': 'orange', 'name': '커브'}
}



# Define a custom colormap for styling
cmap_sum = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#648FFF', '#FFFFFF', '#FFB000'])

column_config_dict = {
    '투수ID': '투수ID',
    '선수명': '선수명',
    '구종': '구종',
    '투구수': st.column_config.NumberColumn("투구수", format="%.0f"),
    'Stuff+': st.column_config.NumberColumn("Stuff+", format="%.0f"),
    '평균 구속': st.column_config.NumberColumn("평균 구속", format="%.0f"),
    '회전수': st.column_config.NumberColumn("회전수", format="%.0f"),
    '수직 무브': st.column_config.NumberColumn("수직 무브", format="%.0f"),
    '수평 무브': st.column_config.NumberColumn("수평 무브", format="%.0f"),
    '릴리스 높이': st.column_config.NumberColumn("릴리스 높이", format="%.0f"),
    '릴리스 사이드': st.column_config.NumberColumn("릴리스 사이드", format="%.0f"),
    '익스텐션': st.column_config.NumberColumn("익스텐션", format="%.0f"),
    '포심 구속 차이': st.column_config.NumberColumn("포심 구속 차이", format="%.0f"),
    '포심 수직 무브 차이': st.column_config.NumberColumn("포심 수직 무브 차이", format="%.0f"),
    '포심 수평 무브 차이': st.column_config.NumberColumn("포심 수평 무브 차이", format="%.0f"),
    '구종 점수': st.column_config.NumberColumn("구종 점수", format="%.0f"),
    'Location+': st.column_config.NumberColumn("Location+", format="%.0f"),
    'Pitching+': st.column_config.NumberColumn("Pitching+", format="%.0f"),
}

# 탭 생성
tab_titles = ['Stuff+', 'Location+', '설명']
tabs = st.tabs(tab_titles)
 
# 각 탭에 콘텐츠 추가
with tabs[0]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='stf25')
        
    st.markdown("""Stuff+<br>
                투수가 얼마나 더러운(nasty), 위력적인 공을 던졌는지 물리적인 변수로만 평가한 지표<br> <br>
                구속, 회전수, 수직 무브, 수평 무브, 릴리스 높이, 릴리스 사이드, 익스텐션 <br> 변화구는 직구와 구속 차이, 수평 무브 차이, 수직 무브 차이 반영""", unsafe_allow_html=True)
        
    # 구종 목록
    custom_order = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
    
    unique_pitch_types =  [g for g in custom_order if g in df['구종'].unique()]
    unique_pitches = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # 구종 선택 박스 위젯 생성
    selected_pitch_types = st.selectbox('구종 선택', unique_pitch_types, key='pitch_type_selectbox_1')
    
    # 구종 선택 박스 위젯 생성
    selected_n = st.selectbox('투구수 선택', unique_pitches, key='pitches_selectbox_1')
    
    # Filter the DataFrame based on selected pitch types
    if selected_pitch_types != '':
        df_stuff = df[(df['구종'] == selected_pitch_types) & (df['투구수'] >= selected_n)].sort_values('Stuff+', ascending=False)
    
    
    # Apply background gradient styling to specific columns
    styled_df = df_stuff.style.background_gradient(subset=['Stuff+'], cmap=cmap_sum, vmin=80, vmax=120)
    
    st.dataframe(styled_df, hide_index=True, column_config=column_config_dict, width=1500)
    
    st.divider() # --------------------
    
    df = df[df['투구수'] >= 50]
    df = df.sort_values(['선수명'], ascending=[True])
    
    df['투수ID'] = df['투수ID'].apply(str)
    
    # Create dictionaries for pitcher information
    pitcher_id_name = dict(zip(df['투수ID'], df['선수명']))
    pitcher_id_name_id = dict(zip(df['투수ID'], df['선수명'] + ' - ' + df['투수ID']))
    pitcher_name_id_id = dict(zip(df['선수명'] + ' - ' + df['투수ID'], df['투수ID']))
    
    # Create a selectbox widget for pitchers
    pitcher_id_name_select = st.selectbox('투수 선택', sorted(pitcher_name_id_id.keys()), key='pitcher_id_selectbox_1')

    # Get selected pitcher information
    pitcher_id = pitcher_name_id_id[pitcher_id_name_select]
    pitcher_name = pitcher_id_name[pitcher_id]

    import grade_plot

    # Button to update plot
    if st.button('Update Plot', key='button_1'):
        st.session_state.update_plot = True
        grade_plot.stuff_plot(df, pitcher_id, pitcher_name)
        
with tabs[1]:
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name='loc25')
        
    st.markdown("""Location+ <br>
                적절한 위치에 투구할 수 있는 능력 평가한 지표 <br>
                타자 유형, 투수 유형, 수직 로케이션, 수평 로케이션, 카운트 변수 반영""", unsafe_allow_html=True)
        
    # 구종 목록
    custom_order = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
    
    unique_pitch_types =  [g for g in custom_order if g in df['구종'].unique()]
    unique_pitches = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # 구종 선택 박스 위젯 생성
    selected_pitch_types = st.selectbox('구종 선택', unique_pitch_types, key='pitch_type_selectbox_2')
    
    # 구종 선택 박스 위젯 생성
    selected_n = st.selectbox('투구수 선택', unique_pitches, key='pitches_selectbox_2')
    
    # Filter the DataFrame based on selected pitch types
    if selected_pitch_types != '':
        df_loc = df[(df['구종'] == selected_pitch_types) & (df['투구수'] >= selected_n)].sort_values('Location+', ascending=False)
    
    
    # Apply background gradient styling to specific columns
    styled_df = df_loc.style.background_gradient(subset=['Location+'], cmap=cmap_sum, vmin=80, vmax=120)
    
    st.dataframe(styled_df, hide_index=True, column_config=column_config_dict, width=1500)
    
    st.divider() # --------------------
    
    df = df[df['투구수'] >= 50]
    df = df.sort_values(['선수명'], ascending=[True])
    
    df['투수ID'] = df['투수ID'].apply(str)
    
    # Create dictionaries for pitcher information
    pitcher_id_name = dict(zip(df['투수ID'], df['선수명']))
    pitcher_id_name_id = dict(zip(df['투수ID'], df['선수명'] + ' - ' + df['투수ID']))
    pitcher_name_id_id = dict(zip(df['선수명'] + ' - ' + df['투수ID'], df['투수ID']))
    
    # Create a selectbox widget for pitchers
    pitcher_id_name_select = st.selectbox('투수 선택', sorted(pitcher_name_id_id.keys()), key='pitcher_id_selectbox_2')

    # Get selected pitcher information
    pitcher_id = pitcher_name_id_id[pitcher_id_name_select]
    pitcher_name = pitcher_id_name[pitcher_id]

    import grade_plot

    # Button to update plot
    if st.button('Update Plot', key='button_2'):
        st.session_state.update_plot = True
        grade_plot.location_plot(df, pitcher_id, pitcher_name)
        

with tabs[2]:
    st.markdown("""1. 2가지 지표 모두 2022-24 KBO 데이터 학습해 2025 퓨처스 평가 진행 <br>
2. 평가 기준은 xRV(Expected Run Value) 기대 득점 가치. <br> <br>
기대 득점 가치는 스트라이크, 볼, 1~3루타, 홈런, 파울, 삼진, 볼넷 등 모든 이벤트에 대한 득점 가치 산출 후 투구의 예상 득점가치 산출 <br>
ex) Stuff+: 구속: 144km/h, 회전수: 2100rpm, 수직 무브: 50cm, 수평 무브: 35cm, 릴리스 높이: 180cm, 릴리스 사이드: 43cm, 익스텐션: 170cm <br>
-> 해당 공의 기대 득점가치: 0.5 (높을수록 투수에게 부정적, 타자에게 긍정적)
3. 투구의 결과가 아닌 지표별 투구 평가에 활용한 변수 통해 결과 예측""", unsafe_allow_html=True)