# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:25:28 2024

@author: jaebeom.soon
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
#data = pd.read_csv(r'D:\jaebeom.soon\Desktop\aa\example.csv', encoding='cp949')
data = pd.read_csv('https://raw.githubusercontent.com/jaeb0129/practice/refs/heads/master/pitch/example.CSV', encoding='cp949')
#os.chdir(r'C:\Users\jaebeom.soon')

# 구종
conditions = [
    (data['AutoPitchType'] == 'Four-Seam'),
    (data['AutoPitchType'] == 'Sinker'),
    (data['AutoPitchType'] == 'Cutter'), 
    (data['AutoPitchType'] == 'Slider'),
    (data['AutoPitchType'] == 'Changeup'),
    (data['AutoPitchType'] == 'Splitter'),
    (data['AutoPitchType'] == 'Curveball'),
    (data['AutoPitchType'] == 'Kuckleball')
]
choices = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']

data['AutoPitchType'] = np.select(conditions, choices, default="Unknown").astype(object)

order = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']

data['AutoPitchType'] = pd.Categorical(data['AutoPitchType'], categories=order, ordered=True)
data.rename(columns = {'AutoPitchType': '구종', 'InducedVertBreak': '수직 무브먼트',
                       'HorzBreak': '수평 무브먼트', 'PlateLocHeight' : '수직 로케이션',
                       'PlateLocSide': '수평 무브먼트', 'RelSpeed': '구속', 'SpinRate': '회전수', 'Spixis': '회전축',
                       'Extension': '익스텐션', 'RelHeight': '릴리스 높이', 'RelSide': '릴리스 사이드',
                       'PlateLocHeight': '수직 로케이션', 'PlateLocSide': '수평 로케이션'}, inplace = True)

# 타자 유형
conditions_batterside = [
    (data['BatterSide'] == 'Right'),
    (data['BatterSide'] == 'Left')
]
choices_batterside = ['우', '좌']

data['BatterSide'] = np.select(conditions_batterside, choices_batterside, default="Unknown").astype(object)

order_batterside = ['우', '좌']

data['BatterSide'] = pd.Categorical(data['BatterSide'], categories=order_batterside, ordered=True)
data.rename(columns = {'BatterSide': '타자유형', 'PitchCall': '투구결과', 'PlayResult': '타구결과', 'PAofInning': '이닝타석', 'PitchofPA':'타석투구'}, inplace = True)

data['익스텐션'] = data['익스텐션'] * 100
data['릴리스 높이'] = data['릴리스 높이'] * 100
data['릴리스 사이드'] = data['릴리스 사이드'] * 100
data['수직 로케이션'] = round(data['수직 로케이션'] * 100, 1)
data['수평 로케이션'] = round(data['수평 로케이션'] * -100, 1)
data['구속'] = round(data['구속'], 1)
data['수평 무브먼트'] = round(data['수평 무브먼트'], 1)
data['수직 무브먼트'] = round(data['수직 무브먼트'], 1)
data['회전수'] = round(data['회전수'], 0)


### PITCH COLOURS ##

pitch_colours = {
    ## Fastballs ##
    '직구': {'colour': 'red', 'name': '직구'},
    '싱커': {'colour': 'pink', 'name': '싱커'},
    '커터': {'colour': '#67E18D', 'name': '커터'},

    ## Offspeed ##
    '체인지업': {'colour': 'blue', 'name': '체인지업'},
    '스플리터': {'colour': 'purple', 'name': '스플리터'},

    ## Sliders ##
    '슬라이더': {'colour': 'green', 'name': '슬라이더'},

    ## Curveballs ##
    '커브': {'colour': 'orange', 'name': '커브'},

    ## Others ##
    '너클': {'colour': '#867A08', 'name': '너클'}
}

# Create a dictionary mapping pitch types to their colors
dict_colour = dict(zip(pitch_colours.keys(), [pitch_colours[key]['colour'] for key in pitch_colours]))

# Create a dictionary mapping pitch types to their colors
dict_pitch = dict(zip(pitch_colours.keys(), [pitch_colours[key]['name'] for key in pitch_colours]))


# 사이드바에 select box를 활용하여 종을 선택한 다음 그에 해당하는 행만 추출하여 데이터프레임을 만들고자합니다.
st.sidebar.title('옵션')

# select_species 변수에 사용자가 선택한 값이 지정됩니다
select_pitcher = st.sidebar.selectbox(
    '확인하고 싶은 투수를 선택하세요',
    data.Pitcher.unique()
)

select_date = st.sidebar.multiselect(
    '확인하고 싶은 날짜를 선택하세요',
    data[(data.Pitcher == select_pitcher)].Date.unique()
)

st.title(f'{select_pitcher} 투구 대시보드')

def track(Pitcher, Date):
    pdata = data.loc[(data.Pitcher == Pitcher) & (data.Date.isin(Date))]

    table = pdata.groupby('구종')[['Date', '구속', '회전수', '회전축', '수직 무브먼트', '수평 무브먼트', '릴리스 높이', '릴리스 사이드', '익스텐션']].agg({'Date': 'count', '구속': ['mean', 'max'], '회전수': 'mean', '회전축':'mean',
    '수직 무브먼트': 'mean', '수평 무브먼트': 'mean', '릴리스 높이': 'mean', '릴리스 사이드': 'mean', '익스텐션': 'mean'}).dropna().round(1)
            
    table.columns = ['투구수', '평균_구속(km/h)', '최고_구속(km/h)', '회전수(rpm)', '회전축', '수직 무브먼트(cm)', '수평 무브먼트(cm)', '릴리스 높이(cm)', '릴리스 사이드(cm)', '익스텐션(cm)']
    table['구사율(%)'] = round((table['투구수']/sum(table['투구수'])) * 100, 1)
    table['수직 무브먼트(cm)'] = round(table['수직 무브먼트(cm)'], 1)
    table['수평 무브먼트(cm)'] = round(table['수평 무브먼트(cm)'], 1)
    table['평균_구속(km/h)'] = round(table['평균_구속(km/h)'], 1)
    table['최고_구속(km/h)'] = round(table['최고_구속(km/h)'], 1)
    table['릴리스 높이(cm)'] = round(table['릴리스 높이(cm)'], 1)
    table['릴리스 사이드(cm)'] = round(table['릴리스 사이드(cm)'], 1)
    table['익스텐션(cm)'] = round(table['익스텐션(cm)'], 1)
    table['회전수(rpm)'] = table['회전수(rpm)'].astype(int)
    table['회전축'] = table['회전축'].astype(int)
    table = table[['투구수', '구사율(%)', '평균_구속(km/h)', '최고_구속(km/h)', '회전수(rpm)', '회전축', '수직 무브먼트(cm)', '수평 무브먼트(cm)', '릴리스 높이(cm)', '릴리스 사이드(cm)', '익스텐션(cm)']]
    table = table.sort_values(by='구사율(%)', ascending=False).reset_index()
    table = table.style.format({"구사율(%)": "{:.1f}",
                                "수직 무브먼트(cm)": "{:.1f}",
                                "수평 무브먼트(cm)": "{:.1f}",
                                "평균_구속(km/h)": "{:.1f}",
                                "최고_구속(km/h)": "{:.1f}",
                                "릴리스 높이(cm)": "{:.1f}",
                                "릴리스 사이드(cm)": "{:.1f}",
                                "익스텐션(cm)": "{:.1f}".format})
    
    return table

def movement(Pitcher, Date):
    pdata = data.loc[(data.Pitcher == Pitcher) & (data.Date.isin(Date))]
    pdata = pdata.sort_values(by=['구종'])

    fig = px.scatter(data_frame=pdata, 
                     x='수평 무브먼트', 
                     y='수직 무브먼트',
                     color='구종',
                     color_discrete_map=dict_colour,
                     hover_data=['구속', '회전수'],
                     category_orders={'구종': ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']})

    fig.update_layout(xaxis=dict(range=[80, -80], dtick=10, autorange=False, tickfont=dict(size=10)),
                      yaxis=dict(range=[-80, 80], dtick=10, tickfont=dict(size=10)),
                      plot_bgcolor='whitesmoke',
                      width=500,
                      height=500,
                     title=f"{Pitcher} 무브먼트", title_x=0.25)

    fig.add_trace(go.Scatter(x=[0, 0], y=[80, -80], mode='lines', line=dict(color='red'), showlegend=False))
    fig.add_trace(go.Scatter(x=[80, -80], y=[0, 0], mode='lines', line=dict(color='red'), showlegend=False))

    return fig

def release(Pitcher, Date):
    pdata = data.loc[(data.Pitcher == Pitcher) & (data.Date.isin(Date))]
    pdata = pdata.sort_values(by=['구종'])

    fig = px.scatter(data_frame=pdata, 
                     x='릴리스 사이드', 
                     y='릴리스 높이',
                     color='구종',
                     color_discrete_map=dict_colour,
                     category_orders={'구종': ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']})

    fig.update_layout(xaxis=dict(range=[150, -150], dtick=20, autorange=False, tickfont=dict(size=10)),
                      yaxis=dict(range=[140, 200], dtick=10, tickfont=dict(size=10)),
                      plot_bgcolor='whitesmoke',
                      width=500,
                      height=500,
                     title=f"{Pitcher} 릴리스 포인트", title_x=0.25)

    fig.add_trace(go.Scatter(x=[0, 0], y=[120, 220], mode='lines', line=dict(color='red'), showlegend=False))
    fig.add_trace(go.Scatter(x=[150, -150], y=[170, 170], mode='lines', line=dict(color='red'), showlegend=False))

    return fig

def location(Pitcher, Date):
    pdata = data.loc[(data.Pitcher == Pitcher) & (data.Date.isin(Date))]
    pdata = pdata.sort_values(by=['구종'])
    
    L, R = -0.708333 * 30.48, +0.708333 * 30.48
    Bot, Top = 1.5 * 30.48, 3.5 * 30.48
    
    L_p = L+(0.1 * 30.48)
    L_m = L-(0.1 * 30.48)
    R_p = R+(0.1 * 30.48)
    R_m = R-(0.1 * 30.48)
    Center= 0

    S_height = 0
    M_height = -0.6 * 30.48
    E_height = -1.0 * 30.48
    
    fig = px.scatter(data_frame = pdata, 
               x = '수평 로케이션', 
               y = '수직 로케이션',
               color = '구종',
               color_discrete_map=dict_colour,
               facet_row="타자유형",
               facet_col="구종",
               hover_data = ['구속', '회전수', '투구결과', '타구결과', '이닝타석', '타석투구', 'Balls', 'Strikes'])
    
    fig.update_layout(width=3000,
                      height=500)
    
    fig.add_shape(type ='rect',
                  x0 = L, y0 = Bot,
                  x1 = R, y1 = Top,
                  line_width = 3,
                  line_color = 'black',
                  opacity = 0.5, row="all", col="all")
    fig.update_xaxes(range=[L-(2.5*30.48), R+(2.5*30.48)])
    fig.update_yaxes(range=[Bot-(3*30.48), Top+(2*30.48)])
    fig.add_trace(go.Scatter(x=[R_m, L_p, L_m, Center, R_p, R_m], y=[S_height, S_height, M_height, E_height, M_height, S_height], showlegend=False), row="all", col="all")
    return fig

result_table = track(select_pitcher, select_date)
result_movement = movement(select_pitcher, select_date)
result_rlse_point = release(select_pitcher, select_date)
result_location = location(select_pitcher, select_date)

st.plotly_chart(result_movement)
st.plotly_chart(result_rlse_point)
st.plotly_chart(result_location)
st.dataframe(result_table, width=1000, hide_index=True)
