# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:15:53 2024

@author: jaebeom.soon
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
#data = pd.read_csv(r'D:\jaebeom.soon\Desktop\aa\example.csv', encoding='cp949')
data = pd.read_csv('https://raw.githubusercontent.com/jaeb0129/practice/refs/heads/master/pitch/example.csv', encoding='cp949')
#os.chdir(r'C:\Users\jaebeom.soon')

data['y0'] = 50

data['x0'] = data['x0'] * 3.28084
data['z0'] = data['z0'] * 3.28084

data['vx0'] = data['vx0'] * 3.28084
data['vy0'] = data['vy0'] * 3.28084
data['vz0'] = data['vz0'] * 3.28084

data['ax0'] = data['ax0'] * 3.28084
data['ay0'] = data['ay0'] * 3.28084
data['az0'] = data['az0'] * 3.28084

# 구종
conditions = [
    (data['TaggedPitchType'] == 'Four-Seam'),
    (data['TaggedPitchType'] == 'Sinker'),
    (data['TaggedPitchType'] == 'Cutter'), 
    (data['TaggedPitchType'] == 'Slider'),
    (data['TaggedPitchType'] == 'Changeup'),
    (data['TaggedPitchType'] == 'Splitter'),
    (data['TaggedPitchType'] == 'Curveball'),
    (data['TaggedPitchType'] == 'Kuckleball')
]
choices = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']

data['TaggedPitchType'] = np.select(conditions, choices, default="Unknown").astype(object)

order = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']

data['TaggedPitchType'] = pd.Categorical(data['TaggedPitchType'], categories=order, ordered=True)
data.rename(columns = {'TaggedPitchType': '구종', 'InducedVertBreak': '수직 무브먼트',
                       'HorzBreak': '수평 무브먼트', 'PlateLocHeight' : '수직 로케이션',
                       'PlateLocSide': '수평 무브먼트', 'RelSpeed': '구속', 'SpinRate': '회전수', 'SpinAxis': '회전축',
                       'Extension': '익스텐션', 'RelHeight': '릴리스 높이', 'RelSide': '릴리스 사이드',
                       'PlateLocHeight': '수직 로케이션', 'PlateLocSide': '수평 로케이션'}, inplace = True)

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
               facet_col="구종")
    
    # 모든 facet별 x축과 y축 제목 숨기기
    fig.for_each_xaxis(lambda axis: axis.update(title=None))
    fig.for_each_yaxis(lambda axis: axis.update(title=None))
    
    fig.update_layout(
    width=1000,
    height=350,
    plot_bgcolor='white')
    
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split('=')[-1]))
    
    # Facet 제목 스타일 조정
    annotations = fig['layout']['annotations']  # 모든 annotations 가져오기
    for annotation in annotations:
        annotation['font'] = dict(size=16, family='Arial', weight='bold')  # Facet 제목 크기 및 스타일 변경
    
    fig.add_shape(type ='rect',
                  x0 = L, y0 = Bot,
                  x1 = R, y1 = Top,
                  line_width = 3,
                  line_color = 'black',
                  opacity = 0.5, row="all", col="all")
    fig.update_xaxes(range=[L-(2.5*30.48), R+(2.5*30.48)], showline=False, zeroline=False)
    fig.update_yaxes(range=[Bot-(3*30.48), Top+(2*30.48)], showline=False, zeroline=False)
    
    fig.add_trace(go.Scatter(x=[R_m, L_p, L_m, Center, R_p, R_m], y=[S_height, S_height, M_height, E_height, M_height, S_height], showlegend=False, marker_color = 'black'), row="all", col="all")
    return fig

def get_position_at_time(x0, ax0, vx0, y0, ay0, vy0, z0, az0, vz0, t):
    """
    주어진 시간 t에서 x, y, z 위치를 계산하는 함수
    """
    x = x0 + vx0 * t + 0.5 * ax0 * t**2
    y = y0 + vy0 * t + 0.5 * ay0 * t**2
    z = z0 + vz0 * t + 0.5 * az0 * t**2
    return {'x': x, 'y': y, 'z': z}

def pitchloc(t, x0, ax0, vx0, y0, ay0, vy0, z0, az0, vz0):
    """
    Calculates the location (x, y, z) of the pitch at a given time t.
    """
    x = x0 + vx0 * t + 0.5 * ax0 * t ** 2
    y = y0 + vy0 * t + 0.5 * ay0 * t ** 2
    z = z0 + vz0 * t + 0.5 * az0 * t ** 2
    if np.isscalar(t):  # Check if t is a single value
        loc = np.array([x, y, z])
    else:
        loc = np.column_stack((x, y, z))
    return loc

def pitch_trajectory(x0, ax0, vx0, y0, ay0, vy0, z0, az0, vz0, interval=0.001):
    """
    Calculates the trajectory of a pitch from the start to when it crosses the plate.
    """
    # Time at which the ball crosses the plate (y = 0)
    cross_plate_time = (-vy0 - np.sqrt(vy0 ** 2 - 2 * y0 * ay0)) / ay0

    # Time intervals from 0 to cross_plate_time
    t_values = np.arange(0, cross_plate_time, interval)

    # Calculate the trajectory for each time step
    trajectory = np.array([pitchloc(t, x0, ax0, vx0, y0, ay0, vy0, z0, az0, vz0) for t in t_values])

    # Convert to a Pandas DataFrame for easy manipulation and visualization
    tracking = pd.DataFrame(trajectory, columns=["x", "y", "z"])

    return tracking

def plot_pitch_trajectories(Pitcher, Date, time_points=[0.213, 0.100]):
    pdata = data.loc[(data.Pitcher == Pitcher) & (data.Date.isin(Date))]
    pdata = pdata.reset_index()
    pdata = pdata.sort_values(by=['구종'])

    fig = go.Figure()

    # 범례에 추가된 구종 추적
    legend_added = set()

    # 데이터 반복
    for i, row in pdata.iterrows():
        trajectory = pitch_trajectory(row['x0'], row['ax0'], row['vx0'], 
                                       row['y0'], row['ay0'], row['vy0'], 
                                       row['z0'], row['az0'], row['vz0'])
        pitch_type = row['구종']
        color = dict_colour.get(pitch_type, 'black')

        # 범례에 구종이 이미 추가되었는지 확인
        show_legend = pitch_type not in legend_added
        if show_legend:
            legend_added.add(pitch_type)  # 범례에 구종 추가

        # 궤적 추가
        fig.add_trace(go.Scatter3d(
            x=trajectory['x'], y=trajectory['y'], z=trajectory['z'],
            mode='lines',
            name=f"{pitch_type} (Pitch {i+1})",  # 구종 뒤에 투구 번호 추가
            line=dict(color=color, width=4), visible='legendonly'
        ))

    # 홈플레이트 영역 추가
    zones = {
        "zone_front": {'x': [-0.893, 0.893, 0.893, -0.893, -0.893],
                       'y': [1.417, 1.417, 1.417, 1.417, 1.417],
                       'z': [3.30, 3.30, 1.60, 1.60, 3.30],
                       'color': 'skyblue'},
        "zone_mid": {'x': [-0.893, 0.893, 0.893, -0.893, -0.893],
                     'y': [0.7083, 0.7083, 0.7083, 0.7083, 0.7083],
                     'z': [3.30, 3.30, 1.60, 1.60, 3.30],
                     'color': 'brown'},
        "zone_back": {'x': [-0.893, 0.893, 0.893, -0.893, -0.893],
                      'y': [0, 0, 0, 0, 0],
                      'z': [3.30-0.0492, 3.30-0.0492, 1.60-0.0492, 1.60-0.0492, 3.30-0.0492],
                      'color': 'grey'}
    }
    # 2025시즌
    # 3.292283cm / 180cm 기준 55.75%
    # 1.596831cm / 180cm 기준 27.04%
    col_ls = ['']

    for key, zone in zones.items():
        fig.add_trace(go.Scatter3d(
            x=zone['x'], y=zone['y'], z=zone['z'],
            mode='lines',
            name=key,
            line=dict(width=3, color=zone['color']), showlegend=False
        ))

    # 레이아웃 설정
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(range=[-3.5, 3.5]),
            zaxis=dict(range=[0, 8])
        ),
        title="투구 궤적",
        legend_title="구종",
    )

    # 특정 초에 점 추가
    for i, row in pdata.iterrows():
        pitch_type = row['구종']
        for t in time_points:
            point_position = get_position_at_time(
            row['x0'], row['ax0'], row['vx0'],
            row['y0'], row['ay0'], row['vy0'],
            row['z0'], row['az0'], row['vz0'], t)
            fig.add_trace(go.Scatter3d(
                x=[point_position['x']],
                y=[point_position['y']],
                z=[point_position['z']],
                mode='markers',
                name=f"{pitch_type} Pitch {i+1}, Point at {t}s ",
                marker=dict(color='black', size=2, symbol='circle'), visible='legendonly'
            ))
            
    return fig
# 100ms 380
# 213ms 167 / 릴리스 포인트에서 흐른 초 기준
# 100ms 380
# 213ms 167 / 릴리스 포인트에서 흐른 초 기준
result_table = track(select_pitcher, select_date)
result_movement = movement(select_pitcher, select_date)
result_rlse_point = release(select_pitcher, select_date)
result_location = location(select_pitcher, select_date)
result_pitch_trajectories = plot_pitch_trajectories(select_pitcher, select_date)

st.plotly_chart(result_movement)
st.plotly_chart(result_rlse_point)
st.plotly_chart(result_location)
st.dataframe(result_table, width=1000, hide_index=True)
st.plotly_chart(result_pitch_trajectories)