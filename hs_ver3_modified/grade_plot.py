# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:51:04 2026

@author: jaebeom.soon
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import streamlit as st
import matplotlib.font_manager as fm
import matplotlib as mpl
import streamlit as st

@st.cache_resource
def setup_fonts():
    font_dir = "./fonts"
    font_path = os.path.join(font_dir, "NanumGothicBold.ttf")
    os.makedirs(font_dir, exist_ok=True)
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
        urllib.request.urlretrieve(url, font_path)
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()

    mpl.rcParams["font.family"] = font_name

setup_fonts()

# For help with plotting the pitch data, we will use the following dictionary to map pitch types to their corresponding colours
### PITCH COLOURS ###
pitch_colours = {
    '직구': {'colour': 'red', 'name': '직구'},
    '싱커': {'colour': '#FF33FF', 'name': '싱커'},
    
    '커터': {'colour': '#336633', 'name': '커터'},
    '슬라이더': {'colour': '#009933', 'name': '슬라이더'},
    '스위퍼': {'colour': '#999900', 'name': '스위퍼'},

    '체인지업': {'colour': 'blue', 'name': '체인지업'},
    '스플리터': {'colour': '#B266FF', 'name': '스플리터'},
    '커브': {'colour': 'orange', 'name': '커브'}
}

# Create a dictionary mapping pitch types to their colors
dict_colour = dict(zip(pitch_colours.keys(), [pitch_colours[key]['colour'] for key in pitch_colours]))
# Create a dictionary mapping pitch types to their colors
dict_pitch = dict(zip(pitch_colours.keys(), [pitch_colours[key]['name'] for key in pitch_colours]))

# Create a dictionary mapping pitch types to their colors
dict_pitch_desc_type = dict(zip([pitch_colours[key]['name'] for key in pitch_colours],pitch_colours.keys()))


# Create a dictionary mapping pitch types to their colors
dict_pitch_name = dict(zip([pitch_colours[key]['name'] for key in pitch_colours], 
                           [pitch_colours[key]['colour'] for key in pitch_colours]))



required_pitch_types = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
# Create a mapping dictionary from the list
custom_order_dict = {pitch: index for index, pitch in enumerate(required_pitch_types)}


def stuff_plot(df, 투수ID:int, 선수명:str):

    # 한글 폰트 설정 (예시: 'Malgun Gothic', 'AppleGothic' 등 OS에 맞게 선택)
    mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
    # mpl.rcParams['font.family'] = 'AppleGothic'   # 맥

    # 마이너스(-) 깨짐 방지
    mpl.rcParams['axes.unicode_minus'] = False

    sns.set_style("ticks")
    mpl.rcParams['font.family'] = 'Malgun Gothic'
    mpl.rcParams['axes.unicode_minus'] = False
    
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(10, 8), dpi=450)
    gs = GridSpec(5, 3, height_ratios=[0.1, 10, 10, 2, 0.1], width_ratios=[1, 100, 1])
    gs.update(hspace=0.4, wspace=0.1)

    # Add subplots to the grid
    ax0 = fig.add_subplot(gs[1, 1]) 
    ax1 = fig.add_subplot(gs[2, 1])
    ax1_left = fig.add_subplot(gs[:, 0]) 
    ax1_right = fig.add_subplot(gs[:, 2]) 
    ax1_top = fig.add_subplot(gs[0, :])
    ax1_bot = fig.add_subplot(gs[4, 1])
    ax2 = fig.add_subplot(gs[3, 1])
    
    # Filter data for the specific pitcher
    pitcher_df = df[(df['투수ID'] == 투수ID) & (df['투구수'] >= 50)]
    
    #pitcher_df = df[(df['투수ID'] == 64001) & (df['투구수'] >= 50)]
    
    # Create a mapping dictionary from the list
    custom_order = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
    custom_order_dict = {pitch: index for index, pitch in enumerate(custom_order)}
    
    pitcher_df['order'] = pitcher_df['구종'].map(custom_order_dict)
    pitcher_df = pitcher_df.sort_values('order')
                         
    # Get unique pitch types for the pitcher
    pitcher_pitches = pitcher_df['구종'].unique()
    pitcher_pitches = [x for x in custom_order if x in pitcher_pitches]

                     
    # Plot Stuff+ with swarmplot for all players in the same position
    sns.swarmplot(data=df[(df['투구수'] >= 50)].dropna(subset=['구종']),
                x='구종',
                y='Stuff+',
                palette=dict_colour,
                alpha=0.3,
                size=3,
                ax=ax0,
                order=pitcher_pitches)

    # Overlay swarmplot for the specific pitcher
    sns.swarmplot(data=df[(df['투수ID'] == 투수ID) &
                                            (df['투구수'] >= 50)],
                x='구종',
                y='Stuff+',
                palette=dict_colour,
                alpha=1,
                size=16,
                ax=ax0,
                order=pitcher_pitches,
                edgecolor='black',
                linewidth=1)

    # Annotate the median values on the plot
    for index, row in pitcher_df.reset_index(drop=True).iterrows():
        ax0.text(index, 
                row['Stuff+'], 
                f'{row["Stuff+"]:.0f}', 
                color='white', 
                ha="center", 
                va="center",
                fontsize=8,
                weight='bold',
                clip_on=False)

    # Customize ax0
    ax0.set_xlabel('')
    ax0.set_ylabel('Stuff+')
    ax0.grid(False)
    ax0.set_ylim(70, 130)
    ax0.axhline(y=100, color='black', linestyle='--', alpha=0.2, zorder=0)

    # Plot pitch grade with swarmplot for all players in the same position
    sns.swarmplot(data=df[(df['투구수'] >= 50)].dropna(subset=['구종']),
                x='구종',
                y='구종 점수',
                palette=dict_colour,
                alpha=0.3,
                size=3,
                ax=ax1,
                clip_on=False,
                order=pitcher_pitches)

    # Overlay swarmplot for the specific pitcher
    sns.swarmplot(data=df[(df['투수ID'] == 투수ID) & (df['투구수'] >= 50)],
                x='구종',
                y='구종 점수',
                palette=dict_colour,
                alpha=1,
                size=16,
                ax=ax1,
                order=pitcher_pitches,
                edgecolor='black',
                clip_on=False,
                linewidth=1)

    # Annotate the median values on the plot
    for index, row in pitcher_df.reset_index(drop=True).iterrows():
        ax1.text(index, 
                row['구종 점수'], 
                f'{row["구종 점수"]:.0f}', 
                color='white', 
                ha="center", 
                va="center",
                fontsize=8,
                weight='bold',
                clip_on=False,
                zorder=1000)
        
    # Customize ax1
    ax1.set_xlabel('구종')
    ax1.set_ylabel('구종 점수')
    ax1.grid(False)
    ax1.set_ylim(20, 80)
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.2, zorder=0)

    # Hide axes for additional subplots
    ax2.axis('off')
    ax1_left.axis('off')
    ax1_right.axis('off')
    ax1_top.axis('off')
    ax1_bot.axis('off')

    # Add text annotations
    ax1_bot.text(s='Data: KBO, 50구 이상', x=1, y=1, fontsize=12, ha='right')

    ax1_top.text(0.5, 0, f'{선수명} Stuff+',
                fontsize=24, ha='center', va='top')

    ax2.text(x=0.5, y=0.6, s='Stuff+는 구종 상관 없이 동등하게 Expected Run Value (xRV, 기대 득점 가치) 비교한 값\n'
                            'Stuff+는 평균: 100, 표준편차: 10인 정규분포\n'
                            '구종 점수는 구종별로 20-80스케일 적용해 구종 내에서 비교한 점수 (최소값: 20, 최대값: 80, 평균: 50, 표준편차: 10)',
                            ha='center', va='top', fontname='Malgun Gothic', fontsize=10)

    # Adjust subplot layout
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    # fig.set_facecolor('#e0e0e0')
    st.pyplot(fig)
    

def location_plot(df, 투수ID:int, 선수명:str):

    # 한글 폰트 설정 (예시: 'Malgun Gothic', 'AppleGothic' 등 OS에 맞게 선택)
    mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
    # mpl.rcParams['font.family'] = 'AppleGothic'   # 맥

    # 마이너스(-) 깨짐 방지
    mpl.rcParams['axes.unicode_minus'] = False

    sns.set_style("ticks")
    mpl.rcParams['font.family'] = 'Malgun Gothic'
    mpl.rcParams['axes.unicode_minus'] = False
    
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(10, 8), dpi=450)
    gs = GridSpec(5, 3, height_ratios=[0.1, 10, 10, 2, 0.1], width_ratios=[1, 100, 1])
    gs.update(hspace=0.4, wspace=0.1)

    # Add subplots to the grid
    ax0 = fig.add_subplot(gs[1, 1]) 
    ax1 = fig.add_subplot(gs[2, 1])
    ax1_left = fig.add_subplot(gs[:, 0]) 
    ax1_right = fig.add_subplot(gs[:, 2]) 
    ax1_top = fig.add_subplot(gs[0, :])
    ax1_bot = fig.add_subplot(gs[4, 1])
    ax2 = fig.add_subplot(gs[3, 1])
    
    # Filter data for the specific pitcher
    pitcher_df = df[(df['투수ID'] == 투수ID) & (df['투구수'] >= 50)]
    
    #pitcher_df = df[(df['투수ID'] == 64001) & (df['투구수'] >= 50)]
    
    # Create a mapping dictionary from the list
    custom_order = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
    custom_order_dict = {pitch: index for index, pitch in enumerate(custom_order)}
    
    pitcher_df['order'] = pitcher_df['구종'].map(custom_order_dict)
    pitcher_df = pitcher_df.sort_values('order')
                         
    # Get unique pitch types for the pitcher
    pitcher_pitches = pitcher_df['구종'].unique()
    pitcher_pitches = [x for x in custom_order if x in pitcher_pitches]

                     
    # Plot Stuff+ with swarmplot for all players in the same position
    sns.swarmplot(data=df[(df['투구수'] >= 50)].dropna(subset=['구종']),
                x='구종',
                y='Location+',
                palette=dict_colour,
                alpha=0.3,
                size=3,
                ax=ax0,
                order=pitcher_pitches)

    # Overlay swarmplot for the specific pitcher
    sns.swarmplot(data=df[(df['투수ID'] == 투수ID) &
                                            (df['투구수'] >= 50)],
                x='구종',
                y='Location+',
                palette=dict_colour,
                alpha=1,
                size=16,
                ax=ax0,
                order=pitcher_pitches,
                edgecolor='black',
                linewidth=1)

    # Annotate the median values on the plot
    for index, row in pitcher_df.reset_index(drop=True).iterrows():
        ax0.text(index, 
                row['Location+'], 
                f'{row["Location+"]:.0f}', 
                color='white', 
                ha="center", 
                va="center",
                fontsize=8,
                weight='bold',
                clip_on=False)

    # Customize ax0
    ax0.set_xlabel('')
    ax0.set_ylabel('Location+')
    ax0.grid(False)
    ax0.set_ylim(70, 130)
    ax0.axhline(y=100, color='black', linestyle='--', alpha=0.2, zorder=0)

    # Plot pitch grade with swarmplot for all players in the same position
    sns.swarmplot(data=df[(df['투구수'] >= 50)].dropna(subset=['구종']),
                x='구종',
                y='구종 점수',
                palette=dict_colour,
                alpha=0.3,
                size=3,
                ax=ax1,
                clip_on=False,
                order=pitcher_pitches)

    # Overlay swarmplot for the specific pitcher
    sns.swarmplot(data=df[(df['투수ID'] == 투수ID) & (df['투구수'] >= 50)],
                x='구종',
                y='구종 점수',
                palette=dict_colour,
                alpha=1,
                size=16,
                ax=ax1,
                order=pitcher_pitches,
                edgecolor='black',
                clip_on=False,
                linewidth=1)

    # Annotate the median values on the plot
    for index, row in pitcher_df.reset_index(drop=True).iterrows():
        ax1.text(index, 
                row['구종 점수'], 
                f'{row["구종 점수"]:.0f}', 
                color='white', 
                ha="center", 
                va="center",
                fontsize=8,
                weight='bold',
                clip_on=False,
                zorder=1000)
        
    # Customize ax1
    ax1.set_xlabel('구종')
    ax1.set_ylabel('구종 점수')
    ax1.grid(False)
    ax1.set_ylim(20, 80)
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.2, zorder=0)

    # Hide axes for additional subplots
    ax2.axis('off')
    ax1_left.axis('off')
    ax1_right.axis('off')
    ax1_top.axis('off')
    ax1_bot.axis('off')

    # Add text annotations
    ax1_bot.text(s='Data: KBO, 50구 이상', x=1, y=1, fontsize=12, ha='right')

    ax1_top.text(0.5, 0, f'{선수명} Location+',
                fontsize=24, ha='center', va='top')

    ax2.text(x=0.5, y=0.6, s='Location+는 구종 상관 없이 동등하게 Expected Run Value (xRV, 기대 득점 가치) 비교한 값\n'
                            'Location+는 평균: 100, 표준편차: 10인 정규분포\n'
                            '구종 점수는 구종별로 20-80스케일 적용해 구종 내에서 비교한 점수 (최소값: 20, 최대값: 80, 평균: 50, 표준편차: 10)',
                            ha='center', va='top', fontname='Malgun Gothic', fontsize=10)

    # Adjust subplot layout
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    # fig.set_facecolor('#e0e0e0')
    st.pyplot(fig)
    
    
def pitching_plot(df, 투수ID:int, 선수명:str):

    # 한글 폰트 설정 (예시: 'Malgun Gothic', 'AppleGothic' 등 OS에 맞게 선택)
    mpl.rcParams['font.family'] = 'Malgun Gothic'   # 윈도우
    # mpl.rcParams['font.family'] = 'AppleGothic'   # 맥

    # 마이너스(-) 깨짐 방지
    mpl.rcParams['axes.unicode_minus'] = False

    sns.set_style("ticks")
    mpl.rcParams['font.family'] = 'Malgun Gothic'
    mpl.rcParams['axes.unicode_minus'] = False
    
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(10, 8), dpi=450)
    gs = GridSpec(5, 3, height_ratios=[0.1, 10, 10, 2, 0.1], width_ratios=[1, 100, 1])
    gs.update(hspace=0.4, wspace=0.1)

    # Add subplots to the grid
    ax0 = fig.add_subplot(gs[1, 1]) 
    ax1 = fig.add_subplot(gs[2, 1])
    ax1_left = fig.add_subplot(gs[:, 0]) 
    ax1_right = fig.add_subplot(gs[:, 2]) 
    ax1_top = fig.add_subplot(gs[0, :])
    ax1_bot = fig.add_subplot(gs[4, 1])
    ax2 = fig.add_subplot(gs[3, 1])
    
    # Filter data for the specific pitcher
    pitcher_df = df[(df['투수ID'] == 투수ID) & (df['투구수'] >= 50)]
    
    #pitcher_df = df[(df['투수ID'] == 64001) & (df['투구수'] >= 50)]
    
    # Create a mapping dictionary from the list
    custom_order = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
    custom_order_dict = {pitch: index for index, pitch in enumerate(custom_order)}
    
    pitcher_df['order'] = pitcher_df['구종'].map(custom_order_dict)
    pitcher_df = pitcher_df.sort_values('order')
                         
    # Get unique pitch types for the pitcher
    pitcher_pitches = pitcher_df['구종'].unique()
    pitcher_pitches = [x for x in custom_order if x in pitcher_pitches]

                     
    # Plot Stuff+ with swarmplot for all players in the same position
    sns.swarmplot(data=df[(df['투구수'] >= 50)].dropna(subset=['구종']),
                x='구종',
                y='Pitching+',
                palette=dict_colour,
                alpha=0.3,
                size=3,
                ax=ax0,
                order=pitcher_pitches)

    # Overlay swarmplot for the specific pitcher
    sns.swarmplot(data=df[(df['투수ID'] == 투수ID) &
                                            (df['투구수'] >= 50)],
                x='구종',
                y='Pitching+',
                palette=dict_colour,
                alpha=1,
                size=16,
                ax=ax0,
                order=pitcher_pitches,
                edgecolor='black',
                linewidth=1)

    # Annotate the median values on the plot
    for index, row in pitcher_df.reset_index(drop=True).iterrows():
        ax0.text(index, 
                row['Pitching+'], 
                f'{row["Pitching+"]:.0f}', 
                color='white', 
                ha="center", 
                va="center",
                fontsize=8,
                weight='bold',
                clip_on=False)

    # Customize ax0
    ax0.set_xlabel('')
    ax0.set_ylabel('Pitching+')
    ax0.grid(False)
    ax0.set_ylim(70, 130)
    ax0.axhline(y=100, color='black', linestyle='--', alpha=0.2, zorder=0)

    # Plot pitch grade with swarmplot for all players in the same position
    sns.swarmplot(data=df[(df['투구수'] >= 50)].dropna(subset=['구종']),
                x='구종',
                y='구종 점수',
                palette=dict_colour,
                alpha=0.3,
                size=3,
                ax=ax1,
                clip_on=False,
                order=pitcher_pitches)

    # Overlay swarmplot for the specific pitcher
    sns.swarmplot(data=df[(df['투수ID'] == 투수ID) & (df['투구수'] >= 50)],
                x='구종',
                y='구종 점수',
                palette=dict_colour,
                alpha=1,
                size=16,
                ax=ax1,
                order=pitcher_pitches,
                edgecolor='black',
                clip_on=False,
                linewidth=1)

    # Annotate the median values on the plot
    for index, row in pitcher_df.reset_index(drop=True).iterrows():
        ax1.text(index, 
                row['구종 점수'], 
                f'{row["구종 점수"]:.0f}', 
                color='white', 
                ha="center", 
                va="center",
                fontsize=8,
                weight='bold',
                clip_on=False,
                zorder=1000)
        
    # Customize ax1
    ax1.set_xlabel('구종')
    ax1.set_ylabel('구종 점수')
    ax1.grid(False)
    ax1.set_ylim(20, 80)
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.2, zorder=0)

    # Hide axes for additional subplots
    ax2.axis('off')
    ax1_left.axis('off')
    ax1_right.axis('off')
    ax1_top.axis('off')
    ax1_bot.axis('off')

    # Add text annotations
    ax1_bot.text(s='Data: KBO, 50구 이상', x=1, y=1, fontsize=12, ha='right')

    ax1_top.text(0.5, 0, f'{선수명} Pitching+',
                fontsize=24, ha='center', va='top')

    ax2.text(x=0.5, y=0.6, s='Pitching+는 구종 상관 없이 동등하게 Expected Run Value (xRV, 기대 득점 가치) 비교한 값\n'
                            'Pitching+는 평균: 100, 표준편차: 10인 정규분포\n'
                            '구종 점수는 구종별로 20-80스케일 적용해 구종 내에서 비교한 점수 (최소값: 20, 최대값: 80, 평균: 50, 표준편차: 10)',
                            ha='center', va='top', fontname='Malgun Gothic', fontsize=10)

    # Adjust subplot layout
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    # fig.set_facecolor('#e0e0e0')
    st.pyplot(fig)
