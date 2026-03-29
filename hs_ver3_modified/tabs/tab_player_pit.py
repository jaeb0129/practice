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
    
    rslt['PitcherId'] = rslt['PitcherId'].astype(str)
    master['PLER_TRKNG_ID'] = master['PLER_TRKNG_ID'].astype(str)
    
    data = pd.merge(rslt, master.loc[:,['PLER_TRKNG_ID','PLER_NAME', 'BKNO', 'TEAM_NM'] ],  left_on='PitcherId', right_on='PLER_TRKNG_ID', how='left')
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

    data['TaggedPitchType'] = np.select(conditions, choices, default="Unknown").astype(object)

    order = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']

    data['TaggedPitchType'] = pd.Categorical(data['TaggedPitchType'], categories=order, ordered=True)
    
    data['BatterSide'] = data['BatterSide'].map({'Right': '우타', 'Left': '좌타'})
    order_stand = ['우타', '좌타']
    data['BatterSide'] = pd.Categorical(data['BatterSide'], categories=order_stand, ordered=True)
    
    data.rename(columns = {'TaggedPitchType': '구종', 'InducedVertBreak': '수직 무브먼트',
                           'HorzBreak': '수평 무브먼트', 'RelSpeed': '구속', 'SpinRate': '회전수', 'SpinAxis': '회전축',
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


    data["name_bk"] = data["PLER_NAME"] + "_" + data["BKNO"].astype(str)
    data['year'] = pd.to_datetime(data['Date'], errors='coerce').dt.year

    st.sidebar.title('투수')
    
    select_year = st.sidebar.selectbox('확인하고 싶은 연도를 선택하세요',
                                         sorted(data.year.dropna().unique()))

    select_school = st.sidebar.selectbox(
        '확인하고 싶은 학교를 선택하세요',
        sorted(data[(data.year == select_year)].TEAM_NM.dropna().unique())
    )

    select_pitcher = st.sidebar.selectbox(
        '확인하고 싶은 투수를 선택하세요',
       sorted(data[(data.year == select_year) & (data.TEAM_NM == select_school)].name_bk.dropna().unique())
    )

    select_date = st.sidebar.multiselect(
        '확인하고 싶은 날짜를 선택하세요',
        data[(data.year == select_year) & (data.TEAM_NM == select_school) & (data.name_bk == select_pitcher)].Date.unique()
    )

    st.title(f'{select_pitcher} 투구 대시보드')

    def track(Year, Pitcher, School, Date):
        pdata = data.loc[
            (data.year == Year) &
            (data.name_bk == Pitcher) &
            (data.Date.isin(Date)) &
            (data.TEAM_NM == School),
            :
        ]
    
        # S%, 헛스윙% 계산을 위한 플래그 추가
        strike_events = ["StrikeCalled", "StrikeSwinging", "InPlay", "FoulBall"]
        swing_events = ["StrikeSwinging", "InPlay", "FoulBall"]
        pdata["is_strike"] = pdata["PitchCall"].isin(strike_events).astype(int)
        pdata["is_swing"] = pdata["PitchCall"].isin(swing_events).astype(int)
        pdata["is_whiff"] = (pdata["PitchCall"] == "StrikeSwinging").astype(int)
    
        # 구종별 집계
        table = pdata.groupby('구종')[[
            'Date', '구속', '회전수', '회전축', '수직 무브먼트', '수평 무브먼트',
            '릴리스 높이', '릴리스 사이드', '익스텐션',
            'is_strike', 'is_swing', 'is_whiff'
        ]].agg({
            'Date': 'count',
            '구속': ['mean', 'max'],
            '회전수': 'mean', '회전축':'mean',
            '수직 무브먼트': 'mean', '수평 무브먼트': 'mean',
            '릴리스 높이': 'mean', '릴리스 사이드': 'mean', '익스텐션': 'mean',
            'is_strike': 'sum',
            'is_swing': 'sum',
            'is_whiff': 'sum'
        }).dropna().round(1)
    
        # 컬럼명 정리
        table.columns = [
            '투구수', '평균_구속(km/h)', '최고_구속(km/h)', '회전수(rpm)', '회전축',
            '수직 무브먼트(cm)', '수평 무브먼트(cm)', '릴리스 높이(cm)',
            '릴리스 사이드(cm)', '익스텐션(cm)',
            '스트라이크', '스윙', '헛스윙'
        ]
    
        # 구사율 계산
        table['구사율(%)'] = round((table['투구수']/sum(table['투구수'])) * 100, 1)
    
        # S%, 헛스윙% 추가
        table['S%'] = round((table['스트라이크'] / table['투구수']) * 100, 1)
        table['헛스윙%'] = round((table['헛스윙'] / table['스윙']) * 100, 1)
    
        # 기타 format 적용 및 컬럼 정리
        table['수직 무브먼트(cm)'] = round(table['수직 무브먼트(cm)'], 1)
        table['수평 무브먼트(cm)'] = round(table['수평 무브먼트(cm)'], 1)
        table['평균_구속(km/h)'] = round(table['평균_구속(km/h)'], 1)
        table['최고_구속(km/h)'] = round(table['최고_구속(km/h)'], 1)
        table['릴리스 높이(cm)'] = round(table['릴리스 높이(cm)'], 1)
        table['릴리스 사이드(cm)'] = round(table['릴리스 사이드(cm)'], 1)
        table['익스텐션(cm)'] = round(table['익스텐션(cm)'], 1)
        table['회전수(rpm)'] = table['회전수(rpm)'].astype(int)
        table['회전축'] = table['회전축'].astype(int)
    
        # 최종 컬럼 순서 지정
        table = table[[
            '투구수', '구사율(%)',
            '평균_구속(km/h)', '최고_구속(km/h)', '회전수(rpm)', '회전축',
            '수직 무브먼트(cm)', '수평 무브먼트(cm)', '릴리스 높이(cm)',
            '릴리스 사이드(cm)', '익스텐션(cm)', 'S%', '헛스윙%'
        ]]
        table = table.sort_values(by='구사율(%)', ascending=False).reset_index()
        table = table.style.format({
            "구사율(%)": "{:.1f}",
            "S%": "{:.1f}",
            "헛스윙%": "{:.1f}",
            "수직 무브먼트(cm)": "{:.1f}",
            "수평 무브먼트(cm)": "{:.1f}",
            "평균_구속(km/h)": "{:.1f}",
            "최고_구속(km/h)": "{:.1f}",
            "릴리스 높이(cm)": "{:.1f}",
            "릴리스 사이드(cm)": "{:.1f}",
            "익스텐션(cm)": "{:.1f}"
        })
    
        return table

    def movement(Year, Pitcher, School, Date):
        pdata = data.loc[(data.year == Year) & (data.name_bk == Pitcher) & (data.Date.isin(Date)) & (data.TEAM_NM == School),:]
        pdata = pdata.sort_values(by=['구종'])

        order = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']
        actual_order = [p for p in order if p in pdata['구종'].values]
        
        pdata['구종'] = pdata['구종'].astype(str)

        fig = px.scatter(data_frame=pdata, 
                     x='수평 무브먼트', 
                     y='수직 무브먼트',
                     color='구종',
                     color_discrete_map=dict_colour,
                     hover_data=['구속', '회전수'],
                     category_orders={'구종': actual_order})

        fig.update_layout(xaxis=dict(range=[80, -80], dtick=10, autorange=False, tickfont=dict(size=10)),
                          yaxis=dict(range=[-80, 80], dtick=10, tickfont=dict(size=10)),
                          plot_bgcolor='whitesmoke',
                          width=700,
                          height=700,
                         title=f"구종별 무브먼트", title_x=0)

        fig.add_trace(go.Scatter(x=[0, 0], y=[80, -80], mode='lines', line=dict(color='red'), showlegend=False))
        fig.add_trace(go.Scatter(x=[80, -80], y=[0, 0], mode='lines', line=dict(color='red'), showlegend=False))

        return fig

    def release(Year, Pitcher, School, Date):
        pdata = data.loc[(data.year == Year) & (data.name_bk == Pitcher) & (data.Date.isin(Date)) & (data.TEAM_NM == School),:]
        pdata = pdata.sort_values(by=['구종'])

        
        order = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']
        actual_order = [p for p in order if p in pdata['구종'].values]
    
        pdata['구종'] = pdata['구종'].astype(str)

        fig = px.scatter(data_frame=pdata, 
                     x='릴리스 사이드',
                     y='릴리스 높이',
                     color='구종',
                     color_discrete_map=dict_colour,
                     hover_data=['구속', '회전수'],
                     category_orders={'구종': actual_order})

        fig.update_layout(xaxis=dict(range=[150, -150], dtick=20, autorange=False, tickfont=dict(size=10)),
                          yaxis=dict(range=[120, 220], dtick=10, tickfont=dict(size=10)),
                          plot_bgcolor='whitesmoke',
                          width=700,
                          height=700,
                         title=f"구종별 릴리스 포인트", title_x=0)

        fig.add_trace(go.Scatter(x=[0, 0], y=[120, 220], mode='lines', line=dict(color='red'), showlegend=False))
        fig.add_trace(go.Scatter(x=[150, -150], y=[170, 170], mode='lines', line=dict(color='red'), showlegend=False))

        return fig

    def location(Year, Pitcher, School, Date):
        pdata = data.loc[(data.year == Year) & (data.name_bk == Pitcher) & (data.Date.isin(Date)) & (data.TEAM_NM == School),:]
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
        
        order = ['직구', '싱커', '커터', '슬라이더', '체인지업', '스플리터', '커브', '너클']
        actual_order = [p for p in order if p in pdata['구종'].values]
        pdata['구종'] = pdata['구종'].astype(str)

        fig = px.scatter(data_frame=pdata, 
                     x='수평 로케이션', 
                     y='수직 로케이션',
                     color='구종',
                     color_discrete_map=dict_colour,
                     facet_col='구종',
                     facet_row='BatterSide',
                     category_orders={'구종': actual_order,
                                      'BatterSide': ['우타', '좌타']},
                     title = '구종별 로케이션')
        
        # 모든 facet별 x축과 y축 제목 숨기기
        fig.for_each_xaxis(lambda axis: axis.update(title=None))
        fig.for_each_yaxis(lambda axis: axis.update(title=None))
        
        fig.update_layout(
        width=1000,
        height=500,
        plot_bgcolor='white')
        
        fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split('=')[-1]))
        
        # Facet 제목 스타일 조정
        annotations = fig['layout']['annotations']  # 모든 annotations 가져오기
        for annotation in annotations:
            annotation['font'] = dict(size=16, family='Arial')  # Facet 제목 크기 및 스타일 변경
        
        fig.add_shape(type ='rect',
                      x0 = L, y0 = Bot,
                      x1 = R, y1 = Top,
                      line_width = 3,
                      line_color = 'black',
                      opacity = 0.5, row="all", col="all")
        fig.update_xaxes(range=[L-(2.5*30.48), R+(2.5*30.48)], showline=False, zeroline=False)
        fig.update_yaxes(range=[Bot-(3*30.48), Top+(3*30.48)], showline=False, zeroline=False)
        
        fig.add_trace(go.Scatter(x=[R_m, L_p, L_m, Center, R_p, R_m], y=[S_height, S_height, M_height, E_height, M_height, S_height], showlegend=False, marker_color = 'black'), row="all", col="all")
        return fig
    
    def location2(Year, Pitcher, School, Date):
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        pdata = data.loc[
            (data.name_bk == Pitcher) &
            (data.Date.isin(Date)) &
            (data.TEAM_NM == School) &
            (data.year == Year)
        ].copy()
    
        # 데이터가 없을 경우 처리
        if pdata.empty or pdata['구종'].nunique() == 0 or pdata['BatterSide'].nunique() == 0:
            import streamlit as st
            st.warning("해당 조건에 맞는 데이터가 없습니다.")
            return None
    
        pdata = pdata.sort_values(by=['구종', 'BatterSide'])
    
        if hasattr(pdata['구종'], 'cat'):
            pdata['구종'] = pdata['구종'].cat.remove_unused_categories()
    
        # 스트존 및 배터박스 좌표
        L, R = -0.708333 * 30.48, +0.708333 * 30.48
        Bot, Top = 1.5 * 30.48, 3.5 * 30.48
        L_p, L_m = L + (0.1 * 30.48), L - (0.1 * 30.48)
        R_p, R_m = R + (0.1 * 30.48), R - (0.1 * 30.48)
        Center = 0
        S_height, M_height, E_height = 0, -0.6 * 30.48, -1.0 * 30.48
        x = [R_m, L_p, L_m, Center, R_p, R_m]
        y = [S_height, S_height, M_height, E_height, M_height, S_height]
        x2 = [L, R, R, L, L]
        y2 = [Top, Top, Bot, Bot, Top]
    
        import matplotlib
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.rc('font', family='Malgun Gothic')
        plt.rc('figure', titlesize=20)
        plt.figure(figsize=(10, 10), dpi=100)
    
        # FacetGrid 생성 (row=BatterSide, col=구종)
        g = sns.FacetGrid(
            pdata,
            row="BatterSide",
            col="구종",
            sharex=True,
            sharey=True,
            ylim=(Bot - (3 * 30.48), Top + (2 * 30.48)),
            xlim=(L - (2.5 * 30.48), R + (2.5 * 30.48))
        )
    
        # 커스텀 그리기 함수 (색상 적용)
        def kde_or_scatter(data, color=None, **kwargs):
            ax = plt.gca()
            if len(data) < 10:
                pitch_type = data['구종'].iloc[0] if not data['구종'].empty else '기타'
                c = dict_colour.get(pitch_type, 'gray')
                ax.scatter(
                    data['수평 로케이션'], data['수직 로케이션'],
                    color=c, s=80, alpha=0.8, edgecolor='black', linewidth=0.5
                )
            else:
                sns.kdeplot(
                    x=data['수평 로케이션'],
                    y=data['수직 로케이션'],
                    fill=True,
                    cmap="coolwarm",
                    levels=20,
                    bw_adjust=1.5,
                    ax=ax
                )
    
        g.map_dataframe(kde_or_scatter)
    
        # 스트존, 배터박스 선 그리기
        g.map_dataframe(lambda data, **kwargs: plt.plot(x2, y2, '-', linewidth=2, color='black'))
        g.map_dataframe(lambda data, **kwargs: plt.plot(x, y, '-', linewidth=0.5, color='black'))
    
        # 축 및 제목 설정
        g.set_axis_labels('수평 로케이션', '수직 로케이션')
        g.set_titles(col_template="{col_name}", row_template="{row_name}", fontsize=30)
    
        def clean_title(text):
            return text.replace("우 |", "").replace("좌 |", "").strip()
    
        for ax in g.axes.flat:
            ax.set_title(clean_title(ax.get_title()), fontsize=20)
    
        g.fig.set_size_inches(20, 5.5)
        return g.fig

    result_table = track(select_year, select_pitcher, select_school, select_date)
    result_movement = movement(select_year, select_pitcher, select_school, select_date)
    result_rlse_point = release(select_year, select_pitcher, select_school, select_date)
    result_location = location(select_year, select_pitcher, select_school, select_date)
    result_location2 = location2(select_year, select_pitcher, select_school, select_date)
    
    
    st.subheader('구종별 트래킹')
    st.dataframe(result_table, width=1500, hide_index=True)
    st.plotly_chart(result_movement)
    st.plotly_chart(result_rlse_point)
    st.plotly_chart(result_location)
    st.pyplot(result_location2)
    
