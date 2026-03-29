import streamlit as st
import pandas as pd

import streamlit as st
import pandas as pd
import matplotlib

def render():
    # 컬러맵 정의 (matplotlib)
    cmap_sum = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#648FFF', '#FFFFFF', '#FFB000'])

    # 데이터프레임 column_config 설정
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

    st.markdown("""
    <div class="section-title">구종 Stuff+ 및 Location+ 평가</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader('구종 데이터 업로드 (.xlsx)', type=['xlsx'], key='ptype_file')
    if uploaded_file is not None:
        # Sheet 읽기
        df_stf = pd.read_excel(uploaded_file, sheet_name='stf')
        df_loc = pd.read_excel(uploaded_file, sheet_name='loc')

        # 구종 목록
        custom_order = ['직구', '싱커', '커터', '슬라이더', '스위퍼', '체인지업', '스플리터', '커브']
        unique_pitch_types_stf = [g for g in custom_order if g in df_stf['구종'].unique()]
        unique_pitch_types_loc = [g for g in custom_order if g in df_loc['구종'].unique()]
        unique_pitches = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        tab1, tab2 = st.tabs(['Stuff+', 'Location+'])

        # Stuff+ 탭
        with tab1:
            st.markdown(
                "#### Stuff+<br>"
                "투수가 얼마나 위력적인 공을 던졌는지 물리적인 변수로만 평가한 지표<br>"
                "구속·회전수·수직/수평 무브·릴리스 높이/사이드·익스텐션 등 반영<br>"
                "변화구는 직구와의 구속/무브 차이도 반영", unsafe_allow_html=True)

            selected_pitch_type = st.selectbox('구종 선택', unique_pitch_types_stf, key='ptype_stf')
            selected_n = st.selectbox('투구수 기준', unique_pitches, key='ptype_stf_n')
            df_stf_disp = df_stf[(df_stf['구종'] == selected_pitch_type) & (df_stf['투구수'] >= selected_n)]
            df_stf_disp = df_stf_disp.sort_values('Stuff+', ascending=False)
            # gradient 스타일 적용
            styled_df = df_stf_disp.style.background_gradient(subset=['Stuff+'], cmap=cmap_sum, vmin=80, vmax=120)
            st.dataframe(styled_df, hide_index=True, column_config=column_config_dict, width=1500)

            pitcher_options = sorted(df_stf_disp['선수명'] + ' - ' + df_stf_disp['투수ID'].astype(str))
            if pitcher_options:
                selected_pitcher = st.selectbox('투수 선택', pitcher_options, key='ptype_stf_pitcher')
                if st.button('Stuff+ 그래프 보기', key='ptype_stf_plot'):
                    pitcher_id = selected_pitcher.split(' - ')[-1]
                    from grade_plot import stuff_plot
                    stuff_plot(df_stf_disp, pitcher_id, selected_pitcher.split(' - ')[0])

        # Location+ 탭
        with tab2:
            st.markdown(
                "#### Location+<br>"
                "투구 위치 적합성 평가 지표<br>"
                "타자 타석 방향, 투수 투구 방향, 수직/수평 로케이션, 카운트 등 반영", unsafe_allow_html=True)

            selected_pitch_type = st.selectbox('구종 선택', unique_pitch_types_loc, key='ptype_loc')
            selected_n = st.selectbox('투구수 기준', unique_pitches, key='ptype_loc_n')
            df_loc_disp = df_loc[(df_loc['구종'] == selected_pitch_type) & (df_loc['투구수'] >= selected_n)]
            df_loc_disp = df_loc_disp.sort_values('Location+', ascending=False)
            # gradient 스타일 적용
            styled_df = df_loc_disp.style.background_gradient(subset=['Location+'], cmap=cmap_sum, vmin=80, vmax=120)
            st.dataframe(styled_df, hide_index=True, column_config=column_config_dict, width=1500)

            pitcher_options = sorted(df_loc_disp['선수명'] + ' - ' + df_loc_disp['투수ID'].astype(str))
            if pitcher_options:
                selected_pitcher = st.selectbox('투수 선택', pitcher_options, key='ptype_loc_pitcher')
                if st.button('Location+ 그래프 보기', key='ptype_loc_plot'):
                    pitcher_id = selected_pitcher.split(' - ')[-1]
                    from grade_plot import location_plot
                    location_plot(df_loc_disp, pitcher_id, selected_pitcher.split(' - ')[0])

        st.markdown(
            """
            <hr>
            <b>Stuff+, Location+ 설명:</b><br>
            - 2022~24년 KBO 데이터를 학습해 고교 평가 진행<br>
            - 평가 기준은 xRV(Expected Run Value) 기대 득점 가치<br>
            - 투구 결과가 아닌, 변수 기반 투구 평가<br>
            """, unsafe_allow_html=True)
    else:
        st.info("구종 Stuff+, Location+ 데이터(.xlsx)를 업로드해 주세요.")
