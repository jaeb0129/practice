"""
data_loader.py
──────────────
CSV 파일 업로드 로직을 한 곳에서 관리합니다.
- render_upload_ui() : 사이드바에 업로드 위젯 렌더링 (app.py 최상단에서 1회 호출)
- get_raw_df()       : hs_data   (raw 트래킹 데이터) 반환
- get_master_df()    : hs_profile (선수 마스터) 반환
"""

import streamlit as st
import pandas as pd


def render_upload_ui():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📂 데이터 업로드")

        raw_file = st.file_uploader(
            "① Raw 트래킹 데이터 (hs_data)",
            type=["csv"],
            key="uploader_raw",
            help="hs_data 테이블에 해당하는 CSV 파일",
        )
        master_file = st.file_uploader(
            "② 선수 마스터 (hs_profile)",
            type=["csv"],
            key="uploader_master",
            help="hs_profile 테이블에 해당하는 CSV 파일",
        )

        if raw_file is not None:
            try:
                st.session_state["df_raw"] = pd.read_csv(raw_file)
                st.success(f"✅ Raw 로드 완료 ({len(st.session_state['df_raw']):,}행)")
            except Exception as e:
                st.error(f"Raw CSV 읽기 오류: {e}")

        if master_file is not None:
            try:
                st.session_state["df_master"] = pd.read_csv(master_file, encoding='euc-kr')
                st.success(f"✅ 마스터 로드 완료 ({len(st.session_state['df_master']):,}행)")
            except Exception as e:
                st.error(f"Master CSV 읽기 오류: {e}")

        raw_ok    = "df_raw"    in st.session_state
        master_ok = "df_master" in st.session_state

        if raw_ok and master_ok:
            st.info(
                f"📊 Raw: {len(st.session_state['df_raw']):,}행\n\n"
                f"👤 Master: {len(st.session_state['df_master']):,}행"
            )
        else:
            missing = []
            if not raw_ok:    missing.append("① Raw 트래킹 데이터")
            if not master_ok: missing.append("② 선수 마스터")
            st.warning("⚠️ 미업로드:\n\n" + "\n\n".join(missing))


def get_raw_df() -> pd.DataFrame:
    if "df_raw" not in st.session_state:
        st.error("❌ Raw 트래킹 데이터(hs_data)가 업로드되지 않았습니다. 사이드바에서 CSV를 업로드해 주세요.")
        st.stop()
    return st.session_state["df_raw"].copy()


def get_master_df() -> pd.DataFrame:
    if "df_master" not in st.session_state:
        st.error("❌ 선수 마스터(hs_profile)가 업로드되지 않았습니다. 사이드바에서 CSV를 업로드해 주세요.")
        st.stop()
    return st.session_state["df_master"].copy()
