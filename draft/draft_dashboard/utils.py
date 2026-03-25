# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 02:04:53 2026

@author: jaeb0
"""

import streamlit as st
from googleapiclient.discovery import build
from google.oauth2 import service_account
import io

@st.cache_data
def load_raw():
    import io
    import pandas as pd
    import streamlit as st
    
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    # ==============================
    # 🔑 설정
    # ==============================
    
    FILE_HS = "HS.csv"
    FILE_MASTER = "마스터정보.csv"
    FOLDER_ID = None


    # ==============================
    # 🔐 인증
    # ==============================

    def get_drive_service():
        SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )

        service = build('drive', 'v3', credentials=creds)
        return service


    # ==============================
    # 🔍 파일 찾기
    # ==============================

    def find_file_id(service, filename, folder_id=None):
        if folder_id:
            query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        else:
            query = f"name = '{filename}' and trashed = false"

        results = service.files().list(
            q=query,
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc"
        ).execute()

        files = results.get('files', [])

        if not files:
            raise FileNotFoundError(f"{filename} not found")

        return files[0]['id']


    # ==============================
    # 📥 다운로드
    # ==============================

    def download_file(service, file_id):
        request = service.files().get_media(fileId=file_id)

        file_data = io.BytesIO()
        downloader = MediaIoBaseDownload(file_data, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        file_data.seek(0)
        return file_data


    # ==============================
    # 📊 CSV 로드 (캐시 ⭐)
    # ==============================

    @st.cache_data
    def load_csv(filename):
        service = get_drive_service()
        file_id = find_file_id(service, filename, FOLDER_ID)

        file_data = download_file(service, file_id)
        df = pd.read_csv(file_data)

        return df


    # ==============================
    # 📊 데이터 로드
    # ==============================

    hs25 = load_csv(FILE_HS)
    master = load_csv(FILE_MASTER)


    # ==============================
    # 🔗 병합
    # ==============================

    cols = ['tm_player_id', 'player_name', 'player_backno', 'kor_teamname', 'pos_eng']

    data_pit = pd.merge(
        hs25,
        master.loc[:, cols],
        left_on='PitcherId',
        right_on='tm_player_id',
        how='left'
    )

    data_bat = pd.merge(
        hs25,
        master.loc[:, cols],
        left_on='BatterId',
        right_on='tm_player_id',
        how='left'
    )

    # ==============================
    # 🏷️ 이름 + 등번호
    # ==============================

    data_pit["name_bk"] = data_pit["player_name"].astype(str) + "_" + data_pit["player_backno"].astype(str)
    data_bat["name_bk"] = data_bat["player_name"].astype(str) + "_" + data_bat["player_backno"].astype(str)


    return data_pit, data_bat