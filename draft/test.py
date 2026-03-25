import io
import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# ==============================
# 🔑 설정
# ==============================

FILENAME = "25HS.csv"   # Drive에 있는 파일 이름
FOLDER_ID = None           # 폴더 제한 (없으면 None)


# ==============================
# 🔐 인증 (Streamlit secrets 사용)
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
        spaces='drive',
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc"
    ).execute()

    files = results.get('files', [])

    if not files:
        raise FileNotFoundError(f"{filename} not found in Google Drive")

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
# 📊 데이터 로드 (캐시 ⭐)
# ==============================

@st.cache_data
def load_data(filename):
    service = get_drive_service()
    file_id = find_file_id(service, filename, FOLDER_ID)

    file_data = download_file(service, file_id)

    df = pd.read_csv(file_data)
    return df


# ==============================
# 🎨 Streamlit UI
# ==============================

st.title("⚾ MLB Data Viewer")

try:
    df = load_data(FILENAME)

    st.success(f"Loaded {len(df):,} rows")

    st.dataframe(df)

except Exception as e:
    st.error(f"Error: {e}")