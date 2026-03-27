import streamlit as st

@st.cache_data

def get_sql(sql, param=[]):
    import pymysql
    import pandas as pd
    
    try: 
        conn = pymysql.connect(user = st.secrets["user"], passwd = st.secrets["password"], host = st.secrets["host"], port = st.secrets["port"], db = st.secrets["name"])
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql, param)
        data = cursor.fetchall()
    
    finally:
        cursor.close()
        conn.close()

    return data


