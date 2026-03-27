import streamlit as st

# DB 연결 선언
sUser = 'dsdata'
sPswd = 'entksqnstjrN2^'
sHost = 'dbears-tas-dw.ckwpzrkzl2lx.ap-northeast-2.rds.amazonaws.com'
nPort = 13306
sDBnm = 'dbears_dw'

@st.cache_data
# 쿼리문으로 DB 데이터 추출
def get_sql(sql, param=[]):
    import pymysql
    import pandas as pd
    
    try: 
        conn = pymysql.connect(user = sUser, passwd = sPswd, host = sHost, port = nPort, db = sDBnm)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(sql, param)
        data = cursor.fetchall()
    
    finally:
        cursor.close()
        conn.close()

    return data


