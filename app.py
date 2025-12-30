import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

st.set_page_config(page_title="בדיקת גישה ל-Sheets", layout="wide")

def extract_sheet_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s

def get_client():
    if "gcp_service_account" not in st.secrets:
        st.error("חסר Secrets בשם gcp_service_account")
        st.stop()

    info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)

    st.success("Secrets נטענו")
    st.write("SERVICE ACCOUNT RUNNING AS:", creds.service_account_email)

    return gspread.authorize(creds)

st.title("בדיקת גישה ל-Google Sheet")

url = st.text_input("הדבק קישור ל-Sheet")
if st.button("בדוק"):
    sheet_id = extract_sheet_id(url)
    st.write("SHEET ID:", sheet_id)

    gc = get_client()
    try:
        sh = gc.open_by_key(sheet_id)
        st.success(f"נפתח! שם קובץ: {sh.title}")
        st.write("Tabs:", [w.title for w in sh.worksheets()])
    except Exception as e:
        st.error("נכשל לפתוח את הקובץ")
        st.exception(e)
        st.info("אם זה PermissionError => או שלא שיתפת למייל שמודפס למעלה, או שזה Shared Drive שדורש הוספה כחבר.")
