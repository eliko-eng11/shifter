import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

st.set_page_config(page_title="בדיקת חיבור לגוגל שיטס", layout="wide")

def extract_sheet_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s

def get_gspread_client():
    if "gcp_service_account" not in st.secrets:
        raise ValueError("חסר Secrets בשם [gcp_service_account] ב-Streamlit.")
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=SCOPES
    )
    return gspread.authorize(creds)

st.title("🔎 בדיקת חיבור ל-Google Sheets")

sheet_link = st.text_input("הדבק קישור ל-Google Sheet")

if st.button("בדוק חיבור"):
    sheet_id = extract_sheet_id(sheet_link)
    gc = get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    st.success(f"✅ התחברתי! שם הקובץ: {sh.title}")
    st.write("הטאבים שיש בקובץ:")
    st.write([ws.title for ws in sh.worksheets()])
