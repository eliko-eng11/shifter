import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError
import streamlit as st
st.write("APP VERSION:", "2025-12-30 20:30")  # תשנה כל פעם


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

st.set_page_config(page_title="בדיקת גישה ל-Google Sheets", layout="wide")
st.title("🔧 בדיקה אמיתית של הרשאות Google Sheets")

def extract_sheet_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if not s:
        return ""
    if "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s  # אם הדבקת ID

def get_creds_and_email():
    if "gcp_service_account" not in st.secrets:
        raise ValueError("❌ חסר ב-Secrets: gcp_service_account")
    info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return creds, getattr(creds, "service_account_email", None)

def open_sheet(creds, sheet_id: str):
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)

sheet_link = st.text_input("הדבק קישור / ID של Google Sheet", value="")

if st.button("בדוק עכשיו"):
    try:
        # 1) Sheet ID
        sheet_id = extract_sheet_id(sheet_link)
        if not sheet_id:
            st.error("❌ לא זיהיתי Sheet ID. הדבק קישור מלא או ID.")
            st.stop()

        # 2) Creds + Email בפועל
        creds, sa_email = get_creds_and_email()
        st.success("✅ Secrets נטענו בהצלחה")
        st.write("📧 **Service Account שהקוד משתמש בו בפועל:**")
        st.code(sa_email or "לא הצלחתי לחלץ אימייל מה-credentials")

        st.warning("👉 עכשיו תוודא שזה בדיוק המייל ששיתפת אליו את ה-Sheet (Editor).")

        # 3) נסיון פתיחה
        st.info("מנסה לפתוח את הקובץ...")
        sh = open_sheet(creds, sheet_id)
        st.success(f"✅ נפתח! שם הקובץ: {sh.title}")

        # 4) טאבים
        tabs = [ws.title for ws in sh.worksheets()]
        st.write("📑 טאבים קיימים:")
        st.write(tabs)

    except APIError as e:
        st.error("❌ Google APIError")
        st.write("זה כמעט תמיד אומר: לא שיתפת נכון / API לא מופעל / Service Account לא נכון.")
        st.write("פירוט מלא (חשוב!):")
        st.exception(e)

    except Exception as e:
        st.error("❌ שגיאה כללית")
        st.exception(e)

