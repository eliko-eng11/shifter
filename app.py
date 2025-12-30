import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# =============================
# חובה: חייב להיות ראשון
# =============================
st.set_page_config(page_title="מערכת שיבוץ משמרות (Google Sheets)", layout="wide")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def get_gspread_client():
    if "gcp_service_account" not in st.secrets:
        raise ValueError("חסר Secrets בשם [gcp_service_account] ב-Streamlit Cloud.")
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=SCOPES
    )
    return gspread.authorize(creds)

def extract_sheet_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s

def read_sheet_as_df(sh, worksheet_name: str) -> pd.DataFrame:
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = [h.strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    df.columns = df.columns.str.strip()
    return df

def write_df_to_worksheet(sh, worksheet_name: str, df: pd.DataFrame):
    # אם קיים, ננקה; אם לא קיים, ניצור
    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=30)

    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(data)

# -----------------------------
# Greedy assignment
# -----------------------------
def simple_assignment(cost_matrix):
    used_rows, used_cols = set(), set()
    assignments = []
    rows = len(cost_matrix)
    cols = len(cost_matrix[0]) if rows > 0 else 0

    for _ in range(min(rows, cols)):
        best = None
        best_cost = 10 ** 18
        for i in range(rows):
            if i in used_rows:
                continue
            for j in range(cols):
                if j in used_cols:
                    continue
                c = cost_matrix[i][j]
                if c < best_cost:
                    best_cost = c
                    best = (i, j)
        if best is None:
            break
        r, c = best
        assignments.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    if not assignments:
        return [], []
    rr, cc = zip(*assignments)
    return list(rr), list(cc)

# -----------------------------
# Build schedule
# -----------------------------
def build_schedule(workers_df, req_df, pref_df, week_number):
    # clean
    for df in (workers_df, req_df, pref_df):
        df.columns = df.columns.str.strip()

    # enforce expected headers
    # workers: worker
    # requirements: day, shift, required
    # preferences: worker, day, shift, preference

    workers_df["worker"] = workers_df["worker"].astype(str).str.strip()
    req_df["day"] = req_df["day"].astype(str).str.strip()
    req_df["shift"] = req_df["shift"].astype(str).str.strip()
    pref_df["worker"] = pref_df["worker"].astype(str).str.strip()
    pref_df["day"] = pref_df["day"].astype(str).str.strip()
    pref_df["shift"] = pref_df["shift"].astype(str).str.strip()

    workers = workers_df["worker"].dropna().tolist()
    if not workers:
        raise ValueError("אין עובדים בטאב workers")

    req_df["required"] = pd.to_numeric(req_df["required"], errors="coerce").fillna(0).astype(int)

    shift_slots = []
    day_shift_pairs = []
    for _, row in req_df.iterrows():
        d, s, r = row["day"], row["shift"], int(row["required"])
        if r <= 0:
            continue
        pair = (d, s)
        if pair not in day_shift_pairs:
            day_shift_pairs.append(pair)
        for i in range(r):
            shift_slots.append((d, s, i))

    if not shift_slots:
        raise ValueError("אין דרישות בטאב requirements")

    ordered_days = list(dict.fromkeys([d for d, _, _ in shift_slots]))
    full_shifts = list(dict.fromkeys([s for _, s, _ in shift_slots]))

    # pref dict
    pref_dict = {}
    for _, row in pref_df.iterrows():
        try:
            p = int(row["preference"])
        except Exception:
            continue
        pref_dict[(row["worker"], row["day"], row["shift"])] = p

    # worker copies where pref >= 0
    worker_copies = []
    for w in workers:
        for (d, s) in day_shift_pairs:
            p = pref_dict.get((w, d, s), -1)
            if p >= 0:
                worker_copies.append((w, d, s))

    if not worker_copies:
        raise ValueError("אין העדפות חוקיות (>=0) בטאב preferences")

    # cost matrix
    cost_matrix = []
    for w, d, s in worker_copies:
        row_costs = []
        for sd, ss, _ in shift_slots:
            if (d, s) == (sd, ss):
                pref = pref_dict.get((w, d, s), 0)
                row_costs.append(100 if pref == 0 else 4 - pref)  # pref 3 -> 1 cost
            else:
                row_costs.append(1e6)
        cost_matrix.append(row_costs)

    cost_matrix = np.array(cost_matrix, dtype=float)
    row_ind, col_ind = simple_assignment(cost_matrix)

    assignments = []
    used_slots = set()
    worker_shift_count = {w: 0 for w in workers}
    worker_daily_shifts = {w: {d: [] for d in ordered_days} for w in workers}
    worker_day_shift_assigned = set()

    max_shifts_per_worker = len(shift_slots) // len(workers) + 1

    pairs = list(zip(row_ind, col_ind))
    pairs.sort(key=lambda x: cost_matrix[x[0], x[1]])

    for r, c in pairs:
        worker, _, _ = worker_copies[r]
        slot_day, slot_shift, slot_i = shift_slots[c]
        slot = (slot_day, slot_shift, slot_i)
        wds_key = (worker, slot_day, slot_shift)

        if cost_matrix[r][c] >= 1e6:
            continue
        if wds_key in worker_day_shift_assigned:
            continue
        if slot in used_slots:
            continue
        if worker_shift_count[worker] >= max_shifts_per_worker:
            continue

        # no adjacent shifts same day
        try:
            current_shift_index = full_shifts.index(slot_shift)
        except ValueError:
            current_shift_index = 0

        if any(abs(full_shifts.index(x) - current_shift_index) == 1 for x in worker_daily_shifts[worker][slot_day]):
            continue

        used_slots.add(slot)
        worker_day_shift_assigned.add(wds_key)

        assignments.append({"שבוע": int(week_number), "יום": slot_day, "משמרת": slot_shift, "עובד": worker})
        worker_shift_count[worker] += 1
        worker_daily_shifts[worker][slot_day].append(slot_shift)

    df = pd.DataFrame(assignments)
    if df.empty:
        raise ValueError("לא נוצר שיבוץ. בדוק נתונים/העדפות.")

    df["יום_מספר"] = df["יום"].apply(lambda x: ordered_days.index(x))
    df = df.sort_values(by=["שבוע", "יום_מספר", "משמרת", "עובד"])
    df = df[["שבוע", "יום", "משמרת", "עובד"]]
    return df

# =============================
# UI
# =============================
st.title("🛠️ מערכת שיבוץ משמרות (Google Sheets)")

sheet_link = st.text_input("הדבק קישור Google Sheet")
week_number = st.number_input("מספר שבוע לשיבוץ", min_value=1, step=1, value=1)

if st.button("🚀 בצע שיבוץ וכתוב חזרה ל-Google Sheet"):
    try:
        sheet_id = extract_sheet_id(sheet_link)
        if not sheet_id:
            st.error("לא זיהיתי Sheet ID. הדבק קישור מלא של Google Sheets.")
            st.stop()

        gc = get_gspread_client()
        sh = gc.open_by_key(sheet_id)

        workers_df = read_sheet_as_df(sh, "workers")
        req_df = read_sheet_as_df(sh, "requirements")
        pref_df = read_sheet_as_df(sh, "preferences")

        # טיפ: להכריח מספרים בעדיפות/required
        if "required" in req_df.columns:
            req_df["required"] = pd.to_numeric(req_df["required"], errors="coerce").fillna(0).astype(int)
        if "preference" in pref_df.columns:
            pref_df["preference"] = pd.to_numeric(pref_df["preference"], errors="coerce").fillna(-1).astype(int)

        schedule_df = build_schedule(workers_df, req_df, pref_df, int(week_number))

        out_tab = f"שבוע {int(week_number)}"
        write_df_to_worksheet(sh, out_tab, schedule_df)

        st.success(f"✅ נכתב לטאב: {out_tab}")
        st.dataframe(schedule_df, use_container_width=True)

    except Exception as e:
        st.exception(e)
        st.info("בדוק: שיתפת את ה-Sheet למייל של ה-service account עם הרשאת Editor.")
