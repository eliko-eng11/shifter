# =============================
# 2) AUTH (SQLite) - Login/Register
# =============================
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return dk.hex()

def create_user(username: str, password: str) -> bool:
    """יצירת משתמש חדש במסד הנתונים"""
    username = username.strip()
    if not username or not password:
        return False
    
    salt = os.urandom(16).hex()
    p_hash = hash_password(password, salt)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", 
                    (username, p_hash, salt))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # שם המשתמש כבר קיים
        return False

def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username.strip(),))
    row = cur.fetchone()
    conn.close()
    if not row: return False
    stored_hash, salt = row
    return hmac.compare_digest(stored_hash, hash_password(password, salt))

def auth_gate():
    init_db()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    if st.session_state.logged_in:
        st.sidebar.success(f"שלום, {st.session_state.username}")
        if st.sidebar.button("התנתקות"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
        return True

    # עיצוב דף הכניסה
    st.title("🔐 כניסה למערכת השיבוץ")
    
    tab1, tab2 = st.tabs(["התחברות", "רישום משתמש חדש"])

    with tab1:
        u_login = st.text_input("שם משתמש", key="l_user")
        p_login = st.text_input("סיסמה", type="password", key="l_pass")
        if st.button("התחבר"):
            if verify_user(u_login, p_login):
                st.session_state.logged_in = True
                st.session_state.username = u_login
                st.success("מתחבר...")
                st.rerun()
            else:
                st.error("שם משתמש או סיסמה לא נכונים")

    with tab2:
        st.info("כאן ניתן להירשם למערכת בפעם הראשונה")
        u_reg = st.text_input("בחר שם משתמש", key="r_user")
        p_reg = st.text_input("בחר סיסמה", type="password", key="r_pass")
        p_reg_confirm = st.text_input("אימות סיסמה", type="password", key="r_pass_conf")
        
        if st.button("צור משתמש"):
            if not u_reg or not p_reg:
                st.warning("יש למלא את כל השדות")
            elif p_reg != p_reg_confirm:
                st.error("הסיסמאות לא תואמות!")
            elif len(p_reg) < 4:
                st.error("הסיסמה חייבת להכיל לפחות 4 תווים")
            else:
                if create_user(u_reg, p_reg):
                    st.success("המשתמש נוצר בהצלחה! כעת עבור ללשונית התחברות.")
                else:
                    st.error("שם המשתמש כבר קיים במערכת, בחר שם אחר.")
    
    st.stop() # עוצר את הרצת שאר הקוד עד להתחברות
