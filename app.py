import streamlit as st
import pandas as pd
from PIL import Image
import os
import time
import requests

# Fast API Backend URL
API_URL = "http://127.0.0.1:8000"

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOC-INTEL | AI Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global Styles ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base dark background ── */
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #1E293B; }

    /* ── Hide default Streamlit header chrome when showing auth pages ── */
    .auth-hide-header header { visibility: hidden; }

    /* ── Generic button override ── */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #6366F1);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        width: 100%;
        height: 50px;
        font-size: 1rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1E293B;
        color: #E2E8F0;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* ── Auth card container ── */
    .auth-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 48px 56px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        max-width: 480px;
        margin: auto;
    }

    /* ── Auth card heading ── */
    .auth-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .auth-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 32px;
    }

    /* ── Divider ── */
    .auth-divider {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #475569;
        font-size: 0.85rem;
        margin: 18px 0;
    }
    .auth-divider::before, .auth-divider::after {
        content: "";
        flex: 1;
        border-top: 1px solid #334155;
    }

    /* ── Secondary link button ── */
    .auth-link {
        background: none !important;
        border: 1px solid #334155 !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    .auth-link:hover { border-color: #6366F1 !important; color: #E2E8F0 !important; }

    /* ── Feature badge ── */
    .feature-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 50px;
        padding: 6px 16px;
        font-size: 0.82rem;
        color: #94A3B8;
        margin: 4px;
    }

    /* ── Hero gradient text ── */
    .hero-text {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1.15;
        background: linear-gradient(135deg, #60A5FA 0%, #818CF8 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Main app textarea styling ── */
    .stTextArea > div > div > textarea { color: #10B981; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Session helpers
# ══════════════════════════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        "auth_page": "login",   # "login" | "signup"
        "token": None,
        "user": None,
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def _api_signup(name: str, email: str, password: str):
    try:
        r = requests.post(
            f"{API_URL}/api/v1/auth/signup",
            json={"name": name, "email": email, "password": password},
            timeout=10,
        )
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Cannot reach the backend API. Is it running?"}


def _api_login(email: str, password: str):
    try:
        r = requests.post(
            f"{API_URL}/api/v1/auth/login",
            json={"email": email, "password": password},
            timeout=10,
        )
        return r.status_code, r.json()
    except requests.exceptions.ConnectionError:
        return 503, {"detail": "Cannot reach the backend API. Is it running?"}


# ══════════════════════════════════════════════════════════════════════════════
# Auth pages  (full-page, NOT sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def _render_login_page():
    """Beautiful full-page login form."""
    # Two-column layout: branding left, form right
    left, _, right = st.columns([1.1, 0.15, 0.9])

    with left:
        st.markdown("<div style='padding-top:80px;'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-text'>DOC-INTEL<br>AI Platform</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#94A3B8; font-size:1.1rem; margin-top:16px;'>"
            "Enterprise-grade handwriting & document intelligence<br>powered by state-of-the-art AI.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style='margin-top:32px;'>
                <span class='feature-badge'>🔍 OCR Intelligence</span>
                <span class='feature-badge'>☁️ Cloud AI</span>
                <span class='feature-badge'>📋 Auto-Structure</span>
                <span class='feature-badge'>🔒 Secure Auth</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.markdown("<div class='auth-title'>Welcome back 👋</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-subtitle'>Sign in to your DOC-INTEL account</div>",
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("📧 Email address", placeholder="you@example.com")
            password = st.text_input("🔑 Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In →")

        if submitted:
            if not email or not password:
                st.error("Please fill in all fields.")
            else:
                with st.spinner("Authenticating..."):
                    code, data = _api_login(email, password)
                if code == 200:
                    st.session_state.token = data["access_token"]
                    st.session_state.user = data["user"]
                    st.success(f"✅ Welcome back, {data['user']['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"❌ {data.get('detail', 'Login failed.')}")

        st.markdown("<div class='auth-divider'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("Create Account", key="goto_signup"):
            st.session_state.auth_page = "signup"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _render_signup_page():
    """Beautiful full-page signup form."""
    left, _, right = st.columns([1.1, 0.15, 0.9])

    with left:
        st.markdown("<div style='padding-top:80px;'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-text'>Join<br>DOC-INTEL<br>Today</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#94A3B8; font-size:1.1rem; margin-top:16px;'>"
            "Create your free account and start processing documents<br>"
            "with the power of AI in minutes.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style='margin-top:32px;'>
                <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:12px;'>✅ Free to start</div>
                <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:12px;'>✅ No credit card required</div>
                <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:12px;'>✅ Unlimited document scans</div>
                <div style='color:#94A3B8; font-size:0.9rem;'>✅ Cloud & local AI engines</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.markdown("<div class='auth-title'>Create Account 🚀</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-subtitle'>Start your AI document journey for free</div>",
            unsafe_allow_html=True,
        )

        with st.form("signup_form", clear_on_submit=False):
            name = st.text_input("👤 Full name", placeholder="John Doe")
            email = st.text_input("📧 Email address", placeholder="you@example.com")
            password = st.text_input("🔑 Password", type="password", placeholder="At least 6 characters")
            confirm = st.text_input("🔒 Confirm password", type="password", placeholder="Repeat password")
            submitted = st.form_submit_button("Create Account →")

        if submitted:
            if not name or not email or not password or not confirm:
                st.error("Please fill in all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                with st.spinner("Creating your account..."):
                    code, data = _api_signup(name, email, password)
                if code == 201:
                    st.session_state.token = data["access_token"]
                    st.session_state.user = data["user"]
                    st.success(f"🎉 Account created! Welcome, {data['user']['name']}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"❌ {data.get('detail', 'Signup failed.')}")

        st.markdown("<div class='auth-divider'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Sign In Instead", key="goto_login"):
            st.session_state.auth_page = "login"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main application  (shown after authentication)
# ══════════════════════════════════════════════════════════════════════════════

def _render_main_app():
    """The main DOC-INTEL dashboard (shown after login)."""
    user = st.session_state.user or {}

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center; padding:12px 0;'>"
            f"<div style='font-size:2.5rem;'>🚀</div>"
            f"<div style='font-weight:700; color:#60A5FA;'>DOC-INTEL</div>"
            f"<div style='color:#94A3B8; font-size:0.8rem;'>v5.0 Vanguard</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(f"👤 **{user.get('name', 'User')}**")
        st.markdown(f"<span style='color:#94A3B8; font-size:0.82rem;'>{user.get('email', '')}</span>", unsafe_allow_html=True)
        st.markdown("---")

        st.title("🛡️ Vanguard Shields")
        medical_mode = st.toggle("Medical Context Correction", value=True)
        high_fidelity = st.toggle(
            "High-Fidelity Ink Scan", value=True,
            help="Uses TrOCR for deep handwriting verification.",
        )
        turbo_mode = st.toggle(
            "🚀 Vanguard Turbo Mode", value=False,
            help="MAX SPEED: Skips high-fidelity verification.",
        )
        st.markdown("---")
        st.markdown("### Platform Status")
        st.success("✅ Vision Engine Online")
        st.success("✅ OCR Engine Online")
        st.success("✅ FastAPI Backend")
        st.markdown("---")
        if st.button("🚪 Sign Out"):
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🚀 DOC-INTEL AI COMMAND CENTER")
    st.subheader("Enterprise-Grade Handwriting Intelligence (v5.0 DocTR)")
    st.info("⚡ **Vanguard 5.0 (Enterprise Client View):** Powered by High-Speed FastAPI Backend")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("📸 High-Res Document Upload")
        uploaded_file = st.file_uploader(
            "Upload Image (Auto-Enhancement Enabled)", type=["jpg", "png", "jpeg"]
        )
        file_bytes = None
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Loaded Source", use_container_width=True)
            file_bytes = uploaded_file.getvalue()

    with col2:
        st.header("🏆 Vanguard AI Intelligence (v2.0)")
        tab1, tab2 = st.tabs(["🔍 Live Scanner", "📂 Document History (Database)"])

        effective_hf = high_fidelity if not turbo_mode else False

        with tab1:
            st.markdown("---")
            st.subheader("☁️ Cloud Super Mode (FREE & Paid)")
            cloud_provider = st.selectbox(
                "Cloud Provider",
                ["Gemini (FREE)", "OpenAI GPT-4 (Paid)"],
                help="Gemini: Free tier (15 req/min). OpenAI: Paid but more accurate.",
            )
            cloud_mode = st.toggle("Enable Cloud OCR", value=False, help="Uses cloud AI for maximum accuracy")

            if cloud_mode:
                if "Gemini" in cloud_provider:
                    api_key_input = st.text_input(
                        "Gemini API Key (FREE)",
                        value=st.session_state.gemini_api_key,
                        type="password",
                        help="Get free key from: https://aistudio.google.com/app/apikey",
                    )
                    if api_key_input:
                        st.session_state.gemini_api_key = api_key_input
                        os.environ["GEMINI_API_KEY"] = api_key_input
                        st.success("✅ Gemini API Key set!")
                    elif not st.session_state.gemini_api_key:
                        st.info("💡 Get FREE API key: https://aistudio.google.com/app/apikey")
                else:
                    api_key_input = st.text_input(
                        "OpenAI API Key",
                        value=st.session_state.openai_api_key,
                        type="password",
                    )
                    if api_key_input:
                        st.session_state.openai_api_key = api_key_input
                        os.environ["OPENAI_API_KEY"] = api_key_input
                        st.success("✅ OpenAI API Key set!")
                    elif not st.session_state.openai_api_key:
                        st.warning("⚠️ Please enter your OpenAI API key")

            extraction_mode = st.selectbox(
                "Intelligence Mode",
                ["Auto-Detect (AI Recommendation)", "Force Paragraph (Notes/Letters)", "Force Table (Forms/Bills)"],
            )

            if uploaded_file and st.button("🚀 EXECUTE VANGUARD DEEP SCAN"):
                with st.status("🛸 Routing to Vanguard API Gateway...", expanded=True) as status:
                    st.write("📡 Step 1: Securely transmitting encrypted file to Backend server...")

                    provider = "gemini" if "Gemini" in cloud_provider else "openai"
                    api_key_to_use = ""
                    if cloud_mode:
                        api_key_to_use = (
                            st.session_state.gemini_api_key if provider == "gemini"
                            else st.session_state.openai_api_key
                        )

                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    data_payload = {
                        "cloud_mode": cloud_mode,
                        "provider": provider,
                        "high_fidelity": effective_hf,
                    }
                    if "Force Paragraph" in extraction_mode:
                        data_payload["force_mode"] = "HANDWRITTEN_NOTE"
                    elif "Force Table" in extraction_mode:
                        data_payload["force_mode"] = "STRUCTURED_FORM"

                    st.write(f"🧠 Step 2: Running Deep AI Neural Layers ({'Cloud' if cloud_mode else 'Local'})...")

                    try:
                        if cloud_mode and api_key_to_use:
                            os.environ[f"{provider.upper()}_API_KEY"] = api_key_to_use

                        headers = {}
                        if st.session_state.token:
                            headers["Authorization"] = f"Bearer {st.session_state.token}"

                        response = requests.post(
                            f"{API_URL}/api/v1/process_document",
                            files=files,
                            data=data_payload,
                            headers=headers,
                            timeout=300,
                        )
                        response.raise_for_status()
                        result = response.json()

                        if result["status"] == "success":
                            doc_analysis = result["analysis"]
                            data = result["structured_data"]
                            final_type = doc_analysis["type"]
                            paths = result["paths"]

                            st.write("💾 Step 3: Analysis Complete. Saved to Server DB.")
                            status.update(
                                label=f"💎 Vanguard Analysis: {final_type} | Record ID: #{result['record_id']}",
                                state="complete",
                                expanded=False,
                            )

                            if os.path.exists(paths["processed"]):
                                st.image(paths["processed"], caption="Vanguard Enhanced Vision", use_container_width=True)
                        else:
                            st.error("API Error: Unknown status returned.")
                            st.stop()

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Backend API Connection Failed! Is the FastAPI server running on {API_URL}?")
                        st.info(f"Technical Details: {e}")
                        status.update(label="API Offline", state="error")
                        st.stop()

                m1, m2, m3 = st.columns(3)
                m1.metric("Handwriting Score", f"{doc_analysis.get('confidence_score', 0) * 100:.1f}%")
                m2.metric("Intelligence Layer", "Vanguard 5.0 API")
                m3.metric("Context Hits", "Medical Standard")

                if final_type == "HANDWRITTEN_NOTE" or not data.get("table"):
                    st.subheader("📝 Digitized Paper (Vanguard Reconstruction)")
                    text = data.get("full_text", "")
                    st.markdown(
                        f"""<div style="background-color:#f0f4f8; padding:30px; border:2px solid #3498db;
                        border-radius:8px; color:#1a2a3a; font-family:'Inter',sans-serif;
                        line-height:1.8; font-size:1.2em; white-space:pre-wrap;">{text}</div>""",
                        unsafe_allow_html=True,
                    )
                    st.download_button("📥 DOWNLOAD VANGUARD TEXT", text, "vanguard_note.txt")
                else:
                    st.subheader("📋 Professional Data Matrix")
                    kv = data.get("kv", {})
                    if kv:
                        with st.expander("📌 Context Fields", expanded=True):
                            if isinstance(kv, dict):
                                for k, v in kv.items():
                                    st.write(f"**{k}:** `{v}`")
                            else:
                                st.write(str(kv))

                    table_data = data.get("table", [])
                    if table_data:
                        if all(isinstance(x, str) for x in table_data):
                            df = pd.DataFrame({"Text content": table_data})
                        else:
                            df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.download_button("📥 DOWNLOAD VANGUARD CSV", df.to_csv(index=False), "vanguard_data.csv")

        with tab2:
            st.subheader("🗄️ Enterprise Database Records")
            try:
                response = requests.get(f"{API_URL}/api/v1/history", timeout=10)
                if response.status_code == 200:
                    records = response.json()
                    if records:
                        st.write(f"Total documents Server-Synced: **{len(records)}**")
                        history_data = [
                            {
                                "ID": r["id"],
                                "File": r["filename"],
                                "Type": r["document_type"],
                                "Engine": r["ocr_provider"],
                                "Time": r["upload_time"],
                            }
                            for r in records
                        ]
                        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)

                        st.markdown("---")
                        st.markdown("### 🔎 Inspect Specific Record")
                        selected_id = st.selectbox("Select Record ID to view details", [r["id"] for r in records])
                        if selected_id:
                            selected_record = next(r for r in records if r["id"] == selected_id)
                            col_a, col_b = st.columns([1, 1])
                            with col_a:
                                st.write(f"**Filename:** {selected_record['filename']}")
                                st.write(f"**Type:** {selected_record['document_type']}")
                                st.write(f"**Confidence:** {selected_record['confidence_score'] * 100:.1f}%")
                                if os.path.exists(selected_record.get("processed_image_path", "")):
                                    st.image(selected_record["processed_image_path"], caption="Processed Image", width=300)
                            with col_b:
                                if selected_record.get("extracted_json"):
                                    st.write("**Structured Data JSON:**")
                                    st.json(selected_record["extracted_json"])
                                else:
                                    st.write("**Extracted Text:**")
                                    st.text(selected_record.get("extracted_text", ""))
                    else:
                        st.info("No records found in database. Scan a document first!")
                else:
                    st.error("Failed to fetch history from API Server.")
            except Exception as e:
                st.error(f"Error connecting to Database API: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

_init_session()

if not _is_authenticated():
    if st.session_state.auth_page == "signup":
        _render_signup_page()
    else:
        _render_login_page()
else:
    _render_main_app()

