import streamlit as st
import pandas as pd
from PIL import Image
import os
import time
import requests
from translation_engine import DocumentTranslationEngine, LANGUAGE_NAME_BY_CODE

# Fast API Backend URL
API_URL = "http://127.0.0.1:8000"

# Translation options for frontend selection.
LANGUAGE_OPTIONS = {"Original (No Translation)": "original"}
LANGUAGE_OPTIONS.update({name: code for code, name in LANGUAGE_NAME_BY_CODE.items()})
TRANSLATION_ENGINE = DocumentTranslationEngine()


def resolve_local_image_path(raw_path: str):
    """Resolve image paths that may come from another OS/workspace."""
    if not raw_path:
        return None

    path_str = str(raw_path).strip()
    if os.path.exists(path_str):
        return path_str

    # Normalize separators from Windows-style DB entries.
    normalized = path_str.replace("\\", "/")
    file_name = os.path.basename(normalized)
    if not file_name:
        return None

    candidates = [
        os.path.abspath(os.path.join("uploads", file_name)),
    ]

    if file_name.startswith("clean_"):
        candidates.append(os.path.abspath(os.path.join("uploads", file_name[len("clean_") :])))
    else:
        candidates.append(os.path.abspath(os.path.join("uploads", f"clean_{file_name}")))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def fetch_api_json(path: str, method: str = "GET", **kwargs):
    response = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    response.raise_for_status()
    return response.json()


def get_backend_status():
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        response.raise_for_status()
        payload = response.json()
        return True, f"Backend online · v{payload.get('version', 'unknown')}"
    except requests.exceptions.ConnectionError:
        return False, "Backend offline"
    except requests.exceptions.Timeout:
        return False, "Backend timed out"
    except Exception as exc:
        return False, f"Backend error: {exc}"


def get_error_message(exc: Exception, fallback: str) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "FastAPI backend is offline. Start api.py and try again."
    if isinstance(exc, requests.exceptions.Timeout):
        return "Backend request timed out. If the server just started, wait for model loading to finish."
    if isinstance(exc, requests.exceptions.HTTPError) and getattr(exc, "response", None) is not None:
        try:
            detail = exc.response.json().get("detail")
            if detail:
                return str(detail)
        except Exception:
            pass
        return f"Request failed with status {exc.response.status_code}."
    return f"{fallback}: {exc}"


def confidence_to_percent(value):
    return f"{float(value or 0.0) * 100:.1f}%"


def render_field_predictions(predictions):
    if not predictions:
        return

    rows = []
    for item in predictions:
        rows.append({
            "Field": item.get("field_name", ""),
            "Type": item.get("field_type", ""),
            "Value": item.get("corrected_value") or item.get("predicted_value", ""),
            "Confidence": confidence_to_percent(item.get("confidence_score", 0.0)),
            "Status": item.get("review_status", ""),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_extraction_blocks(blocks, limit=25):
    if not blocks:
        return

    rows = []
    for item in blocks[:limit]:
        rows.append({
            "Block": item.get("block_type", ""),
            "Text": item.get("text", ""),
            "Confidence": confidence_to_percent(item.get("confidence_score", 0.0)),
            "BBox": item.get("bbox", ""),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Page Configuration 
st.set_page_config(page_title="DOC-INTEL | AI Intelligence", page_icon="🚀", layout="wide")

# Custom Styling (Premium Look)
st.markdown("""
    <style>
    .main { background-color: #0F172A; color: white; }
    .stButton>button { 
        background-color: #3B82F6; color: white; border-radius: 10px; 
        font-weight: bold; width: 100%; height: 50px;
    }
    .auth-card {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
    }
    .auth-title {
        font-size: 18px;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 4px;
    }
    .auth-sub {
        font-size: 12px;
        color: #9ca3af;
    }
    .status-chip {
        padding: 10px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .status-online {
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #d1fae5;
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fee2e2;
    }
    .stTextArea>div>div>textarea { background-color: #1E293B; color: #10B981; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🚀 DOC-INTEL AI COMMAND CENTER")
st.subheader("Enterprise-Grade Handwriting Intelligence (v5.0 DocTR)")

# Show info about new architecture
st.info("⚡ **Vanguard 5.0 (Enterprise Client View):** Powered by High-Speed FastAPI Backend")

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = ""

backend_online, backend_status_text = get_backend_status()

with st.sidebar:
    st.markdown(
        """
        <div class="auth-card">
            <div class="auth-title">Secure Workspace Access</div>
            <div class="auth-sub">Login or create an account to use the document scanner.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_class = "status-online" if backend_online else "status-offline"
    st.markdown(
        f'<div class="status-chip {status_class}">{backend_status_text}</div>',
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.caption("Start the API server before logging in:")
        st.code("source .venv/bin/activate && uvicorn api:app --host 127.0.0.1 --port 8000")

    if st.session_state.auth_user is None:
        auth_mode = st.radio("Access", ["Login", "Sign Up"], horizontal=True)
        if auth_mode == "Login":
            with st.form("login_form", clear_on_submit=False):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submitted = st.form_submit_button("Login", disabled=not backend_online)

            if login_submitted:
                try:
                    response = fetch_api_json(
                        "/api/v1/auth/login",
                        method="POST",
                        json={"email": login_email, "password": login_password},
                        timeout=20,
                    )
                    st.session_state.auth_user = {
                        "id": response["user_id"],
                        "username": response["username"],
                        "email": response["email"],
                    }
                    st.session_state.auth_token = response["token"]
                    st.success(f"Welcome back, {response['username']}.")
                    st.rerun()
                except Exception as e:
                    st.error(get_error_message(e, "Login failed"))
        else:
            with st.form("signup_form", clear_on_submit=False):
                signup_username = st.text_input("Username", key="signup_username")
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input("Password", type="password", key="signup_password")
                signup_confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
                signup_submitted = st.form_submit_button("Create Account", disabled=not backend_online)

            if signup_submitted:
                if signup_password != signup_confirm_password:
                    st.error("Passwords do not match.")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        response = fetch_api_json(
                            "/api/v1/auth/signup",
                            method="POST",
                            json={
                                "username": signup_username,
                                "email": signup_email,
                                "password": signup_password,
                            },
                            timeout=20,
                        )
                        st.session_state.auth_user = {
                            "id": response["user_id"],
                            "username": response["username"],
                            "email": response["email"],
                        }
                        st.session_state.auth_token = response["token"]
                        st.success(f"Account created. Welcome, {response['username']}.")
                        st.rerun()
                    except Exception as e:
                        st.error(get_error_message(e, "Signup failed"))
    else:
        st.success(f"Logged in as {st.session_state.auth_user['username']}")
        st.caption(st.session_state.auth_user["email"])
        if st.button("Logout", key="logout_btn"):
            st.session_state.auth_user = None
            st.session_state.auth_token = ""
            st.rerun()

if st.session_state.auth_user is None:
    st.warning("Please login or sign up from the sidebar to continue.")
    st.stop()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("📸 High-Res Document Upload")
    uploaded_file = st.file_uploader("Upload Image (Auto-Enhancement Enabled)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Loaded Source", use_container_width=True)
        # Instead of saving locally, we'll stream it to the API
        file_bytes = uploaded_file.getvalue()

with col2:
    st.header("🏆 Vanguard AI Intelligence (v2.0)")
    
    # --- TAB INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["🔍 Live Scanner", "📂 Document History (Database)", "🧑‍⚖️ Human Review"])

    with tab1:
        # --- VANGUARD SETTINGS ---
        with st.sidebar:
            st.title("🛡️ Vanguard Shields")
            medical_mode = st.toggle("Medical Context Correction", value=True)
            high_fidelity = st.toggle("High-Fidelity Ink Scan", value=True, help="Uses TrOCR for deep handwriting verification. REQUIRED for best note accuracy.")
            turbo_mode = st.toggle("🚀 Vanguard Turbo Mode", value=False, help="MAX SPEED: Skips high-fidelity verification. Better for printed bills.")
            
            st.markdown("---")
            st.subheader("⚙️ Processing Stack")
            processing_stack = st.selectbox(
                "Inference Route",
                ["Enterprise OCR + Reasoning", "Local OCR"],
                help="Enterprise route uses specialized OCR providers configured on backend environment.",
            )
            enterprise_mode = processing_stack == "Enterprise OCR + Reasoning"
            enterprise_ocr_provider = st.selectbox(
                "Enterprise OCR Provider",
                ["documentai", "textract", "azure"],
                disabled=not enterprise_mode,
            )

            st.markdown("---")
            st.subheader("🌐 Translation")
            target_language_name = st.selectbox(
                "Output Language",
                list(LANGUAGE_OPTIONS.keys()),
                index=0,
                help="Detect source language automatically and optionally translate output."
            )
            target_language_code = LANGUAGE_OPTIONS[target_language_name]
        
        # Determine final high_fidelity flag based on settings
        # If Turbo Mode is ON, High-Fidelity is OFF (for speed)
        effective_hf = high_fidelity if not turbo_mode else False

        extraction_mode = st.selectbox("Intelligence Mode", 
                                     ["Auto-Detect (AI Recommendation)", 
                                      "Force Paragraph (Notes/Letters)", 
                                      "Force Table (Forms/Bills/Certificates)"] ,
                                     help="Choose table mode for any document that resembles a form or certificate so the layout is preserved.")

        if uploaded_file and st.button("🚀 EXECUTE VANGUARD DEEP SCAN"):
            with st.status("🛸 Routing to Vanguard API Gateway...", expanded=True) as status:
                st.write("📡 Step 1: Securely transmitting encrypted file to Backend server...")

                # Configure payload
                files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                data_payload = {
                    "cloud_mode": False,
                    "enterprise_mode": enterprise_mode,
                    "enterprise_ocr_provider": enterprise_ocr_provider,
                    "high_fidelity": effective_hf
                }
                
                # Setup custom force mode
                if "Force Paragraph" in extraction_mode: data_payload["force_mode"] = "HANDWRITTEN_NOTE"
                elif "Force Table" in extraction_mode: data_payload["force_mode"] = "STRUCTURED_FORM"

                st.write(f"🧠 Step 2: Running Deep AI Neural Layers ({'Enterprise' if enterprise_mode else 'Local'})...")
                
                # Call the FastAPI Endpoint
                try:
                    response = requests.post(f"{API_URL}/api/v1/process_document", files=files, data=data_payload, timeout=300)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result["status"] == "success":
                        doc_analysis = result["analysis"]
                        data = result["structured_data"]
                        output_data = data
                        field_predictions = result.get("field_predictions", [])
                        review_summary = result.get("review_summary", {})
                        template_summary = result.get("template", {})
                        layout_summary = result.get("layout_summary", {})
                        final_type = doc_analysis["type"]
                        paths = result["paths"]
                        filled_form_paths = result.get("filled_form_paths", [])
                        preserved_template_paths = result.get("preserved_template_paths", [])

                        if target_language_code != "original":
                            source_code, source_name, source_conf = TRANSLATION_ENGINE.detect_language(
                                data.get("full_text", "")
                            )
                            if source_code != target_language_code:
                                try:
                                    output_data = TRANSLATION_ENGINE.translate_data(
                                        data,
                                        target_language_code,
                                        source_code,
                                    )
                                except Exception as trans_err:
                                    st.warning(f"Translation failed, showing original text: {trans_err}")
                            st.caption(
                                f"Detected source language: {source_name} ({source_code}) | confidence: {source_conf * 100:.1f}%"
                            )
                        
                        st.write("💾 Step 3: Analysis Complete. Saved to Server DB.")
                        status.update(label=f"💎 Vanguard Analysis: {final_type} | Record ID: #{result['record_id']}", state="complete", expanded=False)
                        
                        # Show the processed image returned by backend if locally running
                        processed_preview_path = resolve_local_image_path(paths.get("processed", ""))
                        if processed_preview_path:
                            st.image(processed_preview_path, caption="Vanguard Enhanced Vision (Ink-Preserved)", use_container_width=True)
                        elif paths.get("processed"):
                            st.info(f"Processed image saved at backend path: {paths['processed']}")
                    else:
                        st.error("API Error: Unknown status returned.")
                        st.stop()
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Backend API Connection Failed! Is the FastAPI server running on {API_URL}?")
                    st.info(f"Technical Details: {e}")
                    status.update(label="API Offline", state="error")
                    st.stop()

            # --- VANGUARD METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Handwriting Score", f"{doc_analysis.get('confidence_score', 0)*100:.1f}%")
            m2.metric("Intelligence Layer", "Vanguard 5.0 Backend")
            m3.metric("Context Hits", "Medical Standard")
            m4.metric("Review Queue", review_summary.get("open_tasks", 0))

            if field_predictions:
                st.subheader("🧾 Field Confidence Grid")
                render_field_predictions(field_predictions)

            if review_summary:
                st.caption(
                    f"Document review status: {review_summary.get('document_review_status', 'unknown')} | Auto-approved fields: {review_summary.get('auto_approved_fields', 0)}"
                )

            if template_summary:
                st.info(
                    f"Template: {template_summary.get('template_name', 'n/a')} | Domain: {template_summary.get('document_domain', 'general')} | Match: {confidence_to_percent(template_summary.get('match_score', 0.0))}"
                )

            if layout_summary:
                st.caption(
                    f"Layout blocks captured: {layout_summary.get('block_count', 0)} | Source engine: {layout_summary.get('source_engine', 'unknown')}"
                )

            if final_type == "HANDWRITTEN_NOTE" or not data.get("table"):
                st.subheader("📝 Digitized Paper (Vanguard Reconstruction)")
                text = output_data.get("full_text", "")
                st.markdown(f"""
                <div style="background-color: #f0f4f8; padding: 30px; border: 2px solid #3498db; border-radius: 8px; color: #1a2a3a; font-family: 'Inter', sans-serif; line-height: 1.8; font-size: 1.2em; white-space: pre-wrap;">
                {text}
                </div>
                """, unsafe_allow_html=True)
                st.download_button("📥 DOWNLOAD VANGUARD TEXT", text, "vanguard_note.txt")

            if filled_form_paths:
                st.subheader("📄 Filled Form (Digital Conversion)")
                st.write("*Original form layout preserved; only handwritten entries are digitized.*")
                for idx, filled_path in enumerate(filled_form_paths, start=1):
                    if filled_path and os.path.exists(filled_path):
                        st.image(filled_path, caption=f"Filled Form Page {idx}")
                        with open(filled_path, "rb") as fp:
                            st.download_button(
                                f"📥 DOWNLOAD FILLED FORM PAGE {idx}",
                                fp.read(),
                                file_name=f"filled_form_page_{idx}.jpg",
                                mime="image/jpeg"
                            )
            elif preserved_template_paths:
                st.subheader("📄 Preserved Form Template")
                st.write("*Form layout kept exactly same as original.*")
                for idx, form_path in enumerate(preserved_template_paths, start=1):
                    if form_path and os.path.exists(form_path):
                        st.image(form_path, caption=f"Form Template Page {idx}")
                        with open(form_path, "rb") as fp:
                            st.download_button(
                                f"📥 DOWNLOAD PRESERVED FORM PAGE {idx}",
                                fp.read(),
                                file_name=f"preserved_form_page_{idx}.jpg",
                                mime="image/jpeg"
                            )

            if final_type != "HANDWRITTEN_NOTE":
                st.subheader("📋 Professional Data Matrix")

                kv = output_data.get("kv", {})
                if kv:
                    with st.expander("📌 Context Fields", expanded=True):
                        # Ensure dictionary type just in case the JSON model goes off track
                        if isinstance(kv, dict):
                            for k, v in kv.items():
                                st.write(f"**{k}:** `{v}`")
                        else:
                            st.write(str(kv))
                
                table_data = output_data.get("table", [])
                if table_data:
                    # In case LLM returns a list of strings instead of dicts for paragraphs
                    if all(isinstance(x, str) for x in table_data):
                        df = pd.DataFrame({"Text content": table_data})
                    else:
                        df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("📥 DOWNLOAD VANGUARD CSV", df.to_csv(index=False), "vanguard_data.csv")
                    
        with tab2:
            st.subheader("🗄️ Enterprise Database Records")
            try:
                records = fetch_api_json("/api/v1/history", timeout=10)
                
                if records:
                    st.write(f"Total documents Server-Synced: **{len(records)}**")
                    
                    history_data = []
                    for r in records:
                        history_data.append({
                            "ID": r["id"],
                            "File": r["filename"],
                            "Type": r["document_type"],
                            "Engine": r["ocr_provider"],
                            "Review": r.get("review_status", "unknown"),
                            "Time": r["upload_time"]
                        })
                    st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("### 🔎 Inspect Specific Record")
                    selected_id = st.selectbox("Select Record ID to view details", [r["id"] for r in records], key="history_record_select")
                    
                    if selected_id:
                        selected_record = fetch_api_json(f"/api/v1/documents/{selected_id}", timeout=15)
                        
                        col_a, col_b = st.columns([1,1])
                        with col_a:
                            st.write(f"**Filename:** {selected_record['filename']}")
                            st.write(f"**Type:** {selected_record['document_type']}")
                            st.write(f"**Confidence:** {selected_record['confidence_score']*100:.1f}%")
                            st.write(f"**Review Status:** {selected_record.get('review_status', 'unknown')}")
                            st.write(f"**Domain:** {selected_record.get('document_domain', 'general')}")
                            st.write(f"**Template Match:** {selected_record.get('template_match_score', 0.0) * 100:.1f}%")
                            
                            processed_record_path = resolve_local_image_path(selected_record.get('processed_image_path', ''))
                            original_record_path = resolve_local_image_path(selected_record.get('original_image_path', ''))
                            if processed_record_path:
                                st.image(processed_record_path, caption="Processed Image", width=300)
                            elif original_record_path:
                                st.image(original_record_path, caption="Original Image", width=300)
                            else:
                                st.warning("Preview image not available locally for this older record.")
                                
                        with col_b:
                            if selected_record.get('extracted_json'):
                                st.write("**Structured Data JSON:**")
                                st.json(selected_record['extracted_json'])
                            else:
                                st.write("**Extracted Text:**")
                                st.text(selected_record.get('extracted_text', ''))

                        if selected_record.get("field_predictions"):
                            st.markdown("### Field Predictions")
                            render_field_predictions(selected_record["field_predictions"])

                        if selected_record.get("template"):
                            st.markdown("### Template Match")
                            template_data = selected_record["template"]
                            st.write(f"**Template Name:** {template_data.get('template_name', 'n/a')}")
                            st.write(f"**Domain:** {template_data.get('document_domain', 'general')}")
                            st.write(f"**Samples Learned:** {template_data.get('sample_count', 0)}")
                            with st.expander("Template Fingerprint"):
                                st.json(template_data.get("fingerprint", {}))

                        if selected_record.get("extraction_blocks"):
                            st.markdown("### Layout Blocks")
                            render_extraction_blocks(selected_record["extraction_blocks"])

                        if selected_record.get("review_tasks"):
                            review_rows = []
                            for task in selected_record["review_tasks"]:
                                review_rows.append({
                                    "Task ID": task["id"],
                                    "Status": task["status"],
                                    "Priority": task["priority"],
                                    "Predicted": task.get("predicted_value", ""),
                                    "Corrected": task.get("corrected_value", ""),
                                })
                            st.markdown("### Review Tasks")
                            st.dataframe(pd.DataFrame(review_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No records found in database. Scan a document first!")
            except Exception as e:
                st.error(f"Error connecting to Database API: {e}")

        with tab3:
            st.subheader("🧑‍⚖️ Human Review Queue")
            try:
                review_tasks = fetch_api_json("/api/v1/review/tasks?status=open&limit=100", timeout=15)
                if review_tasks:
                    queue_rows = []
                    for task in review_tasks:
                        queue_rows.append({
                            "Task ID": task["id"],
                            "Document ID": task["document_id"],
                            "Priority": task["priority"],
                            "Predicted": task.get("predicted_value", ""),
                            "Created": task["created_at"],
                        })
                    st.dataframe(pd.DataFrame(queue_rows), use_container_width=True, hide_index=True)

                    selected_task_id = st.selectbox(
                        "Select review task",
                        [task["id"] for task in review_tasks],
                        key="review_task_select",
                    )
                    selected_task = next(task for task in review_tasks if task["id"] == selected_task_id)
                    selected_doc = fetch_api_json(f"/api/v1/documents/{selected_task['document_id']}", timeout=15)
                    selected_prediction = next(
                        (item for item in selected_doc.get("field_predictions", []) if item["id"] == selected_task["field_prediction_id"]),
                        None,
                    )

                    preview_col, form_col = st.columns([1, 1])
                    with preview_col:
                        preview_path = resolve_local_image_path(selected_doc.get("processed_image_path", "")) or resolve_local_image_path(selected_doc.get("original_image_path", ""))
                        if preview_path:
                            st.image(preview_path, caption=f"Document #{selected_doc['id']}", use_container_width=True)
                        st.json(selected_doc.get("extracted_json", {}))

                    with form_col:
                        field_name = selected_prediction.get("field_name", "unknown") if selected_prediction else "unknown"
                        st.write(f"**Field:** {field_name}")
                        st.write(f"**Confidence:** {confidence_to_percent(selected_prediction.get('confidence_score', 0.0) if selected_prediction else 0.0)}")
                        st.write(f"**Predicted Value:** {selected_task.get('predicted_value', '')}")

                        resolution = st.selectbox("Resolution", ["corrected", "approved"], key=f"resolution_{selected_task_id}")
                        corrected_value = st.text_area(
                            "Corrected Value",
                            value=selected_task.get("predicted_value", ""),
                            key=f"corrected_value_{selected_task_id}",
                        )
                        reviewer_name = st.text_input("Reviewer Name", key=f"reviewer_name_{selected_task_id}")
                        review_notes = st.text_area("Review Notes", key=f"review_notes_{selected_task_id}")

                        if st.button("Submit Review Decision", key=f"submit_review_{selected_task_id}"):
                            payload = {
                                "corrected_value": corrected_value,
                                "reviewer_name": reviewer_name,
                                "review_notes": review_notes,
                                "resolution": resolution,
                            }
                            completion = fetch_api_json(
                                f"/api/v1/review/tasks/{selected_task_id}/complete",
                                method="POST",
                                json=payload,
                                timeout=20,
                            )
                            st.success(
                                f"Review saved for document #{completion['document_id']}. Status: {completion['document_review_status']}"
                            )
                            st.rerun()
                else:
                    st.success("No open review tasks. All current fields are approved.")
            except Exception as e:
                st.error(f"Error loading review queue: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 1 Crore Project Status")
st.sidebar.success("✅ Vision Engine Online")
st.sidebar.success("✅ OCR Engine Online")
st.sidebar.success("✅ FastAPI Backend Architecture")
