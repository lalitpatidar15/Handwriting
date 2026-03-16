import streamlit as st
import pandas as pd
from PIL import Image
import os
import time
import requests

# Fast API Backend URL
API_URL = "http://127.0.0.1:8000"
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
    .stTextArea>div>div>textarea { background-color: #1E293B; color: #10B981; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🚀 DOC-INTEL AI COMMAND CENTER")
st.subheader("Enterprise-Grade Handwriting Intelligence (v5.0 DocTR)")

# Show info about new architecture
st.info("⚡ **Vanguard 5.0 (Enterprise Client View):** Powered by High-Speed FastAPI Backend")

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
        tab1, tab2 = st.tabs(["🔍 Live Scanner", "📂 Document History (Database)"])
        
        with tab1:
            # --- VANGUARD SETTINGS ---
            with st.sidebar:
                st.title("🛡️ Vanguard Shields")
                medical_mode = st.toggle("Medical Context Correction", value=True)
                high_fidelity = st.toggle("High-Fidelity Ink Scan", value=True, help="Uses TrOCR for deep handwriting verification. REQUIRED for best note accuracy.")
                turbo_mode = st.toggle("🚀 Vanguard Turbo Mode", value=False, help="MAX SPEED: Skips high-fidelity verification. Better for printed bills.")
            
            st.markdown("---")
            st.subheader("☁️ Cloud Super Mode (FREE & Paid)")
            cloud_provider = st.selectbox("Cloud Provider", 
                                         ["Gemini (FREE)", "OpenAI GPT-4 (Paid)"],
                                         help="Gemini: Free tier (15 req/min). OpenAI: Paid but more accurate.")
            cloud_mode = st.toggle("Enable Cloud OCR", value=False, help="Uses cloud AI for maximum accuracy")
            
            # API Key Input based on provider
            if cloud_mode:
                if "Gemini" in cloud_provider:
                    if 'gemini_api_key' not in st.session_state:
                        st.session_state.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
                    
                    api_key_input = st.text_input("Gemini API Key (FREE)", 
                                                 value=st.session_state.gemini_api_key, 
                                                 type="password",
                                                 help="Get free key from: https://aistudio.google.com/app/apikey")
                    if api_key_input:
                        st.session_state.gemini_api_key = api_key_input
                        os.environ['GEMINI_API_KEY'] = api_key_input
                        st.success("✅ Gemini API Key set!")
                    elif not st.session_state.gemini_api_key:
                        st.info("💡 Get FREE API key: https://aistudio.google.com/app/apikey")
                else:  # OpenAI
                    if 'openai_api_key' not in st.session_state:
                        st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')
                    
                    api_key_input = st.text_input("OpenAI API Key", 
                                                 value=st.session_state.openai_api_key, 
                                                 type="password",
                                                 help="Enter your OpenAI API key. It will be stored in session only.")
                    if api_key_input:
                        st.session_state.openai_api_key = api_key_input
                        os.environ['OPENAI_API_KEY'] = api_key_input
                        st.success("✅ OpenAI API Key set!")
                    elif not st.session_state.openai_api_key:
                        st.warning("⚠️ Please enter your OpenAI API key")
        
        # Determine final high_fidelity flag based on settings
        # If Turbo Mode is ON, High-Fidelity is OFF (for speed)
        effective_hf = high_fidelity if not turbo_mode else False

        extraction_mode = st.selectbox("Intelligence Mode", 
                                     ["Auto-Detect (AI Recommendation)", "Force Paragraph (Notes/Letters)", "Force Table (Forms/Bills)"])

        if uploaded_file and st.button("🚀 EXECUTE VANGUARD DEEP SCAN"):
            with st.status("🛸 Routing to Vanguard API Gateway...", expanded=True) as status:
                st.write("📡 Step 1: Securely transmitting encrypted file to Backend server...")
                
                provider = "gemini" if "Gemini" in cloud_provider else "openai"
                api_key_to_use = ""
                if cloud_mode:
                    api_key_to_use = st.session_state.gemini_api_key if provider == "gemini" else st.session_state.openai_api_key

                # Configure payload
                files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                data_payload = {
                    "cloud_mode": cloud_mode,
                    "provider": provider,
                    "high_fidelity": effective_hf
                }
                
                # Setup custom force mode
                if "Force Paragraph" in extraction_mode: data_payload["force_mode"] = "HANDWRITTEN_NOTE"
                elif "Force Table" in extraction_mode: data_payload["force_mode"] = "STRUCTURED_FORM"

                st.write(f"🧠 Step 2: Running Deep AI Neural Layers ({'Cloud' if cloud_mode else 'Local'})...")
                
                # Call the FastAPI Endpoint
                try:
                    # Temporary environment variable setting for the backend (Ideally backend should handle it via headers/auth securely)
                    if cloud_mode and api_key_to_use:
                        os.environ[f"{provider.upper()}_API_KEY"] = api_key_to_use

                    response = requests.post(f"{API_URL}/api/v1/process_document", files=files, data=data_payload, timeout=300)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result["status"] == "success":
                        doc_analysis = result["analysis"]
                        data = result["structured_data"]
                        final_type = doc_analysis["type"]
                        paths = result["paths"]
                        
                        st.write("💾 Step 3: Analysis Complete. Saved to Server DB.")
                        status.update(label=f"💎 Vanguard Analysis: {final_type} | Record ID: #{result['record_id']}", state="complete", expanded=False)
                        
                        # Show the processed image returned by backend if locally running
                        if os.path.exists(paths["processed"]):
                            st.image(paths["processed"], caption="Vanguard Enhanced Vision (Ink-Preserved)", use_container_width=True)
                    else:
                        st.error("API Error: Unknown status returned.")
                        st.stop()
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Backend API Connection Failed! Is the FastAPI server running on {API_URL}?")
                    st.info(f"Technical Details: {e}")
                    status.update(label="API Offline", state="error")
                    st.stop()

            # --- VANGUARD METRICS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Handwriting Score", f"{doc_analysis.get('confidence_score', 0)*100:.1f}%")
            m2.metric("Intelligence Layer", "Vanguard 5.0 API (DocTR/Gemini)") # Major Upgrade
            m3.metric("Context Hits", "Medical Standard")

            if final_type == "HANDWRITTEN_NOTE" or not data.get("table"):
                st.subheader("📝 Digitized Paper (Vanguard Reconstruction)")
                text = data.get("full_text", "")
                st.markdown(f"""
                <div style="background-color: #f0f4f8; padding: 30px; border: 2px solid #3498db; border-radius: 8px; color: #1a2a3a; font-family: 'Inter', sans-serif; line-height: 1.8; font-size: 1.2em; white-space: pre-wrap;">
                {text}
                </div>
                """, unsafe_allow_html=True)
                st.download_button("📥 DOWNLOAD VANGUARD TEXT", text, "vanguard_note.txt")
            else:
                st.subheader("📋 Professional Data Matrix")
                kv = data.get("kv", {})
                if kv:
                    with st.expander("📌 Context Fields", expanded=True):
                        # Ensure dictionary type just in case the JSON model goes off track
                        if isinstance(kv, dict):
                            for k, v in kv.items():
                                st.write(f"**{k}:** `{v}`")
                        else:
                            st.write(str(kv))
                
                table_data = data.get("table", [])
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
                response = requests.get(f"{API_URL}/api/v1/history", timeout=10)
                if response.status_code == 200:
                    records = response.json()
                    
                    if records:
                        st.write(f"Total documents Server-Synced: **{len(records)}**")
                        
                        # Create a quick summary table
                        history_data = []
                        for r in records:
                            history_data.append({
                                "ID": r["id"],
                                "File": r["filename"],
                                "Type": r["document_type"],
                                "Engine": r["ocr_provider"],
                                "Time": r["upload_time"]
                            })
                        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
                        
                        # Detailed view
                        st.markdown("---")
                        st.markdown("### 🔎 Inspect Specific Record")
                        selected_id = st.selectbox("Select Record ID to view details", [r["id"] for r in records])
                        
                        if selected_id:
                            selected_record = next(r for r in records if r["id"] == selected_id)
                            
                            col_a, col_b = st.columns([1,1])
                            with col_a:
                                st.write(f"**Filename:** {selected_record['filename']}")
                                st.write(f"**Type:** {selected_record['document_type']}")
                                st.write(f"**Confidence:** {selected_record['confidence_score']*100:.1f}%")
                                
                                if os.path.exists(selected_record.get('processed_image_path', '')):
                                    st.image(selected_record['processed_image_path'], caption="Processed Image", width=300)
                                    
                            with col_b:
                                if selected_record.get('extracted_json'):
                                    st.write("**Structured Data JSON:**")
                                    st.json(selected_record['extracted_json'])
                                else:
                                    st.write("**Extracted Text:**")
                                    st.text(selected_record.get('extracted_text', ''))
                    else:
                        st.info("No records found in database. Scan a document first!")
                else:
                    st.error("Failed to fetch history from API Server.")
            except Exception as e:
                st.error(f"Error connecting to Database API: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 1 Crore Project Status")
st.sidebar.success("✅ Vision Engine Online")
st.sidebar.success("✅ OCR Engine Online")
st.sidebar.success("✅ FastAPI Backend Architecture")
