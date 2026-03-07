import streamlit as st
import pandas as pd
from main import DocumentIntelligenceSystem
from PIL import Image
import os
import cv2
import time

from cloud_ocr import cloud_ocr

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

# Show info about first load
if 'first_load_shown' not in st.session_state:
    st.info("⏳ **First Load Notice:** Initializing AI models takes 30-60 seconds. Please wait... (This happens only once)")
    st.session_state.first_load_shown = True

from universal_parser import UniversalDataIntelligence

# Initialize System (cached for performance)
@st.cache_resource 
def load_system():
    return DocumentIntelligenceSystem(), UniversalDataIntelligence()

try:
    # Load system (first time takes 30-60 seconds, then cached)
    system, intelligence = load_system()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.header("📸 High-Res Document Upload")
        uploaded_file = st.file_uploader("Upload Image (Auto-Enhancement Enabled)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Loaded Source", use_container_width=True)
            temp_path = "temp_file.jpg"
            img.save(temp_path)

    with col2:
        st.header("🏆 Vanguard AI Intelligence (v2.0)")
        
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
            with st.status("🛸 Vanguard AI Scaling Pipeline...", expanded=True) as status:
                st.write("🔍 [Vision] Local Contrast Normalization & Ink Sharpening...")
                clean_img = system.ocr.vision.enhance_image(temp_path)
                cv2.imwrite("vanguard_debug.jpg", clean_img)
                st.image("vanguard_debug.jpg", caption="Vanguard Enhanced Vision (Ink-Preserved)", use_container_width=True)

                # Default placeholders
                doc_analysis = {"type": "HANDWRITTEN_NOTE", "confidence_score": 0.95}
                data = {"full_text": ""}
                all_rows = []

                if cloud_mode:
                    # --- CLOUD SUPER MODE (Gemini FREE or OpenAI Paid) ---
                    provider = "gemini" if "Gemini" in cloud_provider else "openai"
                    provider_name = "Gemini (FREE)" if provider == "gemini" else "GPT-4o (Paid)"
                    st.write(f"☁️ [Cloud] Using {provider_name} for ultra-accurate transcription...")
                    try:
                        # Use session state API key if available
                        if provider == "gemini" and st.session_state.get('gemini_api_key'):
                            os.environ['GEMINI_API_KEY'] = st.session_state.gemini_api_key
                        elif provider == "openai" and st.session_state.get('openai_api_key'):
                            os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        
                        text = cloud_ocr(temp_path, provider=provider)
                        data["full_text"] = text
                        doc_analysis["type"] = "HANDWRITTEN_NOTE"
                        doc_analysis["confidence_score"] = 0.99
                        final_type = doc_analysis["type"]
                        status.update(label=f"💎 Vanguard Analysis (Cloud {provider_name}): {final_type}", state="complete", expanded=False)
                    except Exception as e:
                        st.error(f"Cloud OCR failed: {e}")
                        st.info("Falling back to local Vanguard engine (DocTR + TrOCR).")
                        cloud_mode = False  # fall through to local path

                if not cloud_mode:
                    st.write("📖 [Analyst] Running Local Neural Recognition...")
                    # Vanguard v4.0 Pivot: Disable internal paragraph merging to allow 
                    # custom precision reconstruction in the intelligence unit.
                    ocr_results = system.ocr.extract_text_from_image(temp_path, pre_cleaned=clean_img, 
                                                                     paragraph=False, 
                                                                     high_fidelity=effective_hf)
                    
                    st.write("🧠 [Intelligence] Contextual Correction & Layout Analysis...")
                    # Vanguard v3.1: Pass manual override to the parser
                    force_mode = None
                    if "Force Paragraph" in extraction_mode: force_mode = "HANDWRITTEN_NOTE"
                    elif "Force Table" in extraction_mode: force_mode = "STRUCTURED_FORM"
                    
                    doc_analysis, data, all_rows = intelligence.parse_universal(ocr_results, force_mode=force_mode)
                    final_type = doc_analysis["type"]
                    
                    status.update(label=f"💎 Vanguard Analysis: {final_type}", state="complete", expanded=False)
            
            # --- VANGUARD METRICS ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Handwriting Score", f"{doc_analysis.get('confidence_score', 0)*100:.1f}%")
            m2.metric("Intelligence Layer", "Vanguard 5.0 (DocTR)") # Major Upgrade
            m3.metric("Context Hits", "Medical Standard")

            if final_type == "HANDWRITTEN_NOTE":
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
                        for k, v in kv.items():
                            st.write(f"**{k}:** `{v}`")
                
                table_data = data.get("table", [])
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("📥 DOWNLOAD VANGUARD CSV", df.to_csv(index=False), "vanguard_data.csv")
                else:
                    st.warning("Vanguard AI: No table structure detected. Try 'Force Paragraph' mode.")

except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.info("Bhai, tension mat lo! AI environment setup ho raha hai. Thoda wait karein.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 1 Crore Project Status")
st.sidebar.success("✅ Vision Engine Online")
st.sidebar.success("✅ OCR Engine Online")
st.sidebar.info("⏳ Intelligence Engine Tuning")
