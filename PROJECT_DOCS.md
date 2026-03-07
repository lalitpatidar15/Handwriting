# 📄 DOC-INTEL (Vanguard AI): Complete Technical Documentation

**Version**: 4.1 (Precision Edition)  
**Author**: Banti Kevat  
**Project Type**: Intelligent Handwritten Document Digitization System

---

## 1. Project Overview
**DOC-INTEL (Vanguard AI)** ek AI-powered system hai jo handwritten documents, medical bills, prescriptions, aur notes ko digital structured data mein convert karta hai.

### System Goal:
- **Messy Photos** → Clean Documents
- **Handwritten Text** → Readable Text
- **Unstructured Data** → Structured Tables
- **Medical Handwriting** → Correct Medicine Names

### Target Industries:
- Hospitals & Medical Stores
- Insurance Companies
- Pathology Labs
- Document Digitization Services

---

## 2. Core Objectives
System ko ye major problems solve karni hain:
1. **Handwriting Complexity**: Doctors ki handwriting difficult hoti hai.
2. **Irregular Layouts**: Medical bills mein tables irregular hote hain.
3. **Image Noise**: Photos messy aur shadow waali hoti hain.
4. **Data Accuracy**: Medicine names galat padhe jaate hain.

### Final Output (Example):
```json
{
 "patient_name": "Rahul Sharma",
 "date": "12-03-2026",
 "items": [
   {
     "medicine": "Paracetamol",
     "qty": 2,
     "price": 50
   }
 ]
}
```

---

## 3. System Architecture
System ko 5 major AI layers mein divide kiya gaya hai:
1. **User Upload** (Streamlit UI)
2. **Vision Engine** (Preprocessing)
3. **Layout Detection Engine** (Contour-based Intelligence)
4. **OCR Engine** (Hybrid: EasyOCR + TrOCR)
5. **Intelligence Parser** (Dynamic Mapping & Line Grouping)
6. **Medical AI Correction** (Fuzzy/Context Logic)

---

## 4. Vision Engine (Computer Vision Layer)
**Purpose**: Raw image ko clean digital document mein convert karna.
- **Pipeline**: Bilateral Filtering → Adaptive Threshold → Vanguard v4.1 Soft Sharpening (1.2 / -0.2 weights).
- **Libraries**: OpenCV, NumPy.

---

## 5. OCR Engine (Text Recognition)
**Purpose**: Clean image se text read karna.
- **Hybrid Strategy**:
    - **Fast OCR**: EasyOCR (Printed text)
    - **Handwriting**: TrOCR (Microsoft Baseline)
- **High-Fidelity Mode**: TrOCR precision-verification for ambiguous handwriting.

---

## 6. Intelligence Parser (Vanguard v4.1 Refinement)
**Purpose**: OCR text ko structured data mein convert karna.
- **Precision Logic**: 
    - **Y-Threshold (30px)**: Words ko lines mein accurately group karna.
    - **X-Gap (120px)**: Handwriting spacing ko accurately maintain karna.
- **Medical Logic Patch**: Fuzzy correction ab sirf "Force Table" mode mein chalta hai taaki standard handwriting (like "something") mis-correct na ho.

---

## 7. Medical Intelligence Layer (The Specialist)
**Purpose**: OCR mistakes ko correct karke standard database se match karna.
- **Libraries**: rapidfuzz, fuzzywuzzy.

---

## 8. AI Verification & Deployment
- **Port**: Live on **8506**.
- **Dependencies**: Verified Python 3.14 stability with `accelerate` & `torchvision`.

---

## ⚡ Developer Advice (For AI Agents)
*This documentation is designed to be understood by ChatGPT, Claude, Cursor, and Devin style agents. It clearly outlines the modular architecture and specific engine requirements.*

---

## ⚡ Developer Advice (For AI Agents)
*This documentation is designed to be understood by ChatGPT, Claude, Cursor, and Devin style agents. It clearly outlines the modular architecture and specific engine requirements.*
