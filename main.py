from ocr_engine import DocumentOCREngine
from universal_parser import UniversalDataIntelligence
import pandas as pd
import os

class DocumentIntelligenceSystem:
    """
    The 'Manager' of our system. 
    It coordinates the Vision and OCR engines to deliver 
    high-quality digital results.
    """
    
    def __init__(self):
        print("======== [DOC-INTEL] Systems Initialized ========")
        self.ocr = DocumentOCREngine()
        self.intelligence = UniversalDataIntelligence()

    def process_document(self, image_path, output_format='excel', force_mode=None):
        """
        Processes a raw handwritten document and exports standard digital data.
        """
        print(f"[Doc-Intel] Processing: {os.path.basename(image_path)}")
        
        # Run the full AI Pipeline
        ocr_results = self.ocr.extract_text_from_image(image_path)
        
        # FIXED: Use intelligence layer for proper parsing
        doc_analysis, data, rows = self.intelligence.parse_universal(ocr_results, force_mode=force_mode)
        
        # Convert to DataFrame (Professional Standard)
        if doc_analysis.get("type") == "HANDWRITTEN_NOTE":
            # For paragraph mode, just return text
            df = pd.DataFrame([{"text": data.get("full_text", "")}])
        else:
            # For table mode, use the table data
            table_data = data.get("table", [])
            if table_data:
                df = pd.DataFrame(table_data)
            else:
                df = pd.DataFrame(data.get("kv", {}), index=[0])
        
        # Exporting data
        filename = os.path.basename(image_path).split('.')[0]
        if output_format == 'excel':
            output_file = f"{filename}_digitized.xlsx"
            df.to_excel(output_file, index=False)
            print(f"[Doc-Intel] Data successfully digitized into: {output_file}")
            return output_file
        
        return df

if __name__ == "__main__":
    # Standard launch sequence
    print("Document Intelligence System - Command Center")
