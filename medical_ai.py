from rapidfuzz import process, fuzz
import re

class MedicalIntelligence:
    """
    The 'Specialist Physician' of our system.
    Handles medication name correction, abbreviation expansion, and 
    contextual medical validation.
    """

    def __init__(self):
        # Premium Medical Dictionary (Indian Market Focus)
        self.drug_database = [
            "Paracetamol", "Amoxicillin", "Ciprofloxacin", "Metformin", "Atorvastatin",
            "Amlodipine", "Omeprazole", "Azithromycin", "Pantoprazole", "Diclofenac",
            "Aceclofenac", "Cetirizine", "Levocetirizine", "Montelukast", "Telmisartan",
            "Losartan", "Glimepiride", "Teneligliptin", "Vildagliptin", "Metoprolol",
            "Cilnidipine", "Tulsar", "Amodep", "Zifi", "Taxim", "Monocef", "Clavam",
            "Augmentin", "Dolo 650", "Calpol", "Sumo", "Nimulid", "Combiflam", "Meftal-Spas",
            "Digene", "Gelusil", "Omee", "Pan-D", "Pantocid", "Rabipur", "Cilnisave",
            "MP-15", "Zeds", "AT", "Mxt"
        ]
        
        # Common Medical Abbreviations & Mapping
        self.abbreviations = {
            "PCM": "Paracetamol",
            "AMOX": "Amoxicillin",
            "CIPRO": "Ciprofloxacin",
            "TAB.": "Tablet",
            "CAP.": "Capsule",
            "SYP.": "Syrup",
            "INJ.": "Injection",
            "OD": "Once a day",
            "BD": "Twice a day",
            "TDS": "Thrice a day",
            "QID": "Four times a day",
            "HS": "At bedtime",
            "AC": "Before meals",
            "PC": "After meals"
        }

    def correct_medication(self, text: str) -> str:
        """
        Uses Fuzzy Matching and Abbreviation expansion to clean medical terms.
        """
        if not text: return ""
        
        # 1. Clean and Normalize
        text = text.strip().upper()
        
        # 2. Check Abbreviation Mapping
        if text in self.abbreviations:
            return self.abbreviations[text]
            
        # 3. Fuzzy Match against Database
        # scorer=fuzz.ratio works well for simple typos
        match = process.extractOne(text, self.drug_database, scorer=fuzz.WRatio)
        
        if match:
            best_match, score, _ = match
            # If 80% confident, return the corrected name
            if score > 80:
                return best_match
                
        return text.capitalize()

    def parse_dosage_line(self, text: str):
        """
        Extracts Qty, Name, and Strength using smart regex.
        Example: '2 Tab Dolo 650mg' -> {'qty': '2', 'name': 'Dolo', 'strength': '650mg'}
        """
        data = {"qty": "", "name": "", "strength": ""}
        
        # Heuristic Regex for medical lines
        qty_match = re.search(r'^(\d+)', text)
        strength_match = re.search(r'(\d+\s*(MG|G|ML|MCG))', text, re.I)
        
        if qty_match: data["qty"] = qty_match.group(1)
        if strength_match: data["strength"] = strength_match.group(1)
        
        # The rest is likely the name (after cleaning)
        clean_name = re.sub(r'^\d+', '', text)
        clean_name = re.sub(r'\d+\s*(MG|G|ML|MCG)', '', clean_name, flags=re.I).strip()
        data["name"] = self.correct_medication(clean_name)
        
        return data

if __name__ == "__main__":
    intel = MedicalIntelligence()
    print(f"Correction [PCM]: {intel.correct_medication('PCM')}")
    print(f"Correction [Paracitamol]: {intel.correct_medication('Paracitamol')}")
    print(f"Parsed Line: {intel.parse_dosage_line('2 Tab Dolo 650mg')}")
