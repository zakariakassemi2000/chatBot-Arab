import json
import re
import unicodedata
from rapidfuzz import process, fuzz
import os
import logging

class Extractor:
    def __init__(self, db_path="data/medicaments_maroc.json"):
        self.db_path = db_path
        self.medications = []
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # New format: {"medicaments": [{dci, noms_commerciaux, ...}]}
                if isinstance(data, dict) and "medicaments" in data:
                    for entry in data["medicaments"]:
                        dci = entry.get("dci", "")
                        if dci:
                            self.medications.append(dci)
                        for cn in entry.get("noms_commerciaux", []):
                            self.medications.append(cn)
                # Old format: flat list [{nom, principe, ...}]
                elif isinstance(data, list):
                    for item in data:
                        if "nom_commercial" in item:
                            self.medications.append(item["nom_commercial"])
                        elif "nom" in item:
                            self.medications.append(item["nom"])
                        if "dci" in item:
                            self.medications.append(item["dci"])
                        elif "principe" in item:
                            self.medications.append(item["principe"])
        else:
            logging.warning(f"Database {self.db_path} not found. Fuzzy matching will be limited.")

    def _normalize_text(self, text: str) -> str:
        """Normalize text: lowercasing + unicode normalization."""
        text = text.lower()
        # Remove accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        # Remove arbitrary punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def extract_entities(self, text: str, ocr_confidence: float) -> dict:
        """
        Uses RegEx to extract dosage, posology, and duration.
        Uses RapidFuzz to match the drug name.
        Applies confidence fusion.
        """
        normalized_text = self._normalize_text(text)
        
        # 1. Regex Extractions
        dosage_match = re.search(r'(\d+)\s*(mg|g|ml|µg|ui)', text, re.IGNORECASE)
        dosage = dosage_match.group(0).strip() if dosage_match else None

        posology_match = re.search(r'(\d+)\s*(fois|gelule|comprimé|sachet)[s]?\s*(par jour|/j)', text, re.IGNORECASE)
        posology = posology_match.group(0).strip() if posology_match else None

        duration_match = re.search(r'pendant\s*(\d+)\s*(jour[s]?|mois|semaine[s]?)', text, re.IGNORECASE)
        duration = duration_match.group(0).strip() if duration_match else None

        # 2. Fuzzy Matching for Drug Name
        drug_name = None
        fuzzy_score = 0.0
        
        if self.medications:
            # We match the entire line text to our DB
            best_match = process.extractOne(normalized_text, self.medications, scorer=fuzz.token_set_ratio)
            if best_match:
                drug_match_str, match_score, _ = best_match
                fuzzy_score = match_score / 100.0  # normalize to 0-1
                if fuzzy_score > 0.65: # threshold for accepting fuzzy match
                    drug_name = drug_match_str

        # 3. Confidence Fusion
        # final_confidence = 0.6 * ocr_confidence + 0.4 * fuzzy_score
        final_confidence = (0.6 * ocr_confidence) + (0.4 * fuzzy_score)

        return {
            "drug_name": drug_name,
            "dosage": dosage,
            "posology": posology,
            "duration": duration,
            "confidence": round(final_confidence, 4)
        }
