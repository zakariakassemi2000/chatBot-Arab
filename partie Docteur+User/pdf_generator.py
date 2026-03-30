# pdf_generator.py

from fpdf import FPDF
from datetime import datetime
import os

class PDFGenerator:
    @staticmethod
    def generate_ordonnance(patient_info, prescription, medecin_info=None):
        """Génère une ordonnance PDF"""
        pdf = FPDF()
        pdf.add_page()
        
        # En-tête
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'ORDONNANCE MÉDICALE', 0, 1, 'C')
        pdf.ln(10)
        
        # Informations patient
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Patient: {patient_info.get('full_name', 'Non renseigné')}", 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d/%m/%Y')}", 0, 1)
        
        if patient_info.get('date_naissance'):
            pdf.cell(0, 10, f"Né(e) le: {patient_info['date_naissance']}", 0, 1)
        
        pdf.ln(10)
        
        # Prescription
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Prescription:', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, prescription)
        
        pdf.ln(10)
        
        # Médecin
        if medecin_info:
            pdf.set_font('Arial', 'I', 11)
            pdf.cell(0, 10, f"Dr. {medecin_info.get('full_name', 'Médecin')}", 0, 1)
            pdf.cell(0, 10, f"Signature:", 0, 1)
        
        # Pied de page
        pdf.set_y(-30)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, "Document généré par AI Shifa Pro - À valider par un médecin", 0, 0, 'C')
        
        # Sauvegarder
        filename = f"ordonnances/ordo_{patient_info.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        
        return filename
    
    @staticmethod
    def generate_rapport_analyse(patient_info, analyse_result):
        """Génère un rapport d'analyse PDF"""
        pdf = FPDF()
        pdf.add_page()
        
        # En-tête
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "RAPPORT D'ANALYSE MÉDICALE", 0, 1, 'C')
        pdf.ln(10)
        
        # Informations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Patient: {patient_info.get('full_name', 'Non renseigné')}", 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1)
        pdf.ln(10)
        
        # Résultat
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Résultat de l'analyse:", 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Diagnostic: {analyse_result['resultat']}", 0, 1)
        pdf.cell(0, 10, f"Confiance: {analyse_result['confiance']:.1%}", 0, 1)
        
        if analyse_result.get('urgent'):
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, "⚠️ RÉSULTAT URGENT", 0, 1)
            pdf.set_text_color(0, 0, 0)
        
        pdf.ln(10)
        
        # Recommandations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Recommandations:", 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, analyse_result['recommandations'])
        
        filename = f"reports/rapport_{patient_info.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        
        return filename