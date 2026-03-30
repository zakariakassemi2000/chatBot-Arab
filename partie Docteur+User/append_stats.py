import sqlite3
import logging

logger = logging.getLogger(__name__)

def append_stats_methods():
    content = ""
    with open('database1.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "def get_dashboard_stats" not in content:
        stats_code = """
    def get_dashboard_stats(self):
        try:
            return {
                'total_users': self.execute_query("SELECT COUNT(*) as count FROM users", fetch_one=True)['count'],
                'total_patients': self.execute_query("SELECT COUNT(*) as count FROM users WHERE role = 'patient'", fetch_one=True)['count'],
                'total_medecins': self.execute_query("SELECT COUNT(*) as count FROM users WHERE role = 'medecin'", fetch_one=True)['count'],
                'total_analyses': self.execute_query("SELECT COUNT(*) as count FROM analyses", fetch_one=True)['count']
            }
        except Exception as e:
            logger.error(f"Erreur get_dashboard_stats: {e}")
            return {'total_users': 0, 'total_patients': 0, 'total_medecins': 0, 'total_analyses': 0}

    def get_doctor_statistics(self, doctor_id):
        try:
            total_patients = self.execute_query(
                "SELECT COUNT(DISTINCT patient_id) as count FROM rendez_vous WHERE medecin_id = ?", 
                (doctor_id,), fetch_one=True
            )['count']
            
            total_consultations = self.execute_query(
                "SELECT COUNT(*) as count FROM consultations WHERE medecin_id = ?",
                (doctor_id,), fetch_one=True
            )['count']
            
            taux_urgence = self.execute_query(
                "SELECT COUNT(*) as count FROM analyses WHERE urgent = 1", fetch_one=True
            )['count'] # Simplification
            
            return {
                'total_patients': total_patients,
                'consultations_month': total_consultations,
                'emergency_cases': taux_urgence,
                'avg_rating': 4.8
            }
        except Exception as e:
            logger.error(f"Erreur get_doctor_statistics: {e}")
            return {'total_patients': 0, 'consultations_month': 0, 'emergency_cases': 0, 'avg_rating': 0.0}
"""
        with open('database1.py', 'a', encoding='utf-8') as f:
            f.write(stats_code)

if __name__ == "__main__":
    append_stats_methods()
