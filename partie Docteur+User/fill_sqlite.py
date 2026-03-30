import sqlite3
import re

def fill_sqlite(sql_path, db_path):
    print("Connecting to DB...")
    # Add a timeout to wait for locks if the DB is busy
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()
    
    # First, clear the database tables
    print("Clearing database tables...")
    tables = [
        "activity_logs", "analyses", "consultations", "conversations", 
        "conversation_users", "dossiers", "logs", "medecin_specialites",
        "messages", "notes_medicales", "notifications", "ordonnances",
        "parametres", "photos_patient", "rendez_vous", "specialites", "users"
    ]
    for t in tables:
        try:
            cursor.execute(f"DELETE FROM {t}")
        except Exception as e:
            print(f"Failed to clear {t}: {e}")
            
    conn.commit()

    print("Reading SQL dump...")
    with open(sql_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    inserts = re.findall(r'^(INSERT INTO `[^`]+`.*?VALUES.*?);$', content, re.MULTILINE | re.DOTALL)
    print(f"Found {len(inserts)} INSERT blocks.")
    
    success = 0
    errors = 0
    for stmt in inserts:
        try:
            stmt_clean = stmt.replace("\\'", "''")
            cursor.execute(stmt_clean)
            success += 1
        except Exception as e:
            print(f"Error executing insert: {e}\n{stmt[:100]}...")
            errors += 1
            
    conn.commit()
    conn.close()
    print(f"Finished. Success: {success}, Errors: {errors}")

if __name__ == "__main__":
    sql_path = r"c:\Users\zakar\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\37AD10E0F8F9B0C3067FC6B8D65D9F509AB0868E\transfers\2026-13\ai_shifa_pro.sql"
    db_path = r"d:\chatBot Arab\partie Docteur+User\ai_shifa_pro.db"
    fill_sqlite(sql_path, db_path)
