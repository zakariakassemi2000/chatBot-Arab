import sqlite3
import re
import os
import sys

def convert_mysql_to_sqlite(mysql_file, sqlite_db):
    print(f"Reading MySQL dump from: {mysql_file}")
    with open(mysql_file, 'r', encoding='utf-8') as f:
        sql = f.read()

    # Nettoyage de la syntaxe MySQL
    print("Converting syntax...")
    
    # 1. Enlever les infos de ENGINE, DEFAULT CHARSET, etc.
    sql = re.sub(r'ENGINE=InnoDB.*?COLLATE=.*?_ci;', ';', sql)
    sql = re.sub(r'ENGINE=InnoDB.*?CHARSET=.*?;', ';', sql)
    
    # 2. AUTO_INCREMENT -> AUTOINCREMENT
    # Note: Dans SQLite, AUTOINCREMENT ne peut être utilisé qu'avec INTEGER PRIMARY KEY
    sql = re.sub(r'\bint\(\d+\)\s+NOT NULL\s+AUTO_INCREMENT', 'INTEGER NOT NULL AUTOINCREMENT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bint\(\d+\)\s+AUTO_INCREMENT', 'INTEGER AUTOINCREMENT', sql, flags=re.IGNORECASE)
    
    # Cleanup keys / indexes in CREATE TABLE which SQLite doesn't support inside the table definition
    # This requires processing line by line or complex regex.
    # Fortunately, we can just extract the INSERT statements or use regex to drop KEY/UNIQUE KEY lines at the end of CREATE TABLE.
    
    # Simplify: since this is a complex RegExp task, another approach is to let python's sqlite3 execute standard statements,
    # but SQLite doesn't support ALTER TABLE ... ADD CONSTRAINT.
    
    statements = sql.split(';')
    clean_statements = []
    
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt: continue
        if stmt.startswith('/*!') or stmt.startswith('--') or stmt.startswith('SET ') or stmt.startswith('START TRANSACTION') or stmt.startswith('COMMIT'):
            continue
            
        # If it is a CREATE TABLE
        if stmt.startswith('CREATE TABLE'):
            # Replace types
            stmt = re.sub(r'\bint\(\d+\)', 'INTEGER', stmt, flags=re.IGNORECASE)
            stmt = re.sub(r'\bdatetime\(\)', 'DATETIME', stmt, flags=re.IGNORECASE)
            # Replace AUTO_INCREMENT
            stmt = re.sub(r'AUTO_INCREMENT', 'AUTOINCREMENT', stmt, flags=re.IGNORECASE)
            
            # Remove ON UPDATE ...
            stmt = re.sub(r'ON UPDATE current_timestamp\(\)', '', stmt, flags=re.IGNORECASE)
            
            # Remove PRIMARY KEY definition if AUTOINCREMENT is used because SQLite expects it inline
            # Actually, standard sqlite expects `id INTEGER PRIMARY KEY AUTOINCREMENT`
            # Let's fix id columns
            stmt = re.sub(r'`id` INTEGER NOT NULL AUTOINCREMENT', '`id` INTEGER PRIMARY KEY AUTOINCREMENT', stmt, flags=re.IGNORECASE)
            stmt = re.sub(r'`id` INTEGER AUTOINCREMENT', '`id` INTEGER PRIMARY KEY AUTOINCREMENT', stmt, flags=re.IGNORECASE)
            
            # Remove stand-alone PRIMARY KEY (`id`) or similar if we already put it inline
            stmt = re.sub(r',\s*PRIMARY KEY\s*\(`id`\)', '', stmt, flags=re.IGNORECASE)
            
            # Remove other KEY/INDEX declarations from inside CREATE TABLE
            stmt = re.sub(r',\s*KEY\s+`[^`]+`\s*\([^)]+\)', '', stmt, flags=re.IGNORECASE)
            stmt = re.sub(r',\s*UNIQUE KEY\s+`[^`]+`\s*\([^)]+\)', '', stmt, flags=re.IGNORECASE)
            stmt = re.sub(r',\s*CONSTRAINT\s+`[^`]+`\s*FOREIGN KEY\s*\([^)]+\)\s*REFERENCES\s*`[^`]+`\s*\([^)]+\)(?:\s*ON DELETE CASCADE)?(?:\s*ON UPDATE CASCADE)?', '', stmt, flags=re.IGNORECASE)
            
            # MySQL ENUM -> VARCHAR
            stmt = re.sub(r'enum\([^)]+\)', 'VARCHAR(255)', stmt, flags=re.IGNORECASE)
            
            clean_statements.append(stmt)
            
        elif stmt.startswith('INSERT INTO'):
            clean_statements.append(stmt)
            
        # Skip ALTER TABLE for constraints as SQLite handles them differently or we ignore them for local dev
        elif stmt.startswith('ALTER TABLE'):
            pass
            
    print(f"Connecting to SQLite: {sqlite_db}")
    if os.path.exists(sqlite_db):
        os.remove(sqlite_db)
        
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    
    print("Executing statements...")
    success = 0
    errors = 0
    for s in clean_statements:
        try:
            cursor.execute(s)
            success += 1
        except Exception as e:
            # Uncomment for debugging specific errors
            # print(f"Error on statement: {s[:100]}...\n{e}")
            errors += 1
            
    conn.commit()
    conn.close()
    print(f"Migration completed! {success} statements executed successfully. {errors} ignored/failed.")

if __name__ == "__main__":
    sql_path = r"c:\Users\zakar\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\37AD10E0F8F9B0C3067FC6B8D65D9F509AB0868E\transfers\2026-13\ai_shifa_pro.sql"
    db_path = r"d:\chatBot Arab\partie Docteur+User\ai_shifa_pro.db"
    convert_mysql_to_sqlite(sql_path, db_path)
