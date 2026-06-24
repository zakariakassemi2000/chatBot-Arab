# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Unified Launcher
  Lance les deux applications Streamlit simultanément :
    • Projet principal (Chatbot arabe, Scanner IA)  →  http://localhost:8501
    • Espace Docteur + Patient (Auth, RDV, Dossiers) →  http://localhost:8503
═══════════════════════════════════════════════════════════════════════
  Usage :  python launch_shifa.py
═══════════════════════════════════════════════════════════════════════
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MAIN_APP = PROJECT_ROOT / "app.py"
DOCTOR_APP = PROJECT_ROOT / "partie Docteur+User" / "main.py"

# ── Port Config ────────────────────────────────────────────────
MAIN_PORT = "8501"
DOCTOR_PORT = "8503"

# ── Colors for terminal ───────────────────────────────────────
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def banner():
    print(f"""
{GREEN}{BOLD}
  ===========================================================
                 SHIFA AI - Unified Launcher            
  ===========================================================
    Chatbot Medical (arabe)    ->  http://localhost:{MAIN_PORT}     
    Espace Docteur + Patient   ->  http://localhost:{DOCTOR_PORT}     
  ===========================================================
{RESET}""")


def launch():
    banner()

    processes = []

    # ── 1. Launch main SHIFA AI app ──
    if MAIN_APP.exists():
        print(f"{CYAN}[1/2]{RESET} Lancement de SHIFA AI (port {MAIN_PORT})...")
        p1 = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(MAIN_APP),
             "--server.port", MAIN_PORT,
             "--server.headless", "true"],
            cwd=str(PROJECT_ROOT)
        )
        processes.append(("SHIFA AI", p1))
        print(f"  {GREEN}[OK] PID {p1.pid}{RESET}")
    else:
        print(f"  {RED}[ERR] Fichier introuvable : {MAIN_APP}{RESET}")

    time.sleep(2)

    # ── 2. Launch Docteur+User app ──
    if DOCTOR_APP.exists():
        print(f"{CYAN}[2/2]{RESET} Lancement de l'Espace Docteur+Patient (port {DOCTOR_PORT})...")
        p2 = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(DOCTOR_APP),
             "--server.port", DOCTOR_PORT,
             "--server.headless", "true"],
            cwd=str(DOCTOR_APP.parent)
        )
        processes.append(("Docteur+Patient", p2))
        print(f"  {GREEN}[OK] PID {p2.pid}{RESET}")
    else:
        print(f"  {RED}[ERR] Fichier introuvable : {DOCTOR_APP}{RESET}")

    if not processes:
        print(f"\n{RED}Aucune application lancée. Vérifiez les fichiers.{RESET}")
        return

    print(f"\n{GREEN}{BOLD}=== Les deux applications sont lancees ! ==={RESET}")
    print(f"{YELLOW}Appuyez sur Ctrl+C pour arreter toutes les applications.{RESET}\n")

    # ── Wait & Cleanup ──
    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"{RED}[!] {name} s'est arrêté (code: {proc.returncode}){RESET}")
            time.sleep(3)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Arrêt de toutes les applications...{RESET}")
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"  {GREEN}[OK] {name} arrêté{RESET}")
            except Exception:
                proc.kill()
                print(f"  {RED}[ERR] {name} forcé à s'arrêter{RESET}")
        print(f"\n{GREEN}Toutes les applications sont arrêtées. Au revoir !{RESET}")


if __name__ == "__main__":
    launch()
