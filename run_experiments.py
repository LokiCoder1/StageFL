#!/usr/bin/env python3

import subprocess
import time
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from timing_utils import load_last_timing

# Percorso al file pyproject.toml e al file dei timings
TOML_PATH = os.path.join("pytorchtest", "pyproject.toml")
TIMINGS_FILE = Path("/mnt/shared/dcannizzaro/pytorchtest/pytorchtest/experiment_timings.json")

def print_separator(char="=", length=80):
    print(char * length)

def print_header(message):
    print_separator("=")
    print(f"🚀 {message}")
    print_separator("=")

def print_section(message):
    print_separator("-", 60)
    print(f"📋 {message}")
    print_separator("-", 60)

def print_step(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"⏰ [{timestamp}] {message}")

def run_command(cmd):
    print_step(f"Eseguo comando: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Errore durante l'esecuzione di: {cmd}")
        return False
    print(f"✅ Comando completato con successo")
    return True

def update_toml(num_rounds, local_epochs, nodes, fraction):
    print_step(f"Aggiorno pyproject.toml: nodes={nodes}, rounds={num_rounds}, epochs={local_epochs}, fraction-fit={fraction}")
    try:
        with open(TOML_PATH, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("num-server-rounds"):
                new_lines.append(f"num-server-rounds = {num_rounds}\n")
            elif stripped.startswith("local-epochs"):
                new_lines.append(f"local-epochs = {local_epochs}\n")
            elif stripped.startswith("num-nodes"):
                new_lines.append(f"num-nodes = {nodes}\n")
            elif stripped.startswith("fraction-fit"):
                new_lines.append(f"fraction-fit = {fraction}\n")
            else:
                new_lines.append(line)
        
        with open(TOML_PATH, "w") as f:
            f.writelines(new_lines)
        
        print("✅ File TOML aggiornato")
        return True
    except Exception as e:
        print(f"❌ Errore aggiornando pyproject.toml: {e}")
        return False

def wait_for_new_timing(last_id: int, timeout: int = 3600, check_interval: int = 5):
    """Aspetta che venga scritto un nuovo timing nel JSON"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with TIMINGS_FILE.open("r") as f:
                data = json.load(f)
            timings = data.get("timings", [])
            if timings and timings[-1].get("experiment_id", -1) > last_id:
                return True
        except Exception:
            pass
        time.sleep(check_interval)
    return False

def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    elif ',' in range_str:
        return [int(x.strip()) for x in range_str.split(',')]
    else:
        return [int(range_str)]

def main():
    parser = argparse.ArgumentParser(description="Script per test automatizzati")
    parser.add_argument('--nodes', default='2-3')
    parser.add_argument('--rounds', default='3-4')
    parser.add_argument('--epochs', default='2-4')
    parser.add_argument('--fraction', default='0.5')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    nodes_range = parse_range(args.nodes)
    rounds_range = parse_range(args.rounds)
    epochs_range = parse_range(args.epochs)
    fraction_fit = float(args.fraction)
    
    print_header("CONFIGURAZIONE TEST")
    print(f"📊 Nodi: {nodes_range}")
    print(f"🔄 Round: {rounds_range}")
    print(f"📈 Epoche: {epochs_range}")
    print(f"⚖️  Frazione: {fraction_fit}")
    
    total_tests = len(nodes_range) * len(epochs_range) * len(rounds_range)
    print(f"🎯 Numero totale di test: {total_tests}")
    
    if not args.dry_run:
        input("\n⏸️  Premi INVIO per continuare o CTRL+C per annullare...")
    
    test_counter = 0
    
    for nodes in nodes_range:
        print_header(f"TESTING CON {nodes} NODI")
        
        for local_epochs in epochs_range:
            print_section(f"Configurazione: {nodes} nodi, {local_epochs} epoche")
            
            for num_rounds in rounds_range:
                test_counter += 1
                
                print_step(f"TEST {test_counter}/{total_tests}")
                print(f"   🖥️ Nodi: {nodes}")
                print(f"   🔄 Round: {num_rounds}")
                print(f"   📈 Epoche: {local_epochs}")
                print(f"   ⚖️ Frazione: {fraction_fit}")

                if args.dry_run:
                    print("   🔍 [DRY-RUN] Test simulato completato")
                    continue

                if not update_toml(num_rounds, local_epochs, nodes, fraction_fit):
                    return

                os.environ["EXPERIMENT_NODES"] = str(nodes)

                # Ricava l'ultimo id
                last_timing = load_last_timing()
                last_id = last_timing["experiment_id"] if last_timing else -1

                if not run_command(f"make start NODES={nodes}"):
                    return

                print_step("⏳ Aspetto che il nuovo timing venga scritto...")
                if wait_for_new_timing(last_id):
                    print("✅ Nuovo timing trovato, procedo con stop")
                    time.sleep(10)  # Piccola pausa per sicurezza
                else:
                    print("⚠️ Timeout, procedo comunque con stop")

                if not run_command(f"make stop NODES={nodes}"):
                    return

                print("✅ Test completato\n")
    
    print_header("TUTTI I TEST COMPLETATI!")
    print(f"🎉 Eseguiti {total_tests} test con successo")

if __name__ == "__main__":
    main()
