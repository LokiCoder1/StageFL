#!/usr/bin/env python3

import subprocess
import time
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Percorso al file pyproject.toml e al file dei timings
TOML_PATH = os.path.join("pytorchtest", "pyproject.toml")
TIMINGS_FILE = Path("pytorchtest/experiment_timings.json")

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

def update_toml(num_rounds, local_epochs, nodes, fraction, scaling_mode, samples_per_client):
    print_step(f"Aggiorno pyproject.toml: nodes={nodes}, rounds={num_rounds}, epochs={local_epochs}, fraction-fit={fraction}")
    print_step(f"  Scaling: {scaling_mode}, samples={samples_per_client}")
    try:
        with open(TOML_PATH, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        scaling_found = False
        samples_found = False
        
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
            elif stripped.startswith("scaling-mode"):
                new_lines.append(f'scaling-mode = "{scaling_mode}"\n')
                scaling_found = True
            elif stripped.startswith("samples-per-client"):
                new_lines.append(f"samples-per-client = {samples_per_client}\n")
                samples_found = True
            else:
                new_lines.append(line)
        
        # Se non esistono, aggiungili nella sezione [tool.flwr.app.config]
        if not scaling_found or not samples_found:
            print("⚠️ Parametri scaling non trovati, li aggiungo...")
            for i, line in enumerate(new_lines):
                if "[tool.flwr.app.config]" in line:
                    if not scaling_found:
                        new_lines.insert(i+1, f'scaling-mode = "{scaling_mode}"\n')
                    if not samples_found:
                        new_lines.insert(i+2 if not scaling_found else i+1, f"samples-per-client = {samples_per_client}\n")
                    break
        
        with open(TOML_PATH, "w") as f:
            f.writelines(new_lines)
        
        print("✅ File TOML aggiornato")
        print("[DEBUG] timing_file path:", TIMINGS_FILE.resolve())
        return True
    except Exception as e:
        print(f"❌ Errore aggiornando pyproject.toml: {e}")
        return False
    
def load_last_experiment():
    try:
        with TIMINGS_FILE.open("r") as f:
            data = json.load(f)
          
            # Cerca la lista dei timings
            if 'timings' in data:
                timings = data['timings']
                if timings:
                    last_id = timings[-1]['experiment_id']
                    print(f"ℹ️ Ultimo experiment_id trovato: {last_id}")
                    return last_id
            
            print("⚠️ Nessun timing trovato")
            return -1
                
    except Exception as e:
        print(f"❌ Errore: {e}")
        return -1
    

def wait_for_new_experiment(last_id: int, timeout: int = 7200, check_interval: int = 5):
    """Aspetta che venga scritto un nuovo timing nel JSON"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with TIMINGS_FILE.open("r") as f:
                data = json.load(f)
            if 'timings' in data:
                timings = data['timings']
                if timings:
                    current_id = timings[-1]['experiment_id']
                    if current_id > last_id:
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
    parser = argparse.ArgumentParser(description="Script per test automatizzati con Strong/Weak Scaling")
    parser.add_argument('--nodes', default='2-3', help='Range di nodi (es: 2-3 o 2,4,8)')
    parser.add_argument('--rounds', default='3-4', help='Range di round (es: 3-4 o 3,5,10)')
    parser.add_argument('--epochs', default='2-4', help='Range di epoche (es: 2-4 o 2,5,10)')
    parser.add_argument('--fraction', default='0.5', help='Frazione di client da selezionare')
    parser.add_argument('--scaling', default='strong', choices=['strong', 'weak'], 
                       help='Modalità di scaling: strong (dataset fisso) o weak (campioni fissi per client)')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Numero di campioni per client (solo per weak scaling)')
    parser.add_argument('--dry-run', action='store_true', help='Simula i test senza eseguirli')
    args = parser.parse_args()
    
    nodes_range = parse_range(args.nodes)
    rounds_range = parse_range(args.rounds)
    epochs_range = parse_range(args.epochs)
    fraction_fit = float(args.fraction)
    scaling_mode = args.scaling
    samples_per_client = args.samples
    
    print_header("CONFIGURAZIONE TEST")
    print(f"📊 Nodi: {nodes_range}")
    print(f"🔄 Round: {rounds_range}")
    print(f"📈 Epoche: {epochs_range}")
    print(f"⚖️  Frazione: {fraction_fit}")
    print(f"⚡ Scaling: {scaling_mode.upper()}")
    if scaling_mode == "weak":
        print(f"📦 Campioni per client: {samples_per_client}")
    
    total_tests = len(nodes_range) * len(epochs_range) * len(rounds_range)
    print(f"🎯 Numero totale di test: {total_tests}")
    
    if not args.dry_run:
        input("\n⏸️  Premi INVIO per continuare o CTRL+C per annullare...")
    
    test_counter = 0
    
    for nodes in nodes_range:
        print_header(f"TESTING CON {nodes} NODI ({scaling_mode.upper()} SCALING)")
        
        for local_epochs in epochs_range:
            print_section(f"Configurazione: {nodes} nodi, {local_epochs} epoche")
            
            for num_rounds in rounds_range:
                test_counter += 1
                
                print_step(f"TEST {test_counter}/{total_tests}")
                print(f"   🖥️  Nodi: {nodes}")
                print(f"   🔄 Round: {num_rounds}")
                print(f"   📈 Epoche: {local_epochs}")
                print(f"   ⚖️  Frazione: {fraction_fit}")
                print(f"   ⚡ Scaling: {scaling_mode}")
                if scaling_mode == "weak":
                    print(f"   📦 Campioni/client: {samples_per_client}")

                if args.dry_run:
                    print("   🔍 [DRY-RUN] Test simulato completato")
                    continue

                if not update_toml(num_rounds, local_epochs, nodes, fraction_fit, 
                                 scaling_mode, samples_per_client):
                    return

                os.environ["EXPERIMENT_NODES"] = str(nodes)

                # Ricava l'ultimo id
                last_experiment = load_last_experiment()
                
                if not run_command(f"make start NODES={nodes}"):
                    return

                print_step("⏳ Aspetto che il nuovo esperimento venga scritto...")
                if wait_for_new_experiment(last_experiment):
                    print("✅ Nuovo esperimento trovato, procedo con stop")
                    time.sleep(10)  # Piccola pausa per sicurezza
                else:
                    print("⚠️ Timeout, procedo comunque con stop")

                if not run_command(f"make stop NODES={nodes}"):
                    return

                print("✅ Test completato\n")
    
    print_header("TUTTI I TEST COMPLETATI!")
    print(f"🎉 Eseguiti {total_tests} test con successo")
    print(f"📊 Modalità di scaling utilizzata: {scaling_mode.upper()}")

if __name__ == "__main__":
    main()