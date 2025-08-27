#!/usr/bin/env python3

import subprocess
import time
import os
import argparse
from datetime import datetime

# Percorso al file pyproject.toml
TOML_PATH = os.path.join("pytorchtest", "pyproject.toml")

def print_separator(char="=", length=80):
    """Stampa una linea separatrice per migliorare la leggibilità"""
    print(char * length)

def print_header(message):
    """Stampa un header ben formattato"""
    print_separator("=")
    print(f"🚀 {message}")
    print_separator("=")

def print_section(message):
    """Stampa una sezione ben formattata"""
    print_separator("-", 60)
    print(f"📋 {message}")
    print_separator("-", 60)

def print_step(message):
    """Stampa un passo dell'esecuzione"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"⏰ [{timestamp}] {message}")

def run_command(cmd):
    """Esegue un comando shell e gestisce gli errori"""
    print_step(f"Eseguo comando: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Errore durante l'esecuzione di: {cmd}")
        return False
    print(f"✅ Comando completato con successo")
    return True

def update_toml(num_rounds, local_epochs, nodes):
    """Aggiorna il file pyproject.toml con i nuovi parametri"""
    print_step(f"Aggiorno pyproject.toml: nodes={nodes}, num-server-rounds={num_rounds}, local-epochs={local_epochs}")
    
    try:
        with open(TOML_PATH, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        updated_rounds = False
        updated_epochs = False
        updated_nodes = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("num-server-rounds"):
                new_lines.append(f"num-server-rounds = {num_rounds}\n")
                updated_rounds = True
            elif stripped.startswith("local-epochs"):
                new_lines.append(f"local-epochs = {local_epochs}\n")
                updated_epochs = True
            elif stripped.startswith("num-nodes"):
                new_lines.append(f"num-nodes = {nodes}\n")
                updated_nodes = True
            else:
                new_lines.append(line)
        
        # Se num-nodes non esisteva, aggiungilo nella sezione config
        if not updated_nodes:
            # Trova dove inserirlo (dopo local-epochs)
            for i, line in enumerate(new_lines):
                if line.strip().startswith("local-epochs"):
                    new_lines.insert(i + 1, f"num-nodes = {nodes}\n")
                    break
        
        if not updated_rounds or not updated_epochs:
            print(f"⚠️  Attenzione: alcuni parametri potrebbero non essere stati trovati nel file TOML")
        
        with open(TOML_PATH, "w") as f:
            f.writelines(new_lines)
        
        print(f"✅ File TOML aggiornato con successo")
        return True
        
    except FileNotFoundError:
        print(f"❌ Errore: File {TOML_PATH} non trovato")
        return False
    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento del file TOML: {e}")
        return False

def calculate_wait_times(num_rounds, local_epochs, nodes):
    """
    Calcola i tempi di attesa in modo più funzionale basandosi sui parametri
    
    Args:
        num_rounds: Numero di round del server
        local_epochs: Numero di epoche locali
        nodes: Numero di nodi
    
    Returns:
        tuple: (wait_start_to_stop, wait_stop_to_next)
    """
    # Tempo base per l'inizializzazione 
    base_init_time = 90  
    
    # Tempo stimato per round e epoche
    # Questi valori sono ipotetici e possono essere regolati in base ai test
    time_per_round = 30 + (local_epochs * 6) + (nodes * 3) 
    
    # Tempo totale stimato per tutti i round
    total_round_time = num_rounds * time_per_round
    
    # Tempo di attesa dopo start (con margine di sicurezza del 10%)
    wait_start_to_stop = base_init_time + int(total_round_time * 1.1)
    
    # Tempo di attesa dopo stop 
    wait_stop_to_next = 45 + (nodes * 3) 
    
    return wait_start_to_stop, wait_stop_to_next

def parse_range(range_str):
    """
    Converte una stringa come "3-6" o "3,4,5" in una lista di numeri
    
    Args:
        range_str: Stringa che rappresenta un range (es. "3-6" o "3,4,5")
    
    Returns:
        list: Lista di numeri
    """
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    elif ',' in range_str:
        return [int(x.strip()) for x in range_str.split(',')]
    else:
        return [int(range_str)]

def main():
    parser = argparse.ArgumentParser(
        description="Script per test automatizzati con diversi parametri",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python3 script.py --nodes 2-4 --rounds 3-6 --epochs 2,3,4
  python3 script.py --nodes 2,3,4 --rounds 5 --epochs 2-4
  python3 script.py --nodes 3 --rounds 3-5 --epochs 3
        """
    )
    
    parser.add_argument(
        '--nodes', 
        default='2-3',
        help='Range di nodi da testare (es. "2-4" o "2,3,4"). Default: 2-3'
    )
    
    parser.add_argument(
        '--rounds', 
        default='3-4',
        help='Range di round da testare (es. "3-6" o "3,4,5"). Default: 3-4'
    )
    
    parser.add_argument(
        '--epochs', 
        default='2-4',
        help='Range di epoche da testare (es. "2-4" o "2,3,4"). Default: 2-4'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Mostra solo quello che verrebbe eseguito senza eseguirlo realmente'
    )
    
    args = parser.parse_args()
    
    # Parsing dei range
    try:
        nodes_range = parse_range(args.nodes)
        rounds_range = parse_range(args.rounds)
        epochs_range = parse_range(args.epochs)
    except ValueError as e:
        print(f"❌ Errore nel parsing dei parametri: {e}")
        return
    
    # Mostra la configurazione
    print_header("CONFIGURAZIONE TEST AUTOMATIZZATI")
    print(f"📊 Nodi da testare: {nodes_range}")
    print(f"🔄 Round da testare: {rounds_range}")
    print(f"📈 Epoche da testare: {epochs_range}")
    print(f"🏃‍♂️ Modalità dry-run: {'Sì' if args.dry_run else 'No'}")
    
    # Calcola il numero totale di test
    total_tests = len(nodes_range) * len(epochs_range) * len(rounds_range)
    print(f"🎯 Numero totale di test: {total_tests}")
    
    if not args.dry_run:
        input("\n⏸️  Premi INVIO per continuare o CTRL+C per annullare...")
    
    test_counter = 0
    
    for nodes in nodes_range:
        print_header(f"TESTING CON {nodes} NODI")
        
        for local_epochs in epochs_range:
            print_section(f"Configurazione: {nodes} nodi, {local_epochs} epoche locali")
            
            for num_rounds in rounds_range:
                test_counter += 1
                
                print_step(f"TEST {test_counter}/{total_tests}")
                print(f"   🖥️  Nodi: {nodes}")
                print(f"   🔄 Round: {num_rounds}")
                print(f"   📈 Epoche: {local_epochs}")
                
                # Calcola i tempi di attesa
                wait_start_to_stop, wait_stop_to_next = calculate_wait_times(
                    num_rounds, local_epochs, nodes
                )
                
                print(f"   ⏱️  Tempo attesa dopo start: {wait_start_to_stop}s")
                print(f"   ⏱️  Tempo attesa dopo stop: {wait_stop_to_next}s")
                
                if args.dry_run:
                    print("   🔍 [DRY-RUN] Test simulato completato")
                    continue
                
                # Aggiorna pyproject.toml
                if not update_toml(num_rounds, local_epochs, nodes):
                    print("❌ Errore nell'aggiornamento del file TOML, esco.")
                    return
                
                # Imposta la variabile d'ambiente per il numero di nodi
                os.environ["EXPERIMENT_NODES"] = str(nodes)
                
                # Lancia make start
                start_cmd = f"make start NODES={nodes}"
                if not run_command(start_cmd):
                    print("❌ Errore nel comando start, esco.")
                    return
                
                # Attendi dopo start
                print_step(f"Attendo {wait_start_to_stop} secondi per il completamento...")
                time.sleep(wait_start_to_stop)
                
                # Lancia make stop
                stop_cmd = f"make stop NODES={nodes}"
                if not run_command(stop_cmd):
                    print("❌ Errore nel comando stop, esco.")
                    return
                
                # Attendi dopo stop
                print_step(f"Attendo {wait_stop_to_next} secondi per il cleanup...")
                time.sleep(wait_stop_to_next)
                
                print("✅ Test completato con successo")
                print()
    
    print_header("TUTTI I TEST COMPLETATI!")
    print(f"🎉 Eseguiti {total_tests} test con successo")
    
if __name__ == "__main__":
    main()