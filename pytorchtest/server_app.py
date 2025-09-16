"""pytorchtest: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

class TimedFedAvg(FedAvg):
    """FedAvg con tracking del tempo di esecuzione e metriche avanzate"""
    
    def __init__(self, experiment_id, group_name, nodes, rounds, epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_id = experiment_id
        self.group_name = group_name
        self.nodes = nodes
        self.rounds = rounds
        self.epochs = epochs
        self.start_time = None
        self.end_time = None
        
        # Initialize experiment metrics tracker
        self.exp_tracker = get_experiment_tracker()
        self.exp_tracker.start_experiment(experiment_id, group_name, nodes, rounds, epochs)
    
    def initialize_parameters(self, client_manager):
        """Chiamato all'inizio del training"""
        import time
        self.start_time = time.time()
        print(f"⏱️  Inizio training: {self.group_name}")
        return super().initialize_parameters(client_manager)
    
    def evaluate(self, server_round, parameters):
        """Chiamato alla fine di ogni round"""
        result = super().evaluate(server_round, parameters)
        
        # Track round metrics if available
        if result is not None and len(result) >= 2:
            loss = result[0]
            metrics_dict = result[1] if len(result) > 1 else {}
            accuracy = metrics_dict.get("accuracy", 0.0)
            
            # Track this round in experiment metrics
            self.exp_tracker.track_round(
                round_num=server_round,
                aggregated_loss=loss,
                aggregated_accuracy=accuracy
            )
        
        # Se è l'ultimo round, calcola il tempo totale
        if server_round >= int(self.rounds):
            import time
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            # Converte stringhe in numeri
            def safe_int(value, default=0):
                try:
                    return int(value) if value != "unknown" else default
                except (ValueError, TypeError):
                    return default
            
            # Log del timing
            timing_data = {
                "experiment_id": self.experiment_id,
                "nodes": safe_int(self.nodes),
                "rounds": safe_int(self.rounds),
                "epochs": safe_int(self.epochs),
                "fraction-fit": self.fraction_fit,
                "execution_time_seconds": round(execution_time, 2),
                "execution_time_minutes": round(execution_time / 60, 2)
            }
            
            # Salva timing
            timing_file = "pytorchtest/experiment_timings.json"
            timings = load_json_safe(timing_file, {"timings": []})
            timings["timings"].append(timing_data)
            save_json_safe(timing_file, timings)
            
            print(f"⏱️  Training completato in: {execution_time:.1f}s ({execution_time/60:.1f} min)")
            
            # Finalize experiment metrics
            self.exp_tracker.finalize_experiment()
            
            # Print comparison report
            print("\n" + "="*70)
            print(self.exp_tracker.print_comparison_report())
        
        return result
from task import Net, get_weights
import os
import json
from datetime import datetime
from experiment_metrics import get_experiment_tracker

# Percorsi ai file di configurazione
GROUP_PATH = "pytorchtest/group_id.json"
CURRENT_PATH = "pytorchtest/current_id.json"
CLIENT_PATH = "pytorchtest/client_id.json"


def load_json_safe(path, default=None):
    """Carica un file JSON in modo sicuro, gestendo file mancanti o corrotti"""
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"⚠️  Warning: File {path} corrotto o non leggibile, uso default")
        return default if default is not None else {}

def save_json_safe(path, data):
    """Salva un file JSON in modo sicuro"""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except IOError as e:
        print(f"❌ Errore nel salvare {path}: {e}")
        return False

def get_experiment_metadata():
    """
    Estrae i metadati dell'esperimento dal file pyproject.toml
    """
    toml_path = "pytorchtest/pyproject.toml"
    metadata = {
        "nodes": "unknown",
        "rounds": "unknown", 
        "epochs": "unknown"
    }
    
    if not os.path.exists(toml_path):
        print(f"⚠️  Warning: {toml_path} non trovato, uso valori di default")
        return metadata
    
    try:
        with open(toml_path, "r") as f:
            content = f.read()
            
        # Parsing del TOML per estrarre i valori
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("num-server-rounds"):
                metadata["rounds"] = line.split("=")[1].strip()
            elif line.startswith("local-epochs"):
                metadata["epochs"] = line.split("=")[1].strip()
            elif line.startswith("num-nodes"):
                metadata["nodes"] = line.split("=")[1].strip()
                
    except Exception as e:
        print(f"⚠️  Warning: Errore nel leggere {toml_path}: {e}")
    
    return metadata

def load_last_experiment_id():
    """Carica l'ultimo ID esperimento usato"""
    data = load_json_safe(GROUP_PATH, {"last_experiment_id": 0})
    return data.get("last_experiment_id", 0)

def save_last_experiment_id(experiment_id):
    """Salva l'ultimo ID esperimento"""
    data = load_json_safe(GROUP_PATH, {})
    data["last_experiment_id"] = experiment_id
    return save_json_safe(GROUP_PATH, data)

def save_current_experiment_info(experiment_info):
    """Salva le informazioni dell'esperimento corrente"""
    return save_json_safe(CURRENT_PATH, experiment_info)

def clear_client_registry():
    """Pulisce il registro dei client per un nuovo esperimento"""
    return save_json_safe(CLIENT_PATH, {})

def generate_experiment_group_name(nodes, rounds, epochs):
    """
    Genera un nome di gruppo comprensibile e strutturato per l'esperimento
    
    Args:
        nodes: Numero di nodi partecipanti
        rounds: Numero di round del server
        epochs: Numero di epoche locali
        
    Returns:
        tuple: (experiment_id, group_name, experiment_info)
    """
    # Incrementa l'ID esperimento
    last_id = load_last_experiment_id()
    experiment_id = last_id + 1
    save_last_experiment_id(experiment_id)
    
    # Genera un nome di gruppo descrittivo
    # Formato: EXP_{ID:03d}_N{nodes}_R{rounds}_E{epochs}
    group_name = f"EXP_{experiment_id:03d}_N{nodes}_R{rounds}_E{epochs}"
    
    # Informazioni complete dell'esperimento
    experiment_info = {
        "experiment_id": experiment_id,
        "group_name": group_name,
        "nodes": nodes,
        "rounds": rounds,
        "epochs": epochs,
        "timestamp": datetime.now().isoformat(),
        "description": f"Federated Learning: {nodes} nodi, {rounds} round, {epochs} epoche locali"
    }
    
    print(f"🚀 Nuovo esperimento: {group_name}")
    print(f"   📊 Configurazione: {nodes} nodi, {rounds} round, {epochs} epoche")
    
    return experiment_id, group_name, experiment_info

def server_fn(context: Context):
    """Funzione principale del server con gestione migliorata degli esperimenti"""
    
    # Lettura parametri di configurazione dal context
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    local_epochs = context.run_config["local-epochs"]
    
    # Estrazione metadati dell'esperimento
    metadata = get_experiment_metadata()
    
    # Usa i valori dal context se disponibili, altrimenti dai metadati
    nodes = metadata["nodes"]
    rounds = str(num_rounds)
    epochs = str(local_epochs)
    
    print(f"📋 Server: {nodes} nodi, {rounds} round, {epochs} epoche, fraction: {fraction_fit}")
    
    # Inizializzazione modello
    print("🧠 Inizializzazione del modello...")
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Generazione e gestione del gruppo esperimento
    experiment_id, group_name, experiment_info = generate_experiment_group_name(
        nodes, rounds, epochs
    )
    
    # Strategia federata con timing
    strategy = TimedFedAvg(
        experiment_id=experiment_id,
        group_name=group_name,
        nodes=nodes,
        rounds=rounds,
        epochs=epochs,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    
    # Salvataggio delle informazioni dell'esperimento corrente
    if not save_current_experiment_info(experiment_info):
        print("⚠️  Warning: Impossibile salvare le informazioni dell'esperimento corrente")
    
    # Pulizia del registro client per il nuovo esperimento
    if not clear_client_registry():
        print("⚠️  Warning: Impossibile pulire il registro dei client")
    
    
    print(f"✅ Server pronto per: {group_name}")
    
    return ServerAppComponents(strategy=strategy, config=config)

# Istanza dell'app
app = ServerApp(server_fn=server_fn)