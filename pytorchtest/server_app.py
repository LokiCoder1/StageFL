"""pytorchtest: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from task import Net, get_weights
import os
import json
from datetime import datetime

# Percorsi ai file di configurazione (relativi alla directory pytorchtest)
GROUP_PATH = "group_id.json"
CURRENT_PATH = "current_id.json"
CLIENT_PATH = "client_id.json"
EXPERIMENT_PATH = "experiment_log.json"

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
    per sincronizzarsi con lo script di automazione
    """
    toml_path = "pyproject.toml"
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
            
        # Parsing semplice del TOML per estrarre i valori
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("num-server-rounds"):
                metadata["rounds"] = line.split("=")[1].strip()
            elif line.startswith("local-epochs"):
                metadata["epochs"] = line.split("=")[1].strip()
                
        # Il numero di nodi viene passato via environment variable dallo script di automazione
        metadata["nodes"] = os.environ.get("EXPERIMENT_NODES", "unknown")
        
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

def log_experiment_start(experiment_info):
    """Registra l'inizio di un nuovo esperimento nel log"""
    log_data = load_json_safe(EXPERIMENT_PATH, {"experiments": []})
    
    experiment_entry = {
        "experiment_id": experiment_info["experiment_id"],
        "group_name": experiment_info["group_name"],
        "nodes": experiment_info["nodes"],
        "rounds": experiment_info["rounds"],
        "epochs": experiment_info["epochs"],
        "start_time": datetime.now().isoformat(),
        "status": "running"
    }
    
    log_data["experiments"].append(experiment_entry)
    return save_json_safe(EXPERIMENT_PATH, log_data)

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
    
    print(f"📋 Parametri server:")
    print(f"   🖥️  Nodi: {nodes}")
    print(f"   🔄 Round: {rounds}")
    print(f"   📈 Epoche locali: {epochs}")
    print(f"   📊 Frazione fit: {fraction_fit}")
    
    # Inizializzazione modello
    print("🧠 Inizializzazione del modello...")
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Strategia federata
    print("⚙️  Configurazione strategia FedAvg...")
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    
    # Generazione e gestione del gruppo esperimento
    print("🏷️  Generazione identificativo esperimento...")
    experiment_id, group_name, experiment_info = generate_experiment_group_name(
        nodes, rounds, epochs
    )
    
    # Salvataggio delle informazioni dell'esperimento corrente
    if not save_current_experiment_info(experiment_info):
        print("⚠️  Warning: Impossibile salvare le informazioni dell'esperimento corrente")
    
    # Pulizia del registro client per il nuovo esperimento
    if not clear_client_registry():
        print("⚠️  Warning: Impossibile pulire il registro dei client")
    else:
        print("🧹 Registro client pulito per il nuovo esperimento")
    
    # Log dell'inizio dell'esperimento
    if not log_experiment_start(experiment_info):
        print("⚠️  Warning: Impossibile loggare l'inizio dell'esperimento")
    else:
        print("📝 Esperimento registrato nel log")
    
    print(f"✅ Server configurato per esperimento: {group_name}")
    print("🎯 Avvio del training federato...")
    
    return ServerAppComponents(strategy=strategy, config=config)

# Istanza dell'app
app = ServerApp(server_fn=server_fn)