"""pytorchtest: A Flower / PyTorch app."""

import torch
import wandb
import os
import hashlib
import json
import socket
import time
from datetime import datetime

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import Net, get_weights, load_data, set_weights, test, train

# wandb integration
os.environ["WANDB_API_KEY"] = "c9ecc4c3eeac8445768b6c97a55298ddd835562d"
os.environ["WANDB_SILENT"] = "true"

# Percorsi ai file condivisi
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
        return default if default is not None else {}

def save_json_safe(path, data):
    """Salva un file JSON in modo sicuro"""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except IOError:
        return False

def wait_for_experiment_info(timeout=30):
    """
    Attende che le informazioni dell'esperimento siano disponibili
    
    Args:
        timeout: Tempo massimo di attesa in secondi
        
    Returns:
        dict: Informazioni dell'esperimento o None se timeout
    """
    start_time = time.time()
    printed_waiting = False
    
    while time.time() - start_time < timeout:
        if os.path.exists(CURRENT_PATH):
            experiment_info = load_json_safe(CURRENT_PATH)
            if experiment_info and "group_name" in experiment_info:
                return experiment_info
        
        if not printed_waiting:
            print("⏳ Attendo informazioni esperimento...")
            printed_waiting = True
        time.sleep(1)
    
    return None

def register_client_in_experiment(experiment_info):
    """
    Registra questo client nell'esperimento corrente
    
    Args:
        experiment_info: Informazioni dell'esperimento
        
    Returns:
        tuple: (group_name, client_name)
    """
    group_name = experiment_info["group_name"]
    hostname = socket.gethostname()
    
    # Carica il registro dei client
    client_registry = load_json_safe(CLIENT_PATH, {})
    
    # Inizializza il gruppo se non esiste
    if group_name not in client_registry:
        client_registry[group_name] = {
            "clients": {},
            "experiment_info": experiment_info,
            "created_at": datetime.now().isoformat()
        }
    
    group_clients = client_registry[group_name]["clients"]
    
    # Se il client è già registrato, restituisce il nome esistente
    if hostname in group_clients:
        client_name = group_clients[hostname]["client_name"]
        print(f"🔄 Client già registrato come: {client_name}")
        return group_name, client_name
    
    # Altrimenti, registra un nuovo client
    client_number = len(group_clients) + 1
    client_name = f"{group_name}_CLIENT_{client_number:02d}"
    
    # Registra il client con informazioni dettagliate
    group_clients[hostname] = {
        "client_name": client_name,
        "hostname": hostname,
        "registered_at": datetime.now().isoformat(),
        "client_number": client_number
    }
    
    # Salva il registro aggiornato
    if save_json_safe(CLIENT_PATH, client_registry):
        print(f"✅ Client registrato: {client_name} (#{client_number})")
    
    return group_name, client_name

def generate_wandb_id(name):
    """Genera un ID deterministic per wandb basato sul nome"""
    return hashlib.md5(name.encode()).hexdigest()

def setup_wandb_tracking(group_name, client_name, experiment_info):
    """
    Configura il tracking wandb per questo client
    
    Args:
        group_name: Nome del gruppo esperimento
        client_name: Nome univoco del client
        experiment_info: Informazioni complete dell'esperimento
        
    Returns:
        wandb.Run: Oggetto run di wandb
    """
    # Converti stringhe in numeri per wandb
    def safe_int(value, default=0):
        try:
            return int(value) if value != "unknown" else default
        except (ValueError, TypeError):
            return default
    
    # Tags (rimangono stringhe)
    tags = [
        f"nodes_{experiment_info.get('nodes', 'unknown')}",
        f"rounds_{experiment_info.get('rounds', 'unknown')}",
        f"epochs_{experiment_info.get('epochs', 'unknown')}",
        "federated_learning",
        "pytorch"
    ]
    
    # Config (solo numeri per evitare errori)
    config = {
        "experiment_id": experiment_info.get("experiment_id", 0),
        "nodes": safe_int(experiment_info.get("nodes")),
        "server_rounds": safe_int(experiment_info.get("rounds")),
        "local_epochs": safe_int(experiment_info.get("epochs")),
        "partition_id": None,  # Verrà impostato dal client
        "total_partitions": None,  # Verrà impostato dal client
    }
    
    # Configura wandb
    run = wandb.init(
        project="CNN_Stage",
        entity="damiano-cannizzaro-universit-di-torino",
        group=group_name,
        name=client_name,
        id=generate_wandb_id(client_name),
        tags=tags,
        config=config,
        notes=experiment_info.get("description", ""),
        reinit=False,
        resume="allow",
    )
    
    # Aggiorna config con info che sono stringhe
    wandb.config.update({
        "hostname": socket.gethostname(),
        "client_name": client_name,
    }, allow_val_change=True)
    
    print(f"📊 Wandb: {group_name} → {client_name}")
    return run

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id, run, num_partitions, client_name):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.run = run
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.client_name = client_name
        
        # File unico per tutti i round
        self.rounds_file = "pytorchtest/rounds_tracker.json"
        self.client_key = f"{client_name}_{partition_id}"
        
        print(f"💻 {client_name} pronto - Device: {self.device}, Samples: {len(trainloader.dataset)}")
        
        # Aggiorna wandb config con info specifiche del client
        run.config.update({
            "partition_id": partition_id,
            "total_partitions": num_partitions,
        }, allow_val_change=True)
    
    def get_and_increment_round(self, operation):
        """Gestisce il contatore dei round tramite file unico"""
        rounds_data = load_json_safe(self.rounds_file, {})
        
        # Inizializza il client se non esiste
        if self.client_key not in rounds_data:
            rounds_data[self.client_key] = {"train_rounds": 0, "eval_rounds": 0}
        
        client_rounds = rounds_data[self.client_key]
        
        if operation == "train":
            client_rounds["train_rounds"] += 1
            current_round = client_rounds["train_rounds"]
        else:  # evaluate
            client_rounds["eval_rounds"] += 1
            current_round = client_rounds["eval_rounds"]
        
        save_json_safe(self.rounds_file, rounds_data)
        return current_round

    def fit(self, parameters, config):
        """Training del modello locale"""
        current_round = self.get_and_increment_round("train")
        
        print(f"🏋️ [{self.client_name}] Round {current_round} - Training...")
        
        #Training status
        # self.run.log({f"{self.client_name}_status": 1}, commit = False)

        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        
        # Log su wandb (solo metriche numeriche)
        self.run.log({"train_loss": train_loss}, commit=False)
        
        #Idle status 
        # self.run.log({f"{self.client_name}_status": 0})

        print(f"✅ Training Loss: {train_loss:.4f}")
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Valutazione del modello"""
        current_round = self.get_and_increment_round("eval")
        print(f"🧪 [{self.client_name}] Round {current_round} - Valutazione...")
        
        #Evaluation status
        # self.run.log({f"{self.client_name}_status": 2}, commit=False)

        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        # Log finale su wandb (solo metriche numeriche)
        self.run.log({
            "evaluate_loss": loss,
            "evaluate_accuracy": accuracy,
        })
        
        #Idle status 
        # self.run.log({f"{self.client_name}_status": 0})

        print(f"✅ Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        try:
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        finally:
            print(f"🏁 Round {current_round} completato")
            self.run.finish()

def client_fn(context: Context):
    """Funzione principale del client - ogni cshiamata è un nuovo processo"""
    
    print("🚀 Inizializzazione client Flower...")
    print("🧠 Caricamento modello e dati...")
    
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    
    print(f"📊 Partition: {partition_id}/{num_partitions}, Epoche: {local_epochs}")
    
    # Caricamento informazioni esperimento (timeout ridotto per evitare attese)
    experiment_info = wait_for_experiment_info(timeout=30)
    
    if not experiment_info:
        print("❌ Fallback naming")
        group_name = f"FALLBACK_GROUP_{int(time.time())}"
        client_name = f"{group_name}_CLIENT_{socket.gethostname()}"
        run = wandb.init(
            project="CNN_Stage", 
            entity="damiano-cannizzaro-universit-di-torino",
            group=group_name, 
            name=client_name, 
            id=generate_wandb_id(client_name),
            reinit=False, 
            resume="allow"
        )
    else:
        group_name, client_name = register_client_in_experiment(experiment_info)
        run = setup_wandb_tracking(group_name, client_name, experiment_info)
    
    # Creazione client
    client = FlowerClient(
        net, trainloader, valloader, local_epochs, 
        partition_id, run, num_partitions, client_name
    ).to_client()
    
    return client

# Flower ClientApp
app = ClientApp(
    client_fn,
)