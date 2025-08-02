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

# Percorsi ai file condivisi (relativi alla directory pytorchtest)
CURRENT_PATH = "current_id.json"      
CLIENT_PATH = "client_id.json"     

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

def wait_for_experiment_info(timeout=60):
    """
    Attende che le informazioni dell'esperimento siano disponibili
    
    Args:
        timeout: Tempo massimo di attesa in secondi
        
    Returns:
        dict: Informazioni dell'esperimento o None se timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if os.path.exists(CURRENT_PATH):
            experiment_info = load_json_safe(CURRENT_PATH)
            if experiment_info and "group_name" in experiment_info:
                print(f"✅ Informazioni esperimento caricate: {experiment_info['group_name']}")
                return experiment_info
        
        print("⏳ In attesa delle informazioni dell'esperimento...")
        time.sleep(2)
    
    print(f"⚠️  Timeout: impossibile caricare le informazioni dell'esperimento dopo {timeout}s")
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
        print(f"✅ Client registrato come: {client_name}")
        print(f"   🖥️  Hostname: {hostname}")
        print(f"   📊 Numero client: {client_number}")
    else:
        print("⚠️  Warning: Impossibile salvare il registro dei client")
    
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
    # Costruisci tags descrittivi
    tags = [
        f"nodes_{experiment_info.get('nodes', 'unknown')}",
        f"rounds_{experiment_info.get('rounds', 'unknown')}",
        f"epochs_{experiment_info.get('epochs', 'unknown')}",
        "federated_learning",
        "pytorch"
    ]
    
    # Configura wandb
    run = wandb.init(
        project="CNN_Stage",
        entity="damiano-cannizzaro-universit-di-torino",
        group=group_name,
        name=client_name,
        id=generate_wandb_id(client_name),
        tags=tags,
        config={
            "experiment_id": experiment_info.get("experiment_id"),
            "nodes": experiment_info.get("nodes"),
            "server_rounds": experiment_info.get("rounds"),
            "local_epochs": experiment_info.get("epochs"),
            "description": experiment_info.get("description", ""),
            "hostname": socket.gethostname()
        },
        notes=experiment_info.get("description", ""),
        reinit=False,
        resume="allow",
    )
    
    print(f"📊 Wandb configurato:")
    print(f"   🏷️  Gruppo: {group_name}")
    print(f"   📝 Nome: {client_name}")
    print(f"   🏷️  Tags: {', '.join(tags)}")
    
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
        
        print(f"💻 Client {client_name} inizializzato:")
        print(f"   🖥️  Device: {self.device}")
        print(f"   📊 Partition ID: {partition_id}/{num_partitions}")
        print(f"   📈 Epoche locali: {local_epochs}")
        print(f"   📚 Training samples: {len(trainloader.dataset)}")
        print(f"   🧪 Validation samples: {len(valloader.dataset)}")

    def fit(self, parameters, config):
        """Training del modello locale"""
        print(f"🏋️ [{self.client_name}] Inizio training locale...")
        
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        
        # Log su wandb
        self.run.log(
            {
                "train_loss": train_loss,
                "client_name": self.client_name,
                "partition_id": self.partition_id
            },
            commit=False,
        )
        
        print(f"✅ [{self.client_name}] Training completato - Loss: {train_loss:.4f}")
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Valutazione del modello"""
        print(f"🧪 [{self.client_name}] Inizio valutazione...")
        
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        # Log finale su wandb
        self.run.log({
            "evaluate_loss": loss,
            "evaluate_accuracy": accuracy,
            "client_name": self.client_name,
            "partition_id": self.partition_id
        })
        
        print(f"✅ [{self.client_name}] Valutazione completata - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        try:
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        finally:
            print(f"🏁 [{self.client_name}] Chiusura wandb run")
            self.run.finish()

def client_fn(context: Context):
    """Funzione principale del client con gestione migliorata degli esperimenti"""
    
    print("🚀 Inizializzazione client Flower...")
    
    # Caricamento modello e dati
    print("🧠 Caricamento modello e dati...")
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    
    print(f"📊 Configurazione client:")
    print(f"   🆔 Partition ID: {partition_id}")
    print(f"   📊 Num partitions: {num_partitions}")
    print(f"   📈 Local epochs: {local_epochs}")
    
    # Attesa e caricamento delle informazioni dell'esperimento
    print("⏳ Caricamento informazioni esperimento...")
    experiment_info = wait_for_experiment_info(timeout=120)
    
    if not experiment_info:
        print("❌ Impossibile caricare le informazioni dell'esperimento, uso fallback")
        # Fallback con naming basico
        group_name = f"FALLBACK_GROUP_{int(time.time())}"
        client_name = f"{group_name}_CLIENT_{socket.gethostname()}"
    else:
        # Registrazione del client nell'esperimento
        print("📝 Registrazione client nell'esperimento...")
        group_name, client_name = register_client_in_experiment(experiment_info)
    
    # Setup del tracking wandb
    print("📊 Configurazione tracking wandb...")
    if experiment_info:
        run = setup_wandb_tracking(group_name, client_name, experiment_info)
    else:
        # Fallback wandb config
        run = wandb.init(
            project="CNN_Stage",
            entity="damiano-cannizzaro-universit-di-torino",
            group=group_name,
            name=client_name,
            id=generate_wandb_id(client_name),
            reinit=False,
            resume="allow",
        )
    
    # Creazione e return del client
    print("✅ Client configurato e pronto per il training federato")
    client = FlowerClient(
        net, trainloader, valloader, local_epochs, 
        partition_id, run, num_partitions, client_name
    ).to_client()
    
    return client

# Flower ClientApp
app = ClientApp(
    client_fn,
)