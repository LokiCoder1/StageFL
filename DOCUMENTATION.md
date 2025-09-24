# StageFL - Documentazione Completa

## 📋 Indice

1. [Panoramica del Progetto](#panoramica-del-progetto)
2. [Struttura del Progetto](#struttura-del-progetto)
3. [Moduli Python](#moduli-python)
   - [server_app.py](#server_apppy)
   - [client_app.py](#client_apppy)
   - [task.py](#taskpy)
   - [run_experiments.py](#run_experimentspy)
4. [Script Shell](#script-shell)
5. [File di Configurazione](#file-di-configurazione)

---

## 🎯 Panoramica del Progetto

StageFL è un'implementazione di Federated Learning basata sul framework Flower (flwr) con PyTorch. Il sistema permette l'addestramento distribuito di un modello CNN sul dataset CIFAR-10 attraverso multiple macchine virtuali usando Docker.

### Caratteristiche Principali:
- **Federated Learning** con strategia FedAvg
- **Tracking avanzato** degli esperimenti con metriche dettagliate
- **Integrazione Weights & Biases** per il monitoraggio in tempo reale
- **Deployment automatizzato** su cluster di VM via SSH
- **Sistema di naming intelligente** per esperimenti e client

---

## 📁 Struttura del Progetto

```
StageFL/
├── pytorchtest/              # Modulo principale Flower
│   ├── server_app.py         # Server di federazione
│   ├── client_app.py         # Client di federazione
│   ├── task.py               # Modello e utilities ML
│   └── pyproject.toml        # Configurazione Flower
├── Makefile                  # Automazione Docker e deployment
├── Dockerfile                # Immagine container
├── requirements.txt          # Dipendenze Python
├── run_experiments.py        # Script automazione esperimenti
└── *.sh                      # Script di deployment e utility
```

---

## 🐍 Moduli Python

### server_app.py

Modulo del server di federazione che coordina il processo di training distribuito.

#### Classi

##### `TimedFedAvg`
Estensione di FedAvg con tracking del tempo e metriche avanzate.

```python
class TimedFedAvg(FedAvg):
    def __init__(self, experiment_id, group_name, nodes, rounds, epochs, *args, **kwargs)
```
- **Parametri:**
  - `experiment_id`: ID univoco dell'esperimento
  - `group_name`: Nome formattato dell'esperimento (es. EXP_001_N5_R50_E10)
  - `nodes`: Numero di nodi partecipanti
  - `rounds`: Numero di round del server
  - `epochs`: Numero di epoche locali per client
- **Funzionalità:**
  - Traccia il tempo di esecuzione totale
  - Salva i timing in `experiment_timings.json`
  - Log delle metriche per ogni round

**Metodi principali:**
- `initialize_parameters()`: Avvia il timer e inizializza i parametri
- `evaluate()`: Valuta il modello e traccia metriche per round

#### Funzioni Principali

##### `load_json_safe(path, default=None)`
Carica file JSON gestendo errori e file mancanti.
- **Return:** Contenuto JSON o valore di default

##### `save_json_safe(path, data)`
Salva dati in formato JSON con gestione errori.
- **Return:** True se successo, False altrimenti

##### `get_experiment_metadata()`
Estrae configurazione esperimento da `pyproject.toml`.
- **Return:** Dict con nodes, rounds, epochs

##### `load_last_experiment_id()`
Carica l'ultimo ID esperimento utilizzato.
- **Return:** ID dell'ultimo esperimento (int)

##### `save_last_experiment_id(experiment_id)`
Salva l'ultimo ID esperimento utilizzato.
- **Parametri:** experiment_id (int)
- **Return:** True se salvato con successo

##### `save_current_experiment_info(experiment_info)`
Salva le informazioni dell'esperimento corrente.
- **Parametri:** experiment_info (dict)
- **Return:** True se salvato con successo

##### `clear_client_registry()`
Pulisce il registro dei client per un nuovo esperimento.
- **Return:** True se pulito con successo

##### `generate_experiment_group_name(nodes, rounds, epochs)`
Genera nome strutturato per l'esperimento.
- **Return:** Tuple (experiment_id, group_name, experiment_info)
- **Formato nome:** `EXP_{ID:03d}_N{nodes}_R{rounds}_E{epochs}`

##### `server_fn(context: Context)`
Funzione principale del server Flower.
- **Parametri:** Context con configurazione run
- **Return:** ServerAppComponents con strategia e config
- **Funzionalità:**
  - Inizializza modello CNN
  - Crea esperimento con naming automatico
  - Configura strategia TimedFedAvg
  - Salva metadati esperimento

---

### client_app.py

Modulo client che gestisce il training locale e la comunicazione con il server.

#### Classi

##### `FlowerClient(NumPyClient)`
Client Flower personalizzato con tracking avanzato.

```python
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, 
                 partition_id, run, num_partitions, client_name)
```

**Metodi:**
- `fit()`: Esegue training locale e restituisce pesi aggiornati
- `evaluate()`: Valuta il modello sul validation set
- `get_and_increment_round()`: Gestisce contatore round per client

#### Funzioni Principali

##### `load_json_safe(path, default=None)`
Carica file JSON gestendo errori e file mancanti.
- **Return:** Contenuto JSON o valore di default

##### `save_json_safe(path, data)`
Salva dati in formato JSON con gestione errori.
- **Return:** True se successo, False altrimenti

##### `wait_for_experiment_info(timeout=30)`
Attende informazioni esperimento dal server.
- **Return:** Dict con info esperimento o None se timeout

##### `register_client_in_experiment(experiment_info)`
Registra client nell'esperimento corrente.
- **Return:** Tuple (group_name, client_name)
- **Formato nome client:** `{group_name}_CLIENT_{number:02d}`

##### `generate_wandb_id(name)`
Genera ID deterministico per run wandb.
- **Return:** Hash MD5 del nome

##### `setup_wandb_tracking(group_name, client_name, experiment_info)`
Configura tracking Weights & Biases.
- **Return:** wandb.Run object
- **Configurazione:**
  - Project: "CNN_Stage"
  - Group: nome esperimento
  - Tags: nodes, rounds, epochs, federated_learning, pytorch

##### `client_fn(context: Context)`
Funzione principale del client.
- **Return:** FlowerClient configurato
- **Funzionalità:**
  - Carica modello e dati partizionati
  - Attende e registra in esperimento
  - Configura wandb tracking
  - Crea e restituisce client

---

### task.py

Modulo contenente modello CNN e utilities per training/testing.

#### Classi

##### `Net(nn.Module)`
CNN semplice per classificazione CIFAR-10.

```python
class Net(nn.Module):
    # Architettura:
    # Conv2d(3, 6, 5) → ReLU → MaxPool2d(2, 2)
    # Conv2d(6, 16, 5) → ReLU → MaxPool2d(2, 2)
    # Linear(16*5*5, 120) → ReLU
    # Linear(120, 84) → ReLU
    # Linear(84, 10)
```

#### Funzioni Principali

##### `load_data(partition_id: int, num_partitions: int)`
Carica partizione del dataset CIFAR-10.
- **Return:** Tuple (trainloader, testloader)
- **Configurazione:**
  - Partizionamento IID
  - 80% train, 20% test
  - Batch size: 32
  - Normalizzazione: mean=0.5, std=0.5

##### `train(net, trainloader, epochs, device)`
Esegue training del modello.
- **Return:** Average training loss
- **Optimizer:** Adam con lr=0.01
- **Loss:** CrossEntropyLoss

##### `test(net, testloader, device)`
Valuta il modello.
- **Return:** Tuple (loss, accuracy)

##### `get_weights(net)`
Estrae pesi del modello come numpy arrays.
- **Return:** List di numpy arrays

##### `set_weights(net, parameters)`
Imposta pesi del modello da numpy arrays.

---

### run_experiments.py

Script automatizzazione per esecuzione batch di esperimenti.

#### Funzioni Principali

##### `print_separator(char="=", length=80)`
Stampa una linea separatrice.
- **Parametri:** carattere e lunghezza della linea

##### `print_header(message)`
Stampa un header formattato con separatori.
- **Parametri:** messaggio da visualizzare

##### `print_section(message)`
Stampa una sezione con separatori ridotti.
- **Parametri:** messaggio della sezione

##### `print_step(message)`
Stampa un messaggio con timestamp.
- **Parametri:** messaggio del passo corrente

##### `run_command(cmd)`
Esegue un comando shell e gestisce errori.
- **Parametri:** comando da eseguire
- **Return:** True se successo, False se errore

##### `update_toml(num_rounds, local_epochs, nodes, fraction)`
Aggiorna configurazione in pyproject.toml.
- **Return:** True se successo

##### `load_last_experiment()`
Carica l'ID dell'ultimo esperimento dal file dei timing.
- **Return:** ID ultimo esperimento o -1 se errore

##### `wait_for_new_experiment(last_id, timeout=3600, check_interval=5)`
Attende completamento esperimento monitorando timing file.
- **Parametri:** 
  - last_id: ID dell'ultimo esperimento
  - timeout: tempo massimo di attesa in secondi
  - check_interval: intervallo tra i controlli
- **Return:** True se nuovo esperimento trovato

##### `parse_range(range_str)`
Parsing di range numerici (es. "2-5" → [2,3,4,5]).
- **Return:** Lista di interi

##### `main()`
Funzione principale che:
1. Parsea argomenti CLI
2. Genera combinazioni di configurazioni
3. Esegue esperimenti in sequenza
4. Aggiorna pyproject.toml per ogni config
5. Avvia e ferma container automaticamente

**Argomenti CLI:**
- `--nodes`: Range nodi (default: '2-3')
- `--rounds`: Range round (default: '3-4')
- `--epochs`: Range epoche (default: '2-4')
- `--fraction`: Frazione fit (default: '0.5')
- `--dry-run`: Simula esecuzione

---

## 🐚 Script Shell

### Makefile
Automazione principale per gestione container e deployment.

**Target principali:**

#### `setup NODES=N`
Configura ambiente su tutti i nodi.
```bash
make setup NODES=5
```
- Chiama `env_setup.sh` per setup completo

#### `build`
Costruisce immagine Docker locale.
```bash
docker build -t fd-coordinator:5000/pytorch_image:latest .
```

#### `run T=server/client`
Avvia container in background.
- **Server**: `flower-superlink --insecure`
- **Client**: `flower-supernode` con parametri PARTITION e NUM_PARTITIONS

#### `start NODES=N`
Sequenza completa di avvio:
1. Avvia container server
2. Attende 15 secondi
3. Avvia N container client su VM remote
4. Inizia training automaticamente

#### `stop NODES=N`
Ferma e pulisce tutti i container:
1. Stop container server
2. Chiama `stop_containers.sh` per fermare client remoti
3. Esegue `docker container prune -f`

#### `train`
Esegue training nel container server.
```bash
docker exec -it pytorch_project_server sh -c "cd /app/pytorchtest && flwr run"
```
- Output salvato in `{T}_output.log`

#### `ssh NODES=N`
Deploy su VM remote:
1. Chiama `run_nodes.sh` per avviare container
2. Chiama `wandb_setup.sh` per configurare wandb

#### `shell T=server/client`
Accesso shell interattiva al container.

---

### run_nodes.sh
**Script di deployment parallelo su VM**

**Parametri:**
- `$1 (NODES)`: Numero di nodi da attivare

**Funzionamento:**
```bash
for i in $(seq 1 $NODES); do
    PARTITION=$((i - 1))  # Calcola partition ID (0-based)
    
    # Nome nodo con padding zero
    if [ $i -lt 11 ]; then
        NODE_NAME="fd-0$PARTITION"  # fd-00, fd-01, ..., fd-09
    else
        NODE_NAME="fd-$PARTITION"   # fd-10, fd-11, ...
    fi
    
    # SSH e avvia container
    ssh -tt $NODE_NAME "cd pytorchtest/ && \
        make run T=client SUPERLINK=fd-coordinator:9092 \
        PARTITION=$PARTITION NUM_PARTITIONS=$TOT_NODES"
done
```

---

### env_setup.sh
**Setup completo ambiente su tutti i nodi**

**Parametri:**
- `$1 (NODES)`: Numero di nodi da configurare

**Operazioni per ogni nodo:**
1. **Pulizia Docker**:
   ```bash
   docker system prune -af
   docker volume prune -f
   ```
2. **Build immagine**:
   ```bash
   cd pytorchtest/ && make build
   ```
3. **Avvio container e setup wandb**:
   ```bash
   make run T=client SUPERLINK=fd-coordinator:9092 ...
   docker exec -it pytorch_project_client pip install wandb
   ```
4. **Cleanup finale**: Stop di tutti i container

---

### stop_containers.sh
**Arresto coordinato container client**

**Parametri:**
- `$1 (NODES)`: Numero di nodi da fermare

**Variabili:**
- `PROJECT_NAME="pytorch_project"`
- `T="client"`

**Operazioni:**
```bash
for i in $(seq 1 $NODES); do
    # Calcola nome nodo (stesso algoritmo di run_nodes.sh)
    ssh -tt $NODE_NAME "docker stop ${PROJECT_NAME}_${T} && \
                        docker container prune -f"
done
```

---

### wandb_setup.sh
**Configurazione Weights & Biases su tutti i nodi**

**Parametri:**
- `$1 (NODES)`: Numero di nodi da configurare

**Variabili:**
- `WANDB_API_KEY`: API key hardcoded

**Operazioni:**
```bash
for i in $(seq 1 $NODES); do
    # Calcola nome nodo
    ssh -tt $NODE_NAME "docker exec -it pytorch_project_client \
                        sh -c \"wandb login $WANDB_API_KEY\""
done
```

---

### c3s_deploy.sh
**Deploy su cluster SLURM HPC**

**Funzionamento:**
1. **Estrazione nodi dal job SLURM**:
   ```bash
   nodelist=$(scontrol show job "$SLURM_JOB_ID" | \
             awk -F= '/NodeList|ExcNodeList/ {print $2}')
   nodes=($(scontrol show hostnames "$nodelist"))
   ```

2. **Filtraggio nodi broadwell**:
   ```bash
   for i in "${!nodes[@]}"; do
       if [[ "${nodes[i]}" == broadwell* ]]; then
           start_index=$i
           break
       fi
   done
   ```

3. **Setup nodi**:
   - Primo nodo = Coordinator (build locale)
   - Altri nodi = Workers (build via SSH)
   ```bash
   ssh "$node" "cd Stage/StageFL/ && make build ."
   ```

---

### run_client.sh
**Avvio manuale client Flower**

**Parametri posizionali:**
- `$1`: Partition ID
- `$2`: Numero totale partizioni

**Comando eseguito:**
```bash
flower-supernode --insecure \
    --superlink fd-coordinator:9092 \
    --node-config "partition=${1} num-partitions=${2}" \
    2>&1 | tee client_output_$1.log
```

---

### start_training.sh
**Script incompleto per avvio training**

**Note:**
- TODO: fix command inside container
- Attualmente solo scheletro:
  ```bash
  make shell T=server
  cd pytorchtest/
  flwr run
  ```

---

## ⚙️ File di Configurazione

### pyproject.toml
Configurazione principale Flower app.

```toml
[tool.flwr.app]
publisher = "dcannizzaro"

[tool.flwr.app.components]
serverapp = "server_app:app"
clientapp = "client_app:app"

[tool.flwr.app.config]
num-server-rounds = 2      # Round di training
fraction-fit = 1.0         # Frazione client per fit
fraction-evaluate = 1      # Frazione client per evaluate  
local-epochs = 1           # Epoche locali per client
num-nodes = 2              # Numero nodi totali
```

### requirements.txt
Dipendenze Python del progetto.
```
flwr[simulation]>=1.11.0
flwr-datasets[vision]>=0.3.0
torch
torchvision
wandb
```

### docker-compose.yml
Configurazione Docker Compose (alternativa a Makefile).
- Servizio flower-server
- Porta 9092 esposta
- Network bridge dedicata

### Dockerfile
Immagine Docker basata su pytorch/pytorch:latest.
- Working directory: /app
- Installa requirements.txt
- Espone porte 9091-9093
- Include WANDB_API_KEY

---

## 📊 File di Stato e Metriche

### File JSON Generati

1. **group_id.json**: Ultimo experiment ID utilizzato
2. **current_id.json**: Info esperimento corrente
3. **client_id.json**: Registro client per esperimento
4. **experiment_timings.json**: Tempi esecuzione esperimenti
5. **rounds_tracker.json**: Contatore round per client

### Struttura Dati Esperimento

```json
{
  "experiment_id": 1,
  "group_name": "EXP_001_N5_R50_E10",
  "nodes": 5,
  "rounds": 50,
  "epochs": 10,
  "timestamp": "2024-01-01T10:00:00",
  "description": "Federated Learning: 5 nodi, 50 round, 10 epoche locali"
}
```