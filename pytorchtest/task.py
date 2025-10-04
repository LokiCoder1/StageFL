"""pytorchtest: A Flower / PyTorch app with Strong/Weak Scaling support."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
import wandb
import time
import os
import numpy as np

os.environ["WANDB_API_KEY"] ="c9ecc4c3eeac8445768b6c97a55298ddd835562d"
os.environ["WANDB_SILENT"] = "true"


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, scaling_mode: str = "strong", 
              samples_per_client: int = 5000, run_config: dict = None):
    """Load partition CIFAR10 data with Strong or Weak Scaling.
    
    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        scaling_mode: "strong" or "weak"
            - strong: Fixed total dataset size, divided by num_partitions
            - weak: Fixed samples per client, total dataset grows with num_partitions
        samples_per_client: Number of samples per client (only used in weak scaling)
    """
    global fds
    
    if scaling_mode == "strong":
        # STRONG SCALING: Dataset totale fisso, diviso tra N worker
        # Più worker = meno dati per worker
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
            fds_mode = "strong"
        partition = fds.load_partition(partition_id)
        print(f"[STRONG SCALING] Client {partition_id}: ~{len(partition)} samples total")
        
    elif scaling_mode == "weak":
        # WEAK SCALING: Samples per client fissi, dataset totale cresce con N
        # Ogni worker ha sempre lo stesso numero di campioni
        if fds is None:
            partitioner = IidPartitioner(num_partitions=1)
            # Carica dataset completo senza partizionamento
            fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train":partitioner})
        
        fds_mode = "weak"
        # Carica l'intero dataset
        full_dataset = fds.load_partition(0)  # Load full dataset from one partition
        
        # Sampling casuale IID per questo client
        np.random.seed(42 + partition_id)  # Seed diverso per ogni client
        total_available = len(full_dataset)
        
        # Assicurati di non superare la dimensione del dataset
        n_samples = min(samples_per_client, total_available)
        indices = np.random.choice(total_available, size=n_samples, replace=False)
        
        # Crea subset per questo client
        partition = full_dataset.select(indices.tolist())
        print(f"[WEAK SCALING] Client {partition_id}: {n_samples} samples (fixed per client)")
        print(f"  → Total dataset across {num_partitions} clients: ~{n_samples * num_partitions} samples")
    
    else:
        raise ValueError(f"scaling_mode must be 'strong' or 'weak', got '{scaling_mode}'")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
          
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# CONFIG: Setta qui i parametri di scaling
SCALING_CONFIG = {
    "mode": "weak",              # "strong" o "weak"
    "num_partitions": 10,        # Numero totale di client
    "samples_per_client": 3000,  # Solo per weak scaling
}


# Esempio di utilizzo:
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Carica dati secondo la configurazione
    print(f"\n=== {SCALING_CONFIG['mode'].upper()} SCALING ===")
    trainloader, testloader = load_data(
        partition_id=0, 
        num_partitions=SCALING_CONFIG["num_partitions"], 
        scaling_mode=SCALING_CONFIG["mode"],
        samples_per_client=SCALING_CONFIG.get("samples_per_client", 5000)
    )
    
    # Test training
    net = Net()
    print(f"\nTraining model...")
    loss = train(net, trainloader, epochs=1, device=device)
    print(f"Training loss: {loss:.4f}")
    
    # Test evaluation
    test_loss, accuracy = test(net, testloader, device=device)
    print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")