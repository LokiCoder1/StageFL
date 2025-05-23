"""pytorchtest: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from task import Net, get_weights, load_data, set_weights, test, train
import time
import wandb
import os
#
## wandb integration
#os.environ["WANDB_API_KEY"] ="c9ecc4c3eeac8445768b6c97a55298ddd835562d"
#
#group_name = "experiment-" + time.strftime("%Y%m%d-%H%M")
#run_name = "client-" + wandb.util.generate_id()
#
#wandb.init(
#    project="CNN_Stage",    
#    entity="damiano-cannizzaro-universit-di-torino",
#    group = group_name,
#    name = run_name,
#    
#)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        wandb.log({
            "loss": loss,
         #   "epoch": _,
            "accuracy": accuracy,
        })
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
