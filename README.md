# pytorchtest: A Flower / PyTorch app

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `pytorchtest` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)

# STAGE DOCS
## Run containers in different virtual machines

In the `pytorchtest` directory, use `make run T=server` in order to activate a Docker container as a server;
use `make ssh NODES=int` in order to activate a `NODES` number of docker container in different virtual machines

## Start training

``` bash
make train T=String # T can be `server` or `client`
```
## Stop containers

use `make stop` in order to stop and delete pytorch_project_server container;
use `make stop NODES=int` 
