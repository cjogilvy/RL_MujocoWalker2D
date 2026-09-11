# Deep Reinforcement Learning + [Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/)

We train a 2D agent to walk using DRL methods (DDPG, TD3, SAC) with further optimisations (Prioritised Experience Replay, NoisyNetworks).

## Requirements

- Python 3.10 or newer
- [Poetry](https://python-poetry.org/docs/#installation) for local installation
- [Docker](https://docs.docker.com/get-docker/) for containerised installation
- A Linux host with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU Docker runs

The training scripts write checkpoints, rewards, and other results to an `out/` directory. Create this directory before running an experiment:

```sh
mkdir -p out
```

On Windows PowerShell, use `New-Item -ItemType Directory -Force out` instead.

## Local installation

```sh
git clone https://github.com/rl-squad/atdp_walker.git
cd atdp_walker
```

```sh
poetry install
```

## Run experiments

```sh
# Compare SAC with uniform and prioritised replay buffers
poetry run python train_sac_prioritised.py

# Compare the three improvements from DDPG to TD3
poetry run python train_td3_ablation.py

# Run our final agent configuration
poetry run python train_final_agent.py

# Plot result files in ./out/
poetry run python utils/plot_npy_files.py

# Run a single training script. Replace filename with the desired output name.
OUT=filename poetry run python train_td3.py
```

The `OUT=...` form is for Linux and macOS shells. In Windows PowerShell, use:

```powershell
$env:OUT = "filename"; poetry run python train_td3.py
```

## Docker

Docker can be used locally or on a cloud VM/server. The commands below use standard Docker and do not depend on a particular cloud provider.

```sh
# Clone the repository on the machine that will run the training
git clone https://github.com/rl-squad/atdp_walker.git
cd atdp_walker

# Build the image
docker build -t atdp-walker .

# Create a host directory for results and open a shell in the container
mkdir -p out
docker run --rm -it -v "$(pwd)/out:/app/out" atdp-walker

# For an NVIDIA GPU, verify that Docker can access it first:
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi

# Start the container with GPU access
docker run --rm --gpus all -it -v "$(pwd)/out:/app/out" atdp-walker
```

Inside the container, run the training commands as usual:

```sh
# Replace the script and output name as needed
OUT=filename poetry run python train_agent_template.py
```

The `-v` option makes `/app/out` persistent on the host, so results remain available after the container exits. `--rm` removes the stopped container but does not remove the mounted results.

For long-running training on a remote server, SSH and cloud-provider commands are necessarily provider-specific. A typical workflow is:

```sh
# Run these on your local computer; replace the placeholders
ssh <username>@<server-address>
cd <path-to-repository>/atdp_walker
docker build -t atdp-walker .
mkdir -p out
docker run --rm -d --name atdp-training --gpus all \
        -v "$(pwd)/out:/app/out" atdp-walker \
        bash -lc 'OUT=filename poetry run python train_agent_template.py'

# Check output and follow the training logs
docker logs -f atdp-training

# Copy results back to your local computer, in a second local terminal
scp -r <username>@<server-address>:<path-to-repository>/atdp_walker/out ./out
```

The server must have Docker installed and, for GPU runs, a compatible NVIDIA driver and NVIDIA Container Toolkit. Cloud providers may also require firewall, SSH-key, quota, storage, or GPU-instance setup before these commands can work.

### University `hare` servers

If your server provides `hare` as a wrapper around Docker, the original workflow can still be used by replacing `docker` with `hare` and using the image name expected by that server. `hare`, its GPU syntax, and its attach commands are not standard Docker and are therefore not assumed in the general workflow.
