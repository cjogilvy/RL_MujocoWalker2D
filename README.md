# Deep Reinforcement Learning for [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/)

A six-person group coursework project investigating deep reinforcement learning for continuous control in the MuJoCo Walker2d environment. We implemented and evaluated DDPG, TD3, and SAC agents in PyTorch, then studied prioritised experience replay, NoisyNet exploration, and the individual improvements that make up TD3.

The project was primarily an investigation into implementation and experimental understanding rather than a production-ready RL library. The full methodology, results, discussion, and limitations are available in the [project report](Report.pdf).

## What we investigated

- Implemented actor-critic agents and target-network training for continuous actions.
- Compared DDPG, TD3, and SAC under the same Walker2d task.
- Ran a repeated-agent ablation study of TD3's clipped double Q-learning, target policy smoothing, and delayed policy updates.
- Tested proportional prioritised experience replay with importance-sampling correction.
- Tested a NoisyNet policy as an alternative to Gaussian action noise.
- Used vectorised environments, periodic evaluation, checkpointing, and reward logging.
- Applied SHAP to examine which physical state features influenced the learned policy at different stages of training.

## Main findings

The experiments suggested that clipped double Q-learning was the most influential of the three TD3 improvements in this setting. Prioritised replay improved early learning speed, while NoisyNet exploration helped initially but did not produce the strongest final performance. SAC learned quickly in some runs but was less stable over longer training. These findings are specific to this experiment and are discussed in more detail in the [report](Report.pdf).

## My contribution

This was collaborative work, and the DDPG and TD3 implementations were developed and written jointly rather than by one person. My contributions included:

- Contributing to the shared implementation and development of the DDPG and TD3 algorithms.
- Leading the SAC analysis and comparison experiments.
- Contributing background research and the written report.
- Helping with debugging, experiment interpretation, and general group coding tasks.

## Setup

Requirements:

- Python 3.10 or newer
- [Poetry](https://python-poetry.org/docs/#installation)
- MuJoCo through [Gymnasium](https://gymnasium.farama.org/)
- PyTorch

Install the dependencies with:

```sh
poetry install
```

Training scripts write checkpoints, rewards, and other results to `out/`. Create the directory before running an experiment:

```sh
mkdir -p out
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force out
```

## Run experiments

```sh
# Compare SAC with uniform and prioritised replay
poetry run python train_sac_prioritised.py

# Ablate the three improvements from DDPG to TD3
poetry run python train_td3_ablation.py

# Run the final TD3 with prioritised replay configuration
poetry run python train_final_agent.py
```

For a single run, set `OUT` to the output name. On Linux and macOS:

```sh
OUT=filename poetry run python train_td3.py
```

On Windows PowerShell:

```powershell
$env:OUT = "filename"; poetry run python train_td3.py
```

Results in `out/` can be plotted with:

```sh
poetry run python utils/plot_npy_files.py
```

## Docker

The repository includes a Dockerfile for reproducible local or server-based runs. Build and run it with:

```sh
docker build -t walker2d .
mkdir -p out
docker run --rm -it -v "$(pwd)/out:/app/out" walker2d
```

For GPU-enabled Docker runs, the host also needs a compatible NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
