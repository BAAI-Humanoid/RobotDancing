<h1 align="center">RobotDancing</h1>

<p align="center"><strong>Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking</strong></p>

<p align="center">
  <a href="https://baai-humanoid.github.io/RobotDancing/">Project</a> |
  <a href="https://ieeexplore.ieee.org/document/11652927">Paper</a> |
  <a href="#installation">Installation</a> |
  <a href="#training">Training</a> |
  <a href="#citation">Citation</a>
</p>

RobotDancing is a single-stage reinforcement-learning recipe for long-horizon, highly dynamic humanoid motion tracking. Instead of predicting absolute joint targets, the policy applies reference-conditioned residual corrections. Distribution-aware balancing and failure-aware prioritization improve training coverage of difficult motion segments.

The paper has been accepted by IEEE Robotics and Automation Letters (RA-L). RobotDancing trains one policy per reference sequence; this release is not a universal multi-motion controller.

## Platform support

| Platform | Training configuration | Results |
| --- | --- | --- |
| Unitree G1 | Supported | Primary platform. The paper reports Isaac Gym training, MuJoCo evaluation, and zero-shot hardware deployment. |
| Unitree H1 | Supported | Cross-platform simulation results and qualitative hardware evidence. |
| Unitree H1-2 | Supported | Cross-platform simulation results and qualitative hardware evidence. |

This repository provides training and simulation-side research code. Sim2Sim and hardware deployment stack are included in our deployment framework [RobotBridge](https://github.com/BAAI-Humanoid/RobotBridge).

## Repository structure

```text
humanoidverse/
  agents/                 Reinforcement-learning algorithms
  config/                 Hydra experiment and robot configurations
  data/motions/           Retargeted G1, H1, and H1-2 dance motions
  data/robots/            Robot descriptions and meshes
  envs/motion_tracking/   Motion-tracking environment
  simulator/              Simulation backends
  train_agent.py          Training entry point
  eval_agent.py           Evaluation entry point
isaac_utils/              Shared simulation utilities
```

## Installation

The original experiments use Linux, Python 3.8, NVIDIA Isaac Gym Preview 4, PyTorch 1.13.1, CUDA 11.7, and NumPy 1.23.

```bash
conda create -n robotdancing python=3.8 -y
conda activate robotdancing
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy==1.23 -y
```

Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download), extract it outside this repository, and install its Python package:

```bash
pip install -e /path/to/isaacgym/python
pip install -e isaac_utils
pip install -e .
```

Run an Isaac Gym example such as `joint_monkey.py` before starting training.

## Training

All examples below use an included motion and the RobotDancing reward, observation, domain-randomization, and MHPPO defaults.

### Unitree G1

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
  robot=g1/g1_23dof_lock_wrist \
  robot.motion.motion_file=humanoidverse/data/motions/g1/dance1_subject1_0_3945.pkl \
  project_name=RobotDancing \
  experiment_name=g1_dance1_subject1 \
  headless=True
```

### Unitree H1

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
  robot=h1/h1_19dof \
  robot.motion.motion_file=humanoidverse/data/motions/h1/dance1_subject1_0_3945.pkl \
  project_name=RobotDancing \
  experiment_name=h1_dance1_subject1 \
  headless=True
```

### Unitree H1-2

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
  robot=h1_2/h1_2_27dof_anneal_21dof \
  robot.motion.motion_file=humanoidverse/data/motions/h1_2/dance1_subject1_0_3945.pkl \
  project_name=RobotDancing \
  experiment_name=h1_2_dance1_subject1 \
  headless=True
```

Replace the motion filename with another retargeted sequence from the same platform directory. Each platform directory contains the eight LAFAN1 dance references used in the paper.

## Evaluation

Training saves `config.yaml` with each experiment. Evaluation reloads the robot, motion, observation, and reward settings from that configuration:

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
  +checkpoint=/path/to/experiment/model_<iteration>.pt
```

The checkpoint must be accompanied by its saved training configuration.

## Acknowledgements

RobotDancing builds on [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse). We also acknowledge the official [KungfuBot / PBHC](https://github.com/TeleHuman/PBHC) project, which informed the comparative implementation used in the paper. Please cite the respective projects when using their components.

## License

This project is released under the [MIT License](LICENSE). Robot assets and motion data remain subject to their respective upstream terms.

## Citation

```bibtex
@article{sun2026robotdancing,
  title={Robotdancing: Residual-action reinforcement learning enables robust long-horizon humanoid motion tracking},
  author={Sun, Zhenguo and Peng, Yibo and Meng, Yuan and Li, Xukun and Sun, Yu and Jiang, Haojun and Huang, Bo-Sheng and Bing, Zhenshan and Wang, Xinlong and Knoll, Alois},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  publisher={IEEE}
}
```
