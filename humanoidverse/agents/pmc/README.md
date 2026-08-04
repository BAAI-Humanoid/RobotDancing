# PMC (Pretrained Motion Capture) Algorithm

This directory contains the implementation of the PMC algorithm based on the paper ["Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models"](https://tencent-roboticsx.github.io/lifelike-agility-and-play/) by Tencent Robotics X.

## Overview

PMC integrates Vector Quantized Variational AutoEncoder (VQ-VAE) with Proximal Policy Optimization (PPO) to learn lifelike robot behaviors. The algorithm follows a multi-stage training approach:

1. **Pretraining Stage**: VQ-VAE learns latent representations from animal motion data
2. **Adaptation Stage**: PPO trains control policies using the pretrained representations  
3. **Strategic Learning**: Multi-agent games for strategic behavior emergence

## Key Features

- **VQ-VAE Integration**: Learns compressed representations of animal motions
- **PPO-based Policy Learning**: Robust policy optimization with VQ-VAE features
- **Multi-stage Training**: Support for pretraining and fine-tuning phases
- **Lifelike Behaviors**: Emerges natural and strategic robot behaviors
- **Modular Design**: Clean separation between VQ-VAE and policy components

## Architecture

### Core Components

1. **VQVAEEncoder**: Encodes observations into latent space
2. **VectorQuantizer**: Quantizes continuous latents into discrete codes
3. **PMCActor**: Policy network with integrated VQ-VAE
4. **PMCCritic**: Value function estimator
5. **PMC**: Main algorithm orchestrating the training process

### Network Architecture

```
Observation → VQ-VAE Encoder → Vector Quantizer → Concatenate with Obs → Policy Network → Actions
                    ↓
            Quantization Loss (e_latent_loss + q_latent_loss)
```

## Usage

### Basic Usage

```python
from humanoidverse.agents.pmc.pmc import PMC
from humanoidverse.envs.your_env import YourEnv

# Create environment
env = YourEnv(config)

# Create PMC algorithm
pmc = PMC(
    env=env,
    config=config,
    log_dir="./logs/pmc_experiment",
    device="cuda"
)

# Setup and train
pmc.setup()
pmc.learn()
```

### Configuration

See `pmc_config_example.yaml` for a complete configuration example. Key parameters:

#### VQ-VAE Parameters
- `use_vq`: Enable/disable VQ-VAE (set to `false` for standard PPO)
- `latent_dim`: Dimension of latent codes (default: 64)
- `num_embeddings`: Size of codebook (default: 512)
- `vq_loss_coef`: Weight for VQ-VAE loss (default: 0.1)

#### Training Parameters
- `num_learning_iterations`: Total training iterations
- `pretrain_iterations`: Pretraining iterations for VQ-VAE
- `actor_learning_rate`: Learning rate for actor network
- `critic_learning_rate`: Learning rate for critic network

### Multi-stage Training

The algorithm supports different training phases:

1. **VQ-VAE Pretraining**: Train VQ-VAE on motion data
2. **Policy Adaptation**: Train policy with frozen/unfrozen VQ-VAE
3. **Joint Fine-tuning**: Fine-tune both VQ-VAE and policy

```python
# Enable pretraining phase
config.training_phases.pretrain_vqvae.enabled = True
config.training_phases.pretrain_vqvae.iterations = 1000

# Enable adaptation phase  
config.training_phases.adapt_policy.enabled = True
config.training_phases.adapt_policy.freeze_vqvae = False
```

## Implementation Details

### VQ-VAE Loss Function

The VQ-VAE loss consists of:
- **Reconstruction Loss**: MSE between original and reconstructed actions
- **VQ Loss**: Commitment loss for vector quantization
- **Perplexity**: Measure of codebook utilization

```python
total_loss = reconstruction_loss + vq_loss_coef * vq_loss
```

### PPO Integration

The PMC actor integrates VQ-VAE features into the policy:

```python
def update_distribution(self, obs):
    z_q, vq_info = self.encode_obs(obs)  # VQ-VAE encoding
    policy_input = torch.cat([obs, z_q], dim=-1)  # Concatenate features
    mean = self.policy_module(policy_input)  # Policy forward
    self.distribution = Normal(mean, mean*0. + self.std)
    return vq_info
```

### Training Loop

The training loop combines PPO updates with VQ-VAE losses:

```python
# Standard PPO losses
surrogate_loss = ...  # Policy gradient loss
value_loss = ...      # Value function loss  
entropy_loss = ...    # Entropy regularization

# VQ-VAE loss from encoding
vq_loss = vq_info.get('vq_loss', 0.0)

# Combined actor loss
actor_loss = surrogate_loss - entropy_coef * entropy_loss + vq_loss_coef * vq_loss
```

## Paper Implementation Notes

This implementation follows the key ideas from the original paper:

1. **Motion Representation Learning**: VQ-VAE learns discrete latent codes from animal motion data
2. **Behavior Priors**: Pretrained representations provide strong inductive biases
3. **Hierarchical Control**: High-level latent codes guide low-level policy execution
4. **Emergent Strategies**: Multi-agent training leads to strategic behaviors

### Differences from Paper

- Uses modular PyTorch implementation instead of specialized frameworks
- Integrates with existing PPO codebase for compatibility
- Simplified multi-agent setup (can be extended for chase-tag games)
- Configurable VQ-VAE parameters for different robot morphologies

## Evaluation and Analysis

### Metrics to Monitor

- **VQ Loss**: Vector quantization loss (should decrease during pretraining)
- **Perplexity**: Codebook utilization (higher is better, indicates diverse codes)
- **Reconstruction Loss**: Action reconstruction quality
- **Episode Reward**: Task performance
- **Surrogate Loss**: Policy gradient loss

### Visualization

The algorithm logs training metrics to TensorBoard:

```bash
tensorboard --logdir ./logs/pmc_experiment
```

Key plots to monitor:
- `Loss/VQ_Loss`: VQ-VAE training progress
- `Loss/Perplexity`: Codebook diversity  
- `Train/mean_reward`: Task performance
- `Loss/Surrogate`: Policy optimization

## Troubleshooting

### Common Issues

1. **High VQ Loss**: 
   - Check `vq_loss_coef` (try 0.1-1.0)
   - Verify `num_embeddings` is sufficient
   - Ensure proper learning rate for VQ-VAE

2. **Low Perplexity**:
   - Increase `num_embeddings`
   - Adjust commitment cost in VectorQuantizer
   - Check initialization of embedding vectors

3. **Poor Task Performance**:
   - Try training without VQ-VAE first (set `use_vq: false`)
   - Adjust `vq_loss_coef` (lower values give more weight to RL)
   - Verify observation preprocessing and reward design

4. **Training Instability**:
   - Reduce learning rates
   - Enable gradient clipping
   - Use smaller `clip_param` for PPO

### Best Practices

1. **Pretraining**: Always pretrain VQ-VAE on diverse motion data
2. **Gradual Integration**: Start with low `vq_loss_coef` and increase gradually  
3. **Monitor Perplexity**: Aim for perplexity > 64 for good codebook utilization
4. **Ablation Studies**: Compare with/without VQ-VAE to verify benefits

## References

- [Original Paper](https://tencent-roboticsx.github.io/lifelike-agility-and-play/)
- Nature Machine Intelligence, Vol. 6, No. 7, 2024
- [VQ-VAE Paper](https://arxiv.org/abs/1711.00937)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## Citation

If you use this implementation, please cite:

```bibtex
@article{han2024lifelike,
  title={Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models},
  author={Han, Lei and Zhu, Qingxu and Sheng, Jiapeng and Zhang, Chong and Li, Tingguang and Zhang, Yizheng and Zhang, He and Liu, Yuzhen and Zhou, Cheng and Zhao, Rui and others},
  journal={Nature Machine Intelligence},
  volume={6},
  number={7},
  year={2024}
}
``` 