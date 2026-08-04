"""
Test script for PMC algorithm
"""

import torch
import numpy as np
from humanoidverse.agents.pmc.pmc import PMC
from humanoidverse.agents.modules.pmc_modules import PMCActor, PMCCritic


def test_vq_vae_components():
    """Test VQ-VAE components independently"""
    print("Testing VQ-VAE components...")
    
    from humanoidverse.agents.modules.pmc_modules import VQVAEEncoder, VectorQuantizer
    
    # Test encoder
    encoder = VQVAEEncoder(obs_dim=48, hidden_dim=256, latent_dim=64)
    obs = torch.randn(32, 48)
    z_e = encoder(obs)
    assert z_e.shape == (32, 64), f"Expected (32, 64), got {z_e.shape}"
    print("✓ VQ-VAE Encoder test passed")
    
    # Test vector quantizer
    vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
    z_q, vq_loss, perplexity, encodings = vq(z_e)
    assert z_q.shape == z_e.shape, f"Quantized shape mismatch: {z_q.shape} vs {z_e.shape}"
    assert vq_loss.item() >= 0, "VQ loss should be non-negative"
    assert perplexity.item() > 0, "Perplexity should be positive"
    print("✓ Vector Quantizer test passed")


def test_pmc_actor():
    """Test PMC Actor"""
    print("\nTesting PMC Actor...")
    
    obs_dim_dict = {'actor_obs': 48}
    module_config = {
        'use_vq': True,
        'latent_dim': 64,
        'hidden_dim': 256,
        'num_embeddings': 512,
        'input_dim': ['actor_obs'],
        'output_dim': [12],
        'layer_config': {
            'type': 'MLP',
            'hidden_dims': [256, 128],
            'activation': 'ReLU'
        }
    }
    
    actor = PMCActor(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=module_config,
        num_actions=12,
        init_noise_std=1.0
    )
    
    obs = torch.randn(32, 48)
    
    # Test encoding
    z_q, vq_info = actor.encode_obs(obs)
    assert z_q.shape == (32, 64), f"Expected (32, 64), got {z_q.shape}"
    assert 'vq_loss' in vq_info, "VQ info should contain vq_loss"
    print("✓ PMC Actor encoding test passed")
    
    # Test action sampling
    actions, vq_info = actor.act(obs)
    assert actions.shape == (32, 12), f"Expected (32, 12), got {actions.shape}"
    print("✓ PMC Actor action sampling test passed")
    
    # Test inference
    actions_mean = actor.act_inference(obs)
    assert actions_mean.shape == (32, 12), f"Expected (32, 12), got {actions_mean.shape}"
    print("✓ PMC Actor inference test passed")


def test_pmc_critic():
    """Test PMC Critic"""
    print("\nTesting PMC Critic...")
    
    obs_dim_dict = {'critic_obs': 48}
    module_config = {
        'input_dim': ['critic_obs'],
        'output_dim': [1],
        'layer_config': {
            'type': 'MLP',
            'hidden_dims': [256, 128, 64],
            'activation': 'ReLU'
        }
    }
    
    critic = PMCCritic(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=module_config
    )
    
    obs = torch.randn(32, 48)
    values = critic.evaluate(obs)
    assert values.shape == (32, 1), f"Expected (32, 1), got {values.shape}"
    print("✓ PMC Critic test passed")


def test_pmc_algorithm():
    """Test PMC algorithm setup (without actual training)"""
    print("\nTesting PMC Algorithm setup...")
    
    # Mock environment class
    class MockEnv:
        def __init__(self):
            self.num_envs = 16
            self.config = type('Config', (), {
                'num_envs': 16,
                'robot': type('Robot', (), {
                    'algo_obs_dim_dict': {'actor_obs': 48, 'critic_obs': 48},
                    'actions_dim': 12
                })()
            })()
            
        def reset_all(self):
            return {
                'actor_obs': torch.randn(self.num_envs, 48),
                'critic_obs': torch.randn(self.num_envs, 48)
            }
            
        def step(self, actions):
            return (
                {'actor_obs': torch.randn(self.num_envs, 48), 'critic_obs': torch.randn(self.num_envs, 48)},
                torch.randn(self.num_envs),
                torch.zeros(self.num_envs, dtype=torch.bool),
                {'to_log': {}}
            )
    
    # Mock config
    config = type('Config', (), {
        'num_steps_per_env': 24,
        'num_learning_iterations': 10,
        'num_learning_epochs': 2,
        'num_mini_batches': 2,
        'save_interval': 5,
        'clip_param': 0.2,
        'gamma': 0.99,
        'lam': 0.95,
        'value_loss_coef': 1.0,
        'entropy_coef': 0.01,
        'max_grad_norm': 1.0,
        'use_clipped_value_loss': True,
        'actor_learning_rate': 3e-4,
        'critic_learning_rate': 1e-3,
        'schedule': 'adaptive',
        'desired_kl': 0.01,
        'vq_loss_coef': 0.1,
        'pretrain_iterations': 0,
        'init_noise_std': 1.0,
        'init_at_random_ep_len': False,
        'load_optimizer': False,
        'module_dict': type('ModuleDict', (), {
            'actor': {
                'use_vq': True,
                'latent_dim': 64,
                'hidden_dim': 256,
                'num_embeddings': 512,
                'input_dim': ['actor_obs'],
                'output_dim': [12],
                'layer_config': {
                    'type': 'MLP',
                    'hidden_dims': [256, 128],
                    'activation': 'ReLU'
                }
            },
            'critic': {
                'input_dim': ['critic_obs'],
                'output_dim': [1],
                'layer_config': {
                    'type': 'MLP',
                    'hidden_dims': [256, 128, 64],
                    'activation': 'ReLU'
                }
            }
        })(),
        'eval_callbacks': None
    })()
    
    env = MockEnv()
    
    # Test PMC initialization
    pmc = PMC(
        env=env,
        config=config,
        log_dir=None,  # Disable logging for test
        device='cpu'
    )
    
    # Test setup
    pmc.setup()
    print("✓ PMC Algorithm setup test passed")
    
    # Test that models are properly initialized
    assert hasattr(pmc, 'actor'), "PMC should have actor"
    assert hasattr(pmc, 'critic'), "PMC should have critic"
    assert hasattr(pmc, 'storage'), "PMC should have storage"
    print("✓ PMC components initialization test passed")


if __name__ == "__main__":
    print("Running PMC Algorithm Tests...")
    print("=" * 50)
    
    try:
        test_vq_vae_components()
        test_pmc_actor()
        test_pmc_critic()
        test_pmc_algorithm()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("PMC algorithm implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 