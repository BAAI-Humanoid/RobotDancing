import torch
import torch.nn as nn
from torch.distributions import Normal
from copy import deepcopy

from .modules import BaseModule


class PMCEncoder(nn.Module):
    """VQ-VAE Encoder for motion representation learning"""
    def __init__(self, obs_dim, hidden_dim, latent_dim):
        super(PMCEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
    def forward(self, obs):
        return self.encoder(obs)


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encodings


def dict_to_namespace(d):
    """Convert dictionary to namespace object for BaseModule compatibility"""
    if isinstance(d, dict):
        class Config:
            def __init__(self, data):
                for key, value in data.items():
                    if isinstance(value, dict):
                        setattr(self, key, dict_to_namespace(value))
                    else:
                        setattr(self, key, value)
                self._data = data
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __setitem__(self, key, value):
                self._data[key] = value
                setattr(self, key, value)
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        return Config(d)
    return d


class PMCActor(nn.Module):
    """PMC Actor with integrated VQ-VAE"""
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std):
        super(PMCActor, self).__init__()
        
        self.encoder_obs_dim = obs_dim_dict['encoder_obs']
        self.actor_obs_dim = obs_dim_dict['actor_obs']
        self.num_actions = num_actions
        
        # Process module config and convert to namespace
        module_config_dict = self._process_module_config(module_config_dict, num_actions)
        module_config = dict_to_namespace(module_config_dict)
        
        # VQ-VAE configuration
        self.latent_dim = getattr(module_config, 'latent_dim', 64)
        self.hidden_dim = getattr(module_config, 'hidden_dim', 256)
        self.num_embeddings = getattr(module_config, 'num_embeddings', 512)
        self.use_vq = getattr(module_config, 'use_vq', True)
        
        # VQ-VAE components
        if self.use_vq:
            self.encoder = PMCEncoder(self.encoder_obs_dim, self.hidden_dim, self.latent_dim)
            self.vq_layer = VectorQuantizer(self.num_embeddings, self.latent_dim)
        
        # Policy network configuration - add latent dimension to input
        if self.use_vq:
            # Update input dimensions to include latent space
            policy_obs_dim_dict = obs_dim_dict.copy()
            policy_obs_dim_dict['actor_obs'] = self.actor_obs_dim + self.latent_dim
            self.policy_module = BaseModule(policy_obs_dim_dict, module_config)
        else:
            self.policy_module = BaseModule(obs_dim_dict, module_config)
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        
    def _process_module_config(self, module_config_dict, num_actions):
        # Create a copy to avoid modifying the original
        config = module_config_dict.copy()
        
        # Process output dimensions
        if 'output_dim' in config:
            output_dim = []
            for dim in config['output_dim']:
                if dim == 'robot_action_dim':
                    output_dim.append(num_actions)
                else:
                    output_dim.append(dim)
            config['output_dim'] = output_dim
        
        return config
        
    def encode_obs(self, obs):
        """Encode observations using VQ-VAE"""
        if not self.use_vq:
            return obs, {}
            
        # Encode
        z_e = self.encoder(obs)
        
        # Quantize
        z_q, vq_loss, perplexity, encodings = self.vq_layer(z_e)
        
        vq_info = {
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'z_e': z_e,
            'z_q': z_q
        }
        
        return z_q, vq_info
    
    def reset(self, dones=None):
        pass
    
    @property
    def actor(self):
        return self.policy_module
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, encoder_obs, actor_obs):
        vq_info = {}
        if self.use_vq:
            z_q, vq_info = self.encode_obs(encoder_obs)
            policy_input = torch.cat([actor_obs, z_q], dim=-1)
        else:
            policy_input = actor_obs
            
        mean = self.policy_module(policy_input)
        self.distribution = Normal(mean, mean*0. + self.std)
        return vq_info

    def act(self, encoder_obs, actor_obs, **kwargs):
        vq_info = self.update_distribution(encoder_obs, actor_obs)
        action = self.distribution.sample()
        return action, vq_info
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, encoder_obs, actor_obs):
        z_q, _ = self.encode_obs(encoder_obs)
        if self.use_vq:
            policy_input = torch.cat([actor_obs, z_q], dim=-1)
        else:
            policy_input = actor_obs
        actions_mean = self.policy_module(policy_input)
        return actions_mean
    
    def to_cpu(self):
        self.policy_module = deepcopy(self.policy_module).to('cpu')
        self.std.to('cpu')
        if self.use_vq:
            self.encoder = deepcopy(self.encoder).to('cpu')
            self.vq_layer = deepcopy(self.vq_layer).to('cpu')


class PMCCritic(nn.Module):
    """PMC Critic"""
    def __init__(self, obs_dim_dict, module_config_dict):
        super(PMCCritic, self).__init__()
        
        # Convert dict to namespace for BaseModule compatibility
        module_config = dict_to_namespace(module_config_dict)
        self.critic_module = BaseModule(obs_dim_dict, module_config)
    
    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, obs, **kwargs):
        return self.critic_module(obs)
