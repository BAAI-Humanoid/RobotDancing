import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from humanoidverse.agents.modules.pmc_modules import PMCActor, PMCCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.utils.training_progress import (
    completed_iteration,
    log_iteration_metrics,
    restore_training_progress,
    training_progress_state,
)

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()


class PMC(BaseAlgo):
    """
    Pretrained Motion Capture (PMC) algorithm implementation
    
    Based on the paper "Lifelike Agility and Play in Quadrupedal Robots using 
    Reinforcement Learning and Generative Pre-trained Models" by Tencent Robotics X.
    
    This algorithm integrates VQ-VAE with PPO for learning lifelike robot behaviors:
    1. Pretraining stage: VQ-VAE learns latent representations from animal motion data
    2. Adaptation stage: PPO trains control policies using the pretrained representations
    3. Strategic learning: Multi-agent games for strategic behavior emergence
    
    Key features:
    - Vector Quantized Variational AutoEncoder (VQ-VAE) for motion representation
    - PPO-based policy learning with VQ-VAE integration  
    - Support for pretrained motion representations
    - Multi-stage training pipeline
    
    References:
    - Paper: https://tencent-roboticsx.github.io/lifelike-agility-and-play/
    - Nature Machine Intelligence, Vol. 6, No. 7, 2024
    """
    def __init__(self, env: BaseTask, config, log_dir=None, device='cpu'):
        self.device = device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        # Training related Config
        self.num_steps_per_env = self.config.num_steps_per_env
        self.load_optimizer = self.config.load_optimizer
        self.num_learning_iterations = self.config.num_learning_iterations
        self.init_at_random_ep_len = self.config.init_at_random_ep_len
        self.save_interval = self.config.save_interval
        
        # logging related Config
        self.save_interval = self.config.save_interval
        
        # Algorithm related Config
        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.actor_learning_rate = self.config.actor_learning_rate
        self.critic_learning_rate = self.config.critic_learning_rate
        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        
        # PMC specific config
        self.vq_loss_coef = getattr(self.config, 'vq_loss_coef', 0.1)

        # adaptation
        self.counter = 0

    def setup(self):
        logger.info("Setting up PMC")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self):
        self.actor = PMCActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)

        self.critic = PMCCritic(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.critic
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('rewards', shape=(1,), dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        self.storage.register_key('values', shape=(1,), dtype=torch.float)
        self.storage.register_key('returns', shape=(1,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def _train_mode(self):
        self.actor.train()
        self.critic.train()

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
            (
                self.current_learning_iteration,
                self.tot_timesteps,
                self.tot_time,
            ) = restore_training_progress(
                loaded_dict, self.num_steps_per_env, self.env.num_envs, ckpt_path
            )
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        checkpoint = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'infos': infos,
        }
        checkpoint.update(training_progress_state(
            self.current_learning_iteration, self.tot_timesteps, self.tot_time
        ))
        torch.save(checkpoint, path)

    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            
        self._train_mode()

        num_learning_iterations = self.num_learning_iterations
        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(it)
            self.start_time = time.time()

            obs_dict = self._rollout_step(obs_dict)
            loss_dict = self._training_step()

            self.stop_time = time.time()  
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': completed_iteration(it),
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'total_learning_iterations': tot_iter,
            }
            self._post_epoch_logging(log_dict)
            self.current_learning_iteration = log_dict['it']
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(
                    self.log_dir,
                    'model_{}.pt'.format(self.current_learning_iteration),
                ))
            self.ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Get actions and values
                actions, vq_info = self.actor.act(obs_dict["encoder_obs"], obs_dict["actor_obs"])
                values = self.critic.evaluate(obs_dict["critic_obs"]).detach()
                
                # Get action statistics
                action_mean = self.actor.action_mean.detach()
                action_sigma = self.actor.action_std.detach()
                actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
                
                # Store observations and policy states
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])
                
                self.storage.update_key('actions', actions)
                self.storage.update_key('values', values)
                self.storage.update_key('action_mean', action_mean)
                self.storage.update_key('action_sigma', action_sigma)
                self.storage.update_key('actions_log_prob', actions_log_prob)
                
                # Environment step
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * values * infos['time_outs'].unsqueeze(1).to(self.device)
                    
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # Compute returns and advantages
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(
                    values=self.storage.query_key('values'), 
                    dones=self.storage.query_key('dones'), 
                    rewards=self.storage.query_key('rewards')
                )
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.actor.reset(dones)
        self.critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """Compute returns and advantages using GAE"""
        last_values = self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        advantage = 0
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
    
    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        self.counter += 1
        return loss_dict
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict['Value'] = 0
        loss_dict['Surrogate'] = 0
        loss_dict['Entropy'] = 0
        loss_dict['VQ_Loss'] = 0
        loss_dict['Perplexity'] = 0
        return loss_dict
    
    def _update_algo_step(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        encoder_obs_batch = policy_state_dict['encoder_obs']
        actor_obs_batch = policy_state_dict['actor_obs']
        critic_obs_batch = policy_state_dict['critic_obs']

        # Forward pass through actor
        _, vq_info = self.actor.act(encoder_obs_batch, actor_obs_batch)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # Forward pass through critic
        value_batch = self.critic.evaluate(critic_obs_batch)

        # KL divergence for learning rate scheduling
        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                    (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # PPO Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        
        # VQ-VAE loss
        vq_loss = vq_info.get('vq_loss', torch.tensor(0.0, device=self.device))
        perplexity = vq_info.get('perplexity', torch.tensor(0.0, device=self.device))
        
        # Total losses
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss + self.vq_loss_coef * vq_loss
        critic_loss = self.value_loss_coef * value_loss

        # Backward pass
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update loss dict
        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['VQ_Loss'] += vq_loss.item()
        loss_dict['Perplexity'] += perplexity.item()
        
        return loss_dict

    def set_learning_rate(self, actor_learning_rate, critic_learning_rate):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

    @property
    def inference_model(self):
        return {
            "actor": self.actor,
            "critic": self.critic
        }

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        scalar_log_dict = {}
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                scalar_log_dict['Episode/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        train_log_dict = {}
        mean_std = self.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps
        train_log_dict['mean_std'] = mean_std.item()

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}
        scalar_log_dict.update(env_log_dict)

        self._logging_to_writer(log_dict, train_log_dict, scalar_log_dict)

        str = f" \033[1m Learning iteration {log_dict['it']}/{log_dict['total_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict['collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (collection: {log_dict['collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n""")

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / log_dict['it'] * (log_dict['total_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        with Live(Panel(log_string, title="PMC Training Log"), refresh_per_second=4, console=console):
            pass

    def _logging_to_writer(self, log_dict, train_log_dict, scalar_log_dict):
        metrics = dict(scalar_log_dict)
        for loss_key, loss_value in log_dict['loss_dict'].items():
            metrics[f'Loss/{loss_key}'] = loss_value
        metrics.update({
            'Loss/actor_learning_rate': self.actor_learning_rate,
            'Loss/critic_learning_rate': self.critic_learning_rate,
            'Policy/mean_noise_std': train_log_dict['mean_std'],
            'Perf/total_fps': train_log_dict['fps'],
            'Perf/collection time': log_dict['collection_time'],
            'Perf/learning_time': log_dict['learn_time'],
            'Progress/total_timesteps': self.tot_timesteps,
            'Progress/elapsed_seconds': self.tot_time,
        })
        if len(log_dict['rewbuffer']) > 0:
            metrics['Train/mean_reward'] = statistics.mean(log_dict['rewbuffer'])
            metrics['Train/mean_episode_length'] = statistics.mean(log_dict['lenbuffer'])
        log_iteration_metrics(self.writer, metrics, log_dict['it'])

    # Code for Evaluation

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"]['encoder_obs'], actor_state["obs"]['actor_obs'])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def _get_inference_policy(self, device=None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference
