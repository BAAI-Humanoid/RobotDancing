import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
import json
import subprocess
from loguru import logger




from utils.config_utils import *  # noqa: E402, F403


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  
        
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401


    # Isaac Gym must be imported before PyTorch.
    import torch  # noqa: E402
    from utils.common import seeding
    import wandb
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
        
    # Preserve interpolations for evaluation-time overrides.
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb:
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(project=project_name,
                entity=config.wandb.wandb_entity,
                name=run_name,
                sync_tensorboard=False,
                config=unresolved_conf,
                dir=wandb_dir)
    
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed = int(config.seed)
    torch_deterministic = bool(config.get("torch_deterministic", False))
    seeding(seed, torch_deterministic=torch_deterministic)
    logger.info(
        "Initialized Python/NumPy/Torch RNGs with seed={} "
        "(torch_deterministic={})",
        seed,
        torch_deterministic,
    )

    pre_process_config(config)


    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env: BaseEnv = instantiate(config=config.env, device=device)


    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)
    with open(experiment_save_dir / "resolved_config.yaml", "w") as file:
        OmegaConf.save(config, file, resolve=True)
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=hydra.utils.get_original_cwd(),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit_sha = "unknown"
    with open(experiment_save_dir / "run_metadata.json", "w") as file:
        json.dump(
            {
                "commit_sha": commit_sha,
                "seed": seed,
                "torch_deterministic": torch_deterministic,
                "experiment_name": str(config.experiment_name),
                "motion_file": str(config.robot.motion.motion_file),
            },
            file,
            indent=2,
        )
        file.write("\n")

    algo: BaseAlgo = instantiate(device=device, env=env, config=config.algo, log_dir=experiment_save_dir)
    algo.setup()
    if config.checkpoint is not None:
        algo.load(config.checkpoint)

    algo.learn()

    if simulator_type == 'IsaacSim':
        simulation_app.close()

if __name__ == "__main__":
    main()
