"""
PMC (Primitive Motion Controller) Algorithm

This module implements the PMC algorithm from the paper:
"Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning 
and Generative Pre-trained Models" by Tencent Robotics X.

The algorithm integrates VQ-VAE with PPO for learning lifelike robot behaviors.
"""

from .pmc import PMC

__all__ = ['PMC']

__version__ = '1.0.0'
__author__ = 'Based on Tencent Robotics X implementation'
__paper__ = 'https://tencent-roboticsx.github.io/lifelike-agility-and-play/' 