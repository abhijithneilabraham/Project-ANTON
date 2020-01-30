#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 04:29:11 2020

@author: abhijithneilabraham
"""

import gym
import numpy as np

from stable_baselines import DDPG

env = gym.make('FetchPickAndPlace-v1')
#
## the noise objects for DDPG
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

model_class = TD3  # works also with SAC, DDPG and TD3



# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1)
# Train the model
model.learn(3000)

model.save("./pickplace")
'''
 WARNING: you must pass an env
 or wrap your environment with HERGoalEnvWrapper to use the predict method
 '''
model = HER.load('./pickplace', env=env)

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs=env.reset()

   