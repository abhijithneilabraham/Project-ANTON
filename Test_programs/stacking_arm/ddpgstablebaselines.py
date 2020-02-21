#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:32:52 2020

@author: abhijithneilabraham
"""

import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC,DDPG

env = gym.make('FetchPush-v1')

model = DDPG(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_fetchpp")

del model # remove to demonstrate saving and loading

model = DDPG.load("sac_fetchpp")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()