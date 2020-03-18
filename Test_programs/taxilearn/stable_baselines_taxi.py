#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:04:40 2020

@author: abhijithneilabraham
"""

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('Taxi-v2').env
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.encode(3,1,2,0)
done=False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    #env.reset()