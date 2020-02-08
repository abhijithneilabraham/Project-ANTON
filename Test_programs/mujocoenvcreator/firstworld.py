#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 00:01:06 2020

@author: abhijithneilabraham
"""

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from mujoco_worldgen.util.envs import examine_env, load_env

model_class = DDPG  # works also with SAC, DDPG and TD3
env_name='blueprint_construction'
core_dir ='/Users/abhijithneilabraham/Documents/GitHub/multi-agent-emergence-environments/'
envs_dir = 'mae_envs/envs'
xmls_dir = 'xmls'
env,_=load_env(env_name, core_dir=core_dir,
                                           envs_dir=envs_dir, xmls_dir=xmls_dir,
                                           return_args_remaining=True)
# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1)
# Train the model
model.learn(1000)

model.save("./her_bit_env")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load('./her_bit_env', env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()