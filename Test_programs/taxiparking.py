#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:41:33 2019

@author: abhijithneilabraham
"""

import gym
env=gym.make("Taxi-v2").env
i=0
while i<5:
    env.render()
    env.reset()
    print("action space {}".format(env.action_space))
    print("observation space {}".format(env.observation_space))
    i+=1
    
print('state space')
state=env.encode(3,1,2,0)
print('state',state)
env.s=state
env.render()
