#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:14:25 2020

@author: abhijithneilabraham
"""
import gym
env = gym.make("FetchPickAndPlace-v1")
observation = env.reset()
env.render()