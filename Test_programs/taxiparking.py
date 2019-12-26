#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:41:33 2019

@author: abhijithneilabraham
"""

import gym
env=gym.make("Taxi-v2").env
env.render()