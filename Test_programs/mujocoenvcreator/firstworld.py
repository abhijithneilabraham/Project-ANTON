#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:40:34 2020

@author: abhijithneilabraham
"""

from mujoco_worldgen import *
world_params = WorldParams(size=(5, 5, 3.5))
builder = WorldBuilder(world_params, seed)
   # Create a floor
floor = Floor()

# Load geometries from XML, and add to floor
robot = ObjFromXML("particle")
floor.append(robot)
sphere = ObjFromXML("sphere")
floor.append(sphere)

# Create a primitive geometry, and add to floor
box = Geom('box')
floor.append(box)

# Add the root floor to the builder
builder.append(floor)