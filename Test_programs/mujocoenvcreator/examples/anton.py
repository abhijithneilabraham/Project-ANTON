#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:57:13 2020

@author: abhijithneilabraham
"""

import numpy as np
from mujoco_worldgen import Env, WorldParams, WorldBuilder, Floor, ObjFromXML,Geom


def get_reward(sim):
    object_xpos = sim.data.get_site_xpos("object")
    target_xpos = sim.data.get_site_xpos("target")
    ctrl = np.sum(np.square(sim.data.ctrl))
    return -np.sum(np.square(object_xpos - target_xpos)) - 1e-3 * ctrl


def get_sim(seed):
    world_params = WorldParams(size=(10., 6., 4.5))
    builder = WorldBuilder(world_params, seed)
    floor = Floor()
    builder.append(floor)
    obj = ObjFromXML("particle_hinge")
    floor.append(obj)
    floorsize=4.
    # Walls
    wallsize = 0.1
    wall = Geom('box', (wallsize, floorsize, 0.5), name="wall1")
    wall.mark_static()
    floor.append(wall, placement_xy=(0, 0))
    wall = Geom('box', (wallsize, floorsize, 0.5), name="wall2")
    wall.mark_static()
    floor.append(wall, placement_xy=(1, 0))
    wall = Geom('box', (floorsize - wallsize*2, wallsize, 0.5), name="wall3")
    wall.mark_static()
    floor.append(wall, placement_xy=(1/2, 0))
    wall = Geom('box', (floorsize - wallsize*2, wallsize, 0.5), name="wall4")
    wall.mark_static()
    floor.append(wall, placement_xy=(1/2, 1))
    # Add agents
    obj = ObjFromXML("particle", name="agent0")
    floor.append(obj)
    obj.mark("object")
    floor.mark("target", (.5,   .5, 0.05))
    return builder.get_sim()


def make_env():
    return Env(get_sim=get_sim, get_reward=get_reward, horizon=30)
