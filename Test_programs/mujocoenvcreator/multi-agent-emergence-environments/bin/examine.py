#!/usr/bin/env python3
import logging
#import click
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.util.parse_arguments import parse_arguments


#logger = logging.getLogger(__name__)
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper


model_class = DDPG  # works also with SAC, DDPG and TD3
env_name = "hide_and_seek"
kwargs={}
core_dir ='/Users/abhijithneilabraham/Documents/GitHub/multi-agent-emergence-environments/'
envs_dir = 'mae_envs/envs'
xmls_dir = 'xmls'


env,_=load_env(env_name, core_dir=core_dir,
                                   envs_dir=envs_dir, xmls_dir=xmls_dir,
                                   return_args_remaining=True, **kwargs)

# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1)
# Train the model
model.learn(1000)

model.save("./hideandseek")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load('./hideandseek', env=env)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()




   

    


#    print(main.__doc__)


#if __name__ == '__main__':
#    logging.getLogger('').handlers = []
#    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

main()
