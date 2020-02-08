#!/usr/bin/env python3
import logging
import numpy as np
from os.path import abspath, dirname, join
from gym.spaces import Tuple

from mae_envs.viewer.env_viewer import EnvViewer
from mae_envs.wrappers.multi_agent import JoinMultiAgentActions
from mujoco_worldgen.util.envs import examine_env, load_env
from mujoco_worldgen.util.types import extract_matching_arguments



logger = logging.getLogger(__name__)



def main():
    '''
    examine.py is used to display environments and run policies.

    For an example environment jsonnet, see
        mujoco-worldgen/examples/example_env_examine.jsonnet
    You can find saved policies and the in the 'examples' together with the environment they were
    trained in and the hyperparameters used. The naming used is 'examples/<env_name>.jsonnet' for
    the environment jsonnet file and 'examples/<env_name>.npz' for the policy weights file.
    Example uses:
        bin/examine.py hide_and_seek
        bin/examine.py mae_envs/envs/base.py
        bin/examine.py base n_boxes=6 n_ramps=2 n_agents=3
        bin/examine.py my_env_jsonnet.jsonnet
        bin/examine.py my_env_jsonnet.jsonnet my_policy.npz
        bin/examine.py hide_and_seek my_policy.npz n_hiders=3 n_seekers=2 n_boxes=8 n_ramps=1
    '''
    name="blueprint_construction"

    env_name = name
    core_dir ='/Users/abhijithneilabraham/Documents/GitHub/Project-ANTON/Test_programs/mujocoenvcreator/multi-agent-emergence-environments/'
    envs_dir = 'mae_envs/envs'
    xmls_dir = 'xmls'
    kwargs={}

  # examine the environment
    examine_env(env_name, kwargs,
                core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                env_viewer=EnvViewer)



    print(main.__doc__)


if __name__ == '__main__':
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    main()
