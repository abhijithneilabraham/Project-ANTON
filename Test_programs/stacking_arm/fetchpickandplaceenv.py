import gym
env=gym.make('FetchReach-v1')
env.reset()
obs,reward,done,info=env.step(env.action_space.sample())
reward=env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)
print(reward)