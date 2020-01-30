import gym
from rl import DDPG
env=gym.make('FetchReach-v1')
env.reset()
obs,reward,done,info=env.step(env.action_space.sample())
reward=env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)
print(obs)

MAX_EPISODES = 900
MAX_EP_STEPS = 200
a_dim=13
s_dim=9
a_bound=[-1, 1]
rl = DDPG(a_dim, s_dim, a_bound)
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(env.action_space.sample())

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()
train()