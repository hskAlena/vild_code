import numpy as np
import robosuite as suite

import gym
'''
# create environment instance
env = suite.make(
    env_name="SawyerNutAssemblyRound", # try with other tasks like "Stack" and "Door"
    #robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    #action = np.random.randn(env.robots[0].dof) # sample random action
    action = env.action_space.sample() # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

'''

env = gym.make('Humanoid-v2')
env.reset()

for i in range(1000):
    act = env.action_space.sample()
    obs = env.step(act)
    env.render()
    input()

env.close()
