import numpy as np
import robosuite as suite
import random
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
env = gym.make('Pendulum-v0')
env.reset()
for i in range(50):
    act = env.action_space.sample()
    env.step(act)
    env.render()
    input()

env.close()


'''
env = suite.make(
    "SawyerNutAssembly",
    has_renderer=True,           # no on-screen renderer
    has_offscreen_renderer=False, # no off-screen renderer
    use_object_obs=True,          # use object-centric feature
    use_camera_obs=False,         # no camera observations
)


env.reset()

for i in range(50):

    action = np.random.randn(env.dof)  # sample random action
    
    if random.random()<0.5:
        action[-1] = 1
    else:
        action[-1]= 0
    
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
    print(action)
    input("ENTER")
    #print('####################################')
    #obs1 = env._get_observation()
    #print(obs1['robot-state'].shape)
    # print(obs1['object-state'].shape)
    print('####################################')

env.close()

'''
