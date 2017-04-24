from deeprl_hw3.arm_env import TwoLinkArmEnv
import numpy as np
import time
import gym
env= gym.make('TwoLinkArm-v0')
env.reset()
start = time.time()
for i in range(int(10e3)):
    if i % 100 == 0:
    	print(time.time()-start)
    	start = time.time()
    env.step(100*np.ones(2))
    # env.render()