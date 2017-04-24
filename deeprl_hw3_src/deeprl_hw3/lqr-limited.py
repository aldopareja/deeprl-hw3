''' implements lqr

'''
from deeprl_hw3.arm_env import TwoLinkArmEnv
import numpy as np
import time
import gym
from deeprl_hw3.controllers import calc_lqr_input
import matplotlib.pyplot as plt

#params
ENV = 'TwoLinkArm-v1'


env = gym.make(ENV)
sim_env = gym.make(ENV)

state = env.reset()
sim_env.reset()

done = False
i = 0
u0_list = []
u1_list = []
q1_list = []
q0_list = []
q1dot_list = []
q0dot_list = []
rewards = []
while not done:
    u = calc_lqr_input(env,sim_env)
    state,re,done,_ = env.step(u)
    i+=1
    u0_list.append(u[0])
    u1_list.append(u[1])
    q0_list.append(state[0])
    q1_list.append(state[1])
    q0dot_list.append(state[2])
    q1dot_list.append(state[3])
    rewards.append(re)
    if done:
        print('total reward',sum(rewards))
        print('number of steps',i)


plt.figure(1)
plt.subplot(321)
plt.plot(u0_list)
plt.title('u0')

plt.subplot(322)
plt.plot(u1_list)
plt.title('u1')

plt.subplot(323)
plt.plot(q0_list)
plt.title('q0')

plt.subplot(324)
plt.plot(q1_list)
plt.title('q1')

plt.subplot(325)
plt.plot(q0dot_list)
plt.title('dq0')

plt.subplot(326)
plt.plot(q1dot_list)
plt.title('dq1')

plt.show()

plt.figure(2)
plt.plot(rewards)
plt.title('reward')
plt.show()
