import numpy as np
import time
import gym
from . import ilqr
import matplotlib.pyplot as plt

def run_inputs(env, U, ENV):
    u0_list = []
    u1_list = []
    q1_list = []
    q0_list = []
    q1dot_list = []
    q0dot_list = []
    rewards = []
    for i in xrange(U.shape[0]):
        state,re,done,_ = env.step(U[i])
        env.render()
        u = U[i]
        u0_list.append(u[0])
        u1_list.append(u[1])
        q0_list.append(state[0])
        q1_list.append(state[1])
        q0dot_list.append(state[2])
        q1dot_list.append(state[3])
        rewards.append(re)
    print('total reward',sum(rewards))

    plt.figure(1)
    plt.subplot(321)
    plt.plot(u0_list)
    plt.title('u0')

    plt.subplot(322)
    plt.plot(u1_list)
    plt.title('u1')

    plt.subplot(323)
    plt.plot(q0_list)
    plt.plot([env.goal_q[0]]*len(q0_list))
    plt.title('q0')

    plt.subplot(324)
    plt.plot(q1_list)
    plt.plot([env.goal_q[1]]*len(q1_list))
    plt.title('q1')

    plt.subplot(325)
    plt.plot(q0dot_list)
    plt.plot([env.goal_dq[0]]*len(q0dot_list))
    plt.title('dq0')

    plt.subplot(326)
    plt.plot(q1dot_list)
    plt.plot([env.goal_dq[1]]*len(q1dot_list))
    plt.title('dq1')

    # plt.show()
    plt.savefig('ilqr-%s-1.png' % ENV[-2:], bbox_inches='tight')
    plt.clf()

    plt.figure(2)
    plt.plot(rewards)
    plt.title('reward')
    # plt.show()
    plt.savefig('ilqr-%s-2.png' % ENV[-2:], bbox_inches='tight')
    plt.clf()


def run_ilqr(ENV):
    env = gym.make(ENV)
    env.reset()
    sim_env = gym.make(ENV)
    sim_env.reset()
    iter_cost = []
    U = ilqr.calc_ilqr_input(env, sim_env, record_cost=iter_cost)
    run_inputs(env, U, ENV)
    plt.figure(3)
    plt.plot(iter_cost)
    plt.title('cost')
    # plt.show()
    plt.savefig('ilqr-%s-3.png' % ENV[-2:], bbox_inches='tight')
    plt.clf()
    print(iter_cost[:5])



def run_mpc(ENV, n=100):
    mpc = ilqr.MPC(n)
    env = gym.make(ENV)
    env.reset()
    sim_env = gym.make(ENV)
    sim_env.reset()
    u0_list = []
    u1_list = []
    q1_list = []
    q0_list = []
    q1dot_list = []
    q0dot_list = []
    rewards = []
    for i in xrange(n, 0, -1):
        u = mpc(env, sim_env)
        state,re,done,_ = env.step(u)
        env.render()
        u0_list.append(u[0])
        u1_list.append(u[1])
        q0_list.append(state[0])
        q1_list.append(state[1])
        q0dot_list.append(state[2])
        q1dot_list.append(state[3])
        rewards.append(re)
    print('total reward',sum(rewards))

    plt.figure(1)
    plt.subplot(321)
    plt.plot(u0_list)
    plt.title('u0')

    plt.subplot(322)
    plt.plot(u1_list)
    plt.title('u1')

    plt.subplot(323)
    plt.plot(q0_list)
    plt.plot([env.goal_q[0]]*len(q0_list))
    plt.title('q0')

    plt.subplot(324)
    plt.plot(q1_list)
    plt.plot([env.goal_q[1]]*len(q1_list))
    plt.title('q1')

    plt.subplot(325)
    plt.plot(q0dot_list)
    plt.plot([env.goal_dq[0]]*len(q0dot_list))
    plt.title('dq0')

    plt.subplot(326)
    plt.plot(q1dot_list)
    plt.plot([env.goal_dq[1]]*len(q1dot_list))
    plt.title('dq1')

    # plt.show()
    plt.savefig('mpc-%s-1.png' % ENV[-2:], bbox_inches='tight')
    plt.clf()

    plt.figure(2)
    plt.plot(rewards)
    plt.title('reward')
    # plt.show()
    plt.savefig('mpc-%s-2.png' % ENV[-2:], bbox_inches='tight')
    plt.clf()



def main():
    # run_ilqr('TwoLinkArm-v0')
    # run_ilqr('TwoLinkArm-v1')
    run_mpc('TwoLinkArm-v0')
    # run_mpc('TwoLinkArm-v1')

if __name__ == '__main__':
    main()
