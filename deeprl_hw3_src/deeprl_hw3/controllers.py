"""LQR, iLQR and MPC."""

import numpy as np
from scipy.linalg import solve_continuous_are as sol_riccatti

#why would you have a different delta and dt?

def simulate_dynamics(sim_env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    sim_env.state = x.copy()
    # print('sim_env',sim_env.state)
    x_1 = sim_env._step(u,dt)[0]
    # print('x_1', x_1)
    return (x_1-x)/dt
    # return np.zeros(x.shape)


def approximate_A(sim_env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))
    for i in range(A.shape[1]):

      x_perturbed = x.copy()
      x_perturbed[i] -= delta
      # print('x_neg', x_perturbed)
      pert_neg = simulate_dynamics(sim_env,x_perturbed,u,dt)
      # print('out_neg',pert_neg)
      x_perturbed = x.copy()
      x_perturbed[i] += delta
      # print('x_pos', x_perturbed)
      pert_pos = simulate_dynamics(sim_env,x_perturbed,u,dt)
      # print('out_neg',pert_pos)

      A[:,i] = (pert_pos - pert_neg)/(2*delta)

    return A


def approximate_B(sim_env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will need to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))
    for i in range(B.shape[1]):

      u_perturbed = u.copy()
      u_perturbed[i] -= delta
      pert_neg = simulate_dynamics(sim_env,x,u_perturbed,dt)

      u_perturbed = u.copy()
      u_perturbed[i] += delta
      pert_pos = simulate_dynamics(sim_env,x,u_perturbed,dt)

      B[:,i] = (pert_pos - pert_neg)/(2*delta)

    return B


def calc_lqr_input(env, sim_env, u = None):

    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use th  is environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    #custom: added u as argument because I want to test around an specific u
    Returns
    -------
    u: np.array
      The command to execute at this point.
    
    """
    if u is None:
      u = np.zeros(env.DOF)

    x = np.hstack((env.q, env.dq))
    A = approximate_A(sim_env,x,u)
    B = approximate_B(sim_env,x,u)
    R = env.R
    Q = env.Q
    P = sol_riccatti(A, B, Q, R)
    K = np.linalg.inv(R).dot(((B.T).dot(P)))
    u = -K.dot(x-env.goal)
    # print(u)
    # print(u)
    return u
