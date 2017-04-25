"""LQR, iLQR and MPC."""

import numpy as np
import scipy.linalg


def simulate_dynamics_next(env, x, u):
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

    Returns
    -------
    next_x: np.array
    """
    env.state = x.copy()
    x_1 = env.step(u)[0]
    return x_1
  

def approx_F(env, x, u, delta=1e-5, n=10):
    v = np.concatenate((x, u))
    X = np.zeros((n, v.shape[0]))
    Y = np.zeros((n, x.shape[0]))
    x_1 = simulate_dynamics_next(env, x, u)
    for i in xrange(n):
        pert = np.random.normal(size=v.shape)
        pert /= np.sqrt(np.sum(pert * pert))
        pert *= delta
        v_p = v + pert
        X[i] = pert
        Y[i] = simulate_dynamics_next(env, v_p[:x.shape[0]], v_p[x.shape[0]:]) - x_1
    return scipy.linalg.lstsq(X, Y)[0]


def cost_inter(env, x, u):
    """intermediate cost function

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

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    return np.sum(u ** 2), np.zeros((4,)), np.zeros((4, 4)), 2 * u, 2 * np.eye(2), np.zeros((2, 4))


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    return np.sum((x - env.goal)**2) * 1e4, 2 * (x - env.goal) * 1e4, 2 * np.eye(4) * 1e4


def simulate(env, x0, U):
    X = np.zeros((U.shape[0] + 1, 4))
    X[0] = x0
    for i in xrange(1, U.shape[0] + 1):
        X[i] = simulate_dynamics_next(env, X[i-1], U[i-1])
    return X


def total_cost(env, X, U):
    c = 0.0
    for i in xrange(U.shape[0]):
      c += cost_inter(env, X[i], U[i])[0]
    c += cost_final(env, X[-1])[0]
    return c


def calc_ilqr_input(env, sim_env, tN=100, max_iter=1e5):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    U = np.random.normal(size=(tN, 2), scale=10.0)
    alpha = 0.1
    mu = 1.0
    for _ in xrange(int(max_iter)):
        X = simulate(sim_env, env.state, U)
        c_0 = total_cost(env, X, U)
        _, Vx, Vxx = cost_final(env, X[-1])
        k = np.zeros((tN, 2))
        K = np.zeros((tN, 2, 4))
        for i in xrange(tN - 1, -1, -1):
            _, Lx, Lxx, Lu, Luu, Lux = cost_inter(env, X[i], U[i])
            F = approx_F(sim_env, X[i], U[i])
            Fx, Fu = F[:4], F[4:]
            Qx = Lx + np.dot(Fx, Vx)
            Qu = Lu + np.dot(Fu, Vx)
            Qxx = Lxx + np.dot(Fx, np.dot(Vxx, Fx.T))
            Quu = Luu + np.dot(Fu, np.dot(Vxx + mu * np.eye(4), Fu.T))
            Qux = Lux + np.dot(Fu, np.dot(Vxx + mu * np.eye(4), Fx.T))
            k[i] = -scipy.linalg.solve(Quu, Qu)
            K[i] = -scipy.linalg.solve(Quu, Qux)
            Vx = Qx + np.dot(K[i].T, np.dot(Quu, k[i])) + np.dot(K[i].T, Qu) + np.dot(Qux.T, k[i])
            Vxx = Qxx + np.dot(K[i].T, np.dot(Quu, K[i])) + np.dot(K[i].T, Qux) + np.dot(Qux.T, K[i])
        Xp = np.zeros(X.shape)
        Xp[0] = env.state
        Up = np.zeros(U.shape)
        for i in xrange(tN):
            Up[i] = U[i] + alpha * k[i] + np.dot(K[i], (Xp[i] - X[i]))
            Xp[i+1] = simulate_dynamics_next(sim_env, Xp[i], Up[i])
        c_1 = total_cost(env, Xp, Up)
        print c_0, '->', c_1
        U = Up
        if abs(c_1 - c_0) < 1e-3:
          break
    return U
