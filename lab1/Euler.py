import numpy as np
import sys


def EulerSolve(f, U0, T, n):
    t = np.linspace(0, T, n + 1)
    u = np.zeros(n + 1)
    dt = T / float(n)

    u[0] = U0
    for k in range(int(n)):
        u[k + 1] = u[k] + dt * f(u[k], t[k])

    return u, t


def RungeKuttaSolve(f, U0, T, n):
    t = np.linspace(0, T, n + 1)
    u = np.zeros(n + 1)
    dt = T / float(n)

    u[0] = U0
    for k in range(int(n)):
        k1 = dt * f(u[k], t[k])
        k2 = dt * f(u[k] + (1.0 / 2.0) * k1, t[k] + (1.0 / 2.0) * dt)
        k3 = dt * f(u[k] + (1.0 / 2.0) * k2, t[k] + (1.0 / 2.0) * dt)
        k4 = dt * f(u[k] + k3, t[k] + dt)
        u[k + 1] = u[k] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return u, t


def f(u, t):
    return u


def test(U0, T, n):
    
    # U0 = float(sys.argv[1])
    # T = float(sys.argv[2])
    # n = float(sys.argv[3])

    # u, t = EulerSolve(f, U0, T, n)
    u, t = RungeKuttaSolve(f, U0, T, n)

    for i in range(int(n + 1)):
        print("t : %-8.5g u : %-8.5g exact : %-8.5g" %
              (t[i], u[i], np.exp(t[i])))
    import matplotlib.pyplot as mp
    u_exact = np.exp(t)
    mp.plot(t, u, 'r-')
    mp.plot(t, u_exact, 'b-')
    mp.xlabel('t')
    mp.ylabel('u')
    mp.legend(['numerical', 'exact']),
    mp.title("Solution of the ODE u'=u, u(0)=1")
    mp.gcf().set_size_inches(8.5, 6.5, forward=True)
    mp.show()


if __name__ == '__main__':
    test(1, 3, 6)
