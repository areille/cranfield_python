import numpy as np
import sys


class Euler:

    def __init__(self, f, U0, T, n):
        """initialise the data."""
        self.f, self.U0, self.T, self.n = f, U0, T, n
        self.dt = T / float(n)
        self.u = np.zeros(n + 1)
        self.t = np.zeros(n + 1)

    def solve(self):
        """Compute solution for 0 <= t <= T."""
        self.u[0] = float(self.U0)
        self.t[0] = float(0)
        for k in range(int(self.n)):
            self.k = k
            self.t[k + 1] = self.t[k] + self.dt
            self.u[k + 1] = self.advance()
        return self.u, self.t

    def advance(self):
        """Advance the solution one time step."""
        return self.u[self.k] + self.dt * f(self.u[self.k], self.t[self.k])


def f(u, t):
    return u


def test(U0, T, n):

    # U0 = float(sys.argv[1])
    # T = float(sys.argv[2])
    # n = float(sys.argv[3])

    e = Euler(f, U0, T, n)
    u, t = e.solve()

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
    test(1, 3, 30)
