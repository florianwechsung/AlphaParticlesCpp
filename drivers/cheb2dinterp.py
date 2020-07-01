import numpy as np
from numpy.polynomial.chebyshev import chebvander2d, chebval2d


class Cheb2dInterp():

    def __init__(self, fun, n, lower, upper):
        self.chebpoints = np.asarray([np.cos((2*k-1)*np.pi/(2*n)) for k in range(1, n+1)])
        y = x = self.chebpoints
        XX, YY = np.meshgrid(x, y)
        XX = XX.flatten()
        YY = YY.flatten()
        M = chebvander2d(XX, YY, [n-1, n-1])
        rhs = np.zeros((n*n, ))
        for i in range(n*n):
            rhs[i] = fun(
                self.scalefromdefault(XX[i], lower[0], upper[0]),
                self.scalefromdefault(YY[i], lower[1], upper[1]))
        c = np.linalg.solve(M, rhs)
        self.c = c.reshape((n, n))
        self.lower = lower
        self.upper = upper
        self.fun = fun

    def eval(self, x, y):
        return chebval2d(
            self.scaletodefault(x, self.lower[0], self.upper[0]),
            self.scaletodefault(y, self.lower[1], self.upper[1]),
            self.c)

    def scaletodefault(self, p, l, u):
        return -1+2*(p-l)/(u-l)

    def scalefromdefault(self, p, l, u):
        return -0.5*(p-1)*l+0.5*u*(p+1)

    def random_error_estimate(self, k):
        x = np.random.uniform(self.lower[0], self.upper[0], size=(k, 1))
        y = np.random.uniform(self.lower[1], self.upper[1], size=(k, 1))
        err = 0.
        for i in range(k):
            err += (self.fun(x[i], y[i])-self.eval(x[i], y[i]))**2
        return np.sqrt(err/k)


if __name__ == "__main__":
    fun = lambda x, y: np.sin(x) * np.cos(y) * np.exp(x+y)
    a = [-2, -3]
    b = [+2, +4]
    n = 20
    interp = Cheb2dInterp(fun, n, a, b)
    print(interp.random_error_estimate(1000))
