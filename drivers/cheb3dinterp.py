import numpy as np
from numpy.polynomial.chebyshev import chebvander3d, chebval3d


class Cheb3dInterp():

    def __init__(self, fun, n, lower, upper, dim=1):
        self.chebpoints = np.asarray([np.cos((2*k-1)*np.pi/(2*n)) for k in range(1, n+1)])
        z = y = x = self.chebpoints
        XX, YY, ZZ = np.meshgrid(x, y, z)
        XX = XX.flatten()
        YY = YY.flatten()
        ZZ = ZZ.flatten()
        M = chebvander3d(XX, YY, ZZ, [n-1, n-1, n-1])
        rhs = np.zeros((n*n*n, dim))
        for i in range(n*n*n):
          rhs[i, :] = fun(
                self.scalefromdefault(XX[i], lower[0], upper[0]),
                self.scalefromdefault(YY[i], lower[1], upper[1]),
                self.scalefromdefault(ZZ[i], lower[2], upper[2]))
        c = np.linalg.solve(M, rhs)
        self.c = []
        for i in range(dim):
          self.c.append(c[:, i].reshape((n, n, n)))
        self.lower = lower
        self.upper = upper
        self.fun = fun
        self.dim = dim

    def eval(self, x, y, z):
        return np.asarray([
          chebval3d(
            self.scaletodefault(x, self.lower[0], self.upper[0]),
            self.scaletodefault(y, self.lower[1], self.upper[1]),
            self.scaletodefault(z, self.lower[2], self.upper[2]),
            self.c[i]) for i in range(self.dim)])

    def scaletodefault(self, p, l, u):
        return -1+2*(p-l)/(u-l)

    def scalefromdefault(self, p, l, u):
        return -0.5*(p-1)*l+0.5*u*(p+1)

    def random_error_estimate(self, k):
        x = np.random.uniform(self.lower[0], self.upper[0], size=(k, 1))
        y = np.random.uniform(self.lower[1], self.upper[1], size=(k, 1))
        z = np.random.uniform(self.lower[2], self.upper[2], size=(k, 1))
        err = 0.
        for i in range(k):
            err += (self.fun(x[i], y[i], z[i])-self.eval(x[i], y[i], z[i]).reshape((self.dim, )))**2
        return np.sqrt(err/k)


if __name__ == "__main__":
    fun = lambda x, y, z: np.sin(x) * np.cos(y) * np.sin(z) * np.exp(x+y+z)
    a = [-2, -3, -2]
    b = [+2, +4, +2]
    n = 20
    interp = Cheb3dInterp(fun, n, a, b)
    print(interp.eval(1, 2, 3))
    print(interp.random_error_estimate(1000))
