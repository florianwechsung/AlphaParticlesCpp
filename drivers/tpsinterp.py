import numpy as np

def tps(x, x0):
    """
    Evaluates the thin plate spline kernel for x, x0 in R^2,
    given by k(x, x0) = ||x-x0||^2*log||x-x0||

    Params:
        x = 2-entry array / position vector in R^2
        x0 = 2-entry array / position vector in R^2
    """
    x = np.asarray(x)
    x0 = np.asarray(x0)
    r = np.linalg.norm(x-x0)
    if r == 0:
        return 0
    else:
        return r * r * np.log(r)
    

class TPSInterp():

    def __init__(self, fun, n, xmin, xmax, ymin, ymax, dim=1):
        self.fun = fun
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        x = np.linspace(xmin, xmax, num=n)
        y = np.linspace(ymin, ymax, num=n)
        XX, YY = np.meshgrid(x,y)
        XX = XX.flatten()
        YY = YY.flatten()
        M = np.zeros((n*n, n*n))
        for j in range(n*n):
            for i in range(n*n):
                M[i, j] = tps([XX[i],YY[i]], [XX[j],YY[j]])
        rhs = np.zeros((n*n, dim))
        for i in range(n*n):
            rhs[i, :] = fun(XX[i], YY[i])
        c = np.linalg.solve(M, rhs)
        self.c = []
        for i in range(dim):
            self.c.append(c[:, i].reshape((n, n), order='F'))
        self.c = np.asarray(self.c)
        self.dim = dim

    def eval(self, r, z):
        x = np.linspace(self.xmin, self.xmax, num=self.n)
        y = np.linspace(self.ymin, self.ymax, num=self.n)
        result = np.zeros(3)
        for j in range(self.n):
            for i in range(self.n):
                result += self.c[:, i, j] * tps([r, z], [x[i], y[j]])
        return result
