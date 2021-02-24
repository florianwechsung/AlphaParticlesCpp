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
        self.fun = fun # function to interpolate
        self.n = n # number of TPS kernels in each row and column, so total n^2 kernels
        self.xmin = xmin # min x-value in domain
        self.xmax = xmax # max x-value in domain
        self.ymin = ymin # min y-value in domain
        self.ymax = ymax # max y-value in domain
        x = np.linspace(xmin, xmax, num=n)
        y = np.linspace(ymin, ymax, num=n)
        XX, YY = np.meshgrid(x,y)
        XX = XX.flatten() # x-coordinates of all n^2 points in domain
        YY = YY.flatten() # y-coordinates of all n^2 points in domain
        M = np.zeros((n*n, n*n))
        for j in range(n*n):
            for i in range(n*n):
                # populate M with TPS evaluated at each pair of points
                M[i, j] = tps([XX[i],YY[i]], [XX[j],YY[j]])
        rhs = np.zeros((n*n, dim))
        for i in range(n*n):
            rhs[i, :] = fun(XX[i], YY[i]) # function evals at all n^2 points in domain
        c = np.linalg.solve(M, rhs) # flattened coefficient matrix
        self.c = []
        for i in range(dim):
            self.c.append(c[:, i].reshape((n, n), order='F')) # unflatten coefficient matrix; was unsure about "order='F'", but it seems to work
        self.c = np.asarray(self.c)
        self.dim = dim

    def eval(self, r, z):
        x = np.linspace(self.xmin, self.xmax, num=self.n)
        y = np.linspace(self.ymin, self.ymax, num=self.n)
        result = np.zeros(self.dim) # output
        # these nested for loops: is there a faster implementation?
        for j in range(self.n):
            for i in range(self.n):
                result += self.c[:, i, j] * tps([r, z], [x[i], y[j]])
        return result
