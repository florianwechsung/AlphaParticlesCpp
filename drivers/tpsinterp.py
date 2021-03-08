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

    def random_error_estimate(self, k):
        x = np.random.uniform(self.xmin, self.xmax, size=(k,1))
        y = np.random.uniform(self.ymin, self.ymax, size=(k,1))
        err = 0.
        for i in range(k):
            err += (self.fun(x[i], y[i]) - self.eval(x[i], y[i]).reshape((self.dim, )))**2
        return np.sqrt(err/k)


class TPSLinearInterp():

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
        M = np.zeros((n*n, 3*n*n+1)) # each row has n*n kernel evals + n*n position vectors + 1
        for j in range(n*n):
            for i in range(n*n):
                # populate left n^2 x n^2 block of M with TPS evaluated at each pair of points
                M[i, j] = tps([XX[i],YY[i]], [XX[j],YY[j]])
                # populate middle n^2 x n^2 block of M with x-coords
                M[i, n*n+j] = np.linalg.norm([XX[i]-XX[j],YY[i]-YY[j]])
                # populate rightmost column with 1s
                M[i, 2*n*n] = 1
        rhs = np.zeros((n*n, dim))
        for i in range(n*n):
            rhs[i, :] = fun(XX[i], YY[i]) # function evals at all n^2 points in domain
        c = np.linalg.lstsq(M, rhs)[0] # flattened coefficient matrix
        self.c = [] # tps coefficients
        self.beta0 = [] # constants
        self.beta = [] # linear coefficients
        for i in range(dim):
            self.c.append(c[:n*n, i].reshape((n, n), order='F')) # unflatten coefficient matrix; was unsure about "order='F'", but it seems to work
            self.beta.append(c[n*n:2*n*n, i].reshape((n, n), order='F')) # unflatten linear coefficients
            self.beta0 = c[2*n*n]
        self.c = np.asarray(self.c)
        self.beta = np.asarray(self.beta)
        self.beta0 = np.asarray(self.beta0)
        self.dim = dim

    def eval(self, r, z):
        x = np.linspace(self.xmin, self.xmax, num=self.n)
        y = np.linspace(self.ymin, self.ymax, num=self.n)
        result = np.zeros(self.dim) # output
        # these nested for loops: is there a faster implementation?
        for j in range(self.n):
            for i in range(self.n):
                result += self.c[:, i, j] * tps([r, z], [x[i], y[j]]) + self.beta[:, i, j]*np.linalg.norm([r-x[i], z-y[j]])
        result += self.beta0
        return result

    def random_error_estimate(self, k):
        x = np.random.uniform(self.xmin, self.xmax, size=(k,1))
        y = np.random.uniform(self.ymin, self.ymax, size=(k,1))
        err = 0.
        for i in range(k):
            err += (self.fun(x[i], y[i]) - self.eval(x[i], y[i]).reshape((self.dim, )))**2
        return np.sqrt(err/k)

