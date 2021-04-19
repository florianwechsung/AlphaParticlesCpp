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
        tags = np.arange(0, n*n, 1) # array of tags for unwrapping coefficient matrix

        # only preserve XX, YY if that coordinate is in the return region
        rhs = np.zeros((n*n, dim))
        for i in range(n*n):
            rhs[i, :] = fun(XX[i], YY[i]) # function evals at all n^2 points in domain
        XX = XX[rhs[:,0] < 1e7] # x-coords only for particles in return region
        YY = YY[rhs[:,0] < 1e7] # y-coords only for particles in return region
        tags = tags[rhs[:,0] < 1e7]
        rhs = rhs[rhs[:,0] < 1e7, :] # function evals only for particles in return region
        num_coords = rhs.shape[0] # number of (x,y) positions in return region
        
        M = np.zeros((num_coords, num_coords))
        for j in range(num_coords):
            for i in range(num_coords):
                # populate M with TPS evaluated at each pair of points
                M[i, j] = tps([XX[i],YY[i]], [XX[j],YY[j]])
    
        c = np.linalg.solve(M, rhs) # flattened coefficient matrix
        #self.c = np.zeros((n, n, dim))
        self.c = np.zeros((dim, n, n))
        k = 0
        l = 0
        #for k in range(dim):
            #self.c.append(c[:, i].reshape((n, n), order='C')) # unflatten coefficient matrix
            # instead of using numpy's 'reshape' method, have to do something fancier
            # reshape c to n x n x dim tensor by looking at tags array
            # have an incrementing variable k as we iterate over the n_shape rows of c
            # have an incrementing variable l that increments only when nonzeros are added to self.c
            # if k = tags[l], repopulate c normally
            # else, populate c with zeroes
        #print(tags)
        #print(tags.shape)
        for i in range(n):
            for j in range(n):
                if l < tags.shape[0] and k == tags[l]:
                    #self.c[i, j, :] = c[l, :]
                    self.c[:, i, j] = c[l, :]
                    #print(l)
                    l += 1
                else:
                    #self.c[i, j, :] = np.zeros(dim)
                    self.c[:, i, j] = np.zeros(dim)
                k += 1
                
        #self.c = np.asarray(self.c)
        self.dim = dim

    def eval(self, r, z):
        x = np.linspace(self.xmin, self.xmax, num=self.n)
        y = np.linspace(self.ymin, self.ymax, num=self.n)
        XX, YY = np.meshgrid(x, y)
        result = np.zeros(self.dim) # output
        # these nested for loops: is there a faster implementation?
        for j in range(self.n):
            for i in range(self.n):
                result += self.c[:, i, j] * tps([r, z], [XX[i, j], YY[i, j]])
        return result

    def random_error_estimate(self, k):
        x = np.random.uniform(self.xmin, self.xmax, size=k)
        y = np.random.uniform(self.ymin, self.ymax, size=k)
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
        M = np.zeros((n*n, n*n+3)) # each row has n*n kernel evals + 2d position vector + 1
        for j in range(n*n):
            for i in range(n*n):
                # populate left n^2 x n^2 block of M with TPS evaluated at each pair of points
                M[i, j] = tps([XX[i],YY[i]], [XX[j],YY[j]])
        for i in range(n*n):
            # populate columns n^2+1 and n^2+2 with coordinates of position vectors
            M[i, n*n] = XX[i]
            M[i, n*n+1] = YY[i]
            # populate last column with 1s
            M[i, n*n+2] = 1
        
        rhs = np.zeros((n*n, dim))
        for i in range(n*n):
            rhs[i, :] = fun(XX[i], YY[i]) # function evals at all n^2 points in domain
        c = np.linalg.lstsq(M, rhs)[0] # flattened coefficient matrix
        self.c = [] # tps coefficients
        for i in range(dim):
            self.c.append(c[:n*n, i].reshape((n, n), order='C')) # unflatten tps coefficient matrix
        self.beta = c[n*n:n*n+2, :] # 2-by-dim matrix of linear coefficients
        self.beta0 = c[n*n+2, :] # 1-by-dim vector of constants
        self.c = np.asarray(self.c)
        self.beta = np.asarray(self.beta)
        self.beta0 = np.asarray(self.beta0)
        self.dim = dim

    def eval(self, r, z):
        x = np.linspace(self.xmin, self.xmax, num=self.n)
        y = np.linspace(self.ymin, self.ymax, num=self.n)
        XX, YY = np.meshgrid(x, y)
        result = np.zeros(self.dim) # output
        for j in range(self.n):
            for i in range(self.n):
                result += self.c[:, i, j] * tps([r, z], [XX[i, j], YY[i, j]])
        result += r * self.beta[0, :] + z * self.beta[1, :] + self.beta0
        return result

    def random_error_estimate(self, k):
        x = np.random.uniform(self.xmin, self.xmax, size=k)
        y = np.random.uniform(self.ymin, self.ymax, size=k)
        err = 0.
        for i in range(k):
            err += (self.fun(x[i], y[i]) - self.eval(x[i], y[i]).reshape((self.dim, )))**2
        return np.sqrt(err/k)

