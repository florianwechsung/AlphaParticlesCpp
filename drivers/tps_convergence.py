import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dtfrac", type=int, default=32)
parser.add_argument("--angles", type=int, default=2)
args, _ = parser.parse_known_args()

from helpers import get_dommaschk_field
Btin = 5
epsilon = 0.32
B = get_dommaschk_field()

# y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)
#print(mu)

omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

from mapping import apply_map_fullorbit

from tpsinterp import TPSInterp, TPSLinearInterp
fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))

area = 'all'
if (area == 'west'):
  lower = [0.93, -0.01]
  upper = [0.97, 0.01]
elif (area == 'nw'):
  lower = [0.96, 0.01]
  upper = [1.0, 0.02]
elif (area == 'ne'):
  lower = [1.01, 0.0]
  upper = [1.05, 0.02]
elif (area == 'se'):
  lower = [1.01, -0.02]
  upper = [1.05, 0.0]
elif (area == 'sw'):
  lower = [0.96, -0.02]
  upper = [1.0, -0.01]
elif (area == 'center'):
  lower = [+0.98, -0.01]
  upper = [+1.02, +0.01]
else:
  lower = [0.93, -0.02]
  upper = [1.05, 0.02]
  area = 'all'

errs = []
runtimes = []
import time
ns = range(2, 52, 2)
for n in ns:
    np.random.seed(1)
    start_t = time.time()
    #interp = TPSInterp(fun, n, lower[0], upper[0], lower[1], upper[1], dim=3)
    interp = TPSLinearInterp(fun, n, lower[0], upper[0], lower[1], upper[1], dim=3)
    err = interp.random_error_estimate(100)
    runtimes += [time.time() - start_t]
    errs.append(err)
    print(n, err)

errs = np.asarray(errs)
#np.save("tps_rand_error_est", errs)
#np.save("tps_runtimes", runtimes)
np.save("tpslin_rand_error_est", errs)
np.save("tpslin_runtimes", runtimes)
plt.semilogy(ns, errs[:, 0], label="R")
plt.semilogy(ns, errs[:, 1], label="Z")
plt.semilogy(ns, errs[:, 2], label="T")
plt.legend()
plt.show()
    
