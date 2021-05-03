import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dtfrac", type=int, default=32)
parser.add_argument("--angles", type=int, default=2)
args, _ = parser.parse_known_args()

from helpers import get_antoine_field, get_dommaschk_field
Btin = 5
epsilon = 0.32
#B = get_antoine_field(Btin, epsilon=epsilon)
B = get_dommaschk_field()

# y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)
mu = 4e9
print(mu)
#mu = 1289732165.7800171
#total_velocity = 116004.31026474833

omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

from mapping import apply_map_fullorbit, apply_map_gc
 
from cheb2dinterp import Cheb2dInterp
#from cheb3dinterp import Cheb3dInterp
fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))
#fun = lambda x, y, z: np.asarray(apply_map_fullorbit(x, y, total_velocity, z, B, m, q, dT, args.angles))
#mu_low = 1.28e9
#mu_up = 1.3e9
#v_low = 1.15e4
#v_up = 1.17e4

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

#param3_low = v_low
#param3_up = v_up
#lower += [param3_low]
#upper += [param3_up]

errs = []
runtimes = []
import time
ns = range(2, 52, 2)
for n in ns:
  np.random.seed(1)
  #interp = Cheb3dInterp(fun, n, lower, upper, dim=3)
  start_t = time.time()
  interp = Cheb2dInterp(fun, n, lower, upper, dim=3)
  err = interp.random_error_estimate(100)
  runtimes += [time.time() - start_t]
  errs.append(err)
  print("="*20)
  print(n, err)

"""
fun = lambda x, y: np.asarray(apply_map_gc(x, y, total_velocity, mu, B, m, q, dT))
errs_gc = []
for n in ns:
  np.random.seed(1)
  interp = Cheb2dInterp(fun, n, lower, upper, dim=3)
  err = interp.random_error_estimate(100)
  errs_gc.append(err)
  print(n, err)
"""

errs = np.asarray(errs)
#np.save("cheb_rand_error_est", errs)
#np.save("cheb_runtimes", runtimes)
#errs_gc = np.asarray(errs_gc)
np.save("cheb_rand_error_est_mu2e9", errs)
plt.semilogy(ns, errs[:, 0], label="Full Orbit R")
plt.semilogy(ns, errs[:, 1], label="Full Orbit Z")
plt.semilogy(ns, errs[:, 2], label="Full Orbit T")
"""
plt.semilogy(ns, errs_gc[:, 0], label="GC R")
plt.semilogy(ns, errs_gc[:, 1], label="GC Z")
plt.semilogy(ns, errs_gc[:, 2], label="GC T")
"""
plt.legend()
#plt.ylim((1e-16, 1e-1))
plt.show()
#plt.savefig(f"errs-dtfrac-{args.dtfrac}-angles-{args.angles}.png")
#import IPython; IPython.embed()
