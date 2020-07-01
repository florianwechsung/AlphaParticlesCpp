import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dtfrac", type=int, default=32)
parser.add_argument("--angles", type=int, default=2)
args, _ = parser.parse_known_args()

from helpers import get_antoine_field
Btin = 5
epsilon = 0.32
B = get_antoine_field(Btin, epsilon=epsilon)

# y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)


omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

from mapping import apply_map_fullorbit
 
from cheb2dinterp import Cheb2dInterp
fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))
lower = [+0.85, -0.15]
upper = [+1.15, +0.15]
errs = []
ns = range(1, 20, 2)
for n in ns:
  np.random.seed(1)
  interp = Cheb2dInterp(fun, n, lower, upper, dim=3)
  err = interp.random_error_estimate(100)
  errs.append(err)
  print(n, err)

errs = np.asarray(errs)
plt.semilogy(ns, errs[:, 0])
plt.semilogy(ns, errs[:, 1])
plt.semilogy(ns, errs[:, 2])
plt.ylim((1e-16, 1e-1))
plt.savefig(f"errs-dtfrac-{args.dtfrac}-angles-{args.angles}.png")
