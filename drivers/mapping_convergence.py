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


omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

from mapping import apply_map_fullorbit
 
from cheb2dinterp import Cheb2dInterp
fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))
#lower = [+0.85, -0.15]
lower = [+0.98, -0.01] # dommaschk vals
#upper = [+1.15, +0.15]
upper = [+1.02, +0.01] # dommaschk vals

# ===
x_test = np.linspace(0.98, 1.02, num=20)
y_test = np.linspace(-0.01, 0.01, num=20)
RS, ZS = np.meshgrid(x_test, y_test)

RS_out = np.zeros_like(RS)
ZS_out = np.zeros_like(RS)
TS = np.zeros_like(RS)
for i in range(RS.shape[0]):
  for j in range(RS.shape[1]):
    res = apply_map_fullorbit(RS[i ,j], ZS[i, j], total_velocity, mu, B, m, q, dT, args.angles)
    RS_out[i, j] = res[0]
    ZS_out[i, j] = res[1]
    TS[i, j] = res[2]
  print("Progress =", (i+1)/RS.shape[0])

print(np.asarray(RS).shape)

# ===

area = 'c'
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
else:
  area = 'center'

errs = []
ns = range(1, 20, 2)
"""
for n in ns:
  np.random.seed(1)
  interp = Cheb2dInterp(fun, n, lower, upper, dim=3)
  err = interp.random_error_estimate(100)
  errs.append(err)
  print(n, err)
"""
"""
n = 7
interp = Cheb2dInterp(fun, n, lower, upper, dim=3)
c = np.asarray(interp.c)
print(c.shape)
"""

"""
errs = np.asarray(errs)
plt.semilogy(ns, errs[:, 0])
plt.semilogy(ns, errs[:, 1])
plt.semilogy(ns, errs[:, 2])
plt.ylim((1e-16, 1e-1))
plt.legend(['R', 'Z', 'T'])
#field = "antoine"
field = "dommaschk"
plt.savefig(f"errs-dtfrac-{args.dtfrac}-angles-{args.angles}-{field}-{area}.png")
"""
