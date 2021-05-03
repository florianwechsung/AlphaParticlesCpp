import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dtfrac", type=int, default=32)
parser.add_argument("--angles", type=int, default=2)
parser.add_argument("--mu", type=float, default=2e9)
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
mu = args.mu
print("mu:", mu)

omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

# ==================
# ODE system
# ==================
from mapping import apply_map_fullorbit

n = 20
area = 'all'
if (area == 'west'):
  rs = np.linspace(0.93, 0.97, n, endpoint=True)
  zs = np.linspace(-0.01, 0.01, n, endpoint=True)
elif (area == 'nw'):
  rs = np.linspace(0.96, 1.0, n, endpoint=True)
  zs = np.linspace(0.01, 0.02, n, endpoint=True)
elif (area == 'ne'):
  rs = np.linspace(1.01, 1.05, n, endpoint=True)
  zs = np.linspace(0.0, 0.02, n, endpoint=True)
elif (area == 'se'):
  rs = np.linspace(1.01, 1.05, n, endpoint=True)
  zs = np.linspace(-0.02, 0.0, n, endpoint=True)
elif (area == 'sw'):
  rs = np.linspace(0.96, 1.0, n, endpoint=True)
  zs = np.linspace(-0.02, -0.01, n, endpoint=True)
elif (area == 'center'):
  rs = np.linspace(0.98, 1.02, n, endpoint=True)
  zs = np.linspace(-0.01, 0.01, n, endpoint=True)
else:
  rs = np.linspace(0.93, 1.05, n, endpoint=True)
  zs = np.linspace(-0.02, 0.02, n, endpoint=True)
  area = 'all'
print(area)

# increase domain -> have some particles that don't return
rmin = 0.93
rmin = 0.87
rmax = 1.05
rmax = 1.1
zmin = -0.02
zmin = -0.03
zmax = 0.02
zmax = 0.03

rs = np.linspace(rmin, rmax, n, endpoint=True)
zs = np.linspace(zmin, zmax, n, endpoint=True)

RS, ZS = np.meshgrid(rs, zs)
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

# ==============================
# Thin plate spine interpolation
# ==============================

import tpsinterp as tps
fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))

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
elif (area == 'all'):
  lower = [0.93, -0.02]#, mu_low]
  upper = [1.05, 0.02]#, mu_up]
else:
  lower = [+0.98, -0.01] # dommaschk vals
  upper = [+1.02, +0.01] # dommaschk vals
  area = 'center'

lower = [rmin, zmin]
upper = [rmax, zmax]

num = n
#interp = tps.TPSInterp(fun, num, lower[0], upper[0], lower[1], upper[1], dim=3)
interp = tps.TPSLinearInterp(fun, num, lower[0], upper[0], lower[1], upper[1], dim=3)
RS_tps = np.zeros((n, n))
RS_error = np.zeros((n, n))
ZS_tps = np.zeros((n, n))
TS_tps = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        #[RS_tps[i, j], ZS_tps[i, j], TS_tps[i, j]] = interp.eval(rs[i], zs[j])
        [RS_tps[i, j], ZS_tps[i, j], TS_tps[i, j]] = interp.eval(RS[i ,j], ZS[i, j])
        #print(RS[i ,j])

# ===========

def rel_error(exp, act):
  return np.abs((exp - act)/exp)

from matplotlib import ticker

errs = []
errs = interp.random_error_estimate(100)
errs = np.asarray(errs)

print(errs)

from helpers import find_min_max, no_return_region

#RS_out = np.where(no_return_region(RS_out) < 1, RS_out, 0)
RS_tps = np.where(no_return_region(RS_tps) < 1, RS_tps, 1e9)
#ZS_out = np.where(no_return_region(ZS_out) < 1, ZS_out, 0)
ZS_tps = np.where(no_return_region(ZS_tps) < 1, ZS_tps, 1e9)
#TS = np.where(no_return_region(TS) < 1, TS, 0)
TS_tps = np.where(no_return_region(TS_tps) < 1, TS_tps, 1e9)

num_levels = 500
fig, axes = plt.subplots(3, 1, constrained_layout=True)

ax = axes[0]
RS_plot = rel_error(RS_out, RS_tps)
#np.save("RS_rel_mu{:e}".format(mu), RS_plot)
#print(np.sort(RS_plot.flatten())[-10:])
#RS_plot = np.abs(RS_out - RS_tps)
RS_out_min, RS_out_max = find_min_max(RS_plot, threshold=0.9)
print(find_min_max(RS_plot))
RS_levels = np.arange(RS_out_min, RS_out_max, (RS_out_max-RS_out_min)/num_levels)
cs = ax.contourf(RS, ZS, RS_plot, levels=RS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('R')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[1]
ZS_plot = np.abs(ZS_out - ZS_tps)
#np.save("ZS_abs_mu{:e}".format(mu), ZS_plot)
ZS_out_min, ZS_out_max = find_min_max(ZS_plot)
ZS_levels = np.arange(ZS_out_min, ZS_out_max, (ZS_out_max-ZS_out_min)/num_levels)
cs = ax.contourf(RS, ZS, ZS_plot, levels=ZS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('Z')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[2]
TS_plot = rel_error(TS, TS_tps)
#np.save("TS_rel_mu{:e}".format(mu), TS_plot)
#TS_plot = np.abs(TS - TS_tps)
TS_min, TS_max = find_min_max(TS_plot, threshold=0.9)
print(TS_max)
print(np.amax(TS-TS_tps))
TS_levels = np.arange(TS_min, TS_max, (TS_max-TS_min)/num_levels)
cs = ax.contourf(RS, ZS, TS_plot, levels=TS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('t')
ax.set_xlabel('R')
ax.set_ylabel('Z')

plt.show()

