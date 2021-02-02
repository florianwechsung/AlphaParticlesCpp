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

#y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0]) # y = (r, r', p, p', z, z')
q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)
mu = 4.2e9
# good values of mu: 0, 4.8e7, 1.9e9, 2e9, 2.5e9
# bad values of mu: 1.9e8, 1.1e9, 1.5e9, 4.2e9
total_velocity = 1e5

omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

print("mu =", mu)
print("v =", total_velocity)

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
rs = np.linspace(0.87, 1.08, n, endpoint=True)
zs = np.linspace(-0.04, 0.04, n, endpoint=True)
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

#np.save('RS_out', RS_out)
#np.save('ZS_out', ZS_out)
#np.save('TS', TS)

# ======================
# Plotting
# ======================

from matplotlib import ticker

def find_min_max(ar):
  """
  Returns the max and min value of RS_out, ZS_out, and TS, ignoring extremely large 'garbage' values, indicative of an alpha exiting the confinement region

  Param: arr [2d numpy array]
  Returns: min of arr [int]
           max of arr [int]
  """
  ar_flat = ar.flatten()
  ar_flat.sort()
  i = -1
  garbage = True #ar_flat[i] is a garbage value
  while(garbage):
    if ar_flat[i] >= 1e7:
      i -= 1
    else:
      garbage = False
  return ar_flat[0], ar_flat[i]

#RS_out2 = np.load('RS_out.npy')
#ZS_out2 = np.load('ZS_out.npy')
#TS2 = np.load('TS.npy')

num_levels = 500
fig, axes = plt.subplots(3, 1, constrained_layout=True)
ax = axes[0]
#RS_out = RS_out-RS_out2
RS_out_min, RS_out_max = find_min_max(RS_out)
RS_levels = np.arange(RS_out_min, RS_out_max, (RS_out_max-RS_out_min)/num_levels)
cs = ax.contourf(RS, ZS, RS_out, levels=RS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('R')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[1]
#ZS_out = ZS_out-ZS_out2
ZS_out_min, ZS_out_max = find_min_max(ZS_out)
ZS_levels = np.arange(ZS_out_min, ZS_out_max, (ZS_out_max-ZS_out_min)/num_levels)
cs = ax.contourf(RS, ZS, ZS_out, levels=ZS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('Z')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[2]
#TS = TS-TS2
TS_min, TS_max = find_min_max(TS)
TS_levels = np.arange(TS_min, TS_max, (TS_max-TS_min)/num_levels)
cs = ax.contourf(RS, ZS, TS, levels=TS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('t')
ax.set_xlabel('R')
ax.set_ylabel('Z')

plt.show()
