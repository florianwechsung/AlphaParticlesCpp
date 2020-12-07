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

mu_low = 1.28e9
mu_up = 1.3e9

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
  #mus = mp.linspace(mu_low, mu_up, n, endpoint=True)
  area = 'all'
print(area)
rs = np.linspace(0.9, 1.07, n, endpoint=True)
rz = np.linspace(-0.03, 0.03, n, endpoint=True)
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

# ================================
"""
from cheb2dinterp import Cheb2dInterp
from cheb3dinterp import Cheb3dInterp
fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))
#fun = lambda x, y, z: np.asarray(apply_map_fullorbit(x, y, total_velocity, z, B, m, q, dT, args.angles))

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

n = 5
interp = Cheb2dInterp(fun, n, lower, upper, dim=3)
[RS_cheb, ZS_cheb, TS_cheb] = interp.eval(RS, ZS)
#c = np.asarray(interp.c)
#print(c.shape)
"""
# ==============================

levels = 500
#fig = plt.figure()
#ax = plt.subplot(1, 3, 1)
fig, axes = plt.subplots(3, 1, constrained_layout=True)
ax = axes[0]
#cs = ax.contourf(RS, ZS, RS_out-RS_cheb, levels=levels)
cs = ax.contourf(RS, ZS, RS_out, levels=levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('R')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[1]
#cs = ax.contourf(RS, ZS, ZS_out-ZS_cheb, levels=levels)
cs = ax.contourf(RS, ZS, ZS_out, levels=levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('Z')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[2]
#cs = ax.contourf(RS, ZS, TS-TS_cheb, levels=levels)
cs = ax.contourf(RS, ZS, TS, levels=levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('t')
ax.set_xlabel('R')
ax.set_ylabel('Z')

#fig.set_constrained_layout_pads(w_pad=2/72, h_pad=2/72, hspace=0.2, wspace=0.5)
plt.show()

#plt.savefig(f"/home/paco/Documents/sem7/stellarator/magneticmoment/color/{area}.png")
