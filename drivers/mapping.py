import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
epsilon = 0.32
kappa = 1.7
delta = 0.33
A = -0.2
# Btin = 1.5
Btin = 5

B = pp.AntoineField(epsilon, kappa, delta, A, Btin);

# y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
rphiz = np.asarray([y0[0], y0[2], y0[4]])
r, phi, z = rphiz
vrphiz = np.asarray([y0[1], r*y0[3], y0[5]])
_, vxyz = pp.vecfield_cart_to_cyl(rphiz, vrphiz)

total_velocity = np.linalg.norm(vxyz)
mu = 3.2e6


q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(32*omega_c)  # gSize of the time step for numerical ode solver


def apply_map_fullorbit(r, z, total_velocity, mu):
  etas = [p*2*np.pi for p in [0., 0.25, 0.5, 0.75]]
  # etas = [p*2*np.pi for p in [0., 0.5]]
  rphiz = np.asarray([r, 0, z])
  Brphiz = B.B(rphiz[0], rphiz[1], rphiz[2])
  xhat, Bxyz = pp.vecfield_cyl_to_cart(rphiz, Brphiz)
  rnew = 0.
  znew = 0.
  tnew = 0.
  for eta in etas:
    xyz, vxyz = pp.gyro_to_orbit(xhat, mu, total_velocity, eta, Bxyz, m, q)
    rphiz, vrphiz = pp.vecfield_cart_to_cyl(xyz, vxyz)
    y0 = np.asarray([rphiz[0], vrphiz[0], rphiz[1], vrphiz[1]/rphiz[0], rphiz[2], vrphiz[2]])
    t_onerev, y_onerev, gyro_rphiz = pp.compute_single_reactor_revolution(y0, dT, B, m, q)
    rnew += gyro_rphiz[0]
    znew += gyro_rphiz[2]
    tnew += t_onerev
  return rnew/len(etas), znew/len(etas), tnew/len(etas)

print(apply_map_fullorbit(r, z, total_velocity, mu))
from cheb2dinterp import Cheb2dInterp
fun = lambda x, y: apply_map_fullorbit(x, y, total_velocity, mu)[0]
lower = [+0.85, -0.15]
upper = [+1.15, +0.15]
errs = []
ns = range(5, 34, 2)
for n in ns:
  interp = Cheb2dInterp(fun, n, lower, upper)
  np.random.seed(1)
  err = interp.random_error_estimate(100)
  print(n, err)
  errs.append(err)

plt.semilogy(ns, errs)
plt.ylim((1e-12, 1e-4))
plt.savefig(f"errs-dtfrac-{args.dtfrac}-angles-{args.angles}.png")
import sys; sys.exit()
# r = 1.08684211
# z = -0.11842105
# print(apply_map_fullorbit(r, z, total_velocity, mu))

# import sys; sys.exit()
n = 15
rs = np.linspace(0.85, 1.15, n, endpoint=True)
zs = np.linspace(-0.15, 0.15, n, endpoint=True)
RS, ZS = np.meshgrid(rs, zs)

RS_out = np.zeros_like(RS)
ZS_out = np.zeros_like(RS)
TS = np.zeros_like(RS)
for i in range(RS.shape[0]):
  for j in range(RS.shape[1]):
    res = apply_map_fullorbit(RS[i ,j], ZS[i, j], total_velocity, mu)
    RS_out[i, j] = res[0]
    ZS_out[i, j] = res[1]
    TS[i, j] = res[2]
  print("Progress =", (i+1)/RS.shape[0])


import matplotlib.pyplot as plt
levels = 500
fig = plt.figure()
ax = plt.subplot(1, 3, 1)
cs = ax.contourf(RS, ZS, RS_out, levels=levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('R')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = plt.subplot(1, 3, 2)
cs = ax.contourf(RS, ZS, ZS_out, levels=levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('Z')
ax.set_xlabel('R')
ax.set_ylabel('Z')
ax = plt.subplot(1, 3, 3)
cs = ax.contourf(RS, ZS, TS, levels=levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('t')
ax.set_xlabel('R')
ax.set_ylabel('Z')
plt.show()

