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
Btin = 15
epsilon = 0.32
#B = get_antoine_field(Btin, epsilon=epsilon)
B = get_dommaschk_field()

# y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
# y0 = np.asarray([1+epsilon/2, 1e5, 0, 1e7, 0, 0])
q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)


omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

from mapping import apply_map_fullorbit, apply_map_gc

def apply_k_revolutions(r, z, mu, total_velocity, k, method):
  rs = []
  zs = []
  ts = []
  if method == 'full_orbit':
    t_onerev, y_onerev, gyro_rphiz = pp.compute_single_reactor_revolution(y0, dT, B, m, q)
    rphiz = np.asarray([r, 0, z])
    Brphiz = B.B(rphiz[0], rphiz[1], rphiz[2])
    xhat, Bxyz = pp.vecfield_cyl_to_cart(rphiz, Brphiz)
    eta = 0.0
    xyz, vxyz = pp.gyro_to_orbit(xhat, mu, total_velocity, eta, Bxyz, m, q)
    rphiz, vrphiz = pp.vecfield_cart_to_cyl(xyz, vxyz)
    y = np.asarray([rphiz[0], vrphiz[0], rphiz[1], vrphiz[1]/rphiz[0], rphiz[2], vrphiz[2]])
    t = 0.
    for i in range(k):
      if y[2] > np.pi:
        y[2] -= 2*np.pi
      t_onerev, y, _ = pp.compute_single_reactor_revolution(y, dT, B, m, q)
      t += t_onerev
      gyro_rphiz = pp.orbit_to_gyro_cylindrical_helper(y, B, m, q)[0]
      rs.append(gyro_rphiz[0])
      zs.append(gyro_rphiz[2])
      ts.append(t)
  else:
    if method == 'full_orbit_guiding_map':
      t = 0
      for i in range(k):
        r, z, t_onerev = apply_map_fullorbit(r, z, total_velocity, mu, B, m, q, dT, args.angles)
        t += t_onerev
        rs.append(r)
        zs.append(z)
        ts.append(t)
    elif method == 'guiding_approx_map':
      t = 0
      for i in range(k):
        r, z, t_onerev = apply_map_gc(r, z, total_velocity, mu, B, m, q, dT)
        t += t_onerev
        rs.append(r)
        zs.append(z)
        ts.append(t)
    else:
      raise NotImplementedError
  return rs, zs, ts



rotations = 100
res = apply_k_revolutions(1.0, 0., mu, total_velocity, rotations, 'full_orbit') 
res_full = apply_k_revolutions(1.0, 0., mu, total_velocity, rotations, 'full_orbit_guiding_map')
res_gc = apply_k_revolutions(1.0, 0., mu, total_velocity, rotations, 'guiding_approx_map')

res = np.asarray(res)
res_full = np.asarray(res_full)
res_gc = np.asarray(res_gc)

rotations = range(1, rotations+1)
plt.semilogy(rotations, np.abs(res-res_full)[0, :], color="b", label="R Full Orbit Map")
plt.semilogy(rotations, np.abs(res-res_full)[1, :], color="g", label="Z Full Orbit Map")
plt.semilogy(rotations, np.abs(res-res_full)[2, :], color="r", label="T Full Orbit Map")
plt.semilogy(rotations, np.abs(res-res_gc)[0, :], "--", color="b", label="R GC Approximation")
plt.semilogy(rotations, np.abs(res-res_gc)[1, :], "--", color="g", label="Z GC Approximation")
plt.semilogy(rotations, np.abs(res-res_gc)[2, :], "--", color="r", label="T GC Approximation")
plt.xlabel('Revolution')
plt.ylabel('Error')
plt.legend()
plt.show()
import IPython; IPython.embed()
