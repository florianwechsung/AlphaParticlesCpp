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
mu = 4e9
num = 40
print(mu)


omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

from mapping import apply_map_fullorbit, apply_map_gc
import tpsinterp as tps

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
    elif method == 'tps':
      fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))
      rmin = 0.80
      rmax = 1.20
      zmin = -0.10
      zmax = 0.10
      lower = [rmin, zmin]
      upper = [rmax, zmax]
      interp = tps.TPSInterp(fun, num, lower[0], upper[0], lower[1], upper[1], dim=3)
      t = 0
      for i in range(k):
        r, z, t_onerev = interp.eval(r, z)
        t += t_onerev
        rs.append(r)
        zs.append(z)
        ts.append(t)
    elif method == 'tps_linear':
      fun = lambda x, y: np.asarray(apply_map_fullorbit(x, y, total_velocity, mu, B, m, q, dT, args.angles))
      rmin = 0.80
      rmax = 1.20
      zmin = -0.10
      zmax = 0.10
      lower = [rmin, zmin]
      upper = [rmax, zmax]
      interp = tps.TPSLinearInterp(fun, num, lower[0], upper[0], lower[1], upper[1], dim=3)
      t = 0
      for i in range(k):
        r, z, t_onerev = interp.eval(r, z)
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
res_tps = apply_k_revolutions(1.0, 0., mu, total_velocity, rotations, 'tps')
res_tps_lin = apply_k_revolutions(1.0, 0., mu, total_velocity, rotations, 'tps_linear')

res = np.asarray(res)
res_full = np.asarray(res_full)
res_gc = np.asarray(res_gc)
res_tps = np.asarray(res_tps)
res_tps_lin = np.asarray(res_tps_lin)

rotations = range(1, rotations+1)
plotr = False
plotz = False
plott = True

if plotr:
  plt.semilogy(rotations, np.abs(res-res_full)[0, :], color="b", label="R Full Orbit Map")
  plt.semilogy(rotations, np.abs(res-res_gc)[0, :], "--", color="b", label="R GC Approximation")
  plt.semilogy(rotations, np.abs(res-res_tps)[0, :], color="k", label="R TPS")
  plt.semilogy(rotations, np.abs(res-res_tps_lin)[0, :], "--", color="k", label="R TPS + Linear")

if plotz:
  plt.semilogy(rotations, np.abs(res-res_full)[1, :], color="r", label="Z Full Orbit Map")
  plt.semilogy(rotations, np.abs(res-res_gc)[1, :], "--", color="r", label="Z GC Approximation")
  plt.semilogy(rotations, np.abs(res-res_tps)[1, :], color="g", label="Z TPS")
  plt.semilogy(rotations, np.abs(res-res_tps_lin)[1, :], "--", color="g", label="Z TPS + Linear")

if plott:
  plt.semilogy(rotations, np.abs(res-res_full)[2, :], color="c", label="T Full Orbit Map")
  plt.semilogy(rotations, np.abs(res-res_gc)[2, :], "--", color="c", label="T GC Approximation")
  plt.semilogy(rotations, np.abs(res-res_tps)[2, :], color="m", label="T TPS")
  plt.semilogy(rotations, np.abs(res-res_tps_lin)[2, :], "--", color="m", label="T TPS + Linear")
  
plt.xlabel('Revolution')
plt.ylabel('Error')
plt.legend()
plt.show()
#import IPython; IPython.embed()
