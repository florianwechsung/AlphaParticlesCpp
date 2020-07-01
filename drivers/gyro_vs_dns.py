import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
from helpers import get_antoine_field

q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
Btin = 1.5
epsilon = 0.32
B = get_antoine_field(Btin, epsilon=epsilon)
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)


omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
T = 2*np.pi/omega_c  # gCyclotron period
M = 1e7  # gApproximate number of cyclotron periods followed for the particle trajectories

oneev = 1.602176634 * 1e-19
E = 0.5 * m * total_velocity**2
energy_in_ev = E/oneev
print("energy_in_ev %s" % (energy_in_ev))

T_particleTracing = 0.0002
dT = np.pi/(32*omega_c)  # gSize of the time step for numerical ode solver

MM = int(T_particleTracing/dT)  # gNumber of time steps

t_dns, y_dns, v_dns = pp.compute_full_orbit(y0, dT, MM, B, m, q);
y_dns = np.asarray(y_dns)
v_dns = np.asarray(v_dns)
y_dns[:, 1] = np.mod(y_dns[:, 1], 2*np.pi)
y_dns_gyro = np.zeros_like(y_dns)
for i in range(y_dns.shape[0]):
  rphiz = y_dns[i, :]
  vrphiz = v_dns[i, :]
  Brphiz = B.B(rphiz[0], rphiz[1], rphiz[2])
  xyz, Bxyz = pp.vecfield_cyl_to_cart(rphiz, Brphiz)
  _, vxyz = pp.vecfield_cyl_to_cart(rphiz, vrphiz)
  gyro_xyz, _, _, _ = pp.orbit_to_gyro(xyz, vxyz, Bxyz, m, q)
  y_dns_gyro[i, :] = pp.cart_to_cyl(gyro_xyz)

t_onerev, y_onerev, gyro_rphiz = pp.compute_single_reactor_revolution(y0, dT, B, m, q)

dT_gc = dT*32
MM_gc = int(T_particleTracing/dT_gc)
t_gc, y_gc = pp.compute_guiding_center_simple(np.asarray([y0[0], y0[2], y0[4]]), mu, total_velocity, dT_gc, MM_gc, B, m, q)

import matplotlib.pyplot as plt

y_gc = np.asarray(y_gc)
y_gc[:, 1] = np.mod(y_gc[:, 1], 2*np.pi)
fig, axs = plt.subplots(3, 1, figsize=(10, 5))

axs[0].plot(t_dns, y_dns[:, 0], label="R dns")
axs[0].plot(t_dns, y_dns_gyro[:, 0], label="R dns -> GC")
axs[0].plot(t_gc, y_gc[:, 0], "--", label="R GC")
axs[0].scatter([0., t_onerev], [y0[0], y_onerev[0]], label="R onerev")
axs[0].scatter([t_onerev], [gyro_rphiz[0]], label="R onerev")
axs[0].legend()

axs[1].plot(t_dns, y_dns[:, 1], label="Phi dns")
axs[1].plot(t_dns, y_dns_gyro[:, 1], "--", label="Phi dns -> GC")
axs[1].plot(t_gc, y_gc[:, 1], "--", label="Phi GC")
axs[1].scatter([0, t_onerev], [y0[2], y_onerev[2]], label="Phi onerev")
axs[1].scatter([t_onerev], [gyro_rphiz[1]], label="Phi onerev")
axs[1].legend()

axs[2].plot(t_dns, y_dns[:, 2], label="Z dns")
axs[2].plot(t_dns, y_dns_gyro[:, 2], "--",  label="Z dns -> GC")
axs[2].plot(t_gc, y_gc[:, 2], "--",  label="Z GC")
axs[2].scatter([0, t_onerev], [y0[4], y_onerev[4]], label="Z onerev")
axs[2].scatter([t_onerev], [gyro_rphiz[2]], label="Z onerev")
axs[2].legend()
plt.suptitle("B = %.1f" % Btin)
plt.show()
plt.savefig("comparison_B=%.1f.png" % Btin)
