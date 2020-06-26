import numpy as np
import pyparticle as pp
from math import ceil, sin, cos
epsilon = 0.32
kappa = 1.7
delta = 0.33
A = -0.2
Btin = 5.0
Btin = 1.5

B = pp.AntoineField(epsilon, kappa, delta, A, Btin);

# y0 = np.asarray([1+epsilon/2, 5e5, 0, 1e5, 0, 0])
y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
r, phi, z = y0[0], y0[2], y0[4]
v0 = np.asarray([y0[1], r*y0[3], y0[5]])
init_velocity_cart = np.asarray([cos(phi)*v0[0]-sin(phi)*v0[1], sin(phi)*v0[0]+cos(phi)*v0[1], v0[2]])
print("init_velocity_cart %s" % (init_velocity_cart))
init_velocity = np.sqrt(np.sum(init_velocity_cart * init_velocity_cart))
print("init_velocity %s" % (init_velocity))


q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
T = 2*np.pi/omega_c  # gCyclotron period
M = 1e7  # gApproximate number of cyclotron periods followed for the particle trajectories

oneev = 1.602176634 * 1e-19
E = 0.5 * m * init_velocity**2
energy_in_ev = E/oneev
print("energy_in_ev %s" % (energy_in_ev))





T_particleTracing = 2000*T  # gTotal simulation time: 2000 cyclotron periods (just for checking trajectories at the moment)
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

tau = dT;
Delta_T = T_particleTracing/10;

dT *= 32
MM = int(MM/32)
y0shifted = y0.copy()
y0shifted[0] = y_dns_gyro[0, 0]
y0shifted[2] = y_dns_gyro[0, 1]
y0shifted[4] = y_dns_gyro[0, 2]
t_gc, y_gc = pp.compute_guiding_center(y0shifted, dT, MM, B, m, q*1.6e-19);


import matplotlib.pyplot as plt

y_gc = np.asarray(y_gc)
y_gc[:, 1] = np.mod(y_gc[:, 1], 2*np.pi)
fig, axs = plt.subplots(3, 1, figsize=(10, 5))

axs[0].plot(t_dns, y_dns[:, 0], label="R dns")
axs[0].plot(t_dns, y_dns_gyro[:, 0], label="R dns -> GC")
axs[0].plot(t_gc, y_gc[:, 0], "--", label="R GC")
axs[0].legend()

axs[1].plot(t_dns, y_dns[:, 1], label="Phi dns")
axs[1].plot(t_dns, y_dns_gyro[:, 1], "--", label="Phi dns -> GC")
axs[1].plot(t_gc, y_gc[:, 1], "--", label="Phi GC")
axs[1].legend()

axs[2].plot(t_dns, y_dns[:, 2], label="Z dns")
axs[2].plot(t_dns, y_dns_gyro[:, 2], "--",  label="Z dns -> GC")
axs[2].plot(t_gc, y_gc[:, 2], "--",  label="Z GC")
axs[2].legend()
plt.suptitle("B = %.1f" % Btin)
plt.show()
plt.savefig("comparison_B=%.1f.png" % Btin)
