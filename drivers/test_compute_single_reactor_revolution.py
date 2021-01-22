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
B = get_dommaschk_field()

y0 = np.asarray([1+epsilon/2, 1e3, 0, 1e5, 0, 0])
q = 2*1.6e-19  # gParticle charge
m = 6.64e-27  # gParticle mass (2xproton + 2xneutron mass)
gyro, mu, total_velocity, eta = pp.orbit_to_gyro_cylindrical_helper(y0, B, m, q)

omega_c = q*Btin/m  # gCyclotron angular frequency at the inboard midplane
dT = np.pi/(args.dtfrac*omega_c)  # gSize of the time step for numerical ode solver

rgood = 0.95
zgood = 0.0

rbad = 0.88
zbad = -0.02

rs = [rgood, rbad]
zs = [zgood, zbad]

import time
for i in [0, 1]:
    r = rs[i]
    z = zs[i]
    rphiz = np.asarray([r, 0, z])
    Brphiz = B.B(rphiz[0], rphiz[1], rphiz[2])
    xhat, Bxyz = pp.vecfield_cyl_to_cart(rphiz, Brphiz)
    xyz, vxyz = pp.gyro_to_orbit(xhat, mu, total_velocity, eta, Bxyz, m, q)
    rphiz, vrphiz = pp.vecfield_cart_to_cyl(xyz, vxyz)
    y0 = np.asarray([rphiz[0], vrphiz[0], rphiz[1], vrphiz[1]/rphiz[0], rphiz[2], vrphiz[2]])
    start = time.time()
    last_t, y, gyro = pp.compute_single_reactor_revolution(y0, dT, B, m, q)
    end = time.time()
    print("time for run " + str(i) + ": " + str(end-start))
    print("last_t = " + str(last_t))
    #print(y)
    #print(gyro)
  
