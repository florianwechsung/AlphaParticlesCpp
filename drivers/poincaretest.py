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
B = get_antoine_field(Btin, epsilon=epsilon)
B = get_dommaschk_field()

from poincareplot import compute_field_lines

nperiods = 200
spp = 100
# rphiz, xyz = compute_field_lines(
#     B, nperiods=nperiods, batch_size=8, magnetic_axis_radius=1.056,
#     max_thickness=0.2, delta=0.01, steps_per_period=spp
# )
rphiz, xyz = compute_field_lines(
    B, nperiods=nperiods, batch_size=4, magnetic_axis_radius=1.0,
    max_thickness=0.2, delta=0.01, steps_per_period=spp
)
nparticles = rphiz.shape[0]


plt.figure()
for i in range(nparticles):
    plt.scatter(rphiz[i, range(0, nperiods*spp, spp), 0], rphiz[i, range(0, nperiods*spp, spp), 2], s=0.1)
plt.show()
import mayavi.mlab as mlab
# mlab.options.offscreen = True
counter = 0
for i in range(0, nparticles, nparticles//5):
    mlab.plot3d(xyz[i, :, 0], xyz[i, :, 1], xyz[i, :, 2], tube_radius=0.005)
    counter += 1
mlab.view(azimuth=0, elevation=0)
mlab.show()
