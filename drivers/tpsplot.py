import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from helpers import find_min_max

n = 100
rmin = 0.87
rmax = 1.1
zmin = -0.03
zmax = 0.03

rs = np.linspace(rmin, rmax, n, endpoint=True)
zs = np.linspace(zmin, zmax, n, endpoint=True)

RS, ZS = np.meshgrid(rs, zs)

mu = '2.500000e+09'

RS_plot = np.load('RS_left_rel_mu' + mu + '.npy')
ZS_plot = np.load('ZS_left_abs_mu' + mu + '.npy')
TS_plot = np.load('TS_left_rel_mu' + mu + '.npy')
num_levels = 500
fig, axes = plt.subplots(3, 1, constrained_layout=True)

ax = axes[0]
RS_out_min, RS_out_max = find_min_max(RS_plot, threshold=0.9)
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
TS_min, TS_max = find_min_max(TS_plot, threshold=0.9)
TS_levels = np.arange(TS_min, TS_max, (TS_max-TS_min)/num_levels)
cs = ax.contourf(RS, ZS, TS_plot, levels=TS_levels)
cb = fig.colorbar(cs, ax=ax, shrink=0.9)
tick_locator = ticker.MaxNLocator(nbins=5)
cb.locator = tick_locator
cb.update_ticks()
ax.title.set_text('t')
ax.set_xlabel('R')
ax.set_ylabel('Z')

fig.savefig('muleft' + mu + '.png')
