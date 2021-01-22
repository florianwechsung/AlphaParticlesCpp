import numpy as np
import matplotlib.pyplot as plt

RS_out = np.load('RS_out.npy')
ZS_out = np.load('ZS_out.npy')
TS = np.load('TS.npy')

print(np.amin(RS_out))

RS_flat = RS_out.flatten()
RS_flat.sort()
i = -1
second = True
while(second):
    if RS_flat[i] == 1e9:
        i -= 1
    else:
        second = False
#print(RS_flat[i-20:i+1])
print(i)

print(np.amin(RS_out))

ZS_flat = ZS_out.flatten()
ZS_flat.sort()
i = -1
second = True
while(second):
    if ZS_flat[i] == 1e9:
        i -= 1
    else:
        second = False
#print(ZS_flat[i-20:i+1])
print(i)

TS_flat = TS.flatten()
TS_flat.sort()
    
if(0):
    TS_flat = TS.flatten()
    TS_flat.sort()
    plt.plot(TS_flat)
    plt.plot(ZS_flat)
    #plt.show()
    #print(TS_flat[-20:])
    #print(TS_flat[:7])

n = 20
rs = np.linspace(0.9, 1.07, n, endpoint=True)
zs = np.linspace(-0.02, 0.02, n, endpoint=True)
RS, ZS = np.meshgrid(rs, zs)

fig, axes = plt.subplots(3, 1, constrained_layout=True)
ax = axes[0]
RS_out_min = 0.78 # np.amin(RS_out)
RS_out_max = RS_flat[-7] # from while loop above
RS_levels = np.arange(RS_out_min, RS_out_max, (RS_out_max-RS_out_min)/500)
cs = ax.contourf(RS, ZS, RS_out, RS_levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('R')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[1]
ZS_out_min = -0.0536 # np.amin(ZS_out)
ZS_out_max = ZS_flat[-7] # from while loop above
ZS_levels = np.arange(ZS_out_min, ZS_out_max, (ZS_out_max-ZS_out_min)/500)
cs = ax.contourf(RS, ZS, ZS_out, ZS_levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('Z')
ax.set_xlabel('R')
ax.set_ylabel('Z')

ax = axes[2]
TS_min = 0.0
TS_max = TS_flat[-7] # from plot of sorted, flattened TS array
TS_levels = np.arange(TS_min, TS_max, (TS_max-TS_min)/500)
cs = ax.contourf(RS, ZS, TS, TS_levels)
fig.colorbar(cs, ax=ax, shrink=0.9)
ax.title.set_text('t')
ax.set_xlabel('R')
ax.set_ylabel('Z')

plt.show()

