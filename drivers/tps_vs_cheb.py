import numpy as np
import matplotlib.pyplot as plt

cheb = np.load("cheb_rand_error_est.npy")
tps = np.load("tps_rand_error_est.npy")
tpslin = np.load("tpslin_rand_error_est.npy")
cheb_time = np.load("cheb_runtimes.npy")
tps_time = np.load("tps_runtimes.npy")
tpslin_time = np.load("tpslin_runtimes.npy")

cheb_degrees = np.asarray(range(1, 20, 2))
tps_num = np.asarray(range(2, 102, 2))
tpslin_num = np.asarray(range(2, 102, 2))

plt.semilogy(cheb_degrees, cheb[:, 0], label="R, Cheb")
plt.semilogy(tps_num, tps[:, 0], label="R, tps")
plt.semilogy(tpslin_num, tpslin[:, 0], label="R, tps+linear")
plt.legend()
plt.show()

plt.plot(cheb_degrees, cheb_time, label="Cheb")
plt.plot(tps_num, tps_time, label="tps")
plt.plot(tpslin_num, tpslin_time, label="tps+linear")
plt.legend()
plt.show()
