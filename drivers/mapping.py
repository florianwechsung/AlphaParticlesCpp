import numpy as np
import pyparticle as pp


def apply_map_fullorbit(r, z, total_velocity, mu, B, m, q, dT, num_angles):
  etas = [p*2*np.pi/num_angles for p in range(num_angles)]
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

def apply_map_gc(r, z, total_velocity, mu, B, m, q, dT, t=None):
  # if t is passed, then we integrate for a fixed time length. Otherwise we
  # integrate for one revolution.
  rphiz = np.asarray([r, 0, z])
  if t is not None:
    nsteps = int((t/dT)/32)
    dT_gc = t/nsteps
    t_gc, y_gc = pp.compute_guiding_center_simple(rphiz, mu, total_velocity, dT_gc, nsteps, B, m, q)
    rnew = y_gc[-1][0]
    znew = y_gc[-1][2]
    tnew = t_gc[-1]
  else:
    dT_gc = dT*32
    t_gc, y_gc = pp.compute_single_reactor_revolution_gc(rphiz, mu, total_velocity, dT_gc, B, m, q)
    rnew = y_gc[0]
    znew = y_gc[2]
    tnew = t_gc
  return rnew, znew, tnew

# if __name__ == "__main__":
#   r = 1.08684211
#   z = -0.11842105
#   print(apply_map_fullorbit(r, z, total_velocity, mu))
