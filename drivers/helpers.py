import numpy as np
import pyparticle as pp

def get_antoine_field(Btin, epsilon=0.32, kappa=1.7, delta=0.33, A=-0.2):
  B = pp.AntoineField(epsilon, kappa, delta, A, Btin);
  return B
  
def get_dommaschk_field(alpha= 1.98):
  B = pp.DommaschkField(alpha);
  return B

def find_min_max(ar, threshold=1e7):
  """
  Returns the max and min value of RS_out, ZS_out, and TS, ignoring extremely large 'garbage' values, indicative of an alpha exiting the confinement region

  Param: arr [2d numpy array]
         threshold [float]: values above this are 'garbage' values
  Returns: min of arr [int]
           max of arr [int]
  """
  ar_flat = ar.flatten()
  ar_flat.sort()
  i = -1
  garbage = True #ar_flat[i] is a garbage value
  while(garbage):
    if -1*i < len(ar_flat) and ar_flat[i] >= threshold:
      i -= 1
    else:
      garbage = False
  return ar_flat[0], ar_flat[i]

def no_return_region(ar):
    """
    Output of a function (r, z) --> {0, 1}, where 0 is returned if a particle with some initial position (r, z) is still in the confinement region and 1 is returned if the particle has left

    As of 3/28/21, particletracing.cpp marks a particle that has left the confinement region by assigning it large 'garbage' values for (r, rdot, phi, phidot, z, zdot)

    Param: ar [2d numpy array]
    Returns: array with 0's and 1's
    """
    return np.where(ar > 1e7, 1, 0)

