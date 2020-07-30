import pyparticle as pp
def get_antoine_field(Btin, epsilon=0.32, kappa=1.7, delta=0.33, A=-0.2):
  B = pp.AntoineField(epsilon, kappa, delta, A, Btin);
  return B
  
def get_dommaschk_field(alpha= 1.98):
  B = pp.DommaschkField(alpha);
  return B
  
