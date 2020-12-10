#include "particletracing.hpp"
#include "coordhelpers.hpp"
#include "rootfinding.hpp"


Vec3d particle_rhs_guiding_center(const Vec3d& y, MagneticField& B, double v, double mu, double moverq) {
  double r   = y.coeffRef(0);
  double phi = y.coeffRef(1);
  double z   = y.coeffRef(2);
  double AbsB   = B.AbsB(r, phi, z);
  double vperp2 = mu*2*AbsB;
  double vtang2 = v*v - vperp2;
  auto GradAbsB = B.GradAbsB(r, phi, z);
  auto B_ = B.B(r, phi, z);
  auto xyz_Bcart        = vecfield_cyl_to_cart(y, B_);
  auto xyz = xyz_Bcart.first;
  auto Bcart = xyz_Bcart.second;
  auto GradAbsBcart = vecfield_cyl_to_cart(y, GradAbsB).second;
  Vec3d BGB = Bcart.cross(GradAbsBcart);
  Vec3d BcrossGradAbsB_cyl = vecfield_cart_to_cyl(xyz, BGB).second;

  auto b_cdot_grad_par_b = B.b_cdot_grad_par_b(r, phi, z);
  auto b_cdot_grad_par_b_cart = vecfield_cyl_to_cart(y, b_cdot_grad_par_b).second;
  auto B_cross_b_cdot_grad_par_b_cart = Bcart.cross(b_cdot_grad_par_b_cart);
  auto B_cross_b_cdot_grad_par_b_cyl = vecfield_cart_to_cyl(xyz, B_cross_b_cdot_grad_par_b_cart).second;

  //Vec3d res = std::sqrt(vtang2) *  B_/AbsB  + (moverq/(AbsB*AbsB*AbsB)) * (vtang2 + 0.5*vperp2)*BcrossGradAbsB_cyl;
  Vec3d res = std::sqrt(vtang2) *  B_/AbsB  + (moverq/(AbsB*AbsB)) * vtang2 * B_cross_b_cdot_grad_par_b_cyl 
    + (moverq/(AbsB*AbsB*AbsB)) *0.5*vperp2*BcrossGradAbsB_cyl;
  res(1) *= 1/r;
  return res;
}

Vec6d particle_rhs(const Vec6d& y, MagneticField& B, double qoverm) {
  double r = y.coeffRef(0);
  double rdot = y.coeffRef(1);
  double phi = y.coeffRef(2);
  double phidot = y.coeffRef(3);
  double z = y.coeffRef(4);
  double zdot = y.coeffRef(5);
  Vec3d Brphiz = B.B(r, phi, z);
  double Br = Brphiz(0);
  double Bphi = Brphiz(1);
  double Bz = Brphiz(2);

  return Vec6d{
    rdot,
    qoverm * (r * phidot * Bz - zdot*Bphi) + r * phidot * phidot,
    phidot,
    (qoverm/r)*(zdot * Br - rdot*Bz)-2*rdot*phidot/r,
    zdot,
    qoverm*(rdot*Bphi-r*phidot*Br)
  };
}


pair<vector<double>, vector<vector<double>>> compute_guiding_center_simple(Vec3d& x0, double mu, double velocity, double dt, int nsteps, MagneticField& B, double m, double q) {
  auto res_y = vector<vector<double>>(nsteps+1, vector<double>(3, 0.));
  auto res_t = vector<double>(nsteps+1);
  res_y[0] = vector<double>{x0[0], x0[1], x0[2]};
  res_t[0] = 0.;
  Vec3d y = x0;
  double moverq = m/q;
  std::function<Vec3d(const Vec3d&)> rhs = [&B, &velocity, &mu, &moverq](const Vec3d& y){ return particle_rhs_guiding_center(y, B, velocity, mu, moverq);};
  for (int i = 0; i < nsteps; ++i) {
    y = rk4_step(y, dt, rhs);
    res_y[i+1] = vector<double>{y[0], y[1], y[2]};
    res_t[i+1] = res_t[i] + dt;
  }
  return std::make_pair(res_t, res_y);
}

pair<vector<double>, vector<vector<double>>> compute_guiding_center(Vec6d& y0, double dt, int nsteps, MagneticField& B, double m, double q) {
  Vec3d x0 = Vec3d{y0[0], y0[2], y0[4]};
  double r = x0[0];
  double phi = x0[1];
  double z = x0[2];
  Vec3d rphiz = Vec3d{r, phi, z};
  Vec3d v0 = Vec3d{y0[1], r*y0[3], y0[5]};
  Vec3d B0 = B.B(x0[0], x0[1], x0[2]);
  Vec3d Bcart = vecfield_cyl_to_cart(rphiz, B0).second; 
  Vec3d vcart = vecfield_cyl_to_cart(rphiz, v0).second;
  double velocity = vcart.norm();
  double AbsB0 = Bcart.norm();
  double tangential_velocity = vcart.dot(Bcart)/AbsB0;
  double mu = (velocity*velocity - tangential_velocity*tangential_velocity)/(2*AbsB0);
  return compute_guiding_center_simple(x0, mu, velocity, dt, nsteps, B, m, q);
}

tuple<vector<double>, vector<vector<double>>, vector<vector<double>>> compute_full_orbit(Vec6d& y0, double dt, int nsteps, MagneticField& B, double m, double q) {
  auto res_x = vector<vector<double>>(nsteps+1, vector<double>(3, 0.));
  auto res_v = vector<vector<double>>(nsteps+1, vector<double>(3, 0.));
  auto res_t = vector<double>(nsteps+1);
  res_x[0][0] = y0.coeffRef(0);
  res_x[0][1] = y0.coeffRef(2);
  res_x[0][2] = y0.coeffRef(4);
  res_v[0][0] = y0.coeffRef(1);
  res_v[0][1] = y0.coeffRef(0)*y0.coeffRef(3);
  res_v[0][2] = y0.coeffRef(5);
  res_t[0] = 0.;
  Vec6d y = y0;
  double qoverm = q/m;
  std::function<Vec6d(const Vec6d&)> rhs = [&B, &qoverm](const Vec6d& y){ return particle_rhs(y, B, qoverm);};
  for (int i = 0; i < nsteps; ++i) {
    y = rk4_step(y, dt, rhs);
    res_x[i+1] = vector<double>{y[0], y[2], y[4], y[1], y[0]*y[3], y[5]};
    //double energy = y[1]*y[1]+y[0]*y[0]*y[3]*y[3]+y[5]*y[5];
    //if(i % 10000 == 0)
    //  std::cout <<  "energy " <<  energy << std::endl;
    res_t[i+1] = res_t[i] + dt;
  }
  return std::make_tuple(res_t, res_x, res_v);
}


pair<double, Vec3d> compute_single_reactor_revolution_gc(Vec3d& x0, double mu, double velocity, double dt, MagneticField& B, double m, double q) {
  // computes the orbit until we complete a full reactor revolution.  we check
  // whether phi exceeds 2*pi, and once it does, we perform a simple affine
  // interpolation between the state just before and just after phi=2*pi
  Vec3d x = x0;
  Vec3d last_x = x;
  double t = 0.;
  double last_t = t;
  double moverq = m/q;
  std::function<Vec3d(const Vec3d&)> rhs = [&B, &moverq, &velocity, &mu](const Vec3d& y){ return particle_rhs_guiding_center(y, B, velocity, mu, moverq);};
  double phi = x[1];
  while(phi < 2*M_PI){
    last_x = x;
    last_t = t;
    x = rk4_step(x, dt, rhs);
    t += dt;
    phi = x[1];
  }
  std::function<double(double)> phifun = [&last_x,  &rhs](double step){ return 2*M_PI - (rk4_step(last_x, step, rhs)[1]); };
  double dt_final = bisection(phifun, 0, phifun(0), dt, phifun(dt), 1e-14);
  x = rk4_step(last_x, dt_final, rhs);
  t += dt_final;
  return std::make_pair(t, x);
}

tuple<Vec3d, double, double, double> orbit_to_gyro_cylindrical_helper(Vec6d y, MagneticField& B, double m, double q) {
  Vec3d xyz, vxyz;
  Vec3d rphiz = Vec3d {y[0], y[2], y[4] };
  std::tie(xyz, vxyz) = vecfield_cyl_to_cart(rphiz, Vec3d {y[1], y[0]*y[3], y[5]});
  Vec3d Bxyz = std::get<1>(vecfield_cyl_to_cart(rphiz, B.B(y[0], y[2], y[4])));
  Vec3d gyro_xyz;
  double mu, total_velocity, eta;
  std::tie(gyro_xyz, mu, total_velocity, eta) = orbit_to_gyro(xyz, vxyz, Bxyz, m, q);
  return std::make_tuple(cart_to_cyl(gyro_xyz), mu, total_velocity, eta);
}

tuple<double, Vec6d, Vec3d> compute_single_reactor_revolution(Vec6d& y0, double dt, MagneticField& B, double m, double q) {
  // computes the orbit until we complete a full reactor revolution.  we check
  // whether phi exceeds 2*pi, and once it does, we perform a simple affine
  // interpolation between the state just before and just after phi=2*pi
  Vec6d y = y0;
  Vec6d last_y = y;
  Vec3d gyro_location_rphiz, last_gyro_location_rphiz;
  double last_t = 0.;
  double qoverm = q/m;
  std::function<Vec6d(const Vec6d&)> rhs = [&B, &qoverm](const Vec6d& y){ return particle_rhs(y, B, qoverm);};
  double last_phi = y[2];
  double phi = y[2];
  bool use_gyro = false;
  bool timestepreduced = false;
  while(!use_gyro || phi > M_PI){
    last_y = y;
    last_t += dt;
    y = rk4_step(y, dt, rhs);
    last_phi = phi;
    if(!use_gyro && last_phi > 0.98*2*M_PI && last_phi < 0.99*2*M_PI) {
      use_gyro = true;
      last_phi -= 0.1;
    }
    if(use_gyro) {
      last_gyro_location_rphiz = gyro_location_rphiz;
      gyro_location_rphiz = std::get<0>(orbit_to_gyro_cylindrical_helper(y, B, m, q));
      phi = gyro_location_rphiz[1];
    } else {
      phi = y[2];
    }
    if(!timestepreduced && phi < last_phi) {
      y = last_y;
      last_t -= dt;
      phi = last_phi;
      dt *= 1./10000;
      timestepreduced = true;
    }
  }
  double alpha = (2*M_PI-last_phi)/(2*M_PI+phi-last_phi); // alpha=1 if phi = 2*pi, alpha = 0 if last_phi = 2*pi
  if(gyro_location_rphiz[1] < M_PI)
    gyro_location_rphiz[1] += 2*M_PI;
  if(y[2] < M_PI)
    y[2] += 2*M_PI;

  // --- last_phi ------ 2 * PI ----- phi ----
  // ---  last_y  -------------------  y  ----
  y = (1-alpha)*last_y + alpha * y;
  last_t += alpha * dt;
  gyro_location_rphiz = (1-alpha)*last_gyro_location_rphiz + alpha * gyro_location_rphiz;
  return std::make_tuple(last_t, y, gyro_location_rphiz);
}


int main() {

  double epsilon = 0.32;
  double kappa = 1.7;
  double delta = 0.33;
  double A = -0.2;
  double Btin = 1.5;

  auto B = AntoineField(epsilon, kappa, delta, A, Btin);

  Vec6d y0 = Vec6d{
    1.1600000000, 5e5, 0, 1e5, 0, 0,
  };

  double q = 2;//Particle charge, in units of e
  double m = 6.64e-27;//Particle mass
  double omega_c = q*1.6e-19*Btin/m;//Cyclotron angular frequency at the inboard midplane
  double T = 2*M_PI/omega_c;//Cyclotron period
  double M = 1e7;//Approximate number of cyclotron periods followed for the particle trajectories
  double T_particleTracing = 2000*T;//Total simulation time: 2000 cyclotron periods (just for checking trajectories at the moment)
  double dT = M_PI/(32*omega_c);//Size of the time step for numerical ode solver

  int MM = T_particleTracing/dT;//Number of time steps

  auto full_orbit = std::get<1>(compute_full_orbit(y0, dT, MM, B, m, q));
  std::cout << full_orbit[MM][0] << " " << full_orbit[MM][1] << " " << full_orbit[MM][2] << std::endl;
}
