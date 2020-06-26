#include <vector>
using std::vector;
using std::pair;
using std::tuple;
#include <functional>
using std::function;
#include "antoinefield.h"
#include "coordhelpers.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Dense>
typedef Eigen::Matrix<double, 6, 1> Vec6d;
typedef Eigen::Matrix<double, 3, 1> Vec3d;

double kepler_inverse(double x) {
  double x1 = x;
  if(x>0.5)
    double x1 = 1-x;
  double s = 2*M_PI*x1;
  s = std::pow(6*s, 1./3);
  double s2 = s*s;
  double out = s * (1.
                 + s2*(1./60
                     + s2*(1./1400
                         + s2*(1./25200
                             + s2*(43./17248000
                                 + s2*(1213./7207200000
                                     + s2*151439./12713500800000))))));
  while(std::abs(out-sin(out)-2*M_PI*x1) > 1e-13)
    out = out - (out - sin(out)-2*M_PI*x1)/(1-cos(out));
  out = out/(2*M_PI);
  if(x>0.5)
      out = 1-out;
  return out;
}

double cosine_K(double t) {
  return 1. + std::cos(2*M_PI*(t-0.5));
}


Vec3d particle_rhs_guiding_center(const Vec3d& y, AntoineField& B, double v, double mu, double moverq) {
  double r   = y.coeffRef(0);
  double phi = y.coeffRef(1);
  double z   = y.coeffRef(2);
  double AbsB   = B.AbsB(r, phi, z);
  double vperp2 = mu*2*AbsB;
  double vtang2 = v*v - vperp2;
  auto GradAbsB = B.GradAbsB(r, phi, z);
  auto B_ = B.B(r, phi, z);
  Vec3d Bcart        = Vec3d{cos(phi)*B_[0]-sin(phi)*B_[1], sin(phi)*B_[0]+cos(phi)*B_[1], B_[2]};
  Vec3d GradAbsBcart = Vec3d{cos(phi)*GradAbsB[0]-sin(phi)*GradAbsB[1], sin(phi)*GradAbsB[0]+cos(phi)*GradAbsB[1], GradAbsB[2]};
  Vec3d BGB = Bcart.cross(GradAbsBcart);
  Vec3d BcrossGradAbsB_cyl = Vec3d{cos(phi)*BGB(0)+sin(phi)*BGB(1),-sin(phi)*BGB(0)+cos(phi)*BGB(1), BGB(2)};
  Vec3d res = std::sqrt(vtang2) *  B_/AbsB  + (moverq/(AbsB*AbsB*AbsB)) * (vtang2 + 0.5*vperp2)*BcrossGradAbsB_cyl;
  res(1) *= 1/r;
  return res;
}

Vec6d particle_rhs_slow(const Vec6d& y, AntoineField& B) {
  auto Brphiz = B.B(y.coeffRef(0), y.coeffRef(2), y.coeffRef(4));
  return Vec6d{
    y.coeffRef(1),
    y.coeffRef(0)*y.coeffRef(3)*y.coeffRef(3),
    y.coeffRef(3),
    -2*y.coeffRef(1)/y.coeffRef(0)*y.coeffRef(3),
    y.coeffRef(5),
    0,
  };
}

Vec6d particle_rhs(const Vec6d& y, AntoineField& B, double qoverm) {
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

template<class T>
T rk4_step(T& y0, double dt, function<T(const T&)>& rhs){
  T f1 = dt*rhs(y0);
  T f2 = dt*rhs((y0+0.5*f1).eval());
  T f3 = dt*rhs((y0+0.5*f2).eval());
  T f4 = dt*rhs((y0+f3).eval());
  return y0 + (f1+2*(f2+f3)+f4)/6;
}


pair<vector<double>, vector<vector<double>>> compute_guiding_center(Vec6d& y0, double dt, int nsteps, AntoineField& B, double m, double q) {
  Vec3d x0 = Vec3d{y0[0], y0[2], y0[4]};
  double r = x0[0];
  double phi = x0[1];
  double z = x0[2];
  Vec3d v0 = Vec3d{y0[1], r*y0[3], y0[5]};
  Vec3d B0 = B.B(x0[0], x0[1], x0[2]);
  Vec3d Bcart = Vec3d{cos(phi)*B0[0]-sin(phi)*B0[1], sin(phi)*B0[0]+cos(phi)*B0[1], B0[2]};
  Vec3d vcart = Vec3d{cos(phi)*v0[0]-sin(phi)*v0[1], sin(phi)*v0[0]+cos(phi)*v0[1], v0[2]};
  double velocity = vcart.norm();
  double AbsB0 = Bcart.norm();
  double tangential_velocity = vcart.dot(Bcart)/AbsB0;
  double mu = (velocity*velocity - tangential_velocity*tangential_velocity)/(2*AbsB0);
  std::cout << "mu=" << mu << ", velocity=" << velocity << std::endl;
  auto res_y = vector<vector<double>>(nsteps+1, vector<double>(3, 0.));
  auto res_t = vector<double>(nsteps+1);
  res_y[0][0] = x0.coeffRef(0);
  res_y[0][1] = x0.coeffRef(1);
  res_y[0][2] = x0.coeffRef(2);
  res_t[0] = 0.;
  Vec3d y = x0;
  double moverq = m/q;
  std::function<Vec3d(const Vec3d&)> rhs = [&B, &velocity, &mu, &moverq](const Vec3d& y){ return particle_rhs_guiding_center(y, B, velocity, mu, moverq);};
  for (int i = 0; i < nsteps; ++i) {
    y = rk4_step(y, dt, rhs);
    res_y[i+1][0] = y.coeffRef(0);
    res_y[i+1][1] = y.coeffRef(1);
    res_y[i+1][2] = y.coeffRef(2);
    res_t[i+1] = res_t[i] + dt;
  }
  return std::make_pair(res_t, res_y);
}

tuple<vector<double>, vector<vector<double>>, vector<vector<double>>> compute_full_orbit(Vec6d& y0, double dt, int nsteps, AntoineField& B, double m, double q) {
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
    res_x[i+1][0] = y.coeffRef(0);
    res_x[i+1][1] = y.coeffRef(2);
    res_x[i+1][2] = y.coeffRef(4);
    res_v[i+1][0] = y.coeffRef(1);
    res_v[i+1][1] = y.coeffRef(0)*y.coeffRef(3);
    res_v[i+1][2] = y.coeffRef(5);
    //double energy = y[1]*y[1]+y[0]*y[0]*y[3]*y[3]+y[5]*y[5];
    //if(i % 10000 == 0)
    //  std::cout <<  "energy " <<  energy << std::endl;
    res_t[i+1] = res_t[i] + dt;
  }
  return std::make_tuple(res_t, res_x, res_v);
}

pair<vector<double>, vector<vector<double>>> VSHMM(Vec6d& y0, double alpha, double Delta_T, double delta_t, int niter, AntoineField& B, double m, double q) {
  auto res_y = vector<vector<double>>();
  auto res_t = vector<double>();
  Vec6d y = y0;
  double t = 0;
  res_y.push_back(vector<double> {y.coeffRef(0), y.coeffRef(2), y.coeffRef(4)});
  res_t.push_back(0.);
  double qoverm = q/m;
  std::function<Vec6d(const Vec6d&)> rhs = [&B, &qoverm](const Vec6d& y){ return particle_rhs(y, B, qoverm);};
  std::function<Vec6d(const Vec6d&)> rhs_slow = [&B](const Vec6d& y){ return particle_rhs_slow(y, B);};
  for (int i = 0; i < niter; ++i) {
    double tlocal = 0.;
    while(tlocal < Delta_T) {
      double tstep = std::min(delta_t, Delta_T-tlocal);
      y = rk4_step(y, tstep, rhs);
      tlocal = tlocal + tstep;
      t = t + tstep;
      res_y.push_back(vector<double> {y.coeffRef(0), y.coeffRef(2), y.coeffRef(4)});
      res_t.push_back(t);

      double h_t = alpha * delta_t * cosine_K(kepler_inverse(fmod(t, Delta_T)/Delta_T));
      tstep = std::min(h_t, Delta_T-tlocal);
      y = rk4_step(y, tstep, rhs_slow);
      tlocal = tlocal + tstep;
      t = t + tstep;
      res_y.push_back(vector<double> {y.coeffRef(0), y.coeffRef(2), y.coeffRef(4)});
      res_t.push_back(t);
    }
  }
  return std::make_pair(res_t, res_y);
}


pair<double, Vec6d> compute_single_reactor_revolution_gc(Vec6d& y0, double dt, AntoineField& B, double m, double q) {
  // computes the orbit until we complete a full reactor revolution.  we check
  // whether phi exceeds 2*pi, and once it does, we perform a simple affine
  // interpolation between the state just before and just after phi=2*pi
  Vec6d y = y0;
  Vec6d last_y = y;
  double last_t = 0.;
  double qoverm = q/m;
  std::function<Vec6d(const Vec6d&)> rhs = [&B, &qoverm](const Vec6d& y){ return particle_rhs(y, B, qoverm);};
  double last_phi = y[2];
  double phi = y[2];
  bool use_gyro = false;
  while(y[2] < 2*M_PI){
    last_y = y;
    last_t += dt;
    y = rk4_step(y, dt, rhs);
    last_phi = phi;
    if(!use_gyro && last_phi > 0.98*2*M_PI && last_phi < 0.99*2*M_PI) {
      use_gyro = true;
    }
    if(use_gyro) {
      Vec3d xyz, vxyz;
      double r = sqrt(y[0]*y[0]+y[2]*y[2]);
      std::tie(xyz, vxyz) = vecfield_cyl_to_cart(Vec3d {y[0], y[2], y[4] }, Vec3d {y[1], r*y[3], y[5]});
      Vec3d gyro_location = std::get<0>(orbit_to_gyro(xyz, vxyz, B.B(y[0], y[2], y[4]), m, q));
      phi = gyro_location[1];
    } else {
      phi = y[2];
    }
  }
  double alpha = (2*M_PI-last_phi)/(phi-last_phi); // alpha=1 if phi = 2*pi, alpha = 0 if last_phi = 2*pi
  // --- last_phi ------ 2 * PI ----- phi ----
  // ---  last_y  -------------------  y  ----
  y = (1-alpha)*last_y + alpha * y;
  last_t += alpha * dt;
  return std::make_pair(last_t, y);
}
pair<double, Vec6d> compute_single_reactor_revolution(Vec6d& y0, double dt, AntoineField& B, double m, double q) {
  // computes the orbit until we complete a full reactor revolution.  we check
  // whether phi exceeds 2*pi, and once it does, we perform a simple affine
  // interpolation between the state just before and just after phi=2*pi
  Vec6d y = y0;
  Vec6d last_y = y;
  double last_t = 0.;
  double qoverm = q/m;
  std::function<Vec6d(const Vec6d&)> rhs = [&B, &qoverm](const Vec6d& y){ return particle_rhs(y, B, qoverm);};
  double last_phi = y[2];
  double phi = y[2];
  bool use_gyro = false;
  while(y[2] < 2*M_PI){
    last_y = y;
    last_t += dt;
    y = rk4_step(y, dt, rhs);
    last_phi = phi;
    if(!use_gyro && last_phi > 0.9*2*M_PI && last_phi < 0.95*2*M_PI) {
      use_gyro = true;
    }
    if(use_gyro) {
      Vec3d xyz, vxyz;
      double r = sqrt(y[0]*y[0]+y[2]*y[2]);
      std::tie(xyz, vxyz) = vecfield_cyl_to_cart(Vec3d {y[0], y[2], y[4] }, Vec3d {y[1], r*y[3], y[5]});
      Vec3d gyro_location = std::get<0>(orbit_to_gyro(xyz, vxyz, B.B(y[0], y[2], y[4]), m, q));
      phi = gyro_location[1];
    } else {
      phi = y[2];
    }
  }
  double alpha = (2*M_PI-last_phi)/(phi-last_phi); // alpha=1 if phi = 2*pi, alpha = 0 if last_phi = 2*pi
  // --- last_phi ------ 2 * PI ----- phi ----
  // ---  last_y  -------------------  y  ----
  y = (1-alpha)*last_y + alpha * y;
  last_t += alpha * dt;
  return std::make_pair(last_t, y);
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
