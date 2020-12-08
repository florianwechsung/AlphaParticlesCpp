#include <vector>
using std::vector;
using std::pair;
using std::tuple;
#include <functional>
using std::function;
#include "magneticfield.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Dense>
typedef Eigen::Matrix<double, 6, 1> Vec6d;
typedef Eigen::Matrix<double, 3, 1> Vec3d;



Vec3d particle_rhs_guiding_center(const Vec3d& y, MagneticField& B, double v, double mu, double moverq);

Vec6d particle_rhs(const Vec6d& y, MagneticField& B, double qoverm);

template<class T>
T rk4_step(T& y0, double dt, function<T(const T&)>& rhs){
  T f1 = dt*rhs(y0);
  T f2 = dt*rhs((y0+0.5*f1).eval());
  T f3 = dt*rhs((y0+0.5*f2).eval());
  T f4 = dt*rhs((y0+f3).eval());
  return y0 + (f1+2*(f2+f3)+f4)/6;
}


pair<vector<double>, vector<vector<double>>>
compute_guiding_center_simple(Vec3d& x0, double mu, double velocity, double dt, int nsteps, MagneticField& B, double m, double q);

pair<vector<double>, vector<vector<double>>>
compute_guiding_center(Vec6d& y0, double dt, int nsteps, MagneticField& B, double m, double q);

tuple<vector<double>, vector<vector<double>>, vector<vector<double>>>
compute_full_orbit(Vec6d& y0, double dt, int nsteps, MagneticField& B, double m, double q);

pair<double, Vec3d>
compute_single_reactor_revolution_gc(Vec3d& x0, double mu, double velocity, double dt, MagneticField& B, double m, double q);


tuple<Vec3d, double, double, double>
orbit_to_gyro_cylindrical_helper(Vec6d y, MagneticField& B, double m, double q);

tuple<double, Vec6d, Vec3d>
compute_single_reactor_revolution(Vec6d& y0, double dt, MagneticField& B, double m, double q);
