#include "coordhelpers.hpp"

#include <iostream>
tuple<Vec3d, Vec3d, Vec3d> gram_schmidt(Vec3d in1, Vec3d in2, Vec3d in3) {
  Vec3d p1 = in1/in1.norm();
  Vec3d p2 = in2 - in2.dot(p1)*p1;
  p2 *= 1/p2.norm();
  Vec3d p3 = in3 - in3.dot(p1)*p1 - in3.dot(p2)*p2;
  p3 *= 1/p3.norm();
  return std::make_tuple(p1, p2, p3);
}

pair<Vec3d, Vec3d> gyro_to_orbit(Vec3d xhat, double mu, double total_velocity, double eta, Vec3d B, double m, double q) {
  double phi = atan2(xhat[1], xhat[0]);
  Vec3d ex_rotated = Vec3d{cos(phi), sin(phi), 0};
  auto p123 = gram_schmidt(B, Vec3d{0., 0., 1.}, ex_rotated);
  auto p1 = std::get<0>(p123);
  auto p2 = std::get<2>(p123);
  auto p3 = std::get<1>(p123);
  double vperpmag = sqrt(2*mu*B.norm());
  Vec3d vpar = sqrt(total_velocity*total_velocity-vperpmag*vperpmag) * p1;
  double rg = m * vperpmag / (q * B.norm());
  Vec3d x = xhat + rg * sin(eta) * p2 + rg * cos(eta) * p3;
  Vec3d vperp = -vperpmag * cos(eta) * p2 + vperpmag * sin(eta) * p3;
  Vec3d v = vpar + vperp;
  return std::make_pair(x, v);
}

tuple<Vec3d, double, double, double> orbit_to_gyro(Vec3d x, Vec3d v, Vec3d B, double m, double q) {
  double phi = atan2(x[1], x[0]);
  Vec3d ex_rotated = Vec3d{cos(phi), sin(phi), 0};
  auto p123 = gram_schmidt(B, Vec3d{0., 0., 1.}, ex_rotated);
  auto p1 = std::get<0>(p123);
  auto p2 = std::get<2>(p123);
  auto p3 = std::get<1>(p123);
  double vperp_p2 = p2.dot(v);
  double vperp_p3 = p3.dot(v);
  double vperp_mag = sqrt(vperp_p2*vperp_p2 + vperp_p3*vperp_p3);
  double rg = m * vperp_mag / (q * B.norm());
  Vec3d s = vperp_p3 * p2 - vperp_p2 * p3;
  Vec3d xhat = x - rg * s / s.norm();
  double eta = atan2(vperp_p3, -vperp_p2);
  if(eta < 0) {
    eta += 2*M_PI;
  }
  double mu = (vperp_mag*vperp_mag)/(2*B.norm());
  double total_velocity = v.norm();
  return std::make_tuple(xhat, mu, total_velocity, eta);
}

Vec3d cart_to_cyl(Vec3d p) {
  double r = sqrt(p[0]*p[0] + p[1]*p[1]);
  double phi = atan2(p[1], p[0]);
  if(phi < 0) {
    phi += 2*M_PI;
  }
  double z = p[2];
  Vec3d rphiz = Vec3d{ r, phi, z };
  return rphiz;
}

Vec3d cyl_to_cart(Vec3d p) {
  double x = p[0] * cos(p[1]);
  double y = p[0] * sin(p[1]);
  double z = p[2];
  Vec3d xyz = Vec3d{ x, y, z };
  return xyz;
}

pair<Vec3d, Vec3d> vecfield_cart_to_cyl(Vec3d p, Vec3d fp) {
  double r = sqrt(p[0]*p[0] + p[1]*p[1]);
  double phi = atan2(p[1], p[0]);
  double z = p[2];
  double fr = cos(phi) * fp[0] + sin(phi) * fp[1];
  double fphi = -sin(phi) * fp[0] + cos(phi) * fp[1];
  double fz = fp[2];
  Vec3d rphiz = Vec3d{ r, phi, z };
  Vec3d frphiz = Vec3d{ fr, fphi, fz };
  return std::make_pair(rphiz, frphiz);
}

pair<Vec3d, Vec3d> vecfield_cyl_to_cart(Vec3d p, Vec3d fp) {
  double fx = cos(p[1]) * fp[0] - sin(p[1]) * fp[1];
  double fy = sin(p[1]) * fp[0] + cos(p[1]) * fp[1];
  double fz = fp[2];
  
  double x = p[0] * cos(p[1]);
  double y = p[0] * sin(p[1]);
  double z = p[2];
  Vec3d xyz = Vec3d{ x, y, z };
  Vec3d fxyz = Vec3d{ fx, fy, fz };
  return std::make_pair(xyz, fxyz);
}
