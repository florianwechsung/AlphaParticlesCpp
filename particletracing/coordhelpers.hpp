#include <vector>
using std::vector;
using std::pair;
using std::tuple;
#define _USE_MATH_DEFINES
#include <cmath>
using std::sin;
using std::cos;
#include <Eigen/Dense>
typedef Eigen::Matrix<double, 6, 1> Vec6d;
typedef Eigen::Matrix<double, 3, 1> Vec3d;

tuple<Vec3d, Vec3d, Vec3d> gram_schmidt(Vec3d in1, Vec3d in2, Vec3d in3);
pair<Vec3d, Vec3d> gyro_to_orbit(Vec3d xhat, double mu, double total_velocity, double eta, Vec3d B, double m, double q);
tuple<Vec3d, double, double, double> orbit_to_gyro(Vec3d x, Vec3d v, Vec3d B, double m, double q);
Vec3d cart_to_cyl(Vec3d p);
Vec3d cyl_to_cart(Vec3d p);
pair<Vec3d, Vec3d> vecfield_cart_to_cyl(Vec3d p, Vec3d fp);
pair<Vec3d, Vec3d> vecfield_cyl_to_cart(Vec3d p, Vec3d fp);
