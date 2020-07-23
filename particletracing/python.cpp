
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <pybind11/eigen.h>

#include "magneticfield.h"
#include "particletracing.cpp"

namespace py = pybind11;

PYBIND11_MODULE(pyparticle, m) {
  py::class_<AntoineField>(m, "AntoineField")
    .def(py::init<double, double, double, double, double>())
    .def("B", &AntoineField::B);
  py::class_<DommaschkField>(m, "DommaschkField")
    .def(py::init<double>())
    .def("B", &DommaschkField::B);
  m.def("compute_full_orbit", &compute_full_orbit);
  m.def("compute_guiding_center", &compute_guiding_center);
  m.def("compute_guiding_center_simple", &compute_guiding_center_simple);
  m.def("VSHMM", &VSHMM);
  m.def("compute_single_reactor_revolution", &compute_single_reactor_revolution);
  m.def("compute_single_reactor_revolution_gc", &compute_single_reactor_revolution_gc);
  m.def("gyro_to_orbit", &gyro_to_orbit);
  m.def("orbit_to_gyro", &orbit_to_gyro);
  m.def("vecfield_cart_to_cyl", &vecfield_cart_to_cyl);
  m.def("vecfield_cyl_to_cart", &vecfield_cyl_to_cart);
  m.def("cart_to_cyl", &cart_to_cyl);
  m.def("cyl_to_cart", &cyl_to_cart);
  m.def("orbit_to_gyro_cylindrical_helper", &orbit_to_gyro_cylindrical_helper);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
