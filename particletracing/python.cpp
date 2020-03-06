
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <pybind11/eigen.h>

#include "antoinefield.h"
#include "particletracing.cpp"

int add(int i, int j) {
  return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(pyparticle, m) {
  m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
  py::class_<AntoineField>(m, "AntoineField")
    .def(py::init<double, double, double, double, double>());
  m.def("compute_full_orbit", &compute_full_orbit);
  m.def("compute_guiding_center", &compute_guiding_center);
  m.def("VSHMM", &VSHMM);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
