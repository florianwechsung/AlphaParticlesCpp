#pragma once
#include <functional>

double bisection(std::function<double(double)>& f, double l, double fl, double r, double fr, double tol);
