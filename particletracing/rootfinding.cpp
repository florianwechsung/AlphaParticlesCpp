#include "rootfinding.hpp"
#include <stdexcept>


double bisection(std::function<double(double)>& f, double l, double fl, double r, double fr, double tol) {
    if(fl * fr > 0) {
        throw std::invalid_argument("fl * fr has to be non-positive");
    }
    double m = (l+r)/2;
    if(std::abs(fl-fr) < tol){
        return m;
    }
    double fm = f(m);
    if(fm == 0){
        return m;
    }
    if(fm*fl < 0) {
        return bisection(f, l, fl, m, fm, tol);
    } else {
        return bisection(f, m, fm, r, fr, tol);
    }
}
