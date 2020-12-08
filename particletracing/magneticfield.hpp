#pragma once
#include <iostream>
//#include "blaze/Blaze.h"
//typedef blaze::StaticMatrix<double,3UL, 3UL> Mat3d;
//typedef blaze::StaticVector<double,3UL> Vec3d;

#include <Eigen/Dense>
typedef Eigen::Matrix<double, 3, 1> Vec3d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;

class MagneticField{
  public:
    virtual Vec3d B(double R, double phi, double Z)=0;

    double AbsB(double R, double phi, double Z);

    Vec3d GradAbsB(double R, double phi, double Z);

    Vec3d NormalisedB(double R, double phi, double Z);

    Vec3d b_cdot_grad_par_b(double R, double phi, double Z);

    virtual ~MagneticField() = default;

};

class AntoineField: public MagneticField {
  private:
      Vec3d D;
      double A;
      double C;
      double Rout;
      double Rin;
      double Rin2;
      double Btin2;
      double Rtop;
      double Ztop;

  public:	
    AntoineField(double epsilon, double kappa, double delta, double A_, double Btin);
    Vec3d B(double R, double phi, double Z);
    
};

class DommaschkField: public MagneticField  {
  private:
    double alpha;

  public:
    DommaschkField(double alpha_input);
  
    Vec3d B(double R, double phi, double Z);
};
