#pragma once
#include <iostream>
//#include "blaze/Blaze.h"
//typedef blaze::StaticMatrix<double,3UL, 3UL> Mat3d;
//typedef blaze::StaticVector<double,3UL> Vec3d;

#include <Eigen/Dense>
typedef Eigen::Matrix<double, 3, 1> Vec3d;
typedef Eigen::Matrix<double, 3, 3> Mat3d;

class AntoineField {
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
    AntoineField(double epsilon, double kappa, double delta, double A_, double Btin) :
  A(A_)
  {

      C = 1-A;
      Rout = 1+epsilon;
      Rin = 1-epsilon;
      Rtop = 1-delta*epsilon;
      Ztop = kappa*epsilon;

      double Rout2 = Rout*Rout;
      double Rout4 = Rout2*Rout2;
      Rin2 = Rin*Rin;
      double Rin4 = Rin2*Rin2;
      double Rtop2 = Rtop*Rtop;
      double Rtop4 = Rtop2*Rtop2;
      Btin2 = Btin*Btin;

      //Mat3d Amat = Mat3d();
      Mat3d Amat = Mat3d{
        {1, Rout2, Rout4},
        {1, Rin2, Rin4},
        {1, Rtop2, Rtop4-4*Rtop2*Ztop*Ztop}
      };

      Vec3d B = Vec3d{
        -(C/8*Rout4+A*(Rout2*std::log(Rout)/2-Rout4/8)),
        -(C/8*Rin4+A*(Rin2*std::log(Rin)/2-Rin4/8)),
        -(C/8*Rtop4+A*(Rtop2*std::log(Rtop)/2-Rtop4/8)),
      };
      D = Amat.inverse()*B;
    }

    Vec3d B(double R, double phi, double Z){
      Vec3d B;
      B.coeffRef(0) = 8 * D.coeffRef(2) * R * Z;
      auto R2 = R*R;
      auto R4 = R2*R2;
      auto Z2 = Z*Z;
      auto Psi = 0.125*C*R4 + A*(0.5*R2*std::log(R)-0.125*R4) + D.coeffRef(0) + D.coeffRef(1)*R2 + D.coeffRef(2)*(R4-4*R2*Z2);
      auto F = std::sqrt(Rin2*Btin2-2*A*Psi);
      B.coeffRef(1) = F/R;
      B.coeffRef(2) = 0.5*C*R2 + A*(std::log(R)-0.5*R2+0.5) + 2*D.coeffRef(1) + 4*D.coeffRef(2)*(R2-2*Z2);
      return B;
    };

    double AbsB(double R, double phi, double Z){
      Vec3d B_ = B(R, phi, Z);
      return (Vec3d{cos(phi)*B_[0]-sin(phi)*B_[1], sin(phi)*B_[0]+cos(phi)*B_[1], B_[2]}).norm();
    }

    Vec3d GradAbsB(double R, double phi, double Z){
      double eps = 1e-4;
      double dB_dr = (AbsB(R+eps, phi, Z)-AbsB(R-eps, phi, Z))/(2*eps);
      double dB_dphi = (AbsB(R, phi+eps, Z)-AbsB(R, phi-eps, Z))/(2*eps);
      double dB_dz = (AbsB(R, phi, Z+eps)-AbsB(R, phi, Z-eps))/(2*eps);
      return Vec3d { dB_dr, dB_dphi/R, dB_dz };
    }
};
