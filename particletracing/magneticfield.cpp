#include "magneticfield.hpp"

double MagneticField::AbsB(double R, double phi, double Z){
    Vec3d B_ = B(R, phi, Z);
    return (Vec3d{cos(phi)*B_[0]-sin(phi)*B_[1], sin(phi)*B_[0]+cos(phi)*B_[1], B_[2]}).norm();
};

Vec3d MagneticField::GradAbsB(double R, double phi, double Z){
    double eps = 1e-4;
    double dB_dr = (AbsB(R+eps, phi, Z)-AbsB(R-eps, phi, Z))/(2*eps);
    double dB_dphi = (AbsB(R, phi+eps, Z)-AbsB(R, phi-eps, Z))/(2*eps);
    double dB_dz = (AbsB(R, phi, Z+eps)-AbsB(R, phi, Z-eps))/(2*eps);
    return Vec3d { dB_dr, dB_dphi/R, dB_dz };
};

Vec3d MagneticField::NormalisedB(double R, double phi, double Z){
    Vec3d B_ = B(R, phi, Z);
    double AbsB = (Vec3d{cos(phi)*B_[0]-sin(phi)*B_[1], sin(phi)*B_[0]+cos(phi)*B_[1], B_[2]}).norm();
    return B_/AbsB;
};

Vec3d MagneticField::b_cdot_grad_par_b(double R, double phi, double Z) {
    // Computes the material derivative A \cdot (\nabla B), where A and B are
    // both the normalised magnetic field (see
    // https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates
    // for the formula
    double eps = 1e-4;
    Vec3d dB_dr = (NormalisedB(R+eps, phi, Z)-NormalisedB(R-eps, phi, Z))/(2*eps);
    Vec3d dB_dphi = (NormalisedB(R, phi+eps, Z)-NormalisedB(R, phi-eps, Z))/(2*eps);
    Vec3d dB_dz = (NormalisedB(R, phi, Z+eps)-NormalisedB(R, phi, Z-eps))/(2*eps);
    Vec3d A = NormalisedB(R, phi, Z);
    Vec3d B = A;
    return Vec3d {
        A[0]*dB_dr[0] + A[1]*dB_dphi[0]/R + A[2]*dB_dz[0] - A[1]*B[1]/R,
            A[0]*dB_dr[1] + A[1]*dB_dphi[1]/R + A[2]*dB_dz[1] + A[1]*B[0]/R,
            A[0]*dB_dr[2] + A[1]*dB_dphi[2]/R + A[2]*dB_dz[2],
    };
};


AntoineField::AntoineField(double epsilon, double kappa, double delta, double A_, double Btin):
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

Vec3d AntoineField::B(double R, double phi, double Z){
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


DommaschkField::DommaschkField(double alpha_input)
{
    alpha = alpha_input;
};

Vec3d DommaschkField::B(double R, double phi, double Z){
    Vec3d B;
    auto Z2 = Z*Z;
    auto R2 = R*R;
    auto R4 = R2*R2;
    auto R5 = R4*R;
    auto R6 = R4*R2;
    auto Rm4 = 1/R4;
    auto Rm5 = 1/R5;
    auto Rm6 = 1/R6;
    auto phisin = std::sin(5*phi);
    auto phicos = std::cos(5*phi);

    auto pR1 = -(7./48.)*R6+(5./4.)*R4*Z2+(5.*9./480.)*R4-(3./32.)*Rm4-(5./4.)*Z2*Rm6+(7./48.)*Rm6;
    auto pR2 = ((1./2.)*R4+(1./20.)*Rm6)*Z;
    auto pZ1 = (1./2.)*R5*Z+(1./2.)*Z*Rm5;
    auto pZ2 = (1./10.)*R5-(1./10.)*Rm5;
    auto pphi1 = -(1./48.)*R6+(1./4.)*R4*Z2+(3./160.)*R4+(1./32.)*Rm4+(1./4.)*Z2*Rm6-(7./240.)*Rm6;
    auto pphi2 = ((1./10.)*R4-(1./10.)*Rm6)*Z;

    B.coeffRef(0) = alpha*(pR1*phisin+pR2*phicos);
    B.coeffRef(1) = 1/R+alpha*(pphi1*5*phicos-pphi2*5*phisin); 
    B.coeffRef(2) = alpha*(pZ1*phisin+pZ2*phicos);
    return B;
};
