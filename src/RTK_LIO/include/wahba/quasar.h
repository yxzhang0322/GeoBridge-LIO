#ifndef QUASAR_H
#define QUASAR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace quasar {


inline Eigen::Matrix4d Omega1(const Eigen::Vector4d& q) {
    Eigen::Matrix4d omega;
    double qw = q(3), qx = q(0), qy = q(1), qz = q(2);

    omega << qw, -qz,  qy, qx,
             qz,  qw, -qx, qy,
            -qy,  qx,  qw, qz,
            -qx, -qy, -qz, qw;

    return omega;
}


inline Eigen::Matrix4d Omega2(const Eigen::Vector4d& q) {
    Eigen::Matrix4d omega;
    double qw = q(3), qx = q(0), qy = q(1), qz = q(2);

    omega << qw, -qz,  qy, -qx,
             qz,  qw, -qx, -qy,
            -qy,  qx,  qw, -qz,
             qx,  qy,  qz,  qw;

    return omega;
}


inline Eigen::Matrix3d quatToSO3(const Eigen::Vector4d& q) {
    double qw = q(3), qx = q(0), qy = q(1), qz = q(2);
    Eigen::Matrix3d R;

    R << 1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy),
         2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx),
         2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy);

    return R;
}


inline Eigen::Vector3d rotateVector(const Eigen::Vector4d& q, const Eigen::Vector3d& a) {

    Eigen::Vector4d a_hat;
    a_hat << a, 0.0;


    Eigen::Vector4d result = Omega1(q) * Omega2(q).transpose() * a_hat;

    return result.head<3>();
}


inline int blockIndex(int i) {
    return 4 * i;
}


struct SdpResult {
    Eigen::MatrixXd Z;
    double objective;
    bool solved;
    int rank;
    double gap;

    SdpResult() : objective(0.0), solved(false), rank(0), gap(0.0) {}
};


struct QuaternionResult {
    Eigen::Vector4d q;
    bool certified;
    double rank_ratio;
    Eigen::Matrix3d R;

    QuaternionResult() : certified(false), rank_ratio(0.0) {
        q = Eigen::Vector4d(0, 0, 0, 1);
        R = Eigen::Matrix3d::Identity();
    }
};

}

#endif
