#ifndef SDP_SOLVER_H
#define SDP_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace sdp_solver {


struct SdpConstraint {
    enum Type {
        BLOCK_EQUALITY,
        TRACE_EQUALITY,
        SYMMETRY
    };

    Type type;
    int block_i;
    int block_j;
    Eigen::MatrixXd A;
    double b;
};


struct QuasarSdpProblem {
    Eigen::SparseMatrix<double> Q;
    std::vector<SdpConstraint> constraints;
    int dim;
    int N;
};


std::vector<SdpConstraint> build_quasar_constraints(int N);


std::pair<Eigen::MatrixXd, double> solve_sdp_projected_gradient(
    const QuasarSdpProblem& problem,
    int max_iters = 1000,
    double tol = 1e-6);


Eigen::MatrixXd solve_sdp_eigen_approx(
    const Eigen::SparseMatrix<double>& Q,
    int N);


Eigen::MatrixXd project_to_psd(const Eigen::MatrixXd& M);


Eigen::MatrixXd project_to_quasar_constraints(const Eigen::MatrixXd& Z, int N);

}

#endif
