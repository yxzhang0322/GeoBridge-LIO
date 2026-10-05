#include "quasar.h"
#include "sdp_solver.h"
#include <iostream>
#include <cmath>

namespace quasar {


std::vector<Eigen::Triplet<double>> build_Qi_blocks(
    const Eigen::Vector3d& a_i,
    const Eigen::Vector3d& b_i,
    double sigma_i,
    double c_bar,
    int i,
    int N)
{
    std::vector<Eigen::Triplet<double>> triplets;


    Eigen::Vector4d a_hat, b_hat;
    a_hat << a_i, 0.0;
    b_hat << b_i, 0.0;


    double weight = 1.0 / (sigma_i * sigma_i);


    int q_offset = 0;
    int qi_offset = 4 * (i + 1);


    Eigen::Matrix4d A_mat = Omega2(a_hat);
    Eigen::Matrix4d B_mat = Omega2(b_hat);
    Eigen::Matrix4d H_i = (A_mat - B_mat).transpose() * (A_mat - B_mat);


    Eigen::Matrix4d Q_qq_block = (weight * H_i) + (c_bar * c_bar / 2.0) * Eigen::Matrix4d::Identity();
    Eigen::Matrix4d Q_qiqi_block = (c_bar * c_bar / 2.0) * Eigen::Matrix4d::Identity();
    Eigen::Matrix4d Q_qqi_block = -(c_bar * c_bar / 2.0) * Eigen::Matrix4d::Identity();


    for(int row = 0; row < 4; ++row) {
        for(int col = 0; col < 4; ++col) {
            double val = Q_qq_block(row, col);
            if(std::abs(val) > 1e-12) {
                triplets.emplace_back(q_offset + row, q_offset + col, val);
            }
        }
    }


    for(int row = 0; row < 4; ++row) {
        for(int col = 0; col < 4; ++col) {
            double val = Q_qiqi_block(row, col);
            if(std::abs(val) > 1e-12) {
                triplets.emplace_back(qi_offset + row, qi_offset + col, val);
            }
        }
    }


    for(int row = 0; row < 4; ++row) {
        for(int col = 0; col < 4; ++col) {
            double val = Q_qqi_block(row, col);
            if(std::abs(val) > 1e-12) {
                triplets.emplace_back(q_offset + row, qi_offset + col, val);
                triplets.emplace_back(qi_offset + row, q_offset + col, val);
            }
        }
    }

    return triplets;
}


Eigen::SparseMatrix<double> build_Q_total(
    const std::vector<Eigen::Vector3d>& A,
    const std::vector<Eigen::Vector3d>& B,
    const std::vector<double>& sigma,
    double c_bar)
{
    const int N = static_cast<int>(A.size());
    const int dim = 4 * (N + 1);

    std::vector<Eigen::Triplet<double>> all_triplets;


    for(int i = 0; i < N; ++i) {
        double sig = (i < static_cast<int>(sigma.size())) ? sigma[i] : 1.0;

        auto triplets_i = build_Qi_blocks(A[i], B[i], sig, c_bar, i, N);
        all_triplets.insert(all_triplets.end(), triplets_i.begin(), triplets_i.end());
    }


    Eigen::SparseMatrix<double> Q(dim, dim);
    Q.setFromTriplets(all_triplets.begin(), all_triplets.end());

    return Q;
}


QuaternionResult recover_quaternion_from_Z(
    const Eigen::MatrixXd& Z,
    double rank_tol)
{
    QuaternionResult result;


    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Z);
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();


    int n = eigenvalues.size();
    int max_idx = n - 1;
    double lambda_max = eigenvalues(max_idx);


    double lambda_second = (n > 1) ? eigenvalues(n - 2) : 0.0;
    result.rank_ratio = (lambda_max > 1e-12) ? (lambda_second / lambda_max) : 0.0;
    result.certified = (result.rank_ratio < rank_tol);


    Eigen::VectorXd x_star = eigenvectors.col(max_idx);


    result.q = x_star.head<4>();
    result.q.normalize();


    result.R = quatToSO3(result.q);

    return result;
}


QuaternionResult solve_quasar(
    const std::vector<Eigen::Vector3d>& A,
    const std::vector<Eigen::Vector3d>& B,
    const std::vector<double>& sigma,
    double c_bar,
    double rank_tol,
    bool use_eigen_approx)
{
    const int N = static_cast<int>(A.size());


    Eigen::SparseMatrix<double> Q = build_Q_total(A, B, sigma, c_bar);


    Eigen::MatrixXd Z;
    if(use_eigen_approx) {
        Z = sdp_solver::solve_sdp_eigen_approx(Q, N);
    } else {
        sdp_solver::QuasarSdpProblem problem;
        problem.Q = Q;
        problem.dim = 4 * (N + 1);
        problem.N = N;
        problem.constraints = sdp_solver::build_quasar_constraints(N);

        auto result_pair = sdp_solver::solve_sdp_projected_gradient(problem, 500, 1e-6);
        Z = result_pair.first;
    }


    QuaternionResult result = recover_quaternion_from_Z(Z, rank_tol);


    return result;
}

}
