#include "sdp_solver.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace sdp_solver {

std::vector<SdpConstraint> build_quasar_constraints(int N) {
    std::vector<SdpConstraint> constraints;


    SdpConstraint trace_constraint;
    trace_constraint.type = SdpConstraint::TRACE_EQUALITY;
    trace_constraint.block_i = 0;
    trace_constraint.block_j = 0;
    trace_constraint.b = 1.0;
    constraints.push_back(trace_constraint);


    for(int i = 1; i <= N; ++i) {
        SdpConstraint clone_constraint;
        clone_constraint.type = SdpConstraint::BLOCK_EQUALITY;
        clone_constraint.block_i = 0;
        clone_constraint.block_j = i;
        clone_constraint.b = 0.0;
        constraints.push_back(clone_constraint);
    }


    for(int i = 1; i <= N; ++i) {
        SdpConstraint sym_constraint;
        sym_constraint.type = SdpConstraint::SYMMETRY;
        sym_constraint.block_i = 0;
        sym_constraint.block_j = i;
        constraints.push_back(sym_constraint);
    }


    for(int i = 1; i <= N; ++i) {
        for(int j = i + 1; j <= N; ++j) {
            SdpConstraint cross_sym_constraint;
            cross_sym_constraint.type = SdpConstraint::SYMMETRY;
            cross_sym_constraint.block_i = i;
            cross_sym_constraint.block_j = j;
            constraints.push_back(cross_sym_constraint);
        }
    }

    return constraints;
}

Eigen::MatrixXd project_to_psd(const Eigen::MatrixXd& M) {

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();


    for(int i = 0; i < eigenvalues.size(); ++i) {
        if(eigenvalues(i) < 0) {
            eigenvalues(i) = 0;
        }
    }


    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

Eigen::MatrixXd project_to_quasar_constraints(const Eigen::MatrixXd& Z, int N) {
    Eigen::MatrixXd Z_proj = Z;
    int dim = 4 * (N + 1);

    if(Z_proj.rows() != dim || Z_proj.cols() != dim) {
        return Z_proj;
    }


    Eigen::Matrix4d Z_qq = Z_proj.block<4,4>(0, 0);


    double current_trace = Z_qq.trace();
    if(std::abs(current_trace) > 1e-12) {
        Z_proj /= current_trace;
        Z_qq /= current_trace;
    }


    for(int i = 1; i <= N; ++i) {
        int offset = 4 * i;
        Z_proj.block<4,4>(offset, offset) = Z_qq;
    }


    for(int i = 1; i <= N; ++i) {
        int offset = 4 * i;
        Eigen::Matrix4d block = Z_proj.block<4,4>(0, offset);
        Eigen::Matrix4d block_sym = 0.5 * (block + block.transpose());
        Z_proj.block<4,4>(0, offset) = block_sym;
        Z_proj.block<4,4>(offset, 0) = block_sym.transpose();
    }


    for(int i = 1; i <= N; ++i) {
        for(int j = i+1; j <= N; ++j) {
            int offset_i = 4 * i;
            int offset_j = 4 * j;
            Eigen::Matrix4d block = Z_proj.block<4,4>(offset_i, offset_j);
            Eigen::Matrix4d block_sym = 0.5 * (block + block.transpose());
            Z_proj.block<4,4>(offset_i, offset_j) = block_sym;
            Z_proj.block<4,4>(offset_j, offset_i) = block_sym.transpose();
        }
    }

    return Z_proj;
}

std::pair<Eigen::MatrixXd, double> solve_sdp_projected_gradient(
    const QuasarSdpProblem& problem,
    int max_iters,
    double tol)
{
    int dim = problem.dim;


    Eigen::MatrixXd Q_dense = Eigen::MatrixXd(problem.Q);


    Eigen::MatrixXd Z = Eigen::MatrixXd::Identity(dim, dim) / dim;
    Z = project_to_quasar_constraints(Z, problem.N);
    Z = project_to_psd(Z);

    double step_size = 0.1;
    double obj_prev = (Q_dense * Z).trace();

    for(int iter = 0; iter < max_iters; ++iter) {

        Eigen::MatrixXd grad = Q_dense;
        Eigen::MatrixXd Z_new = Z - step_size * grad;


        Z_new = project_to_quasar_constraints(Z_new, problem.N);
        Z_new = project_to_psd(Z_new);


        double obj = (Q_dense * Z_new).trace();


        if(std::abs(obj - obj_prev) < tol) {
            Z = Z_new;
            break;
        }


        if(obj < obj_prev) {
            step_size *= 1.1;
            Z = Z_new;
        } else {
            step_size *= 0.5;
        }

        obj_prev = obj;

        if(iter % 100 == 0) {
        }
    }

    double final_obj = (Q_dense * Z).trace();
    return std::make_pair(Z, final_obj);
}

Eigen::MatrixXd solve_sdp_eigen_approx(
    const Eigen::SparseMatrix<double>& Q,
    int N)
{
    int dim = 4 * (N + 1);


    Eigen::MatrixXd Q_dense = Eigen::MatrixXd(Q);


    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q_dense);
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();


    int min_idx = 0;
    double min_eigenvalue = eigenvalues(0);
    for(int i = 1; i < eigenvalues.size(); ++i) {
        if(eigenvalues(i) < min_eigenvalue) {
            min_eigenvalue = eigenvalues(i);
            min_idx = i;
        }
    }


    Eigen::VectorXd v = eigenvectors.col(min_idx);
    Eigen::MatrixXd Z = v * v.transpose();


    Z = project_to_quasar_constraints(Z, N);
    Z = project_to_psd(Z);

    return Z;
}

}
