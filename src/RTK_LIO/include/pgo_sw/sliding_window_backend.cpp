#include "pgo_sw/sliding_window_backend.h"
#include "runtime_profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace pgo_sw {

namespace {

template <typename T>
inline void QuaternionInverse(const T q[4], T q_inv[4])
{
    q_inv[0] = q[0];
    q_inv[1] = -q[1];
    q_inv[2] = -q[2];
    q_inv[3] = -q[3];
}

struct RelativePoseError {
    RelativePoseError(const Eigen::Vector3d& t_ij,
                      const Eigen::Quaterniond& q_ij,
                      double sigma_pos,
                      double sigma_rot)
        : t_x(t_ij.x()),
          t_y(t_ij.y()),
          t_z(t_ij.z()),
          q_w(q_ij.w()),
          q_x(q_ij.x()),
          q_y(q_ij.y()),
          q_z(q_ij.z()),
          sigma_p(sigma_pos),
          sigma_q(sigma_rot)
    {
    }

    template <typename T>
    bool operator()(const T* q_i, const T* t_i, const T* q_j, const T* t_j, T* residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = t_j[0] - t_i[0];
        t_w_ij[1] = t_j[1] - t_i[1];
        t_w_ij[2] = t_j[2] - t_i[2];

        T q_i_inv[4];
        QuaternionInverse(q_i, q_i_inv);

        T t_i_ij[3];
        ceres::QuaternionRotatePoint(q_i_inv, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(t_x)) / T(sigma_p);
        residuals[1] = (t_i_ij[1] - T(t_y)) / T(sigma_p);
        residuals[2] = (t_i_ij[2] - T(t_z)) / T(sigma_p);

        T q_meas[4] = {T(q_w), T(q_x), T(q_y), T(q_z)};
        T q_meas_inv[4];
        QuaternionInverse(q_meas, q_meas_inv);

        T q_i_j[4];
        ceres::QuaternionProduct(q_i_inv, q_j, q_i_j);

        T q_err[4];
        ceres::QuaternionProduct(q_meas_inv, q_i_j, q_err);

        residuals[3] = T(2.0) * q_err[1] / T(sigma_q);
        residuals[4] = T(2.0) * q_err[2] / T(sigma_q);
        residuals[5] = T(2.0) * q_err[3] / T(sigma_q);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& t_ij,
                                       const Eigen::Quaterniond& q_ij,
                                       double sigma_pos,
                                       double sigma_rot)
    {
        return new ceres::AutoDiffCostFunction<RelativePoseError, 6, 4, 3, 4, 3>(
            new RelativePoseError(t_ij, q_ij, sigma_pos, sigma_rot));
    }

    double t_x, t_y, t_z;
    double q_w, q_x, q_y, q_z;
    double sigma_p, sigma_q;
};

struct RtkPositionError {
    RtkPositionError(const Eigen::Vector3d& pos, double sigma_xy, double sigma_z)
        : p_x(pos.x()),
          p_y(pos.y()),
          p_z(pos.z()),
          sigma_xy(sigma_xy),
          sigma_z(sigma_z)
    {
    }

    template <typename T>
    bool operator()(const T* t, T* residuals) const
    {
        residuals[0] = (t[0] - T(p_x)) / T(sigma_xy);
        residuals[1] = (t[1] - T(p_y)) / T(sigma_xy);
        residuals[2] = (t[2] - T(p_z)) / T(sigma_z);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pos, double sigma_xy, double sigma_z)
    {
        return new ceres::AutoDiffCostFunction<RtkPositionError, 3, 3>(
            new RtkPositionError(pos, sigma_xy, sigma_z));
    }

    double p_x, p_y, p_z;
    double sigma_xy, sigma_z;
};

struct MatchedRtkFactor {
    size_t pose_index = 0;
    const RtkMeas* meas = nullptr;
};

}

SlidingWindowBackend::SlidingWindowBackend() = default;

void SlidingWindowBackend::setWindowSize(size_t size)
{
    window_size_ = std::max<size_t>(2, size);
    while (window_.size() > window_size_) {
        window_.pop_front();
    }
    while (optimized_.size() > window_size_) {
        optimized_.pop_front();
    }
}

void SlidingWindowBackend::setRtkMatchMaxDt(double dt)
{
    rtk_match_max_dt_ = std::max(0.0, dt);
}

void SlidingWindowBackend::setOdomSigmas(double sigma_pos, double sigma_rot)
{
    if (sigma_pos > 0.0) {
        odom_sigma_pos_ = sigma_pos;
    }
    if (sigma_rot > 0.0) {
        odom_sigma_rot_ = sigma_rot;
    }
}

void SlidingWindowBackend::setDefaultRtkSigmas(double sigma_xy, double sigma_z)
{
    if (sigma_xy > 0.0) {
        rtk_sigma_xy_default_ = sigma_xy;
    }
    if (sigma_z > 0.0) {
        rtk_sigma_z_default_ = sigma_z;
    }
}

void SlidingWindowBackend::addLioPose(const Pose& pose)
{
    window_.push_back(pose);
    if (window_.size() > window_size_) {
        window_.pop_front();
    }
}

void SlidingWindowBackend::addRtk(const RtkMeas& meas)
{
    rtk_buffer_.push_back(meas);
    while (rtk_buffer_.size() > window_size_ * 3) {
        rtk_buffer_.pop_front();
    }
}

bool SlidingWindowBackend::optimize()
{
    RUNTIME_TIMER(pgo_timer, "pgo_optimization");
    if (window_.empty()) {
        optimized_.clear();
        latest_rtk_factor_count_ = 0;
        latest_rtk_sigma_scale_ = 1.0;
        return false;
    }

    const size_t n = window_.size();
    const double oldest_time = window_.front().t;
    const double newest_time = window_.back().t;

    while (!rtk_buffer_.empty() && rtk_buffer_.front().t < oldest_time - rtk_match_max_dt_) {
        rtk_buffer_.pop_front();
    }
    while (!rtk_buffer_.empty() && rtk_buffer_.back().t > newest_time + rtk_match_max_dt_ * 10.0) {
        rtk_buffer_.pop_back();
    }

    std::vector<std::array<double, 3>> t_params(n);
    std::vector<std::array<double, 4>> q_params(n);

    for (size_t i = 0; i < n; ++i) {
        Eigen::Quaterniond q = window_[i].q;
        q.normalize();
        q_params[i][0] = q.w();
        q_params[i][1] = q.x();
        q_params[i][2] = q.y();
        q_params[i][3] = q.z();

        t_params[i][0] = window_[i].p.x();
        t_params[i][1] = window_[i].p.y();
        t_params[i][2] = window_[i].p.z();
    }

    ceres::Problem problem;
    ceres::LocalParameterization* quat_param = new ceres::QuaternionParameterization();
    for (size_t i = 0; i < n; ++i) {
        problem.AddParameterBlock(q_params[i].data(), 4, quat_param);
        problem.AddParameterBlock(t_params[i].data(), 3);
    }

    problem.SetParameterBlockConstant(q_params[0].data());
    problem.SetParameterBlockConstant(t_params[0].data());

    for (size_t i = 0; i + 1 < n; ++i) {
        const Pose& a = window_[i];
        const Pose& b = window_[i + 1];

        Eigen::Vector3d t_ij = a.q.conjugate() * (b.p - a.p);
        Eigen::Quaterniond q_ij = a.q.conjugate() * b.q;
        q_ij.normalize();

        ceres::CostFunction* odom_factor =
            RelativePoseError::Create(t_ij, q_ij, odom_sigma_pos_, odom_sigma_rot_);
        problem.AddResidualBlock(odom_factor, nullptr,
                                 q_params[i].data(), t_params[i].data(),
                                 q_params[i + 1].data(), t_params[i + 1].data());
    }

    std::vector<MatchedRtkFactor> rtk_factors;
    rtk_factors.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = window_[i].t;
        double best_dt = std::numeric_limits<double>::infinity();
        const RtkMeas* best = nullptr;
        for (const auto& meas : rtk_buffer_) {
            if (!meas.valid) {
                continue;
            }
            const double dt = std::abs(meas.t - t);
            if (dt < best_dt) {
                best_dt = dt;
                best = &meas;
            }
        }
        if (!best || best_dt > rtk_match_max_dt_) {
            continue;
        }

        MatchedRtkFactor factor;
        factor.pose_index = i;
        factor.meas = best;
        rtk_factors.push_back(factor);
    }

    latest_rtk_factor_count_ = rtk_factors.size();
    latest_rtk_sigma_scale_ = latest_rtk_factor_count_ > 1
                                   ? std::sqrt(static_cast<double>(latest_rtk_factor_count_))
                                   : 1.0;

    ceres::LossFunction* rtk_loss = new ceres::HuberLoss(1.0);
    for (const auto& factor : rtk_factors) {
        const RtkMeas& meas = *factor.meas;
        const double sigma_xy_base = (meas.sigma_xy > 0.0) ? meas.sigma_xy : rtk_sigma_xy_default_;
        const double sigma_z_base = (meas.sigma_z > 0.0) ? meas.sigma_z : rtk_sigma_z_default_;
        const double sigma_xy = sigma_xy_base * latest_rtk_sigma_scale_;
        const double sigma_z = sigma_z_base * latest_rtk_sigma_scale_;
        ceres::CostFunction* rtk_factor = RtkPositionError::Create(meas.p, sigma_xy, sigma_z);
        problem.AddResidualBlock(rtk_factor, rtk_loss, t_params[factor.pose_index].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    optimized_.clear();
    optimized_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        Pose out;
        out.t = window_[i].t;
        out.p = Eigen::Vector3d(t_params[i][0], t_params[i][1], t_params[i][2]);
        Eigen::Quaterniond q(q_params[i][0], q_params[i][1], q_params[i][2], q_params[i][3]);
        q.normalize();
        out.q = q;
        optimized_[i] = out;
    }

    return true;
}

bool SlidingWindowBackend::getLatestOptimized(Pose* out) const
{
    if (!out || optimized_.empty()) {
        return false;
    }
    *out = optimized_.back();
    return true;
}

void SlidingWindowBackend::clearRtk()
{
    rtk_buffer_.clear();
}

void SlidingWindowBackend::clear()
{
    window_.clear();
    optimized_.clear();
    rtk_buffer_.clear();
}

}
