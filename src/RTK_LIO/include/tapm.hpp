#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "common_lib.h"
#include <wahba/alignment.h>

struct TapmConfig
{
  bool enable = false;
  bool normalize_weights = true;
  bool anchor_grace_cutoff = false;
  bool covariance_from_outage = true;
  bool use_analytic_jacobian = true;
  double alpha = 0.5;
  double beta = 1.0;
  double lambda_v = 10.0;
  double lambda_a = 100.0;
  double gamma = 0.01;
  double nominal_dt = 0.1;
  double sigma0_xy = 0.1;
  double sigma0_z = 0.2;
  double v_max = 20.0;
  double v_max_z = 5.0;
  double a_max = 8.0;
  double a_max_z = 3.0;
  double rtk_sigma_min = 1e-3;
  double rtk_match_max_dt = 0.1;
  int grace_frames = 15;
  int max_iterations = 30;
};

struct TapmResult
{
  bool success = false;
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d position_last = Eigen::Vector3d::Zero();
  double cost = 0.0;
  double tau_last = 0.0;
  double weight_tapm_last = 0.0;
  double weight_rtk_last = 0.0;
  bool latest_rtk_used = false;
  double latest_rtk_dt = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d latest_rtk_pos = Eigen::Vector3d::Zero();
  double latest_rtk_epoch = -1.0;
  Eigen::Vector3d latest_rtk_error = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  size_t used_tapm = 0;
  size_t used_rtk = 0;
  std::string message;
};

namespace tapm_detail
{
inline double clamp_non_negative(double value)
{
  return value > 0.0 ? value : 0.0;
}

inline double compute_weight_tapm(double tau, double alpha)
{
  if (tau <= 0.0)
  {
    return 1.0;
  }
  return std::exp(-alpha * tau);
}

inline double compute_weight_rtk(double tau, double beta)
{
  if (tau <= 0.0)
  {
    return 0.0;
  }
  return 1.0 - std::exp(-beta * tau);
}

inline void set_diagonal_jacobian(double* jac, double d0, double d1, double d2)
{
  jac[0] = d0;
  jac[1] = 0.0;
  jac[2] = 0.0;
  jac[3] = 0.0;
  jac[4] = d1;
  jac[5] = 0.0;
  jac[6] = 0.0;
  jac[7] = 0.0;
  jac[8] = d2;
}

struct RtkResidual
{
  RtkResidual(const Eigen::Vector3d& rtk_pos,
              double weight,
              double sigma_xy,
              double sigma_z)
      : rtk_pos_(rtk_pos),
        weight_(weight),
        inv_sigma_xy_(1.0 / sigma_xy),
        inv_sigma_z_(1.0 / sigma_z)
  {
  }

  template <typename T>
  bool operator()(const T* const p, T* residuals) const
  {
    const T w = T(std::sqrt(weight_));
    residuals[0] = w * (p[0] - T(rtk_pos_.x())) * T(inv_sigma_xy_);
    residuals[1] = w * (p[1] - T(rtk_pos_.y())) * T(inv_sigma_xy_);
    residuals[2] = w * (p[2] - T(rtk_pos_.z())) * T(inv_sigma_z_);
    return true;
  }

  Eigen::Vector3d rtk_pos_;
  double weight_;
  double inv_sigma_xy_;
  double inv_sigma_z_;
};

class RtkResidualAnalytic : public ceres::SizedCostFunction<3, 3>
{
public:
  RtkResidualAnalytic(const Eigen::Vector3d& rtk_pos,
                      double weight,
                      double sigma_xy,
                      double sigma_z)
      : rtk_pos_(rtk_pos)
  {
    const double w = std::sqrt(weight);
    coeff_xy_ = w / sigma_xy;
    coeff_z_ = w / sigma_z;
  }

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    const double* p = parameters[0];
    residuals[0] = coeff_xy_ * (p[0] - rtk_pos_.x());
    residuals[1] = coeff_xy_ * (p[1] - rtk_pos_.y());
    residuals[2] = coeff_z_ * (p[2] - rtk_pos_.z());
    if (jacobians && jacobians[0])
    {
      set_diagonal_jacobian(jacobians[0], coeff_xy_, coeff_xy_, coeff_z_);
    }
    return true;
  }

private:
  Eigen::Vector3d rtk_pos_;
  double coeff_xy_ = 0.0;
  double coeff_z_ = 0.0;
};

struct TapmResidual
{
  TapmResidual(const Eigen::Vector3d& anchor_pos,
               double weight,
               double sigma_xy,
               double sigma_z)
      : anchor_pos_(anchor_pos),
        weight_(weight),
        inv_sigma_xy_(1.0 / sigma_xy),
        inv_sigma_z_(1.0 / sigma_z)
  {
  }

  template <typename T>
  bool operator()(const T* const p, T* residuals) const
  {
    const T w = T(std::sqrt(weight_));
    residuals[0] = w * (p[0] - T(anchor_pos_.x())) * T(inv_sigma_xy_);
    residuals[1] = w * (p[1] - T(anchor_pos_.y())) * T(inv_sigma_xy_);
    residuals[2] = w * (p[2] - T(anchor_pos_.z())) * T(inv_sigma_z_);
    return true;
  }

  Eigen::Vector3d anchor_pos_;
  double weight_;
  double inv_sigma_xy_;
  double inv_sigma_z_;
};

class TapmResidualAnalytic : public ceres::SizedCostFunction<3, 3>
{
public:
  TapmResidualAnalytic(const Eigen::Vector3d& anchor_pos,
                       double weight,
                       double sigma_xy,
                       double sigma_z)
      : anchor_pos_(anchor_pos)
  {
    const double w = std::sqrt(weight);
    coeff_xy_ = w / sigma_xy;
    coeff_z_ = w / sigma_z;
  }

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    const double* p = parameters[0];
    residuals[0] = coeff_xy_ * (p[0] - anchor_pos_.x());
    residuals[1] = coeff_xy_ * (p[1] - anchor_pos_.y());
    residuals[2] = coeff_z_ * (p[2] - anchor_pos_.z());
    if (jacobians && jacobians[0])
    {
      set_diagonal_jacobian(jacobians[0], coeff_xy_, coeff_xy_, coeff_z_);
    }
    return true;
  }

private:
  Eigen::Vector3d anchor_pos_;
  double coeff_xy_ = 0.0;
  double coeff_z_ = 0.0;
};

struct VelocityResidual
{
  VelocityResidual(double inv_dt,
                   double weight,
                   double v_max_xy,
                   double v_max_z,
                   const Eigen::Vector3d& reference_delta)
      : inv_dt_(inv_dt),
        weight_(weight),
        inv_v_xy_(1.0 / v_max_xy),
        inv_v_z_(1.0 / v_max_z),
        reference_delta_(reference_delta)
  {
  }

  template <typename T>
  bool operator()(const T* const p_prev, const T* const p_curr, T* residuals) const
  {
    const T w = T(std::sqrt(weight_));
    residuals[0] =
        w * (p_curr[0] - p_prev[0] - T(reference_delta_.x())) * T(inv_dt_) * T(inv_v_xy_);
    residuals[1] =
        w * (p_curr[1] - p_prev[1] - T(reference_delta_.y())) * T(inv_dt_) * T(inv_v_xy_);
    residuals[2] =
        w * (p_curr[2] - p_prev[2] - T(reference_delta_.z())) * T(inv_dt_) * T(inv_v_z_);
    return true;
  }

  double inv_dt_;
  double weight_;
  double inv_v_xy_;
  double inv_v_z_;
  Eigen::Vector3d reference_delta_;
};

class VelocityResidualAnalytic : public ceres::SizedCostFunction<3, 3, 3>
{
public:
  VelocityResidualAnalytic(double inv_dt,
                           double weight,
                           double v_max_xy,
                           double v_max_z,
                           const Eigen::Vector3d& reference_delta)
      : reference_delta_(reference_delta)
  {
    const double w = std::sqrt(weight);
    const double inv_v_xy = 1.0 / v_max_xy;
    const double inv_v_z = 1.0 / v_max_z;
    coeff_xy_ = w * inv_dt * inv_v_xy;
    coeff_z_ = w * inv_dt * inv_v_z;
  }

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    const double* p_prev = parameters[0];
    const double* p_curr = parameters[1];
    residuals[0] = coeff_xy_ * (p_curr[0] - p_prev[0] - reference_delta_.x());
    residuals[1] = coeff_xy_ * (p_curr[1] - p_prev[1] - reference_delta_.y());
    residuals[2] = coeff_z_ * (p_curr[2] - p_prev[2] - reference_delta_.z());
    if (jacobians)
    {
      if (jacobians[0])
      {
        set_diagonal_jacobian(jacobians[0], -coeff_xy_, -coeff_xy_, -coeff_z_);
      }
      if (jacobians[1])
      {
        set_diagonal_jacobian(jacobians[1], coeff_xy_, coeff_xy_, coeff_z_);
      }
    }
    return true;
  }

private:
  double coeff_xy_ = 0.0;
  double coeff_z_ = 0.0;
  Eigen::Vector3d reference_delta_ = Eigen::Vector3d::Zero();
};

struct AccelerationResidual
{
  AccelerationResidual(double inv_dt2,
                       double weight,
                       double a_max_xy,
                       double a_max_z,
                       const Eigen::Vector3d& reference_delta2)
      : inv_dt2_(inv_dt2),
        weight_(weight),
        inv_a_xy_(1.0 / a_max_xy),
        inv_a_z_(1.0 / a_max_z),
        reference_delta2_(reference_delta2)
  {
  }

  template <typename T>
  bool operator()(const T* const p_prev2, const T* const p_prev, const T* const p_curr, T* residuals) const
  {
    const T w = T(std::sqrt(weight_));
    residuals[0] = w * (p_curr[0] - T(2.0) * p_prev[0] + p_prev2[0] -
                         T(reference_delta2_.x())) * T(inv_dt2_) * T(inv_a_xy_);
    residuals[1] = w * (p_curr[1] - T(2.0) * p_prev[1] + p_prev2[1] -
                         T(reference_delta2_.y())) * T(inv_dt2_) * T(inv_a_xy_);
    residuals[2] = w * (p_curr[2] - T(2.0) * p_prev[2] + p_prev2[2] -
                         T(reference_delta2_.z())) * T(inv_dt2_) * T(inv_a_z_);
    return true;
  }

  double inv_dt2_;
  double weight_;
  double inv_a_xy_;
  double inv_a_z_;
  Eigen::Vector3d reference_delta2_;
};

class AccelerationResidualAnalytic : public ceres::SizedCostFunction<3, 3, 3, 3>
{
public:
  AccelerationResidualAnalytic(double inv_dt2,
                               double weight,
                               double a_max_xy,
                               double a_max_z,
                               const Eigen::Vector3d& reference_delta2)
      : reference_delta2_(reference_delta2)
  {
    const double w = std::sqrt(weight);
    const double inv_a_xy = 1.0 / a_max_xy;
    const double inv_a_z = 1.0 / a_max_z;
    coeff_xy_ = w * inv_dt2 * inv_a_xy;
    coeff_z_ = w * inv_dt2 * inv_a_z;
  }

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    const double* p_prev2 = parameters[0];
    const double* p_prev = parameters[1];
    const double* p_curr = parameters[2];
    residuals[0] =
        coeff_xy_ * (p_curr[0] - 2.0 * p_prev[0] + p_prev2[0] - reference_delta2_.x());
    residuals[1] =
        coeff_xy_ * (p_curr[1] - 2.0 * p_prev[1] + p_prev2[1] - reference_delta2_.y());
    residuals[2] =
        coeff_z_ * (p_curr[2] - 2.0 * p_prev[2] + p_prev2[2] - reference_delta2_.z());
    if (jacobians)
    {
      if (jacobians[0])
      {
        set_diagonal_jacobian(jacobians[0], coeff_xy_, coeff_xy_, coeff_z_);
      }
      if (jacobians[1])
      {
        set_diagonal_jacobian(jacobians[1], -2.0 * coeff_xy_, -2.0 * coeff_xy_, -2.0 * coeff_z_);
      }
      if (jacobians[2])
      {
        set_diagonal_jacobian(jacobians[2], coeff_xy_, coeff_xy_, coeff_z_);
      }
    }
    return true;
  }

private:
  double coeff_xy_ = 0.0;
  double coeff_z_ = 0.0;
  Eigen::Vector3d reference_delta2_ = Eigen::Vector3d::Zero();
};

template <typename TimeposContainer>
inline bool find_nearest_timepos(const TimeposContainer& samples,
                                 double timestamp,
                                 double max_dt,
                                 Timepos* nearest,
                                 double* nearest_dt)
{
  if (samples.empty() || max_dt < 0.0)
  {
    return false;
  }

  auto upper = std::lower_bound(
      samples.begin(), samples.end(), timestamp,
      [](const Timepos& sample, double ts)
      { return sample.timestamp < ts; });
  if (upper != samples.begin() && upper != samples.end())
  {
    auto lower = upper;
    --lower;
    const double t0 = lower->timestamp;
    const double t1 = upper->timestamp;
    const double span = t1 - t0;
    const double max_interp_span = std::max(2.5 * max_dt, 0.15);
    if (span > 1e-6 && span <= max_interp_span &&
        timestamp >= t0 && timestamp <= t1)
    {
      const double ratio = (timestamp - t0) / span;
      if (nearest)
      {
        Timepos interp = *lower;
        interp.timestamp = timestamp;
        interp.pos = lower->pos + ratio * (upper->pos - lower->pos);
        interp.h_acc = std::max(lower->h_acc, upper->h_acc);
        interp.v_acc = std::max(lower->v_acc, upper->v_acc);
        interp.vel_acc = std::max(lower->vel_acc, upper->vel_acc);
        interp.fix_type = std::min(lower->fix_type, upper->fix_type);
        if (lower->vel_valid && upper->vel_valid)
        {
          interp.vel = lower->vel + ratio * (upper->vel - lower->vel);
          interp.vel_valid = true;
        }
        else
        {
          interp.vel_valid = false;
        }
        *nearest = interp;
      }
      if (nearest_dt)
      {
        *nearest_dt = 0.0;
      }
      return true;
    }
  }

  bool found = false;
  double best_dt = max_dt;
  for (const Timepos& sample : samples)
  {
    const double dt = std::abs(sample.timestamp - timestamp);
    if (dt <= best_dt)
    {
      best_dt = dt;
      if (nearest)
      {
        *nearest = sample;
      }
      found = true;
    }
  }
  if (found && nearest_dt)
  {
    *nearest_dt = best_dt;
  }
  return found;
}
}

inline TapmResult solve_tapm_translation_sliding(
    const std::vector<Timepos>& world_history_window,
    const std::vector<Eigen::Vector3d>& lio_reference,
    const Timepos& rtk_current,
    double outage_time,
    double recover_time,
    const TapmConfig& config)
{
  TapmResult result;
  if (!config.enable)
  {
    result.message = "disabled";
    return result;
  }
  if (world_history_window.size() < 3 || lio_reference.size() != world_history_window.size() ||
      !std::isfinite(outage_time) || outage_time < 0.0 || !rtk_current.pos.allFinite() ||
      !std::isfinite(recover_time) || recover_time < outage_time ||
      !std::isfinite(rtk_current.h_acc) || !std::isfinite(rtk_current.v_acc) ||
      rtk_current.h_acc <= 0.0 || rtk_current.v_acc <= 0.0)
  {
    result.message = "insufficient_samples";
    return result;
  }

  const size_t sample_count = world_history_window.size();
  const double nominal_dt = config.nominal_dt > 0.0 ? config.nominal_dt : 0.1;
  const double inv_dt = 1.0 / nominal_dt;
  const double inv_dt2 = inv_dt * inv_dt;
  const double grace_time =
      config.grace_frames > 0 ? static_cast<double>(config.grace_frames) * nominal_dt : 0.0;
  const double v_max_xy = config.v_max > 0.0 ? config.v_max : 1.0;
  const double v_max_z = config.v_max_z > 0.0 ? config.v_max_z : v_max_xy;
  const double a_max_xy = config.a_max > 0.0 ? config.a_max : 1.0;
  const double a_max_z = config.a_max_z > 0.0 ? config.a_max_z : a_max_xy;
  const double anchor_time = config.covariance_from_outage
      ? outage_time : world_history_window.front().timestamp;

  ceres::Problem problem;
  std::vector<std::array<double, 3>> positions(sample_count);
  std::vector<Eigen::Vector3d> reference_positions(sample_count);
  for (size_t i = 0; i < sample_count; ++i)
  {
    const Eigen::Vector3d p_init = world_history_window[i].pos;
    reference_positions[i] = lio_reference[i];
    positions[i][0] = p_init.x();
    positions[i][1] = p_init.y();
    positions[i][2] = p_init.z();
    problem.AddParameterBlock(positions[i].data(), 3);
  }

  size_t used_rtk = 0;
  size_t used_tapm = 0;

  auto weights = [&](double tau) {
    double anchor = tapm_detail::compute_weight_tapm(tau, config.alpha);
    double rtk = tapm_detail::compute_weight_rtk(tau, config.beta);
    if (config.anchor_grace_cutoff && grace_time > 0.0 && tau > grace_time)
      anchor = 0.0;
    if (config.normalize_weights)
    {
      const double denominator = anchor + rtk + 1e-12;
      anchor /= denominator;
      rtk /= denominator;
    }
    return std::array<double, 2>{{anchor, rtk}};
  };
  for (size_t i = 0; i + 1 < sample_count; ++i)
  {
    const double timestamp = world_history_window[i].timestamp;
    const double tau = timestamp - recover_time;
    const double weight_tapm = weights(tau)[0];
    if (weight_tapm <= 0.0) continue;
    const double growth = 1.0 + config.gamma *
        tapm_detail::clamp_non_negative(timestamp - anchor_time) / nominal_dt;
    const double sigma_xy = config.sigma0_xy * std::sqrt(growth);
    const double sigma_z = config.sigma0_z * std::sqrt(growth);
    ceres::CostFunction* cost = config.use_analytic_jacobian
        ? static_cast<ceres::CostFunction*>(new tapm_detail::TapmResidualAnalytic(
            world_history_window[i].pos, weight_tapm, sigma_xy, sigma_z))
        : static_cast<ceres::CostFunction*>(new ceres::AutoDiffCostFunction<tapm_detail::TapmResidual, 3, 3>(
            new tapm_detail::TapmResidual(world_history_window[i].pos, weight_tapm, sigma_xy, sigma_z)));
    problem.AddResidualBlock(cost, nullptr, positions[i].data());
    ++used_tapm;
  }
  const double last_tau = world_history_window.back().timestamp - recover_time;
  const double latest_weight_rtk = weights(last_tau)[1];
  if (latest_weight_rtk > 0.0)
  {
    const double sigma_xy = std::max(rtk_current.h_acc, config.rtk_sigma_min);
    const double sigma_z = std::max(rtk_current.v_acc, config.rtk_sigma_min);
    ceres::CostFunction* cost = config.use_analytic_jacobian
        ? static_cast<ceres::CostFunction*>(new tapm_detail::RtkResidualAnalytic(
            rtk_current.pos, latest_weight_rtk, sigma_xy, sigma_z))
        : static_cast<ceres::CostFunction*>(new ceres::AutoDiffCostFunction<tapm_detail::RtkResidual, 3, 3>(
            new tapm_detail::RtkResidual(rtk_current.pos, latest_weight_rtk, sigma_xy, sigma_z)));
    problem.AddResidualBlock(cost, nullptr, positions.back().data());
    used_rtk = 1;
    result.latest_rtk_used = true;
    result.latest_rtk_epoch = rtk_current.timestamp;
    result.latest_rtk_dt = world_history_window.back().timestamp - rtk_current.timestamp;
    result.latest_rtk_pos = rtk_current.pos;
  }

  if (config.lambda_v > 0.0)
  {
    for (size_t i = 1; i < sample_count; ++i)
    {
      ceres::CostFunction* cost = nullptr;
      if (config.use_analytic_jacobian)
      {
        cost = new tapm_detail::VelocityResidualAnalytic(
            inv_dt,
            config.lambda_v,
            v_max_xy,
            v_max_z,
            reference_positions[i] - reference_positions[i - 1]);
      }
      else
      {
        cost = new ceres::AutoDiffCostFunction<tapm_detail::VelocityResidual, 3, 3, 3>(
            new tapm_detail::VelocityResidual(
                inv_dt,
                config.lambda_v,
                v_max_xy,
                v_max_z,
                reference_positions[i] - reference_positions[i - 1]));
      }
      problem.AddResidualBlock(cost, nullptr, positions[i - 1].data(), positions[i].data());
    }
  }

  if (config.lambda_a > 0.0)
  {
    for (size_t i = 2; i < sample_count; ++i)
    {
      ceres::CostFunction* cost = nullptr;
      if (config.use_analytic_jacobian)
      {
        cost = new tapm_detail::AccelerationResidualAnalytic(
            inv_dt2,
            config.lambda_a,
            a_max_xy,
            a_max_z,
            reference_positions[i] - 2.0 * reference_positions[i - 1] +
                reference_positions[i - 2]);
      }
      else
      {
        cost =
            new ceres::AutoDiffCostFunction<tapm_detail::AccelerationResidual, 3, 3, 3, 3>(
                new tapm_detail::AccelerationResidual(
                    inv_dt2,
                    config.lambda_a,
                    a_max_xy,
                    a_max_z,
                    reference_positions[i] - 2.0 * reference_positions[i - 1] +
                        reference_positions[i - 2]));
      }
      problem.AddResidualBlock(
          cost, nullptr, positions[i - 2].data(), positions[i - 1].data(), positions[i].data());
    }
  }


  ceres::Solver::Options options;
  options.max_num_iterations = config.max_iterations > 0 ? config.max_iterations : 30;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 1;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  result.success = summary.IsSolutionUsable();
  result.cost = summary.final_cost;
  result.used_tapm = used_tapm;
  result.used_rtk = used_rtk;
  result.message = summary.BriefReport();

  const Eigen::Vector3d p_last(positions.back()[0], positions.back()[1], positions.back()[2]);
  result.position_last = p_last;
  result.translation = p_last - world_history_window.back().pos;
  result.tau_last =
      (recover_time > 0.0) ? (world_history_window.back().timestamp - recover_time) : 0.0;
  result.weight_tapm_last = weights(result.tau_last)[0];
  result.weight_rtk_last = latest_weight_rtk;
  if (result.latest_rtk_used)
    result.latest_rtk_error = p_last - rtk_current.pos;
  return result;
}
