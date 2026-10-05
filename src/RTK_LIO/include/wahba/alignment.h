#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <common_lib.h>


struct AlignmentMetrics {
    double mean = 0.0;
    double std = 0.0;
    double max = 0.0;
    double min = 0.0;
    double rmse = 0.0;
};


using TrajectoryPoint = Timepos;


struct Transformation
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    double s = 1.0;
    bool valid = true;
};


struct QuasarParams {
    double c_bar = 3.0;
    std::vector<double> sigma;
    bool use_quasar = true;
    double rank_tol = 1e-6;
};


std::vector<TrajectoryPoint> load_trajectory(const std::string& filename);


bool select_corresponding_points_by_time(
    const std::vector<TrajectoryPoint>& traj1,
    const std::vector<TrajectoryPoint>& traj2,
    double time_threshold,
    std::vector<int>& indices1,
    std::vector<int>& indices2
);


Eigen::Matrix3d compute_coarse_alignment(
    const std::vector<TrajectoryPoint>& source_points,
    const std::vector<TrajectoryPoint>& reference_points,
    double velocity_threshold,
    std::vector<int>& valid_indices
);


Transformation compute_fine_alignment(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& reference_points
);


Transformation compute_full_alignment(
    const std::vector<TrajectoryPoint>& source_points,
    const std::vector<TrajectoryPoint>& reference_points,
    double velocity_threshold,
    const Eigen::Vector3d& gravity_source = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& gravity_reference = Eigen::Vector3d::Zero(),
    double gravity_weight = 0.0,
    const Eigen::Matrix3d* fixed_rotation = nullptr
);


void configure_alignment_logging(bool enabled, const std::string& directory);


void configure_alignment_yaw_only(bool enabled);
void configure_alignment_motion_gate(double base_m, double max_speed_mps);
struct VtaDiagnostics
{
    double timestamp = -1.0;
    double sigma1 = 0.0;
    double sigma2 = 0.0;
    double ratio = 0.0;
    double history_sec = 0.0;
    double displacement_m = 0.0;
    int samples = 0;
    bool gravity_used = false;
};
const VtaDiagnostics& alignment_vta_diagnostics();
Transformation gravity_yaw_alignment(const std::vector<Eigen::Vector3d>& source,
    const std::vector<Eigen::Vector3d>& reference, const Eigen::Vector3d& gravity_source);


std::vector<Eigen::Vector3d> apply_transformation(
    const std::vector<Eigen::Vector3d>& points,
    const Transformation& transform
);


AlignmentMetrics compute_alignment_metrics(
    const std::vector<Eigen::Vector3d>& aligned_points,
    const std::vector<Eigen::Vector3d>& reference_points
);


Eigen::Matrix3d wahba_quasar(
    const std::vector<Eigen::Vector3d>& A,
    const std::vector<Eigen::Vector3d>& B,
    const QuasarParams& p);


#endif
