#include "alignment.h"
#include "BSpline.h"
#include "quasar.h"
#include "python_bridge.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <mutex>
#include <atomic>
#include <map>
#include <algorithm>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#ifdef _WIN32
#include <direct.h>
#endif

Eigen::Matrix3d wahba_svd(const std::vector<Eigen::Vector3d>& A, const std::vector<Eigen::Vector3d>& B, const std::vector<double>& weights_vec);
Transformation umeyama_alignment(const std::vector<Eigen::Vector3d>& A_vec, const std::vector<Eigen::Vector3d>& B_vec);
static VtaDiagnostics vta_diagnostics;
const VtaDiagnostics& alignment_vta_diagnostics() { return vta_diagnostics; }

Transformation gravity_yaw_alignment(const std::vector<Eigen::Vector3d>& source,
    const std::vector<Eigen::Vector3d>& reference, const Eigen::Vector3d& gravity_source)
{
    Transformation result;
    result.valid = false;
    if (source.size() < 3 || source.size() != reference.size() || gravity_source.norm() < 1e-9)
        return result;
    const Eigen::Matrix3d level = Eigen::Quaterniond::FromTwoVectors(
        gravity_source.normalized(), -Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < source.size(); ++i) { a += source[i]; b += reference[i]; }
    a /= source.size(); b /= source.size();
    double dot = 0.0, cross = 0.0;
    for (size_t i = 0; i < source.size(); ++i)
    {
        const Eigen::Vector3d x = level * (source[i] - a), y = reference[i] - b;
        dot += x.x() * y.x() + x.y() * y.y();
        cross += x.x() * y.y() - x.y() * y.x();
    }
    if (std::hypot(dot, cross) < 1e-9) return result;
    result.R = Eigen::AngleAxisd(std::atan2(cross, dot), Eigen::Vector3d::UnitZ()).toRotationMatrix() * level;
    result.s = 1.0;
    result.t = b - result.R * a;
    result.valid = result.R.allFinite() && result.t.allFinite();
    return result;
}

namespace {

constexpr double kVelocityNormEpsilon = 1e-6;
constexpr bool kHardGravityConstraint = true;
double g_neighbor_jump_base_m = 1.0;
double g_neighbor_max_speed_mps = 0.0;
constexpr double kTLSPositionSigma = 0.1;
constexpr double kTLSCBarSquared = 11.345;
constexpr int kTLSMaxIterations = 10;
constexpr double kTLSAdaptiveScale = 2.5;

std::mutex g_prior_mutex;
Eigen::Matrix3d g_last_rotation = Eigen::Matrix3d::Identity();
bool g_has_prior_rotation = false;
std::atomic<uint64_t> g_velocity_log_counter{0};
std::mutex g_alignment_log_mutex;
bool g_alignment_log_enabled = false;
std::string g_alignment_log_directory;
std::string g_alignment_run_directory;
std::string g_alignment_source_file;
std::string g_alignment_reference_file;
std::string g_alignment_aligned_file;
double g_alignment_last_source_ts = -std::numeric_limits<double>::infinity();
double g_alignment_last_reference_ts = -std::numeric_limits<double>::infinity();
double g_alignment_last_aligned_ts = -std::numeric_limits<double>::infinity();

Eigen::Matrix3d fetch_prior_rotation()
{
    std::lock_guard<std::mutex> lock(g_prior_mutex);
    return g_has_prior_rotation ? g_last_rotation : Eigen::Matrix3d::Identity();
}

void store_prior_rotation(const Eigen::Matrix3d& R)
{
    std::lock_guard<std::mutex> lock(g_prior_mutex);
    g_last_rotation = R;
    g_has_prior_rotation = true;
}


bool g_yaw_only_enabled = true;


Eigen::Matrix3d project_rotation_to_yaw(const Eigen::Matrix3d& R)
{
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    Eigen::Matrix3d R_yaw;
    R_yaw << c, -s, 0.0,
             s,  c, 0.0,
             0.0, 0.0, 1.0;
    return R_yaw;
}


void apply_yaw_only_constraint(Transformation& T,
                               const Eigen::Vector3d& mean_source,
                               const Eigen::Vector3d& mean_reference)
{
    if (!g_yaw_only_enabled) {
        return;
    }
    T.R = project_rotation_to_yaw(T.R);
    T.t = mean_reference - T.s * (T.R * mean_source);
}

bool directory_exists(const std::string& dir)
{
    if (dir.empty()) return false;
    struct stat info;
    return (stat(dir.c_str(), &info) == 0) && (info.st_mode & S_IFDIR);
}

bool ensure_directory(const std::string& dir)
{
    if (dir.empty()) return false;
    if (directory_exists(dir)) return true;
#ifdef _WIN32
    return (_mkdir(dir.c_str()) == 0) || errno == EEXIST;
#else
    return (::mkdir(dir.c_str(), 0755) == 0) || errno == EEXIST;
#endif
}

uint64_t g_alignment_sequence_counter = 0;

std::string create_alignment_run_directory(const std::string& base_dir)
{
    if (base_dir.empty()) return "";
    uint64_t seq = g_alignment_sequence_counter;
    while (true)
    {
        std::ostringstream oss;
        oss << base_dir << "/alignment_" << std::setw(6) << std::setfill('0') << seq;
        std::string run_dir = oss.str();
        if (!directory_exists(run_dir))
        {
            if (!ensure_directory(run_dir))
            {
                return "";
            }
            g_alignment_sequence_counter = seq + 1;
            return run_dir;
        }
        ++seq;
    }
}

void write_velocity_debug_json(const std::string& label,
                               const std::vector<Timepos>& points,
                               const std::vector<Eigen::Vector3d>& velocities,
                               const std::vector<Eigen::Vector3d>& spline_positions)
{
    if (!BSpline::debug_logging_enabled()) return;
    if (points.size() != velocities.size() || points.empty()) return;

    const std::string dir = std::string(ROOT_DIR) + "data";
    if (!ensure_directory(dir)) return;
    const uint64_t log_id = g_velocity_log_counter.fetch_add(1);
    std::ostringstream filepath;
    filepath << dir << "/velocity_debug_" << std::setw(6) << std::setfill('0') << log_id;
    if (!label.empty())
    {
        filepath << "_" << label;
    }
    filepath << ".json";

    std::ofstream ofs(filepath.str());
    if (!ofs.is_open()) return;

    double time_min = points.front().timestamp;
    double time_max = points.front().timestamp;
    for (const auto& pt : points)
    {
        time_min = std::min(time_min, pt.timestamp);
        time_max = std::max(time_max, pt.timestamp);
    }
    double time_range = std::max(1e-9, time_max - time_min);

    ofs << std::setprecision(15);
    ofs << "{\n";
    ofs << "  \"label\": \"" << label << "\",\n";
    ofs << "  \"call_index\": " << log_id << ",\n";
    ofs << "  \"time_min\": " << time_min << ",\n";
    ofs << "  \"time_range\": " << time_range << ",\n";

    auto write_points = [&](const char* key, bool use_spline)
    {
        ofs << "  \"" << key << "\": [\n";
        for (size_t i = 0; i < points.size(); ++i)
        {
            const Eigen::Vector3d& pos = (use_spline && i < spline_positions.size())
                                             ? spline_positions[i]
                                             : points[i].pos;
            ofs << "    {\"timestamp\": " << points[i].timestamp
                << ", \"position\": [" << pos.x()
                << "," << pos.y()
                << "," << pos.z() << "]}";
            if (i + 1 != points.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ],\n";
    };

    const bool has_spline = !spline_positions.empty() && spline_positions.size() == points.size();
    write_points("raw_points", false);
    write_points("spline_samples", has_spline);

    ofs << "  \"input_velocities\": [\n";
    for (size_t i = 0; i < points.size(); ++i)
    {
        const auto& v = velocities[i];
        ofs << "    {\"timestamp\": " << points[i].timestamp
            << ", \"position\": [" << points[i].pos.x()
            << "," << points[i].pos.y()
            << "," << points[i].pos.z() << "]"
            << ", \"velocity\": [" << v.x() << "," << v.y() << "," << v.z() << "]"
            << ", \"speed\": " << v.norm() << "}";
        if (i + 1 != points.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

void append_sample_line(const std::string& path,
                        double timestamp,
                        const Eigen::Vector3d& pos,
                        double& last_timestamp)
{
    if (timestamp <= last_timestamp) return;
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open())
    {
        return;
    }
    ofs << std::setprecision(15)
        << timestamp << " "
        << pos.x() << " "
        << pos.y() << " "
        << pos.z() << "\n";
    last_timestamp = timestamp;
}

void log_alignment_result(const std::vector<Timepos>& source_points,
                          const std::vector<Timepos>& reference_points,
                          const Transformation& transform)
{
    if (source_points.empty() || reference_points.empty()) return;

    std::string source_file;
    std::string reference_file;
    std::string aligned_file;
    std::lock_guard<std::mutex> lock(g_alignment_log_mutex);
    if (!g_alignment_log_enabled || g_alignment_source_file.empty())
    {
        return;
    }
    source_file = g_alignment_source_file;
    reference_file = g_alignment_reference_file;
    aligned_file = g_alignment_aligned_file;

    const Timepos& src_last = source_points.back();
    const Timepos& ref_last = reference_points.back();
    Eigen::Vector3d aligned_pos = transform.s * (transform.R * src_last.pos) + transform.t;

    append_sample_line(source_file, src_last.timestamp, src_last.pos, g_alignment_last_source_ts);
    append_sample_line(reference_file, ref_last.timestamp, ref_last.pos, g_alignment_last_reference_ts);
    append_sample_line(aligned_file, src_last.timestamp, aligned_pos, g_alignment_last_aligned_ts);
}

}

void configure_alignment_logging(bool enabled, const std::string& directory)
{
    std::lock_guard<std::mutex> lock(g_alignment_log_mutex);
    g_alignment_log_enabled = false;
    g_alignment_log_directory.clear();
    g_alignment_run_directory.clear();
    g_alignment_source_file.clear();
    g_alignment_reference_file.clear();
    g_alignment_aligned_file.clear();
    g_alignment_last_source_ts = -std::numeric_limits<double>::infinity();
    g_alignment_last_reference_ts = -std::numeric_limits<double>::infinity();
    g_alignment_last_aligned_ts = -std::numeric_limits<double>::infinity();
    (void)enabled;
    (void)directory;
}

void configure_alignment_yaw_only(bool enabled)
{
    g_yaw_only_enabled = enabled;
}

void configure_alignment_motion_gate(double base_m, double max_speed_mps)
{
    g_neighbor_jump_base_m = std::max(0.0, base_m);
    g_neighbor_max_speed_mps = std::max(0.0, max_speed_mps);
}


std::vector<TrajectoryPoint> load_trajectory(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::vector<TrajectoryPoint> trajectory;
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        TrajectoryPoint p;
        if (!(ss >> p.timestamp >> p.pos.x() >> p.pos.y() >> p.pos.z()))
        {
            continue;
        }
        trajectory.push_back(p);
    }
    return trajectory;
}

bool select_corresponding_points_by_time(const std::vector<TrajectoryPoint>& traj1,
                                        const std::vector<TrajectoryPoint>& traj2,
                                        double time_threshold, std::vector<int>& indices1,
                                        std::vector<int>& indices2)
{
    indices1.clear();
    indices2.clear();
    for (size_t i = 0; i < traj1.size(); ++i) {
        double min_time_diff = std::numeric_limits<double>::max();
        int best_match_idx = -1;
        for (size_t j = 0; j < traj2.size(); ++j) {
            double time_diff = std::abs(traj1[i].timestamp - traj2[j].timestamp);
            if (time_diff < min_time_diff) {
                min_time_diff = time_diff;
                best_match_idx = j;
            }
        }
        if (best_match_idx != -1 && min_time_diff <= time_threshold) {
            indices1.push_back(i);
            indices2.push_back(best_match_idx);
        }
    }
    return true;
}

std::vector<Eigen::Vector3d> apply_transformation(
    const std::vector<Eigen::Vector3d>& points, const Transformation& transform) {
    std::vector<Eigen::Vector3d> transformed_points;
    transformed_points.reserve(points.size());
    for (const auto& p : points) {
        transformed_points.push_back(transform.s * (transform.R * p) + transform.t);
    }
    return transformed_points;
}

Eigen::Matrix3d compute_coarse_alignment(
    const std::vector<TrajectoryPoint>& source_points,
    const std::vector<TrajectoryPoint>& reference_points,
    double velocity_threshold,
    std::vector<int>& valid_indices)
{
    valid_indices.clear();

    uint64_t source_record_id = BSpline::INVALID_RECORD_ID;
    uint64_t ref_record_id = BSpline::INVALID_RECORD_ID;
    auto vel_source = BSpline::compute_velocities(source_points, "source", &source_record_id);
    auto vel_ref = BSpline::compute_velocities(reference_points, "reference", &ref_record_id);

    std::vector<Eigen::Vector3d> vel_source_valid, vel_ref_valid;
    std::vector<double> weights;

    for (size_t i = 0; i < source_points.size(); ++i)
    {
        double norm_source = vel_source[i].norm();
        double norm_ref = vel_ref[i].norm();
        if (norm_source >= velocity_threshold && norm_ref >= velocity_threshold)
        {
            vel_source_valid.push_back(vel_source[i]);
            vel_ref_valid.push_back(vel_ref[i]);
            weights.push_back(norm_source * norm_ref);
            valid_indices.push_back(i);
        }
    }

    if (vel_source_valid.size() < 3)
    {
        BSpline::finalize_debug_record(source_record_id, false);
        BSpline::finalize_debug_record(ref_record_id, false);
        return Eigen::Matrix3d::Identity();
    }

    BSpline::finalize_debug_record(source_record_id, true);
    BSpline::finalize_debug_record(ref_record_id, true);

    return wahba_svd(vel_source_valid, vel_ref_valid, weights);
}


Transformation compute_fine_alignment(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& reference_points)
{
    return umeyama_alignment(source_points, reference_points);
}

Transformation compute_full_alignment(
    const std::vector<TrajectoryPoint>& source_points,
    const std::vector<TrajectoryPoint>& reference_points,
    double velocity_threshold,
    const Eigen::Vector3d& gravity_source,
    const Eigen::Vector3d& gravity_reference,
    double gravity_weight,
    const Eigen::Matrix3d* fixed_rotation)
{
    Transformation invalid_result;
    invalid_result.valid = false;
    vta_diagnostics = VtaDiagnostics();
    if (!source_points.empty())
    {
        vta_diagnostics.timestamp = source_points.back().timestamp;
        vta_diagnostics.history_sec = source_points.back().timestamp - source_points.front().timestamp;
        vta_diagnostics.displacement_m = (source_points.back().pos - source_points.front().pos).norm();
    }

    if (source_points.size() < 3 || reference_points.size() < 3) {
        return invalid_result;
    }

    const size_t sample_count_pos = std::min(source_points.size(), reference_points.size());
    if (fixed_rotation != nullptr)
    {
        if (sample_count_pos < 3)
        {
            return invalid_result;
        }

        Transformation T_final;
        T_final.R = *fixed_rotation;
        T_final.s = 1.0;
        Eigen::Vector3d mean_source = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_reference = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < sample_count_pos; ++i)
        {
            mean_source += source_points[i].pos;
            mean_reference += reference_points[i].pos;
        }
        mean_source /= static_cast<double>(sample_count_pos);
        mean_reference /= static_cast<double>(sample_count_pos);
        T_final.t = mean_reference - T_final.R * mean_source;
        T_final.valid = true;
        apply_yaw_only_constraint(T_final, mean_source, mean_reference);
        store_prior_rotation(T_final.R);
        log_alignment_result(source_points, reference_points, T_final);
        return T_final;
    }

    auto& bridge = WahbaPythonBridge::instance();
    const double smoothing_factor = BSpline::get_smoothing_factor();

    struct VelocityComputationCache {
        std::vector<Eigen::Vector3d> velocities;
        std::vector<Eigen::Vector3d> spline_positions;
        bool used_python = false;
    };

    auto compute_velocity_with_fallback =
        [&](const std::vector<Timepos>& points,
            const std::string& label,
            VelocityComputationCache& cache)
        {
            cache.velocities.clear();
            cache.spline_positions.clear();
            cache.used_python = false;
            std::string error;
            if (bridge.available() &&
                bridge.compute_bspline_velocities(points,
                                                  cache.velocities,
                                                  &cache.spline_positions,
                                                  label,
                                                  smoothing_factor,
                                                  &error))
            {
                cache.used_python = true;
            }
            else
            {
                if (!error.empty()) {
                }
                uint64_t record_id = BSpline::INVALID_RECORD_ID;
                cache.velocities = BSpline::compute_velocities(points, label, &record_id);
                BSpline::finalize_debug_record(record_id, true);
            }
        };

    VelocityComputationCache source_cache;
    VelocityComputationCache reference_cache;
    compute_velocity_with_fallback(source_points, "source", source_cache);
    compute_velocity_with_fallback(reference_points, "reference", reference_cache);

    if (source_cache.used_python) {
        write_velocity_debug_json("source", source_points, source_cache.velocities, source_cache.spline_positions);
    }
    if (reference_cache.used_python) {
        write_velocity_debug_json("reference", reference_points, reference_cache.velocities, reference_cache.spline_positions);
    }

    const std::vector<Eigen::Vector3d>& vel_source = source_cache.velocities;
    const std::vector<Eigen::Vector3d>& vel_reference = reference_cache.velocities;

    const size_t sample_count = std::min(
        sample_count_pos,
        std::min(vel_source.size(), vel_reference.size()));
    if (sample_count < 3) {
        return invalid_result;
    }

    std::vector<int> valid_indices;
    std::vector<Eigen::Vector3d> vel_source_valid;
    std::vector<Eigen::Vector3d> vel_reference_valid;
    std::vector<double> weights;
    valid_indices.reserve(sample_count);
    vel_source_valid.reserve(sample_count);
    vel_reference_valid.reserve(sample_count);
    weights.reserve(sample_count);

    for (size_t i = 0; i < sample_count; ++i)
    {
        double norm_source = vel_source[i].norm();
        double norm_reference = vel_reference[i].norm();
        if (norm_source >= velocity_threshold && norm_reference >= velocity_threshold &&
            norm_source > kVelocityNormEpsilon && norm_reference > kVelocityNormEpsilon)
        {
            valid_indices.push_back(static_cast<int>(i));
            vel_source_valid.push_back(vel_source[i]);
            vel_reference_valid.push_back(vel_reference[i]);
            weights.push_back(norm_source * norm_reference);
        }
    }


    if (valid_indices.size() >= 2)
    {
        std::vector<int> filtered_indices;
        filtered_indices.reserve(valid_indices.size());
        for (int idx : valid_indices)
        {
            if (idx <= 0)
            {
                filtered_indices.push_back(idx);
                continue;
            }
            const Eigen::Vector3d& src_curr = source_points[idx].pos;
            const Eigen::Vector3d& src_prev = source_points[idx - 1].pos;
            const Eigen::Vector3d& ref_curr = reference_points[idx].pos;
            const Eigen::Vector3d& ref_prev = reference_points[idx - 1].pos;
            const double src_jump = (src_curr - src_prev).norm();
            const double ref_jump = (ref_curr - ref_prev).norm();
            const double dt = std::max(0.0, source_points[idx].timestamp - source_points[idx - 1].timestamp);
            const double limit = g_neighbor_jump_base_m + g_neighbor_max_speed_mps * dt;
            if (src_jump <= limit && ref_jump <= limit)
            {
                filtered_indices.push_back(idx);
            }
        }
        valid_indices.swap(filtered_indices);
        vel_source_valid.clear();
        vel_reference_valid.clear();
        weights.clear();
        vel_source_valid.reserve(valid_indices.size());
        vel_reference_valid.reserve(valid_indices.size());
        weights.reserve(valid_indices.size());
        for (int idx : valid_indices)
        {
            const double norm_source = vel_source[idx].norm();
            const double norm_reference = vel_reference[idx].norm();
            vel_source_valid.push_back(vel_source[idx]);
            vel_reference_valid.push_back(vel_reference[idx]);
            weights.push_back(norm_source * norm_reference);
        }
    }

    if (valid_indices.size() < 3)
    {
        return invalid_result;
    }

    std::vector<Eigen::Vector3d> wahba_source = vel_source_valid;
    std::vector<Eigen::Vector3d> wahba_reference = vel_reference_valid;
    std::vector<double> wahba_weights = weights;
    Eigen::Matrix3d velocity_information = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < weights.size(); ++i)
        velocity_information += weights[i] * vel_reference_valid[i] * vel_source_valid[i].transpose();
    const Eigen::Vector3d singular = velocity_information.jacobiSvd().singularValues();
    vta_diagnostics.sigma1 = singular[0];
    vta_diagnostics.sigma2 = singular[1];
    vta_diagnostics.ratio = singular[0] > 1e-12 ? singular[1] / singular[0] : 0.0;
    vta_diagnostics.samples = static_cast<int>(weights.size());

    if (gravity_weight > 0.0)
    {
        const double norm_source_g = gravity_source.norm();
        const double norm_ref_g = gravity_reference.norm();
        if (norm_source_g > kVelocityNormEpsilon && norm_ref_g > kVelocityNormEpsilon)
        {
            wahba_source.push_back(gravity_source / norm_source_g);
            wahba_reference.push_back(gravity_reference / norm_ref_g);
            wahba_weights.push_back(gravity_weight);
            vta_diagnostics.gravity_used = true;
        }
    }

    if (wahba_source.size() < 3)
    {
        return invalid_result;
    }

    Eigen::Matrix3d R_coarse = wahba_svd(wahba_source, wahba_reference, wahba_weights);
    if (kHardGravityConstraint && gravity_weight > 0.0)
    {
        if (gravity_source.norm() <= kVelocityNormEpsilon || gravity_reference.norm() <= kVelocityNormEpsilon)
            return invalid_result;
        const Eigen::Vector3d up = gravity_reference.normalized();
        const Eigen::Matrix3d level = Eigen::Quaterniond::FromTwoVectors(
            gravity_source.normalized(), up).toRotationMatrix();
        Eigen::Matrix3d skew;
        skew << 0.0, -up.z(), up.y(), up.z(), 0.0, -up.x(), -up.y(), up.x(), 0.0;
        const Eigen::Matrix3d rotated = level * velocity_information.transpose();
        const double cosine = ((Eigen::Matrix3d::Identity() - up * up.transpose()) * rotated).trace();
        const double sine = (skew * rotated).trace();
        if (std::hypot(cosine, sine) <= 1e-12 * std::max(1.0, rotated.norm()))
            return invalid_result;
        R_coarse = Eigen::AngleAxisd(std::atan2(sine, cosine), up).toRotationMatrix() * level;
    }
    Eigen::Matrix3d R_prior = fetch_prior_rotation();
    Eigen::Matrix3d R_stage = R_coarse;

    if (bridge.available())
    {
        std::vector<Eigen::Vector3d> lio_points_for_tls;
        std::vector<Eigen::Vector3d> rtk_points_for_tls;
        lio_points_for_tls.reserve(valid_indices.size());
        rtk_points_for_tls.reserve(valid_indices.size());
        for (int idx : valid_indices)
        {
            lio_points_for_tls.push_back(source_points[idx].pos);
            rtk_points_for_tls.push_back(reference_points[idx].pos);
        }

        Eigen::Matrix3d refined_rotation = R_stage;
        std::vector<bool> tls_mask;
        std::map<std::string, double> tls_stats;
        std::string tls_error;
        bool tls_ok = bridge.tls_refine(
            lio_points_for_tls,
            rtk_points_for_tls,
            R_prior * R_coarse,
            kTLSPositionSigma,
            kTLSCBarSquared,
            kTLSMaxIterations,
            true,
            kTLSAdaptiveScale,
            refined_rotation,
            tls_mask,
            &tls_stats,
            &tls_error);
        if (tls_ok)
        {
            R_stage = refined_rotation;
            if (!tls_mask.empty() && tls_mask.size() == valid_indices.size())
            {
                std::vector<int> filtered_indices;
                filtered_indices.reserve(valid_indices.size());
                for (size_t i = 0; i < tls_mask.size(); ++i)
                {
                    if (tls_mask[i])
                    {
                        filtered_indices.push_back(valid_indices[i]);
                    }
                }
                if (filtered_indices.size() >= 3)
                {
                    valid_indices.swap(filtered_indices);
                }
                else
                {
                }
            }
            auto iter_it = tls_stats.find("iterations");
            auto rmse_improve = tls_stats.find("rmse_improvement");
            if (iter_it != tls_stats.end() || rmse_improve != tls_stats.end())
            {
            }
        }
        else if (!tls_error.empty())
        {
        }
    }

    if (valid_indices.size() < 3) {
        return invalid_result;
    }

    std::vector<Eigen::Vector3d> source_for_fine;
    std::vector<Eigen::Vector3d> ref_for_fine;
    source_for_fine.reserve(sample_count);
    ref_for_fine.reserve(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        source_for_fine.push_back(R_stage * source_points[i].pos);
        ref_for_fine.push_back(reference_points[i].pos);
    }

    if (source_for_fine.size() < 3 || ref_for_fine.size() < 3) {
        return invalid_result;
    }

    Transformation T_fine = compute_fine_alignment(source_for_fine, ref_for_fine);
    Transformation T_final;
    T_final.R = T_fine.R * R_stage;
    T_final.t = T_fine.t;
    T_final.s = T_fine.s;
    T_final.valid = true;


    {
        Eigen::Vector3d mean_source = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_reference = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < sample_count; ++i) {
            mean_source += source_points[i].pos;
            mean_reference += reference_points[i].pos;
        }
        mean_source /= static_cast<double>(sample_count);
        mean_reference /= static_cast<double>(sample_count);
        if (g_yaw_only_enabled)
        {
            std::vector<Eigen::Vector3d> original_source;
            original_source.reserve(sample_count);
            for (size_t i = 0; i < sample_count; ++i) original_source.push_back(source_points[i].pos);

            T_final = gravity_yaw_alignment(original_source, ref_for_fine,
                gravity_weight > 0.0 ? gravity_source : -Eigen::Vector3d::UnitZ());
        }
    }

    store_prior_rotation(T_final.R);
    log_alignment_result(source_points, reference_points, T_final);
    return T_final;
}

AlignmentMetrics compute_alignment_metrics(const std::vector<Eigen::Vector3d>& aligned_points, const std::vector<Eigen::Vector3d>& reference_points)
{
    AlignmentMetrics metrics;
    if (aligned_points.size() != reference_points.size() || aligned_points.empty()) {
        return metrics;
    }

    std::vector<double> errors;
    double sum_sq_err = 0.0;
    double sum_err = 0.0;

    for (size_t i = 0; i < aligned_points.size(); ++i)
    {
        double error = (aligned_points[i] - reference_points[i]).norm();
        errors.push_back(error);
        sum_sq_err += error * error;
        sum_err += error;
    }

    size_t n = errors.size();
    metrics.mean = sum_err / n;
    metrics.rmse = std::sqrt(sum_sq_err / n);
    metrics.min = *std::min_element(errors.begin(), errors.end());
    metrics.max = *std::max_element(errors.begin(), errors.end());

    double sum_std_dev = 0.0;
    for (double err : errors)
    {
        sum_std_dev += (err - metrics.mean) * (err - metrics.mean);
    }
    metrics.std = std::sqrt(sum_std_dev / n);

    return metrics;
}


Eigen::Matrix3d wahba_svd(const std::vector<Eigen::Vector3d>& A, const std::vector<Eigen::Vector3d>& B, const std::vector<double>& weights_vec) {
    if (A.empty() || B.empty()) {
        throw std::runtime_error("Input vectors for Wahba are empty.");
    }

    Eigen::VectorXd weights;
    if (weights_vec.empty()) {
        weights = Eigen::VectorXd::Ones(A.size());
    } else {
        weights = Eigen::Map<const Eigen::VectorXd>(weights_vec.data(), weights_vec.size());
    }

    double w_sum = weights.sum();
    if (w_sum <= 0) throw std::runtime_error("Sum of weights must be positive.");
    weights /= w_sum;


    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < A.size(); ++i) {
        H += weights(i) * B[i] * A[i].transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = U * V.transpose();

    if (R.determinant() < 0) {
        U.col(2) *= -1;
        R = U * V.transpose();
    }
    return R;
}


Transformation umeyama_alignment(const std::vector<Eigen::Vector3d>& A_vec, const std::vector<Eigen::Vector3d>& B_vec) {
    if (A_vec.size() < 3) {
        throw std::runtime_error("Umeyama requires at least 3 point pairs.");
    }
    Eigen::MatrixXd A(3, A_vec.size()), B(3, B_vec.size());
    for(size_t i=0; i<A_vec.size(); ++i) {
        A.col(i) = A_vec[i];
        B.col(i) = B_vec[i];
    }

    Eigen::Vector3d A_mean = A.rowwise().mean();
    Eigen::Vector3d B_mean = B.rowwise().mean();
    Eigen::MatrixXd A_centered = A.colwise() - A_mean;
    Eigen::MatrixXd B_centered = B.colwise() - B_mean;
    Eigen::Matrix3d H = (A_centered * B_centered.transpose()) / A.cols();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d S_diag = Eigen::Matrix3d::Identity();
    if ((V * U.transpose()).determinant() < 0) {
        S_diag(2, 2) = -1;
    }

    Transformation result;
    result.R = V * S_diag * U.transpose();
    result.s = 1.0;
    result.t = B_mean - result.R * A_mean;
    return result;
}
