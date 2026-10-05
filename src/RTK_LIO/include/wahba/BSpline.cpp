
#include "BSpline.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <cerrno>
#include <unordered_map>
#include <unsupported/Eigen/Splines>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif

BSpline::BSpline() {}
BSpline::~BSpline() {}

namespace {

std::atomic<uint64_t> g_debug_call_counter{0};
bool g_debug_logging_enabled = false;
std::string g_debug_output_dir;
std::mutex g_debug_mutex;
std::atomic<double> g_bspline_smoothing_factor{0.0};

struct DebugRecord
{
    uint64_t id = BSpline::INVALID_RECORD_ID;
    std::string label;
    std::vector<TrajectoryPoint> raw_points;
    std::vector<double> normalized_unique_times;
    std::vector<Eigen::Vector3d> unique_positions;
    std::vector<Eigen::Vector3d> velocities;
    std::vector<Eigen::Vector3d> spline_positions;
    std::vector<Eigen::Vector3d> spline_velocities;
    double min_time = 0.0;
    double time_range = 1.0;
};

std::unordered_map<uint64_t, DebugRecord> g_pending_records;

std::string default_debug_dir()
{
    static const std::string dir = "RTK_LIO/data";
    return dir;
}

bool directory_exists(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

std::string normalize_path(const std::string& path)
{
    std::string normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

bool ensure_directory(const std::string& path)
{
    if (path.empty()) return false;
    std::string normalized = normalize_path(path);
    if (normalized.empty()) return false;

    auto create_if_needed = [](const std::string& dir)
    {
        if (dir.empty() || directory_exists(dir)) return true;
#ifdef _WIN32
        if (_mkdir(dir.c_str()) != 0 && errno != EEXIST)
        {
            return false;
        }
#else
        if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST)
        {
            return false;
        }
#endif
        return true;
    };

    std::string current;
    if (!normalized.empty() && normalized[0] == '/')
    {
        current = "/";
    }

    std::stringstream ss(normalized);
    std::string segment;
    while (std::getline(ss, segment, '/'))
    {
        if (segment.empty()) continue;
        if (!current.empty() && current.back() != '/')
        {
            current += "/";
        }
        current += segment;
        if (!create_if_needed(current))
        {
            return false;
        }
    }

    return directory_exists(normalized);
}

std::string vec_to_string(const Eigen::Vector3d& v)
{
    std::ostringstream oss;
    oss << "[" << std::setprecision(15) << v.x() << "," << v.y() << "," << v.z() << "]";
    return oss.str();
}

void write_debug_json(const DebugRecord& record)
{
    if (!g_debug_logging_enabled) return;

    std::string dir;
    {
        std::lock_guard<std::mutex> lock(g_debug_mutex);
        dir = g_debug_output_dir.empty() ? default_debug_dir() : g_debug_output_dir;
    }
    if (!ensure_directory(dir))
    {
        return;
    }

    std::ostringstream filepath;
    filepath << dir << "/velocity_debug_" << std::setw(6) << std::setfill('0') << record.id;
    if (!record.label.empty()) filepath << "_" << record.label;
    filepath << ".json";

    std::ofstream ofs(filepath.str());
    if (!ofs.is_open())
    {
        return;
    }

    ofs << std::setprecision(15);
    ofs << "{\n";
    ofs << "  \"label\": \"" << record.label << "\",\n";
    ofs << "  \"call_index\": " << record.id << ",\n";
    ofs << "  \"time_min\": " << record.min_time << ",\n";
    ofs << "  \"time_range\": " << record.time_range << ",\n";

    ofs << "  \"raw_points\": [\n";
    for (size_t i = 0; i < record.raw_points.size(); ++i)
    {
        ofs << "    {\"timestamp\": " << record.raw_points[i].timestamp
            << ", \"position\": " << vec_to_string(record.raw_points[i].pos) << "}";
        if (i + 1 != record.raw_points.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"unique_points\": [\n";
    for (size_t i = 0; i < record.normalized_unique_times.size(); ++i)
    {
        double ts = record.min_time + record.normalized_unique_times[i] * record.time_range;
        ofs << "    {\"timestamp\": " << ts
            << ", \"normalized_param\": " << record.normalized_unique_times[i]
            << ", \"position\": " << vec_to_string(record.unique_positions[i]) << "}";
        if (i + 1 != record.normalized_unique_times.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"input_velocities\": [\n";
    for (size_t i = 0; i < record.velocities.size(); ++i)
    {
        ofs << "    {\"timestamp\": " << record.raw_points[i].timestamp
            << ", \"position\": " << vec_to_string(record.raw_points[i].pos)
            << ", \"velocity\": " << vec_to_string(record.velocities[i])
            << ", \"speed\": " << record.velocities[i].norm() << "}";
        if (i + 1 != record.velocities.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"spline_samples\": [\n";
    for (size_t i = 0; i < record.spline_positions.size(); ++i)
    {
        double tau = record.spline_velocities.empty() ? 0.0 : static_cast<double>(i) / \
                        static_cast<double>(record.spline_positions.size() - 1);
        double ts = record.min_time + tau * record.time_range;
        ofs << "    {\"timestamp\": " << ts
            << ", \"normalized_param\": " << tau
            << ", \"position\": " << vec_to_string(record.spline_positions[i])
            << ", \"velocity\": " << vec_to_string(record.spline_velocities[i])
            << ", \"speed\": " << record.spline_velocities[i].norm() << "}";
        if (i + 1 != record.spline_positions.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

}


double BSpline::F03(double t) { return 1.0/6 * (-t*t*t + 3*t*t - 3*t + 1); }
double BSpline::F13(double t) { return 1.0/6 * (3*t*t*t - 6*t*t + 4); }
double BSpline::F23(double t) { return 1.0/6 * (-3*t*t*t + 3*t*t + 3*t + 1); }
double BSpline::F33(double t) { return 1.0/6 * t*t*t; }


double BSpline::dF03(double t) { return 1.0/6 * (-3*t*t + 6*t - 3); }
double BSpline::dF13(double t) { return 1.0/6 * (9*t*t - 12*t); }
double BSpline::dF23(double t) { return 1.0/6 * (-9*t*t + 6*t + 3); }
double BSpline::dF33(double t) { return 1.0/6 * (3*t*t); }


std::vector<Eigen::Vector3d> compute_velocities_finite_diff(
    const std::vector<TrajectoryPoint>& points)
{
    std::vector<Eigen::Vector3d> velocities(points.size(), Eigen::Vector3d::Zero());
    if (points.size() < 2) return velocities;

    for (size_t i = 0; i < points.size() - 1; ++i)
    {
        double dt = points[i + 1].timestamp - points[i].timestamp;
        if (dt > 1e-6)
        {
            velocities[i] = (points[i + 1].pos - points[i].pos) / dt;
        }
    }
    velocities.back() = velocities.size() > 1 ?
        velocities[velocities.size() - 2] : Eigen::Vector3d::Zero();
    return velocities;
}

void BSpline::configure_debug_logging(bool enabled, const std::string& directory)
{
    std::lock_guard<std::mutex> lock(g_debug_mutex);
    g_debug_logging_enabled = enabled;
    g_debug_output_dir = directory;
    if (g_debug_logging_enabled)
    {
        std::string target = g_debug_output_dir.empty() ? default_debug_dir() : g_debug_output_dir;
        if (ensure_directory(target))
        {
            g_debug_output_dir = target;
        }
        else
        {
            g_debug_logging_enabled = false;
        }
    }
}

void BSpline::set_smoothing_factor(double factor)
{
    g_bspline_smoothing_factor.store(clamp_smoothing_factor(factor));
}

bool BSpline::debug_logging_enabled()
{
    return g_debug_logging_enabled;
}

double BSpline::get_smoothing_factor()
{
    return g_bspline_smoothing_factor.load();
}

void BSpline::finalize_debug_record(uint64_t record_id, bool keep_record)
{
    if (record_id == INVALID_RECORD_ID) return;
    DebugRecord record;
    bool has_record = false;
    {
        std::lock_guard<std::mutex> lock(g_debug_mutex);
        auto it = g_pending_records.find(record_id);
        if (it != g_pending_records.end())
        {
            record = std::move(it->second);
            g_pending_records.erase(it);
            has_record = true;
        }
    }

    if (!has_record) return;
    if (keep_record && g_debug_logging_enabled)
    {
        write_debug_json(record);
    }
}

std::vector<Eigen::Vector3d> BSpline::compute_velocities(
    const std::vector<TrajectoryPoint>& points,
    const std::string& debug_label,
    uint64_t* debug_record_id)
{
    if (points.size() < 4)
    {
        if (debug_record_id) *debug_record_id = INVALID_RECORD_ID;
        return compute_velocities_finite_diff(points);
    }

    struct OrderedPoint
    {
        double timestamp;
        Eigen::Vector3d position;
    };

    std::vector<OrderedPoint> ordered(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        ordered[i].timestamp = points[i].timestamp;
        ordered[i].position = points[i].pos;
    }
    std::sort(ordered.begin(), ordered.end(), [](const OrderedPoint& a, const OrderedPoint& b) {
        return a.timestamp < b.timestamp;
    });

    std::vector<double> unique_times;
    std::vector<Eigen::Vector3d> unique_positions;
    unique_times.reserve(ordered.size());
    unique_positions.reserve(ordered.size());

    auto flush_group = [&](double time, const Eigen::Vector3d& accum, int count)
    {
        unique_times.push_back(time);
        unique_positions.push_back(accum / static_cast<double>(count));
    };

    double current_time = ordered.front().timestamp;
    Eigen::Vector3d accum = ordered.front().position;
    int count = 1;
    constexpr double kTimeEps = 1e-9;
    for (size_t i = 1; i < ordered.size(); ++i)
    {
        if (std::abs(ordered[i].timestamp - current_time) < kTimeEps)
        {
            accum += ordered[i].position;
            ++count;
        }
        else
        {
            flush_group(current_time, accum, count);
            current_time = ordered[i].timestamp;
            accum = ordered[i].position;
            count = 1;
        }
    }
    flush_group(current_time, accum, count);

    if (unique_times.size() < 4)
    {
        if (debug_record_id) *debug_record_id = INVALID_RECORD_ID;
        return compute_velocities_finite_diff(points);
    }

    using Spline3d = Eigen::Spline<double, 3>;
    Eigen::Matrix<double, 3, Eigen::Dynamic> pts(3, unique_times.size());
    Spline3d::ParameterVectorType params(unique_times.size());
    const double min_time = unique_times.front();
    const double max_time = unique_times.back();
    const double time_range = max_time - min_time;
    if (time_range < 1e-6)
    {
        if (debug_record_id) *debug_record_id = INVALID_RECORD_ID;
        return compute_velocities_finite_diff(points);
    }
    std::vector<Eigen::Vector3d> control_positions = unique_positions;
    double smoothing_factor = g_bspline_smoothing_factor.load();
    smoothing_factor = clamp_smoothing_factor(smoothing_factor);
    if (smoothing_factor > 0.0 && control_positions.size() > 2)
    {
        double alpha = smoothing_factor;
        double keep = 1.0 - alpha;
        std::vector<Eigen::Vector3d> forward = control_positions;
        std::vector<Eigen::Vector3d> backward = control_positions;
        for (size_t i = 1; i < control_positions.size(); ++i)
        {
            forward[i] = alpha * forward[i - 1] + keep * control_positions[i];
        }
        for (size_t idx = control_positions.size() - 1; idx > 0; --idx)
        {
            const size_t cur = idx - 1;
            backward[cur] = alpha * backward[cur + 1] + keep * control_positions[cur];
        }
        for (size_t i = 0; i < control_positions.size(); ++i)
        {
            control_positions[i] = 0.5 * (forward[i] + backward[i]);
        }
    }

    for (size_t i = 0; i < unique_times.size(); ++i)
    {
        pts.col(i) = control_positions[i];
        params(i) = (unique_times[i] - min_time) / time_range;
    }

    Spline3d spline;
    try
    {
        spline = Eigen::SplineFitting<Spline3d>::Interpolate(pts, 3, params);
    }
    catch (const std::exception& e)
    {
        if (debug_record_id) *debug_record_id = INVALID_RECORD_ID;
        return compute_velocities_finite_diff(points);
    }

    std::vector<Eigen::Vector3d> velocities(points.size(), Eigen::Vector3d::Zero());

    for (size_t i = 0; i < points.size(); ++i)
    {
        double query_t = points[i].timestamp;
        double clamped_t = std::min(std::max(query_t, min_time), max_time);
        double normalized_t = (clamped_t - min_time) / time_range;
        auto derivs = spline.derivatives(normalized_t, 1);
        velocities[i] = derivs.col(1).matrix() / time_range;
    }

    if (g_debug_logging_enabled && !debug_label.empty())
    {
        std::vector<double> normalized_unique_times(unique_times.size());
        for (size_t i = 0; i < unique_times.size(); ++i)
        {
            normalized_unique_times[i] = (unique_times[i] - min_time) / time_range;
        }

        DebugRecord record;
        record.id = g_debug_call_counter.fetch_add(1);
        record.label = debug_label;
        record.raw_points = points;
        record.normalized_unique_times = normalized_unique_times;
        record.unique_positions = control_positions;
        record.velocities = velocities;
        record.min_time = min_time;
        record.time_range = time_range;

        const int sample_count = 100;
        record.spline_positions.reserve(sample_count);
        record.spline_velocities.reserve(sample_count);
        for (int i = 0; i < sample_count; ++i)
        {
            double tau = static_cast<double>(i) / static_cast<double>(sample_count - 1);
            record.spline_positions.push_back(spline(tau));
            Eigen::Vector3d vel = spline.derivatives(tau, 1).col(1) / time_range;
            record.spline_velocities.push_back(vel);
        }

        {
            std::lock_guard<std::mutex> lock(g_debug_mutex);
            g_pending_records.emplace(record.id, std::move(record));
        }

        if (debug_record_id) *debug_record_id = record.id;
    }
    else
    {
        if (debug_record_id) *debug_record_id = INVALID_RECORD_ID;
    }

    return velocities;
}

double BSpline::clamp_smoothing_factor(double factor)
{
    if (!std::isfinite(factor)) return 0.0;
    if (factor < 0.0) return 0.0;
    const double upper = 0.999;
    if (factor > upper) return upper;
    return factor;
}
