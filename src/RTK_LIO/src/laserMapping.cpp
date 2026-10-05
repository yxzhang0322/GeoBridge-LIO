#include <omp.h>
#include "runtime_profiler.hpp"
#include <mutex>
#include <memory>
#include <math.h>
#include <array>
#include <cmath>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <limits>
#include <iomanip>
#include <ctime>
#include <deque>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>
#include <cstdio>
#include <cerrno>
#include <cstdint>
#include <cctype>
#include <sys/stat.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "IMU_Processing.hpp"
#include "rtk_height_observation.hpp"
#include "RTK_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/UInt8.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <gnss_comm/gnss_utility.hpp>
#include <ikd-Tree/ikd_Tree.h>


#include <wahba/alignment.h>
#include <wahba/BSpline.h>
#include <wahba/python_bridge.h>


#include "WorldMapTransformManager.hpp"
#include "tapm.hpp"
#include "pgo_sw/sliding_window_backend.h"


bool enable_antenna_lever_arm_compensation = false;
Eigen::Vector3d antenna_offset_T(0.0, 0.0, 0.0);
bool enable_attitude_compensation = false;
bool debug_lever_arm_effect = false;


Eigen::Vector3d compensate_antenna_to_imu(const Eigen::Vector3d &pos_antenna_world,
                                          const Eigen::Matrix3d &rotation_world)
{

    (void)rotation_world;
    return pos_antenna_world;
}

double init_time = 0.1;
bool lidar_stamp_is_end = true;
double laser_point_cov = 0.01;
bool laser_adaptive_variance = true;
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)


bool debug = true;


double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true, world_path_en = true;

bool export_lidar360_enable = false;
bool export_ascii_cloud_enable = false;
bool export_trajectory_enable = false;
bool export_pose_json_enable = false;
bool export_use_smooth_pose = false;
bool export_use_lidar_pose = false;
bool export_hba_enable = false;
bool final_world_body_tf_enable = false;
std::string export_output_dir;
std::string export_pose_json_path;
int export_pcd_name_fill_num = 0;
double export_offset_x = 0.0;
double export_offset_y = 0.0;
double export_offset_z = 100.0;
double export_start_after_sec = 12.0;
enum class ExportTimeBase
{
    GPS_TOW = 0,
    ROS_TIME = 1,
    RELATIVE = 2
};
ExportTimeBase export_time_base = ExportTimeBase::GPS_TOW;
double export_time_zero_unix = -1.0;
bool export_world_tum_enable = false;
bool export_world_tum_raw_enable = false;
bool export_world_tum_smooth_enable = false;
bool export_world_tum_delay_enable = false;
std::string export_world_tum_file_prefix_raw = "world_traj";
std::string export_world_tum_file_prefix_smooth = "world_smooth_traj";
int export_world_tum_flush_every_n = 50;
bool export_world_tum_require_monotonic_stamp = true;
bool export_world_tum_use_wall_time_filename = true;

struct WorldTumStreamState
{
    std::ofstream ofs;
    std::string path;
    bool stream_ready = false;
    size_t write_count = 0;
    double last_stamp = -std::numeric_limits<double>::infinity();
};

WorldTumStreamState world_tum_raw_stream;
WorldTumStreamState world_tum_smooth_stream;
std::string world_tum_session_timestamp;
bool world_tum_stream_warned = false;

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
float mov_threshold = 1.5f;
float plane_fit_threshold = 0.1f;
float point_max_range = 100.0f;
int min_effective_points = 5;
double imu_sync_offset = 0.1;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
bool covariance_debug_enable = false;
bool rtk_diagnostic_log_enable = true;
string rtk_diagnostic_log_path = string(ROOT_DIR) + "Log/rtk_lio_diagnostics.log";
mutex rtk_diagnostic_log_mutex;
std::ofstream rtk_diagnostic_log_file;
size_t rtk_diagnostic_log_line_count = 0;
bool rtk_path_log_enable = true;
string rtk_path_log_path = string(ROOT_DIR) + "Log/rtk_path.txt";
int rtk_path_log_flush_every_n = 50;
mutex rtk_path_log_mutex;
std::ofstream rtk_path_log_file;
size_t rtk_path_log_line_count = 0;
double last_logged_rtk_path_timestamp = -1.0;

enum class RtkFusionState;
const char *rtk_fusion_state_to_string(RtkFusionState state);
extern double lidar_end_time;
extern double first_lidar_time;
extern bool rtk_fix_active;
extern RtkFusionState rtk_fusion_state;
extern bool alignment_rotation_frozen;
extern const char *COLOR_GREEN;
extern const char *RESET_COLOR;

inline void log_measurement_event(const std::string &tag, const std::string &detail)
{
    (void)tag;
    (void)detail;
}

inline void log_alignment_event(const std::string &tag,
                                const Transformation &transform,
                                const std::string &detail)
{
    (void)tag;
    (void)transform;
    (void)detail;
}

static void write_rtk_diagnostic_log(const std::string &tag, const std::string &detail);

inline void log_covariance_sample(double timestamp,
                                  const std::string &tag,
                                  const Eigen::Matrix3d &P,
                                  const std::string &detail = "")
{
    (void)timestamp;
    (void)tag;
    (void)P;
    (void)detail;
}
string map_file_path, lid_topic, imu_topic, gnss_topic, gnss_lla_topic;

struct TapmStatRecord
{
    double magnitude = -1.0;
    double timestamp = -1.0;
    Eigen::Vector3d value = Eigen::Vector3d::Zero();
};

std::array<TapmStatRecord, 5> tapm_top_speed;
std::array<TapmStatRecord, 5> tapm_top_accel;

struct WorldPathPeakMetricsState
{
    bool has_prev_pose = false;
    bool has_prev_vel = false;
    bool has_prev_acc = false;
    double prev_t = -1.0;
    Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond prev_quat = Eigen::Quaterniond::Identity();
    Eigen::Vector3d prev_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d prev_acc = Eigen::Vector3d::Zero();
    double peak_jump_m = -1.0;
    double peak_angle_deg = -1.0;
    double peak_jerk_mps3 = -1.0;
    double peak_jump_time = -1.0;
    double peak_angle_time = -1.0;
    double peak_jerk_time = -1.0;
};

bool world_path_peak_metrics_on_shutdown_enable = true;
double world_path_peak_metrics_start_after_sec = 40.0;
WorldPathPeakMetricsState world_path_peak_metrics_state;

inline void update_tapm_top5(std::array<TapmStatRecord, 5> &records,
                             double magnitude,
                             const Eigen::Vector3d &value,
                             double timestamp)
{
    if (!std::isfinite(magnitude))
    {
        return;
    }
    if (magnitude <= records.back().magnitude)
    {
        return;
    }
    for (size_t i = 0; i < records.size(); ++i)
    {
        if (magnitude > records[i].magnitude)
        {
            for (size_t j = records.size() - 1; j > i; --j)
            {
                records[j] = records[j - 1];
            }
            records[i].magnitude = magnitude;
            records[i].timestamp = timestamp;
            records[i].value = value;
            return;
        }
    }
}

inline void print_tapm_top5(const std::string &label,
                            const std::array<TapmStatRecord, 5> &records)
{
    bool has_record = false;
    for (const auto &rec : records)
    {
        if (rec.magnitude >= 0.0)
        {
            has_record = true;
            break;
        }
    }
    if (!has_record)
    {
        return;
    }
    for (size_t i = 0; i < records.size(); ++i)
    {
        if (records[i].magnitude < 0.0)
        {
            break;
        }
    }
}

inline double quaternion_delta_deg(const Eigen::Quaterniond &q_prev,
                                   const Eigen::Quaterniond &q_curr)
{
    Eigen::Quaterniond q_prev_n = q_prev.normalized();
    Eigen::Quaterniond q_curr_n = q_curr.normalized();
    Eigen::Quaterniond dq = q_prev_n.conjugate() * q_curr_n;
    dq.normalize();
    const double w_abs = std::abs(dq.w());
    const double w_clamped = std::max(-1.0, std::min(1.0, w_abs));
    return std::acos(w_clamped) * 2.0 * 180.0 / M_PI;
}

inline void update_world_path_peak_metrics(double timestamp,
                                           const Eigen::Vector3d &pos_world,
                                           const Eigen::Quaterniond &quat_world)
{
    if (!world_path_peak_metrics_on_shutdown_enable)
    {
        return;
    }
    if (!std::isfinite(timestamp) || !std::isfinite(first_lidar_time))
    {
        return;
    }
    if (timestamp < first_lidar_time + world_path_peak_metrics_start_after_sec)
    {
        return;
    }

    WorldPathPeakMetricsState &st = world_path_peak_metrics_state;
    if (st.has_prev_pose)
    {
        const Eigen::Vector3d delta_pos = pos_world - st.prev_pos;
        const double jump_m = delta_pos.norm();
        if (std::isfinite(jump_m) && jump_m > st.peak_jump_m)
        {
            st.peak_jump_m = jump_m;
            st.peak_jump_time = timestamp;
        }

        const double angle_deg = quaternion_delta_deg(st.prev_quat, quat_world);
        if (std::isfinite(angle_deg) && angle_deg > st.peak_angle_deg)
        {
            st.peak_angle_deg = angle_deg;
            st.peak_angle_time = timestamp;
        }

        const double dt = timestamp - st.prev_t;
        if (dt > 1e-6)
        {
            const Eigen::Vector3d vel = delta_pos / dt;
            if (st.has_prev_vel)
            {
                const Eigen::Vector3d acc = (vel - st.prev_vel) / dt;
                if (st.has_prev_acc)
                {
                    const Eigen::Vector3d jerk = (acc - st.prev_acc) / dt;
                    const double jerk_mps3 = jerk.norm();
                    if (std::isfinite(jerk_mps3) && jerk_mps3 > st.peak_jerk_mps3)
                    {
                        st.peak_jerk_mps3 = jerk_mps3;
                        st.peak_jerk_time = timestamp;
                    }
                }
                st.prev_acc = acc;
                st.has_prev_acc = true;
            }
            st.prev_vel = vel;
            st.has_prev_vel = true;
        }
    }

    st.prev_t = timestamp;
    st.prev_pos = pos_world;
    st.prev_quat = quat_world.normalized();
    st.has_prev_pose = true;
}

inline void print_world_path_peak_metrics_on_shutdown()
{
    if (!world_path_peak_metrics_on_shutdown_enable)
    {
        return;
    }
    auto fmt_value = [](double value) -> std::string
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            return "--";
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << value;
        return oss.str();
    };
    auto fmt_time = [](double value) -> std::string
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            return "--";
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << value;
        return oss.str();
    };

    const WorldPathPeakMetricsState &st = world_path_peak_metrics_state;
    const char *peak_tag_color = "\033[1;36m";
    const char *color_reset = "\033[0m";
    (void)0;
}

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_timestamp_rtk = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<gnss_comm::GnssPVTSolnMsg::ConstPtr> rtk_buffer;
struct LlaTimeSample
{
    double stamp = -1.0;
    double latitude = 0.0;
    double longitude = 0.0;
    double altitude = 0.0;
};
deque<LlaTimeSample> rtk_lla_time_buffer;
deque<gnss_comm::GnssPVTSolnMsg::Ptr> rtk_pending_pvt_buffer;
int rtk_subscriber_queue_size = 100;
int rtk_input_buffer_max_size = 200;
int rtk_lla_time_buffer_max_size = 200;
int rtk_pending_pvt_max_size = 200;
string rtk_time_source = "gpst";
double rtk_lla_match_latlon_tolerance = 1e-8;
double rtk_lla_match_alt_tolerance = 0.05;
double last_published_rtk_odom_timestamp = -1.0;
double last_published_rtk_path_timestamp = -1.0;
bool rtk_wait_for_time_compensated_rtk = false;
double rtk_time_compensation_wait_tolerance = 0.05;
double rtk_time_compensation_wait_max_staleness = 1.0;
size_t rtk_time_compensation_wait_count = 0;
double rtk_last_time_compensation_wait_log = -1.0;
size_t rtk_pvt_callback_count = 0;
size_t rtk_pvt_buffer_push_count = 0;
size_t rtk_lla_callback_count = 0;
size_t rtk_process_call_count = 0;
double rtk_last_pvt_diag_wall = -1.0;
double rtk_last_push_diag_wall = -1.0;
double rtk_last_lla_diag_wall = -1.0;
double rtk_last_process_diag_lidar = -1.0;

bool ekftimepos_state_flag = false, rtktimepos_state_flag = false;
Transformation Global_GNSS_LIO;
WorldMapTransformManager world_map_transform;


WorldState world_state;
WorldState world_publish_state;
bool world_publish_handoff_active = false;
double world_publish_handoff_time = -1.0;
Eigen::Vector3d world_publish_handoff_raw_pos = Eigen::Vector3d::Zero();
Eigen::Quaterniond world_publish_handoff_raw_quat = Eigen::Quaterniond::Identity();
Eigen::Vector3d world_publish_handoff_pub_pos = Eigen::Vector3d::Zero();
Eigen::Quaterniond world_publish_handoff_pub_quat = Eigen::Quaterniond::Identity();
bool world_output_blend_enable = true;
double world_output_blend_max_jump = 0.8;
double world_output_blend_duration = 1.0;
bool world_output_z_guard_enable = false;
double world_output_z_guard_max_speed = 0.5;
double world_output_z_guard_max_step = 0.08;
bool world_output_blend_active = false;
double world_output_blend_start_time = -1.0;
Eigen::Vector3d world_output_blend_start_pos = Eigen::Vector3d::Zero();
Eigen::Quaterniond world_output_blend_start_quat = Eigen::Quaterniond::Identity();
bool world_output_last_valid = false;
double world_output_last_time = -1.0;
Eigen::Vector3d world_output_last_pos = Eigen::Vector3d::Zero();
Eigen::Quaterniond world_output_last_quat = Eigen::Quaterniond::Identity();

Eigen::Vector3d map_position_to_world(const Eigen::Vector3d &pos_map)
{
  if (world_map_transform.isTransformValid())
  {
    return world_map_transform.transformPositionToWorld(pos_map);
  }
  return pos_map;
}

double velocity_threshold = 0.05;
double rtk_lio_time_threshold = 0.1;
double rtk_match_min_ratio = 0.85;
bool rtk_match_at_gnss_epochs = false;
int rtk_alignment_min_pairs = 3;
double last_rtk_match_ratio = 0.0;
M3D Lidar_R_wrt_GNSS = Eye3d();
deque<Timepos> ekftimepos_buffer;
deque<Timepos> rtktimepos_buffer;
deque<Timepos> matched_rtk_window;
deque<Timepos> matched_ekf_window;
std::deque<Timepos> world_history_window;
std::deque<Timepos> tapm_lio_history;

int alignment_window_min = 5;
int alignment_window_max = 10;
int alignment_window_growth_interval = 5;
int alignment_window_reset_fix = alignment_window_min;
int alignment_window_current = alignment_window_min;
int alignment_window_success_counter = 0;
bool alignment_freeze_rotation_at_max_window = true;
bool keep_rotation_frozen_on_loss = true;
double alignment_freeze_rotation_yaw_max_delta_deg = 0.0;
bool alignment_rotation_frozen = false;

Eigen::Matrix3d alignment_frozen_rotation = Eigen::Matrix3d::Identity();
bool rtk_freeze_adaptive_enable = false;
double rtk_freeze_adaptive_max_dist = 0.5;
double rtk_freeze_adaptive_base_weight = 0.5;
bool rtk_freeze_adaptive_log_enable = false;
Transformation last_alignment_no_frozen;
bool last_alignment_no_frozen_valid = false;
double last_dist_no_frozen = std::numeric_limits<double>::quiet_NaN();


const char *COLOR_RED = "\033[31m";
const char *COLOR_GREEN = "\033[32m";
const char *COLOR_BLUE = "\033[34m";
const char *COLOR_YELLOW = "\033[33m";
const char *COLOR_MAGENTA = "\033[35m";
const char *COLOR_CYAN = "\033[36m";
const char *RESET_COLOR = "\033[0m";
int count_rtk_match_fail = 0;
int count_rtk_match_success = 0;

enum class RtkFusionState
{
    NO_FIX = 0,
    WAIT_FIX,
    SMOOTHING,
    CORRECTION
};

enum class TapmRuntimeState
{
    IDLE = 0,
    SLIDING,
    ANCHORED
};

struct RtkFusionEvent
{
    double reference_time = -1.0;
    bool has_new_fix_measurement = false;
    bool has_fix = false;
    bool was_fix_active = false;
};

const char *rtk_fusion_state_to_string(RtkFusionState state);
const char *tapm_runtime_state_to_string(TapmRuntimeState state);
TapmRuntimeState get_tapm_runtime_state();
std::vector<TrajectoryPoint> deque_to_vector(const std::deque<TrajectoryPoint> &src);
void enter_initial_correction_state(double reference_time);
void enter_wait_fix_state(double reference_time);
void enter_recovery_correction_state(double reference_time);
void enter_smoothing_state(double current_time);
void tick_rtk_fusion_state_with_fix(double current_time);
extern MeasureGroup Measures;

double rtk_fix_timeout = 2.0;
bool rtk_fix_active = false;
bool rtk_fix_signal_active = false;
double latest_rtk_timestamp = -1.0;

RtkFusionState rtk_fusion_state = RtkFusionState::NO_FIX;
int rtk_consecutive_fix_count = 0;
int rtk_min_fix_for_smoothing = 5;
const double rtk_wait_fix_duration_sec = 1.0;
double rtk_fix_recover_start_time = -1.0;
bool rtk_fusion_enable = true;
double rtk_loss_start_time = -1.0;
double rtk_last_loss_duration = 0.0;
bool rtk_recovery_verify_enable = true;
double rtk_recover_stable_duration_sec = 3.0;
double rtk_recover_long_loss_threshold_sec = 30.0;
double rtk_recover_long_stable_duration_sec = 8.0;
double rtk_recover_max_h_acc = 0.5;
double rtk_recover_max_v_acc = 1.5;
double rtk_recover_max_sample_dt = 1.0;
double rtk_recover_max_speed_xy = 5.0;
double rtk_recover_max_speed_z = 2.0;
double rtk_recover_max_jump_xy = 2.0;
double rtk_recover_max_jump_z = 0.8;
double rtk_recover_stable_start_time = -1.0;
bool rtk_recover_prev_valid = false;
Timepos rtk_recover_prev_sample;
double rtk_recover_latest_dt = std::numeric_limits<double>::quiet_NaN();
double rtk_recover_latest_jump_xy = std::numeric_limits<double>::quiet_NaN();
double rtk_recover_latest_jump_z = std::numeric_limits<double>::quiet_NaN();
double rtk_recover_latest_speed_xy = std::numeric_limits<double>::quiet_NaN();
double rtk_recover_latest_speed_z = std::numeric_limits<double>::quiet_NaN();
bool rtk_recover_latest_quality_ok = false;
bool rtk_recover_latest_continuity_ok = false;
std::string rtk_recover_latest_reason = "not_checked";
bool rtk_ekf_constraint_enable = true;
std::ofstream update_timing_log;
double diagnostic_sliding_ms = 0.0;
double last_rtk_constraint_epoch = -1.0;
double last_rtk_constraint_lidar_epoch = -1.0;
bool rtk_initial_alignment_done = false;
bool rtk_bootstrap_transform_on_first_fix = false;
double rtk_bootstrap_transform_max_rtk_age = 1.0;


bool rtk_reacquire_enable = true;

bool rtk_reacquire_blocked = false;
bool rtk_fusion_state_pub_enable = false;
uint8_t rtk_fusion_state_last_pub = 255;
Transformation smoothing_alignment;
bool smoothing_alignment_valid = false;
bool smoothing_first_alignment_done = false;
bool smoothing_completion_requested = false;
std::string smoothing_completion_reason;
bool pgo_raw_rtk_allowed = false;
double tapm_recovery_history_start_time = -1.0;
double world_map_transform_alpha = 1.0;
double world_map_transform_max_jump = 0.0;
bool enable_manual_ekf_rtk_offset = false;
Eigen::Vector3d manual_ekf_rtk_offset = Eigen::Vector3d::Zero();
double last_alignment_rmse = -1.0;
double alignment_rmse_jump_ratio = 3.0;
double alignment_rotation_jump_deg = 0.0;
bool alignment_stable = false;
double alignment_stable_enter_time = -1.0;
double alignment_stable_rmse_threshold = 0.1;
double alignment_stable_min_duration = 5.0;
double alignment_stable_rotation_deg_threshold = 2.0;
int alignment_stable_window = 0;
int alignment_latest_window_size = 0;
bool alignment_window_gate_released = false;
bool alignment_rotation_stable = false;
double alignment_rotation_stable_enter_time = -1.0;
double alignment_rotation_stable_deg_threshold = 2.0;
double alignment_rotation_stable_min_duration = 2.0;
bool alignment_freeze_rotation_when_stable = false;
bool alignment_yaw_only_enable = true;
bool alignment_use_gravity_prior = false;
double alignment_gravity_weight = 0.0;
bool alignment_continuity_guard_enable = true;
double alignment_translation_jump_max = 20.0;
double alignment_pose_jump_max = 1.0;
double alignment_max_fit_rmse = 0.0;
bool tapm_enable = false;
bool tapm_sliding_enable = false;
bool tapm_publish_use_z = false;
double tapm_alpha = 0.5;
double tapm_beta = 1.0;
double tapm_lambda_v = 10.0;
double tapm_lambda_a = 100.0;
double tapm_gamma = 0.01;
double tapm_nominal_dt = 0.1;
double tapm_sigma0_xy = 0.1;
double tapm_sigma0_z = 0.2;
double tapm_v_max = 20.0;
double tapm_v_max_z = 5.0;
double tapm_a_max = 8.0;
double tapm_a_max_z = 3.0;
double tapm_rtk_sigma_min = 1e-3;
int tapm_grace_frames = 15;
int tapm_max_iterations = 30;
int tapm_optimization_window = 15;
int tapm_world_history_size = 20;
int tapm_exit_frame_limit = 0;
double tapm_exit_max_rtk_error = 0.05;
double tapm_exit_min_duration = 0.0;
double tapm_exit_max_duration = 0.0;
int tapm_exit_stable_observations = 1;
bool tapm_commit_exit_translation = false;
bool tapm_use_analytic_jacobian = true;
bool tapm_runtime_debug_enable = false;
double tapm_runtime_debug_interval = 0.2;
double tapm_runtime_debug_last_log_time = -1.0;
bool tapm_active = false;
int tapm_frame_count = 0;
bool tapm_anchor_valid = false;
double tapm_anchor_time = -1.0;
double tapm_recover_time = -1.0;
Transformation tapm_anchor_transform;
bool tapm_output_active = false;
bool tapm_recovery_clock_weights = false;
bool tapm_causal_publish_regularization = false;
double tapm_publish_regularization_decay_sec = 0.0;
double tapm_previous_publish_time = -1.0;
double tapm_previous_publish_dt = -1.0;
Eigen::Vector3d tapm_previous_publish_correction = Eigen::Vector3d::Zero();
Eigen::Vector3d tapm_previous_previous_publish_correction = Eigen::Vector3d::Zero();
double tapm_rotation_handover_sec = 0.0;
double tapm_rotation_handover_start = -1.0;
Eigen::Quaterniond tapm_rotation_source = Eigen::Quaterniond::Identity();
Eigen::Quaterniond tapm_rotation_target = Eigen::Quaterniond::Identity();
Eigen::Quaterniond tapm_publish_rotation = Eigen::Quaterniond::Identity();
Eigen::Vector3d tapm_publish_pos = Eigen::Vector3d::Zero();
double tapm_publish_weight = 0.0;

bool pgo_sw_enable = false;
bool pgo_sw_enable_cloud = false;
bool pgo_sw_use_as_system_output = false;
bool pgo_sw_use_tapm_virtual_rtk = true;
int pgo_sw_window_size = 15;
int pgo_sw_optimize_stride = 3;
double pgo_sw_debug_interval = 0.5;
double pgo_sw_rtk_match_max_dt = 0.1;
double pgo_sw_odom_sigma_pos = 0.1;
double pgo_sw_odom_sigma_rot = 0.01;
double pgo_sw_rtk_sigma_xy = 0.05;
double pgo_sw_rtk_sigma_z = 0.10;
double pgo_sw_tapm_sigma_xy = 0.02;
double pgo_sw_tapm_sigma_z = 0.05;
bool pgo_sw_use_rtk_z = false;
bool pgo_sw_use_tapm_z = false;
double pgo_sw_recovery_ramp_duration_sec = 5.0;
double pgo_sw_recovery_long_ramp_duration_sec = 15.0;
double pgo_sw_recovery_ramp_min_weight = 0.15;
double pgo_sw_recovery_z_sigma_scale = 5.0;
double pgo_sw_recovery_ramp_start_time = -1.0;
double pgo_sw_recovery_ramp_duration_active = 0.0;
double pgo_sw_last_rtk_used = -1.0;
pgo_sw::SlidingWindowBackend world_sw_backend;
bool pgo_sw_keep_window_through_recovery = true;
int pgo_sw_frame_counter = 0;
bool pgo_sw_last_delta_valid = false;
Eigen::Vector3d pgo_sw_last_delta = Eigen::Vector3d::Zero();
double pgo_sw_last_opt_time = -1.0;
double pgo_sw_last_opt_cost_ms = 0.0;
double pgo_sw_last_debug_log_time = -1.0;
bool pgo_debug_lio_valid = false;
double pgo_debug_lio_time = -1.0;
Eigen::Vector3d pgo_debug_lio_pos = Eigen::Vector3d::Zero();
bool pgo_debug_output_valid = false;
bool pgo_debug_output_system_applied = false;
double pgo_debug_output_time = -1.0;
Eigen::Vector3d pgo_debug_output_pos = Eigen::Vector3d::Zero();
Eigen::Vector3d pgo_debug_output_delta = Eigen::Vector3d::Zero();
std::string pgo_debug_output_status = "none";
size_t pgo_debug_window_size = 0;
size_t pgo_debug_rtk_buffer_size = 0;
size_t pgo_debug_optimized_size = 0;
size_t pgo_debug_rtk_factor_count = 0;
double pgo_debug_rtk_sigma_scale = 1.0;
bool pgo_debug_constraint_valid = false;
bool pgo_debug_constraint_added = false;
double pgo_debug_constraint_time = -1.0;
double pgo_debug_constraint_dt = std::numeric_limits<double>::quiet_NaN();
Eigen::Vector3d pgo_debug_constraint_pos = Eigen::Vector3d::Zero();
std::string pgo_debug_constraint_source = "none";
std::string pgo_debug_constraint_status = "none";
Eigen::Vector3d tapm_output_pos = Eigen::Vector3d::Zero();

void reset_pgo_sw_runtime_cache()
{
    pgo_sw_frame_counter = 0;
    pgo_sw_last_delta_valid = false;
    pgo_sw_last_delta.setZero();
    pgo_sw_last_opt_time = -1.0;
    pgo_sw_last_opt_cost_ms = 0.0;
    pgo_sw_last_debug_log_time = -1.0;
}

void clear_pgo_sw_for_transition(const std::string &reason, bool force_clear = false)
{
    if (pgo_sw_keep_window_through_recovery && !force_clear)
    {
        write_rtk_diagnostic_log("PGO_SW_KEEP", "reason=" + reason);
        return;
    }

    world_sw_backend.clear();
    reset_pgo_sw_runtime_cache();
    pgo_sw_last_rtk_used = -1.0;
    write_rtk_diagnostic_log("PGO_SW_CLEAR", "reason=" + reason);
}

bool world_pose_log_enable = false;
bool terminal_status_enable = false;
double terminal_status_interval = 0.1;
double terminal_status_last_time = -1.0;
nav_msgs::Path path;
nav_msgs::Path rtk_path;
nav_msgs::Path any_path;
nav_msgs::Path world_ekf_path;
nav_msgs::Path tapm_path;
nav_msgs::Path world_path;

static bool rtk_use_lla_header_time();

double compute_alignment_window_rmse(const Transformation &transform,
                                     const std::deque<Timepos> &ekf_window,
                                     const std::deque<Timepos> &rtk_window)
{
    if (!transform.valid || ekf_window.size() != rtk_window.size() || ekf_window.empty())
    {
        return -1.0;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < ekf_window.size(); ++i)
    {
        const Eigen::Vector3d aligned =
            transform.s * (transform.R * ekf_window[i].pos) + transform.t;
        const double err = (aligned - rtk_window[i].pos).norm();
        sum_sq += err * err;
    }
    return std::sqrt(sum_sq / static_cast<double>(ekf_window.size()));
}

double wrap_angle_rad(double angle)
{
    while (angle > M_PI)
    {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI)
    {
        angle += 2.0 * M_PI;
    }
    return angle;
}

Eigen::Matrix3d compose_zyx_rotation(double yaw, double pitch, double roll)
{
    const Eigen::AngleAxisd yaw_rot(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd pitch_rot(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd roll_rot(roll, Eigen::Vector3d::UnitX());
    return (yaw_rot * pitch_rot * roll_rot).toRotationMatrix();
}

double compute_alignment_rotation_delta_deg(const Transformation &transform)
{
    if (!world_map_transform.isTransformValid())
    {
        return 0.0;
    }
    const Eigen::Matrix3d &R_prior = world_map_transform.getTransform().R;
    Eigen::Matrix3d delta_R = R_prior.transpose() * transform.R;
    Eigen::AngleAxisd delta(delta_R);
    return delta.angle() * 180.0 / M_PI;
}

bool evaluate_alignment_rotation_quality(const Transformation &transform)
{
    if (alignment_rotation_jump_deg <= 0.0)
    {
        return true;
    }
    if (!world_map_transform.isTransformValid())
    {
        return true;
    }
    const double delta_deg = compute_alignment_rotation_delta_deg(transform);
    return delta_deg <= alignment_rotation_jump_deg;
}

const char *rtk_fusion_state_to_string(RtkFusionState state)
{
    switch (state)
    {
    case RtkFusionState::NO_FIX:
        return "NO_FIX";
    case RtkFusionState::WAIT_FIX:
        return "WAIT_FIX";
    case RtkFusionState::SMOOTHING:
        return "SMOOTHING";
    case RtkFusionState::CORRECTION:
        return "CORRECTION";
    default:
        return "UNKNOWN";
    }
}

const char *tapm_runtime_state_to_string(TapmRuntimeState state)
{
    switch (state)
    {
    case TapmRuntimeState::IDLE:
        return "IDLE";
    case TapmRuntimeState::SLIDING:
        return "SLIDING";
    case TapmRuntimeState::ANCHORED:
        return "ANCHORED";
    default:
        return "UNKNOWN";
    }
}

TapmRuntimeState get_tapm_runtime_state()
{
    if (tapm_enable && tapm_sliding_enable && tapm_active)
    {
        return TapmRuntimeState::SLIDING;
    }
    if (tapm_enable && !tapm_sliding_enable &&
        alignment_rotation_frozen &&
        tapm_anchor_valid &&
        tapm_recover_time >= 0.0)
    {
        return TapmRuntimeState::ANCHORED;
    }
    return TapmRuntimeState::IDLE;
}

const char *status_color(bool error, bool warn = false)
{
    if (error)
    {
        return COLOR_RED;
    }
    if (warn)
    {
        return COLOR_YELLOW;
    }
    return COLOR_GREEN;
}

std::string color_status(const std::string &text, bool error, bool warn = false)
{
    std::ostringstream oss;
    oss << status_color(error, warn) << text << RESET_COLOR;
    return oss.str();
}

std::string status_cell(const std::string &text, int width, bool error = false, bool warn = false)
{
    std::string clipped = text;
    if (width > 1 && clipped.size() > static_cast<size_t>(width))
    {
        clipped = clipped.substr(0, static_cast<size_t>(width - 1)) + "~";
    }
    std::ostringstream raw;
    raw << std::left << std::setw(width) << clipped;
    return color_status(raw.str(), error, warn);
}

std::string status_double(double value, int precision = 3)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (!std::isfinite(value))
    {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string matrix_value(double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << std::showpos << std::setw(10) << value << std::noshowpos;
    return oss.str();
}

std::string transform_matrix_row(const Eigen::Matrix3d &R_scaled,
                                 const Eigen::Vector3d &t,
                                 int row)
{
    std::ostringstream oss;
    oss << "T[" << row << "] ["
        << matrix_value(R_scaled(row, 0)) << " "
        << matrix_value(R_scaled(row, 1)) << " "
        << matrix_value(R_scaled(row, 2)) << " "
        << matrix_value(t(row)) << " ]";
    return oss.str();
}

std::string rotation_ypr_text(const Eigen::Matrix3d &R)
{
    const Eigen::Vector3d ypr = R.eulerAngles(2, 1, 0) * 180.0 / M_PI;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "yaw=" << ypr.x()
        << " pitch=" << ypr.y()
        << " roll=" << ypr.z();
    return oss.str();
}

std::string path_xyz_text(const nav_msgs::Path &path)
{
    if (path.poses.empty())
    {
        return "EMPTY";
    }

    const geometry_msgs::Point &p = path.poses.back().pose.position;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "x=" << p.x << " y=" << p.y << " z=" << p.z;
    return oss.str();
}

std::string eigen_xyz_text(const Eigen::Vector3d &p, bool valid)
{
    if (!valid)
    {
        return "EMPTY";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "x=" << p.x() << " y=" << p.y() << " z=" << p.z();
    return oss.str();
}

std::string eigen_delta_text(const Eigen::Vector3d &p, bool valid)
{
    if (!valid)
    {
        return "EMPTY";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "dx=" << p.x() << " dy=" << p.y() << " dz=" << p.z()
        << " norm=" << p.norm();
    return oss.str();
}

std::string panel_line(const std::string &text, bool error = false, bool warn = false)
{
    std::ostringstream oss;
    oss << "| " << status_cell(text, 98, error, warn) << "|\n";
    return oss.str();
}

void publish_terminal_status_panel(double current_time)
{
    if (!terminal_status_enable)
    {
        return;
    }
    if (terminal_status_last_time >= 0.0 &&
        current_time - terminal_status_last_time < terminal_status_interval)
    {
        return;
    }
    terminal_status_last_time = current_time;

    const double rtk_age = (latest_rtk_timestamp > 0.0)
                               ? std::fabs(current_time - latest_rtk_timestamp)
                               : std::numeric_limits<double>::infinity();
    const bool rtk_stale = !std::isfinite(rtk_age) || rtk_age > rtk_fix_timeout;
    const bool match_bad = rtk_fix_active && alignment_latest_window_size > 0 &&
                           last_rtk_match_ratio < rtk_match_min_ratio;
    const bool smoothing_without_tapm =
        rtk_fusion_state == RtkFusionState::SMOOTHING &&
        get_tapm_runtime_state() == TapmRuntimeState::IDLE;
    const bool transform_missing = rtk_initial_alignment_done && !world_map_transform.isTransformValid();
    const bool world_missing = rtk_initial_alignment_done && !world_state.initialized;
    const bool pgo_warn = pgo_sw_use_as_system_output && !pgo_sw_enable;

    std::string issue = "OK";
    bool issue_error = false;
    bool issue_warn = false;
    if (rtk_stale)
    {
        issue = "RTK_STALE";
        issue_error = true;
    }
    else if (smoothing_without_tapm)
    {
        issue = "SMOOTHING_NO_TAPM";
        issue_error = true;
    }
    else if (transform_missing)
    {
        issue = "NO_WORLD_TF";
        issue_error = true;
    }
    else if (world_missing)
    {
        issue = "NO_WORLD_STATE";
        issue_error = true;
    }
    else if (match_bad)
    {
        issue = "MATCH_LOW";
        issue_warn = true;
    }
    else if (pgo_warn)
    {
        issue = "PGO_OUTPUT_DISABLED";
        issue_warn = true;
    }

    const std::string border =
        "+------------------------+------------------------+------------------------+------------------------+\n";
    const bool tf_valid = world_map_transform.isTransformValid();
    const std::string win_text =
        std::to_string(alignment_latest_window_size) + "/" +
        std::to_string(alignment_window_current) + "/" +
        std::to_string(alignment_window_max);
    const std::string pgo_text =
        pgo_sw_enable ? (pgo_sw_use_as_system_output ? "SYSTEM" : "AUX") : "OFF";

    std::ostringstream oss;
    oss << "\n"
        << COLOR_CYAN << border
        << "| RTK-LIO STATUS        | RTK / ALIGN           | TAPM / PGO            | ISSUE                  |\n"
        << border << RESET_COLOR;

    oss << "| state: " << status_cell(rtk_fusion_state_to_string(rtk_fusion_state), 16)
        << "| fix: " << status_cell(rtk_fix_active ? "ACTIVE" : "NO_FIX", 18, !rtk_fix_active)
        << "| tapm: " << status_cell(tapm_runtime_state_to_string(get_tapm_runtime_state()),
                                  17,
                                  smoothing_without_tapm)
        << "| " << status_cell(issue, 22, issue_error, issue_warn) << "|\n";

    oss << "| time:  " << status_cell(status_double(current_time), 16)
        << "| age: " << status_cell(status_double(rtk_age), 18, rtk_stale)
        << "| frame:" << status_cell(std::to_string(tapm_frame_count), 17)
        << "| match: " << status_cell(status_double(last_rtk_match_ratio), 15, false, match_bad)
        << "|\n";

    oss << "| world: " << status_cell(world_state.initialized ? "INIT" : "NOT_INIT", 16, world_missing)
        << "| win: " << status_cell(win_text, 18)
        << "| pgo:  " << status_cell(pgo_text, 17, false, pgo_warn)
        << "| tf:    " << status_cell(tf_valid ? "VALID" : "INVALID", 15, transform_missing)
        << "|\n";

    oss << "| source:" << status_cell(rtk_time_source, 16)
        << "| pvt: " << status_cell(gnss_topic, 18)
        << "| lla:  " << status_cell(rtk_use_lla_header_time() ? gnss_lla_topic : "OFF", 17)
        << "| age_ref:" << status_cell(rtk_use_lla_header_time() ? "LLA_HDR" : "PVT_GPST", 14)
        << "|\n";

    oss << COLOR_CYAN << border
        << panel_line("path_world_ekf: " + path_xyz_text(world_ekf_path), world_ekf_path.poses.empty())
        << panel_line("path_rtk:       " + path_xyz_text(rtk_path), rtk_path.poses.empty() || rtk_stale)
        << panel_line("pgo_lio:        t=" + status_double(pgo_debug_lio_time) + " " +
                          eigen_xyz_text(pgo_debug_lio_pos, pgo_debug_lio_valid),
                      !pgo_debug_lio_valid)
        << panel_line("pgo_buf:        window=" + std::to_string(pgo_debug_window_size) +
                          " rtk_buf=" + std::to_string(pgo_debug_rtk_buffer_size) +
                          " opt_buf=" + std::to_string(pgo_debug_optimized_size))
        << panel_line("pgo_weight:     rtk_factor=" + std::to_string(pgo_debug_rtk_factor_count) +
                          " sigma_scale=" + status_double(pgo_debug_rtk_sigma_scale))
        << panel_line("pgo_out:        status=" + pgo_debug_output_status +
                          " system_applied=" +
                          std::to_string(pgo_debug_output_system_applied ? 1 : 0) +
                          " t=" + status_double(pgo_debug_output_time),
                      !pgo_debug_output_valid,
                      pgo_debug_output_valid && pgo_debug_output_delta.norm() > 0.01)
        << panel_line("pgo_out_xyz:    " +
                          eigen_xyz_text(pgo_debug_output_pos, pgo_debug_output_valid),
                      !pgo_debug_output_valid,
                      pgo_debug_output_valid && pgo_debug_output_delta.norm() > 0.01)
        << panel_line("pgo_out_delta:  " +
                          eigen_delta_text(pgo_debug_output_delta, pgo_debug_output_valid),
                      !pgo_debug_output_valid,
                      pgo_debug_output_valid && pgo_debug_output_delta.norm() > 0.01)
        << panel_line("pgo_pos:        src=" + pgo_debug_constraint_source +
                          " status=" + pgo_debug_constraint_status +
                          " added=" + std::to_string(pgo_debug_constraint_added ? 1 : 0),
                      !pgo_debug_constraint_valid,
                      pgo_debug_constraint_valid && !pgo_debug_constraint_added)
        << panel_line("pgo_pos_time:   t=" + status_double(pgo_debug_constraint_time) +
                          " dt=" + status_double(pgo_debug_constraint_dt),
                      !pgo_debug_constraint_valid,
                      pgo_debug_constraint_valid && !pgo_debug_constraint_added)
        << panel_line("pgo_pos_xyz:    " +
                          eigen_xyz_text(pgo_debug_constraint_pos, pgo_debug_constraint_valid),
                      !pgo_debug_constraint_valid,
                      pgo_debug_constraint_valid && !pgo_debug_constraint_added)
        << border << RESET_COLOR;

    oss << COLOR_CYAN << border
        << panel_line("T_world_map = [ s * R | t ]")
        << border << RESET_COLOR;

    if (tf_valid)
    {
        const Transformation &tf = world_map_transform.getTransform();
        const Eigen::Matrix3d R_scaled = tf.s * tf.R;
        oss << panel_line("T_align rot_deg: " + rotation_ypr_text(tf.R) +
                              " R_frozen=" + std::to_string(alignment_rotation_frozen ? 1 : 0) +
                              " stable=" + std::to_string(alignment_stable ? 1 : 0));
        std::ostringstream last_row;
        last_row << "T[3] [" << matrix_value(0.0) << " "
                 << matrix_value(0.0) << " "
                 << matrix_value(0.0) << " "
                 << matrix_value(1.0) << " ]";
        oss << panel_line(transform_matrix_row(R_scaled, tf.t, 0))
            << panel_line(transform_matrix_row(R_scaled, tf.t, 1))
            << panel_line(transform_matrix_row(R_scaled, tf.t, 2))
            << panel_line(last_row.str());
    }
    else
    {
        oss << panel_line("INVALID: world_map_transform is not available", true);
    }

    oss << COLOR_CYAN << border << RESET_COLOR;
    {
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(6)
             << "state=" << rtk_fusion_state_to_string(rtk_fusion_state)
             << " fix_active=" << (rtk_fix_active ? 1 : 0)
             << " tapm=" << tapm_runtime_state_to_string(get_tapm_runtime_state())
             << " issue=" << issue
             << " time=" << current_time
             << " rtk_age=" << rtk_age
             << " match=" << last_rtk_match_ratio
             << " world_init=" << (world_state.initialized ? 1 : 0)
             << " win=" << alignment_latest_window_size << "/" << alignment_window_current
             << "/" << alignment_window_max
             << " align_window_current=" << alignment_window_current
             << " align_window_max=" << alignment_window_max
             << " rotation_frozen=" << (alignment_rotation_frozen ? 1 : 0)
             << " rotation_stable=" << (alignment_rotation_stable ? 1 : 0)
             << " pgo_mode=" << pgo_text
             << " tf_valid=" << (tf_valid ? 1 : 0)
             << " source=" << rtk_time_source
             << " age_ref=" << (rtk_use_lla_header_time() ? "LLA_HDR" : "PVT_GPST")
             << " path_world_ekf=\"" << path_xyz_text(world_ekf_path) << "\""
             << " path_rtk=\"" << path_xyz_text(rtk_path) << "\""
             << " world_state_pos=[" << world_state.pos.transpose() << "]"
             << " world_publish_pos=[" << world_publish_state.pos.transpose() << "]"
             << " pgo_lio_valid=" << (pgo_debug_lio_valid ? 1 : 0)
             << " pgo_lio_t=" << pgo_debug_lio_time
             << " pgo_lio=[" << pgo_debug_lio_pos.transpose() << "]"
             << " pgo_window=" << pgo_debug_window_size
             << " pgo_rtk_buf=" << pgo_debug_rtk_buffer_size
             << " pgo_opt_buf=" << pgo_debug_optimized_size
             << " pgo_rtk_factor=" << pgo_debug_rtk_factor_count
             << " pgo_rtk_sigma_scale=" << pgo_debug_rtk_sigma_scale
             << " pgo_out_valid=" << (pgo_debug_output_valid ? 1 : 0)
             << " pgo_out_status=" << pgo_debug_output_status
             << " pgo_out_system_applied=" << (pgo_debug_output_system_applied ? 1 : 0)
             << " pgo_out_t=" << pgo_debug_output_time
             << " pgo_out=[" << pgo_debug_output_pos.transpose() << "]"
             << " pgo_out_delta=[" << pgo_debug_output_delta.transpose() << "]"
             << " pgo_out_delta_norm=" << pgo_debug_output_delta.norm()
             << " pgo_pos_valid=" << (pgo_debug_constraint_valid ? 1 : 0)
             << " pgo_pos_source=" << pgo_debug_constraint_source
             << " pgo_pos_status=" << pgo_debug_constraint_status
             << " pgo_pos_added=" << (pgo_debug_constraint_added ? 1 : 0)
             << " pgo_pos_t=" << pgo_debug_constraint_time
             << " pgo_pos_dt=" << pgo_debug_constraint_dt
             << " pgo_pos=[" << pgo_debug_constraint_pos.transpose() << "]";
        if (tf_valid)
        {
            const Transformation &tf = world_map_transform.getTransform();
        diag << " tf_s=" << tf.s
             << " tf_t=[" << tf.t.transpose() << "]"
             << " tf_rot_deg=\"" << rotation_ypr_text(tf.R) << "\"";
        }
        write_rtk_diagnostic_log("STATUS_PANEL", diag.str());
    }
    (void)0;
}

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d());
V3D Lidar_T_wrt_IMU(Zero3d());
M3D Lidar_R_wrt_IMU(Eye3d());


MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

bool evaluate_alignment_continuity_quality(const Transformation &transform,
                                           double &translation_jump,
                                           double &pose_jump,
                                           std::string &reject_reason)
{
    translation_jump = 0.0;
    pose_jump = 0.0;
    reject_reason.clear();

    if (!alignment_continuity_guard_enable ||
        !world_map_transform.isTransformValid() ||
        !transform.valid)
    {
        return true;
    }

    const Transformation &prev_tf = world_map_transform.getTransform();
    translation_jump = (transform.t - prev_tf.t).norm();

    Eigen::Vector3d pos_map_with_offset = state_point.pos;
    if (enable_manual_ekf_rtk_offset)
    {
        pos_map_with_offset += manual_ekf_rtk_offset;
    }

    const Eigen::Vector3d current_world =
        prev_tf.s * (prev_tf.R * pos_map_with_offset) + prev_tf.t;
    const Eigen::Vector3d candidate_world =
        transform.s * (transform.R * pos_map_with_offset) + transform.t;
    pose_jump = (candidate_world - current_world).norm();

    if (alignment_pose_jump_max > 0.0 && pose_jump > alignment_pose_jump_max)
    {
        reject_reason = "pose_jump";
        return false;
    }
    if (alignment_translation_jump_max > 0.0 &&
        translation_jump > alignment_translation_jump_max)
    {
        reject_reason = "translation_jump";
        return false;
    }

    return true;
}

nav_msgs::Odometry odomAftMapped;
nav_msgs::Odometry odomWorld;
nav_msgs::Odometry odomWorldSmooth;
nav_msgs::Odometry odomRtkWorld;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;
geometry_msgs::PoseStamped rtk_msg_body_pose;
geometry_msgs::PoseStamped any_msg_body_pose;
geometry_msgs::PoseStamped world_msg_body_pose;
geometry_msgs::PoseStamped tapm_msg_body_pose;
geometry_msgs::PoseStamped world_smooth_msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());
shared_ptr<RtkProcess> p_rtk(new RtkProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    sig_buffer.notify_all();
}

void print_final_world_body_transform()
{
    if (!final_world_body_tf_enable)
    {
        return;
    }

    if (!world_state.initialized)
    {
        (void)0;
        return;
    }

    Eigen::Quaterniond q_world_imu = world_state.quat;
    const double quat_norm = q_world_imu.norm();
    if (!std::isfinite(quat_norm) || quat_norm < 1e-8)
    {
        (void)0;
        return;
    }
    q_world_imu.normalize();

    const Eigen::Vector3d &t_world_imu = world_state.pos;
    if (!std::isfinite(t_world_imu.x()) ||
        !std::isfinite(t_world_imu.y()) ||
        !std::isfinite(t_world_imu.z()))
    {
        (void)0;
        return;
    }

    const Eigen::Matrix3d R_world_imu = q_world_imu.toRotationMatrix();
    Eigen::Matrix4d T_world_imu = Eigen::Matrix4d::Identity();
    T_world_imu.block<3, 3>(0, 0) = R_world_imu;
    T_world_imu.block<3, 1>(0, 3) = t_world_imu;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "[FINAL_TF] frame=world->body"
        << " stamp=" << lidar_end_time
        << "\n  translation_xyz=[" << t_world_imu.transpose() << "]"
        << "\n  quaternion_wxyz=[" << q_world_imu.w() << " "
        << q_world_imu.x() << " "
        << q_world_imu.y() << " "
        << q_world_imu.z() << "]"
        << "\n  rotation_matrix:\n"
        << R_world_imu
        << "\n  transform_matrix_4x4:\n"
        << T_world_imu;
    (void)0;
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}


inline bool RGBpointBodyToWorldState(PointType const *const pi, PointType *const po)
{
    if (!world_publish_state.initialized)
    {
        return false;
    }

    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Matrix3d R_li_to_imu = state_point.offset_R_L_I.toRotationMatrix();
    Eigen::Vector3d t_li_to_imu = state_point.offset_T_L_I;
    Eigen::Matrix3d R_world_imu = world_publish_state.quat.toRotationMatrix();

    Eigen::Vector3d p_world =
        R_world_imu * (R_li_to_imu * p_body + t_li_to_imu) + world_publish_state.pos;

    po->x = static_cast<float>(p_world.x());
    po->y = static_cast<float>(p_world.y());
    po->z = static_cast<float>(p_world.z());
    po->intensity = pi->intensity;
    return true;
}


inline bool RGBpointBodyToWorldSmooth(PointType const *const pi, PointType *const po)
{
    if (odomWorldSmooth.header.stamp.toSec() <= 0.0)
    {
        return false;
    }

    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Matrix3d R_li_to_imu = state_point.offset_R_L_I.toRotationMatrix();
    Eigen::Vector3d t_li_to_imu = state_point.offset_T_L_I;

    const geometry_msgs::Quaternion &q_msg = odomWorldSmooth.pose.pose.orientation;
    Eigen::Quaterniond q_world_imu(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    Eigen::Matrix3d R_world_imu = q_world_imu.toRotationMatrix();
    Eigen::Vector3d t_world_imu(odomWorldSmooth.pose.pose.position.x,
                                odomWorldSmooth.pose.pose.position.y,
                                odomWorldSmooth.pose.pose.position.z);

    Eigen::Vector3d p_world = R_world_imu * (R_li_to_imu * p_body + t_li_to_imu) + t_world_imu;

    po->x = static_cast<float>(p_world.x());
    po->y = static_cast<float>(p_world.y());
    po->z = static_cast<float>(p_world.z());
    po->intensity = pi->intensity;
    return true;
}

static bool ensure_directory_exists(const std::string &dir_path)
{
    if (dir_path.empty())
    {
        return false;
    }

    struct stat st;
    if (stat(dir_path.c_str(), &st) == 0)
    {
        return S_ISDIR(st.st_mode);
    }

    const size_t last_sep = dir_path.find_last_of('/');
    if (last_sep != std::string::npos)
    {
        const std::string parent = dir_path.substr(0, last_sep);
        if (!parent.empty() && !ensure_directory_exists(parent))
        {
            return false;
        }
    }

    if (mkdir(dir_path.c_str(), 0755) == 0)
    {
        return true;
    }
    if (errno == EEXIST)
    {
        return true;
    }
    return false;
}

static std::string directory_from_file_path(const std::string &file_path)
{
    const size_t last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
    {
        return "";
    }
    return file_path.substr(0, last_sep);
}

static bool initialize_rtk_diagnostic_log()
{
    if (!rtk_diagnostic_log_enable)
    {
        return true;
    }

    if (!rtk_diagnostic_log_path.empty() && rtk_diagnostic_log_path.front() != '/')
    {
        rtk_diagnostic_log_path = root_dir + rtk_diagnostic_log_path;
    }

    const std::string log_dir = directory_from_file_path(rtk_diagnostic_log_path);
    if (!log_dir.empty() && !ensure_directory_exists(log_dir))
    {
        (void)0;
        return false;
    }

    std::lock_guard<mutex> lock(rtk_diagnostic_log_mutex);
    if (rtk_diagnostic_log_file.is_open())
    {
        rtk_diagnostic_log_file.close();
    }
    rtk_diagnostic_log_file.open(rtk_diagnostic_log_path, std::ios::out | std::ios::trunc);
    if (!rtk_diagnostic_log_file.is_open())
    {
        (void)0;
        return false;
    }

    rtk_diagnostic_log_line_count = 0;
    rtk_diagnostic_log_file << "# RTK-LIO diagnostic log\n"
                            << "# file=" << rtk_diagnostic_log_path << "\n";
    rtk_diagnostic_log_file.flush();
    return true;
}

static void write_rtk_diagnostic_log(const std::string &tag, const std::string &detail)
{
    if (!rtk_diagnostic_log_enable)
    {
        return;
    }

    std::lock_guard<mutex> lock(rtk_diagnostic_log_mutex);
    if (!rtk_diagnostic_log_file.is_open())
    {
        rtk_diagnostic_log_file.open(rtk_diagnostic_log_path, std::ios::out | std::ios::app);
    }
    if (!rtk_diagnostic_log_file.is_open())
    {
        return;
    }

    rtk_diagnostic_log_file << std::fixed << std::setprecision(6)
                            << "wall=" << ros::Time::now().toSec()
                            << " lidar=" << lidar_end_time
                            << " tag=" << tag
                            << " " << detail << '\n';
    ++rtk_diagnostic_log_line_count;
    if (rtk_diagnostic_log_line_count % 100 == 0)
    {
        rtk_diagnostic_log_file.flush();
    }
}

static bool initialize_rtk_path_log()
{
    if (!rtk_path_log_enable)
    {
        return true;
    }

    if (rtk_path_log_path.empty())
    {
        rtk_path_log_path = string(ROOT_DIR) + "Log/rtk_path.txt";
    }
    else if (rtk_path_log_path.front() != '/')
    {
        rtk_path_log_path = root_dir + rtk_path_log_path;
    }

    const std::string log_dir = directory_from_file_path(rtk_path_log_path);
    if (!log_dir.empty() && !ensure_directory_exists(log_dir))
    {
        (void)0;
        return false;
    }

    std::lock_guard<mutex> lock(rtk_path_log_mutex);
    if (rtk_path_log_file.is_open())
    {
        rtk_path_log_file.close();
    }
    rtk_path_log_file.open(rtk_path_log_path, std::ios::out | std::ios::trunc);
    if (!rtk_path_log_file.is_open())
    {
        (void)0;
        return false;
    }

    rtk_path_log_line_count = 0;
    last_logged_rtk_path_timestamp = -1.0;
    rtk_path_log_file << "# RTK path log\n"
                      << "# file=" << rtk_path_log_path << "\n"
                      << "# columns: stamp x y z qx qy qz qw fix_type h_acc v_acc "
                      << "vel_x vel_y vel_z vel_acc vel_valid\n";
    rtk_path_log_file.flush();
    (void)0;
    return true;
}

static void write_rtk_path_log(const Timepos &rtk_sample)
{
    if (!rtk_path_log_enable)
    {
        return;
    }
    if (!std::isfinite(rtk_sample.timestamp) ||
        !std::isfinite(rtk_sample.pos.x()) ||
        !std::isfinite(rtk_sample.pos.y()) ||
        !std::isfinite(rtk_sample.pos.z()))
    {
        (void)0;
        return;
    }

    std::lock_guard<mutex> lock(rtk_path_log_mutex);
    if (rtk_sample.timestamp <= last_logged_rtk_path_timestamp + 1e-6)
    {
        return;
    }
    if (!rtk_path_log_file.is_open())
    {
        rtk_path_log_file.open(rtk_path_log_path, std::ios::out | std::ios::app);
    }
    if (!rtk_path_log_file.is_open())
    {
        (void)0;
        return;
    }

    double qx = geoQuat.x;
    double qy = geoQuat.y;
    double qz = geoQuat.z;
    double qw = geoQuat.w;
    const double q_norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (!std::isfinite(q_norm) || q_norm < 1e-8)
    {
        qx = 0.0;
        qy = 0.0;
        qz = 0.0;
        qw = 1.0;
    }
    else
    {
        qx /= q_norm;
        qy /= q_norm;
        qz /= q_norm;
        qw /= q_norm;
    }

    rtk_path_log_file << std::fixed << std::setprecision(9)
                      << rtk_sample.timestamp << " "
                      << rtk_sample.pos.x() << " "
                      << rtk_sample.pos.y() << " "
                      << rtk_sample.pos.z() << " "
                      << qx << " " << qy << " " << qz << " " << qw << " "
                      << rtk_sample.fix_type << " "
                      << rtk_sample.h_acc << " "
                      << rtk_sample.v_acc << " "
                      << rtk_sample.vel.x() << " "
                      << rtk_sample.vel.y() << " "
                      << rtk_sample.vel.z() << " "
                      << rtk_sample.vel_acc << " "
                      << (rtk_sample.vel_valid ? 1 : 0) << "\n";
    if (!rtk_path_log_file.good())
    {
        (void)0;
        return;
    }

    last_logged_rtk_path_timestamp = rtk_sample.timestamp;
    ++rtk_path_log_line_count;
    if (rtk_path_log_flush_every_n > 0 &&
        (rtk_path_log_line_count % static_cast<size_t>(rtk_path_log_flush_every_n) == 0))
    {
        rtk_path_log_file.flush();
    }
}

static void log_rtk_path_samples()
{
    if (!rtk_path_log_enable || Measures.rtk_pos.empty())
    {
        return;
    }

    double last_logged_timestamp = -1.0;
    {
        std::lock_guard<mutex> lock(rtk_path_log_mutex);
        last_logged_timestamp = last_logged_rtk_path_timestamp;
    }
    const auto first_new = std::upper_bound(
        Measures.rtk_pos.begin(),
        Measures.rtk_pos.end(),
        last_logged_timestamp,
        [](double timestamp, const Timepos &sample)
        {
            return timestamp < sample.timestamp;
        });

    for (auto it = first_new; it != Measures.rtk_pos.end(); ++it)
    {
        write_rtk_path_log(*it);
    }
}

static void close_rtk_path_logger()
{
    std::lock_guard<mutex> lock(rtk_path_log_mutex);
    if (!rtk_path_log_file.is_open())
    {
        return;
    }
    rtk_path_log_file.flush();
    rtk_path_log_file.close();
    (void)0;
}

static std::string resolve_export_output_dir()
{
    std::string dir = export_output_dir;
    if (dir.empty())
    {
        dir = root_dir + "data/lidar360";
    }
    else if (dir.front() != '/' && dir.find(':') == std::string::npos)
    {
        dir = root_dir + dir;
    }

    if (!dir.empty() && dir.back() != '/')
    {
        dir.push_back('/');
    }
    return dir;
}

static std::string resolve_export_pose_json_path()
{
    std::string path = export_pose_json_path;
    if (path.empty())
    {
        return resolve_export_output_dir() + "pose.json";
    }

    if (path.front() != '/' && path.find(':') == std::string::npos)
    {
        path = root_dir + path;
    }
    if (path.back() == '/')
    {
        path += "pose.json";
    }
    return path;
}

static bool unix_to_gps_tow(double unix_time, double leap_sec, double time_bias, double *tow_out)
{
    if (tow_out == nullptr)
    {
        return false;
    }
    const double gps_sec = unix_time + leap_sec - time_bias;
    gnss_comm::gtime_t gps_time = gnss_comm::sec2time(gps_sec);
    uint32_t week = 0;
    const double tow = gnss_comm::time2gpst(gps_time, &week);
    if (!std::isfinite(tow))
    {
        return false;
    }
    *tow_out = tow;
    return true;
}

static bool enu_to_lla(const Eigen::Vector3d &enu,
                       const Eigen::Vector3d &ref_lla,
                       Eigen::Vector3d *lla_out)
{
    if (lla_out == nullptr)
    {
        return false;
    }
    const Eigen::Vector3d ref_ecef = gnss_comm::geo2ecef(ref_lla);
    const Eigen::Matrix3d R_ecef_enu = gnss_comm::geo2rotation(ref_lla);
    const Eigen::Vector3d ecef = ref_ecef + R_ecef_enu * enu;
    *lla_out = gnss_comm::ecef2geo(ecef);
    return true;
}

static double normalize_angle_deg(double deg)
{
    double out = std::fmod(deg, 360.0);
    if (out < 0.0)
    {
        out += 360.0;
    }
    return out;
}

static bool export_delay_reached(double current_time)
{
    if (export_start_after_sec <= 0.0)
    {
        return true;
    }
    if (first_lidar_time <= 0.0)
    {
        return false;
    }
    return (current_time - first_lidar_time) >= export_start_after_sec;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

void lasermap_fov_segment()
{

    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;


    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }


    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= mov_threshold * DET_RANGE || dist_to_map_edge[i][1] <= mov_threshold * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;


    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * mov_threshold * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (mov_threshold - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= mov_threshold * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= mov_threshold * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();

    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    RUNTIME_TIMER_STAMP(callback_timer, "lidar_callback", msg->header.stamp.toSec());
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());


    p_pre->process(msg, ptr);
    if (ptr->empty())
    {
        mtx_buffer.unlock();
        (void)0;
        return;
    }


    lidar_buffer.push_back(ptr);


    time_buffer.push_back(msg->header.stamp.toSec() -
                         (lidar_stamp_is_end ? ptr->points.back().curvature / 1000.0 : 0.0));

    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    RUNTIME_TIMER_STAMP(callback_timer, "lidar_callback", msg->header.stamp.toSec());
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + imu_sync_offset - last_timestamp_imu;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;

    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));


    if (abs(timediff_lidar_wrt_imu) > imu_sync_offset && time_sync_en)
    {
        msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

static bool rtk_use_lla_header_time()
{
    return rtk_time_source == "lla_header" || rtk_time_source == "lla" ||
           rtk_time_source == "receive_lla";
}

static bool lla_matches_pvt(const LlaTimeSample &lla,
                            const gnss_comm::GnssPVTSolnMsg &pvt)
{
    return std::fabs(lla.latitude - pvt.latitude) <= rtk_lla_match_latlon_tolerance &&
           std::fabs(lla.longitude - pvt.longitude) <= rtk_lla_match_latlon_tolerance &&
           std::fabs(lla.altitude - pvt.altitude) <= rtk_lla_match_alt_tolerance;
}

static bool rtk_diag_due(size_t count, double now, double &last_log_time, double interval)
{
    if (count <= 5 || count % 100 == 0)
    {
        last_log_time = now;
        return true;
    }
    if (!std::isfinite(last_log_time) || last_log_time < 0.0 || now - last_log_time >= interval)
    {
        last_log_time = now;
        return true;
    }
    return false;
}

static void push_rtk_msg_locked(const gnss_comm::GnssPVTSolnMsg::Ptr &msg)
{
    double unix_timestamp = p_rtk->gpst2time(msg->time.week, msg->time.tow);

    if (unix_timestamp < last_timestamp_rtk)
    {
        rtk_buffer.clear();
        last_published_rtk_odom_timestamp = -1.0;
        last_published_rtk_path_timestamp = -1.0;
    }

    last_timestamp_rtk = unix_timestamp;

    rtk_buffer.push_back(msg);
    ++rtk_pvt_buffer_push_count;
    if (rtk_input_buffer_max_size > 0)
    {
        while (rtk_buffer.size() > static_cast<size_t>(rtk_input_buffer_max_size))
        {
            rtk_buffer.pop_front();
            (void)0;
        }
    }

    if (rtk_diagnostic_log_enable)
    {
        const double now = ros::Time::now().toSec();
        if (rtk_diag_due(rtk_pvt_buffer_push_count, now, rtk_last_push_diag_wall, 1.0))
        {
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "count=" << rtk_pvt_buffer_push_count
                 << " unix=" << unix_timestamp
                 << " week=" << msg->time.week
                 << " tow=" << msg->time.tow
                 << " fix_type=" << static_cast<int>(msg->fix_type)
                 << " valid_fix=" << (msg->valid_fix ? 1 : 0)
                 << " diff_soln=" << (msg->diff_soln ? 1 : 0)
                 << " carr_soln=" << static_cast<int>(msg->carr_soln)
                 << " num_sv=" << static_cast<int>(msg->num_sv)
                 << " h_acc=" << msg->h_acc
                 << " v_acc=" << msg->v_acc
                 << " lat=" << msg->latitude
                 << " lon=" << msg->longitude
                 << " alt=" << msg->altitude
                 << " rtk_buffer=" << rtk_buffer.size()
                 << " pending_pvt=" << rtk_pending_pvt_buffer.size()
                 << " lla_buffer=" << rtk_lla_time_buffer.size()
                 << " time_source=" << rtk_time_source;
            write_rtk_diagnostic_log("RTK_BUFFER_PUSH", diag.str());
        }
    }
}

static bool apply_lla_header_time_to_pvt(gnss_comm::GnssPVTSolnMsg::Ptr &pvt_msg,
                                         const LlaTimeSample &lla)
{
    uint32_t week = 0;
    double tow = 0.0;
    if (!p_rtk->unix2gpst(lla.stamp, week, tow))
    {
        return false;
    }

    pvt_msg->time.week = week;
    pvt_msg->time.tow = tow;
    return true;
}

static bool flush_one_pending_rtk_with_lla_locked()
{
    if (rtk_pending_pvt_buffer.empty())
    {
        return false;
    }

    gnss_comm::GnssPVTSolnMsg::Ptr pvt_msg = rtk_pending_pvt_buffer.front();
    for (auto lla_it = rtk_lla_time_buffer.begin(); lla_it != rtk_lla_time_buffer.end(); ++lla_it)
    {
        if (!lla_matches_pvt(*lla_it, *pvt_msg))
        {
            continue;
        }

        if (apply_lla_header_time_to_pvt(pvt_msg, *lla_it))
        {
            push_rtk_msg_locked(pvt_msg);
        }
        else
        {
            push_rtk_msg_locked(pvt_msg);
            (void)0;
        }

        rtk_pending_pvt_buffer.pop_front();
        rtk_lla_time_buffer.erase(rtk_lla_time_buffer.begin(), std::next(lla_it));
        return true;
    }
    return false;
}

static void flush_pending_rtk_with_lla_locked()
{
    while (flush_one_pending_rtk_with_lla_locked())
    {
    }

    while (rtk_pending_pvt_max_size > 0 &&
           rtk_pending_pvt_buffer.size() > static_cast<size_t>(rtk_pending_pvt_max_size))
    {
        gnss_comm::GnssPVTSolnMsg::Ptr oldest = rtk_pending_pvt_buffer.front();
        rtk_pending_pvt_buffer.pop_front();
        push_rtk_msg_locked(oldest);
        (void)0;
    }
}

void rtk_cbk(const gnss_comm::GnssPVTSolnMsg::ConstPtr &msg_in)
{
    gnss_comm::GnssPVTSolnMsg::Ptr msg(new gnss_comm::GnssPVTSolnMsg(*msg_in));
    ++rtk_pvt_callback_count;

    mtx_buffer.lock();
    if (rtk_use_lla_header_time())
    {
        rtk_pending_pvt_buffer.push_back(msg);
        flush_pending_rtk_with_lla_locked();
    }
    else
    {
        push_rtk_msg_locked(msg);
    }
    if (rtk_diagnostic_log_enable)
    {
        const double now = ros::Time::now().toSec();
        if (rtk_diag_due(rtk_pvt_callback_count, now, rtk_last_pvt_diag_wall, 1.0))
        {
            const double unix_timestamp = p_rtk->gpst2time(msg->time.week, msg->time.tow);
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "count=" << rtk_pvt_callback_count
                 << " unix=" << unix_timestamp
                 << " week=" << msg->time.week
                 << " tow=" << msg->time.tow
                 << " fix_type=" << static_cast<int>(msg->fix_type)
                 << " valid_fix=" << (msg->valid_fix ? 1 : 0)
                 << " diff_soln=" << (msg->diff_soln ? 1 : 0)
                 << " carr_soln=" << static_cast<int>(msg->carr_soln)
                 << " num_sv=" << static_cast<int>(msg->num_sv)
                 << " h_acc=" << msg->h_acc
                 << " v_acc=" << msg->v_acc
                 << " lat=" << msg->latitude
                 << " lon=" << msg->longitude
                 << " alt=" << msg->altitude
                 << " mode=" << (rtk_use_lla_header_time() ? "pending_lla" : "gpst_direct")
                 << " rtk_buffer=" << rtk_buffer.size()
                 << " pending_pvt=" << rtk_pending_pvt_buffer.size()
                 << " lla_buffer=" << rtk_lla_time_buffer.size()
                 << " pushed=" << rtk_pvt_buffer_push_count;
            write_rtk_diagnostic_log("RTK_PVT_CB", diag.str());
        }
    }
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void rtk_lla_cbk(const sensor_msgs::NavSatFix::ConstPtr &msg)
{
    if (!rtk_use_lla_header_time())
    {
        return;
    }

    LlaTimeSample sample;
    sample.stamp = msg->header.stamp.toSec();
    sample.latitude = msg->latitude;
    sample.longitude = msg->longitude;
    sample.altitude = msg->altitude;

    if (!std::isfinite(sample.stamp) || sample.stamp <= 0.0 ||
        !std::isfinite(sample.latitude) ||
        !std::isfinite(sample.longitude) ||
        !std::isfinite(sample.altitude))
    {
        (void)0;
        return;
    }

    mtx_buffer.lock();
    ++rtk_lla_callback_count;
    rtk_lla_time_buffer.push_back(sample);
    while (rtk_lla_time_buffer_max_size > 0 &&
           rtk_lla_time_buffer.size() > static_cast<size_t>(rtk_lla_time_buffer_max_size))
    {
        rtk_lla_time_buffer.pop_front();
    }
    flush_pending_rtk_with_lla_locked();
    if (rtk_diagnostic_log_enable)
    {
        const double now = ros::Time::now().toSec();
        if (rtk_diag_due(rtk_lla_callback_count, now, rtk_last_lla_diag_wall, 1.0))
        {
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "count=" << rtk_lla_callback_count
                 << " stamp=" << sample.stamp
                 << " lat=" << sample.latitude
                 << " lon=" << sample.longitude
                 << " alt=" << sample.altitude
                 << " rtk_buffer=" << rtk_buffer.size()
                 << " pending_pvt=" << rtk_pending_pvt_buffer.size()
                 << " lla_buffer=" << rtk_lla_time_buffer.size()
                 << " pushed=" << rtk_pvt_buffer_push_count;
            write_rtk_diagnostic_log("RTK_LLA_CB", diag.str());
        }
    }
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{

    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }


    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();

        if (meas.lidar->points.size() <= 1)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }

        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {

            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }

        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {


        return false;
    }

    if (rtk_wait_for_time_compensated_rtk && last_timestamp_rtk > 0.0)
    {
        static double waiting_scan = -1.0;
        static std::chrono::steady_clock::time_point waiting_since;
        if (waiting_scan != meas.lidar_beg_time)
        {
            waiting_scan = meas.lidar_beg_time;
            waiting_since = std::chrono::steady_clock::now();
        }
        const double required_rtk_time = lidar_end_time - rtk_time_compensation_wait_tolerance;


        const double rtk_staleness = std::max(lidar_end_time, last_timestamp_imu) - last_timestamp_rtk;
        const double waited = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - waiting_since).count();
        const bool rtk_stream_recent =
            rtk_staleness <= rtk_time_compensation_wait_max_staleness &&
            waited < rtk_time_compensation_wait_max_staleness;
        if (rtk_stream_recent && last_timestamp_rtk < required_rtk_time)
        {
            ++rtk_time_compensation_wait_count;
            if (rtk_diagnostic_log_enable)
            {
                const double now = ros::Time::now().toSec();
                if (rtk_diag_due(rtk_time_compensation_wait_count,
                                 now,
                                 rtk_last_time_compensation_wait_log,
                                 1.0))
                {
                    std::ostringstream diag;
                    diag << std::fixed << std::setprecision(6)
                         << "count=" << rtk_time_compensation_wait_count
                         << " lidar_end=" << lidar_end_time
                         << " last_rtk=" << last_timestamp_rtk
                         << " required_rtk=" << required_rtk_time
                         << " tolerance=" << rtk_time_compensation_wait_tolerance
                         << " staleness=" << rtk_staleness
                         << " max_staleness=" << rtk_time_compensation_wait_max_staleness;
                    write_rtk_diagnostic_log("RTK_TIME_COMP_WAIT", diag.str());
                }
            }
            return false;
        }
    }


    RUNTIME_TIMER_STAMP(sync_timer, "sensor_sync_compute", meas.lidar_beg_time);
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();


    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }


    const double rtk_process_until = lidar_end_time;
    const size_t rtk_buffer_before = rtk_buffer.size();
    const size_t rtk_meas_before = meas.rtk_pos.size();
    const double rtk_front_time_before =
        rtk_buffer.empty() ? -1.0 : p_rtk->gpst2time(rtk_buffer.front()->time.week, rtk_buffer.front()->time.tow);
    const double rtk_back_time_before =
        rtk_buffer.empty() ? -1.0 : p_rtk->gpst2time(rtk_buffer.back()->time.week, rtk_buffer.back()->time.tow);
    p_rtk->Process(meas, rtk_buffer, rtk_process_until);
    ++rtk_process_call_count;
    if (rtk_diagnostic_log_enable)
    {
        const size_t rtk_buffer_after = rtk_buffer.size();
        const size_t rtk_meas_after = meas.rtk_pos.size();
        const size_t rtk_added = (rtk_meas_after >= rtk_meas_before) ? (rtk_meas_after - rtk_meas_before) : 0;
        const bool changed = (rtk_buffer_before != rtk_buffer_after) || (rtk_added > 0);
        const bool log_due = changed ||
                             rtk_process_call_count <= 5 ||
                             rtk_process_call_count % 100 == 0 ||
                             !std::isfinite(rtk_last_process_diag_lidar) ||
                             rtk_last_process_diag_lidar < 0.0 ||
                             lidar_end_time - rtk_last_process_diag_lidar >= 1.0;
        if (log_due)
        {
            rtk_last_process_diag_lidar = lidar_end_time;
            const double rtk_latest_meas_time =
                meas.rtk_pos.empty() ? -1.0 : meas.rtk_pos.back().timestamp;
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "call=" << rtk_process_call_count
                 << " lidar_beg=" << meas.lidar_beg_time
                 << " lidar_end=" << lidar_end_time
                 << " process_until=" << rtk_process_until
                 << " threshold=" << rtk_lio_time_threshold
                 << " buffer_before=" << rtk_buffer_before
                 << " buffer_after=" << rtk_buffer_after
                 << " meas_before=" << rtk_meas_before
                 << " meas_after=" << rtk_meas_after
                 << " added=" << rtk_added
                 << " buffer_front_before=" << rtk_front_time_before
                 << " buffer_back_before=" << rtk_back_time_before
                 << " latest_meas_time=" << rtk_latest_meas_time
                 << " last_timestamp_rtk=" << last_timestamp_rtk
                 << " pvt_cb=" << rtk_pvt_callback_count
                 << " pvt_pushed=" << rtk_pvt_buffer_push_count
                 << " lla_cb=" << rtk_lla_callback_count
                 << " pending_pvt=" << rtk_pending_pvt_buffer.size()
                 << " lla_buffer=" << rtk_lla_time_buffer.size();
            write_rtk_diagnostic_log("RTK_PROCESS", diag.str());
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {

        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "map";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }


    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            (void)0;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}


void publish_frame_world_aligned(const ros::Publisher &pubLaserCloudWorld)
{
    if (!scan_pub_en)
    {
        return;
    }

    if (!world_publish_state.initialized)
    {
        return;
    }

    PointCloudXYZI::Ptr laserCloudSource(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudSource->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorldState(&laserCloudSource->points[i], &laserCloudWorld->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudWorld.publish(laserCloudmsg);
}

static std::string to_lower_ascii(std::string input)
{
    std::transform(input.begin(), input.end(), input.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });
    return input;
}

static ExportTimeBase parse_export_time_base(const std::string &name, bool *valid)
{
    const std::string key = to_lower_ascii(name);
    if (key == "gps" || key == "gps_tow" || key == "tow")
    {
        *valid = true;
        return ExportTimeBase::GPS_TOW;
    }
    if (key == "ros" || key == "ros_time")
    {
        *valid = true;
        return ExportTimeBase::ROS_TIME;
    }
    if (key == "relative" || key == "rel")
    {
        *valid = true;
        return ExportTimeBase::RELATIVE;
    }
    *valid = false;
    return ExportTimeBase::GPS_TOW;
}

static bool export_time_from_unix(double unix_time, double *export_time)
{
    if (export_time_base == ExportTimeBase::ROS_TIME)
    {
        *export_time = unix_time;
        return true;
    }
    if (export_time_base == ExportTimeBase::RELATIVE)
    {
        if (export_time_zero_unix < 0.0)
        {
            export_time_zero_unix = unix_time;
        }
        *export_time = unix_time - export_time_zero_unix;
        return true;
    }

    double gps_tow = 0.0;
    if (!unix_to_gps_tow(unix_time,
                         p_rtk->gpst_leap_seconds_offset_,
                         p_rtk->gpst_time_bias_,
                         &gps_tow))
    {
        return false;
    }
    *export_time = gps_tow;
    return true;
}

static bool get_export_world_pose(Eigen::Vector3d *pos,
                                  Eigen::Quaterniond *quat,
                                  double *pose_time_unix)
{
    if (pos == nullptr || quat == nullptr || pose_time_unix == nullptr)
    {
        return false;
    }
    if (odomWorldSmooth.header.stamp.toSec() > 0.0)
    {
        *pos = Eigen::Vector3d(odomWorldSmooth.pose.pose.position.x,
                               odomWorldSmooth.pose.pose.position.y,
                               odomWorldSmooth.pose.pose.position.z);
        *quat = Eigen::Quaterniond(odomWorldSmooth.pose.pose.orientation.w,
                                   odomWorldSmooth.pose.pose.orientation.x,
                                   odomWorldSmooth.pose.pose.orientation.y,
                                   odomWorldSmooth.pose.pose.orientation.z);
        *pose_time_unix = odomWorldSmooth.header.stamp.toSec();
        return true;
    }
    if (!world_publish_state.initialized)
    {
        return false;
    }
    *pos = world_publish_state.pos;
    *quat = world_publish_state.quat;
    *pose_time_unix = lidar_end_time;
    return true;
}

static void imu_pose_to_lidar_pose(const Eigen::Vector3d &pos_world_imu,
                                   const Eigen::Quaterniond &quat_world_imu,
                                   Eigen::Vector3d *pos_world_lidar,
                                   Eigen::Quaterniond *quat_world_lidar)
{
    if (pos_world_lidar == nullptr || quat_world_lidar == nullptr)
    {
        return;
    }
    const Eigen::Matrix3d R_world_imu = quat_world_imu.toRotationMatrix();
    const Eigen::Matrix3d R_li_to_imu = state_point.offset_R_L_I.toRotationMatrix();
    const Eigen::Vector3d t_li_to_imu = state_point.offset_T_L_I;
    *pos_world_lidar = pos_world_imu + R_world_imu * t_li_to_imu;
    *quat_world_lidar = Eigen::Quaterniond(R_world_imu * R_li_to_imu);
    quat_world_lidar->normalize();
}

static bool get_export_pose(Eigen::Vector3d *pos,
                            Eigen::Quaterniond *quat,
                            double *pose_time_unix)
{
    if (export_use_smooth_pose)
    {
        if (odomWorldSmooth.header.stamp.toSec() <= 0.0)
        {
            return false;
        }
        *pos = Eigen::Vector3d(odomWorldSmooth.pose.pose.position.x,
                               odomWorldSmooth.pose.pose.position.y,
                               odomWorldSmooth.pose.pose.position.z);
        *quat = Eigen::Quaterniond(odomWorldSmooth.pose.pose.orientation.w,
                                   odomWorldSmooth.pose.pose.orientation.x,
                                   odomWorldSmooth.pose.pose.orientation.y,
                                   odomWorldSmooth.pose.pose.orientation.z);
        *pose_time_unix = odomWorldSmooth.header.stamp.toSec();
        return true;
    }

    if (!world_publish_state.initialized)
    {
        return false;
    }
    *pos = world_publish_state.pos;
    *quat = world_publish_state.quat;
    *pose_time_unix = lidar_end_time;
    return true;
}

static bool open_export_stream(std::ofstream &ofs, const std::string &path, const std::string &tag)
{
    if (ofs.is_open())
    {
        return true;
    }
    ofs.open(path.c_str(), std::ios::out | std::ios::app);
    if (!ofs.is_open())
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return false;
    }
    return true;
}

static bool open_export_pose_stream(std::ofstream &ofs,
                                    const std::string &path,
                                    const std::string &tag,
                                    bool *truncated)
{
    if (ofs.is_open())
    {
        return true;
    }

    const bool use_trunc = (truncated != nullptr) && !(*truncated);
    const std::ios::openmode mode = std::ios::out | (use_trunc ? std::ios::trunc : std::ios::app);
    ofs.open(path.c_str(), mode);
    if (!ofs.is_open())
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return false;
    }
    if (truncated != nullptr)
    {
        *truncated = true;
    }
    return true;
}

static std::string build_world_tum_timestamp()
{
    time_t raw_time;
    if (export_world_tum_use_wall_time_filename)
    {
        raw_time = std::time(nullptr);
    }
    else
    {
        const double ros_now = ros::Time::now().toSec();
        raw_time = static_cast<time_t>(ros_now > 0.0 ? ros_now : ros::WallTime::now().toSec());
    }

    struct tm tm_buf;
    localtime_r(&raw_time, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    return oss.str();
}

static bool get_world_tum_raw_pose(Eigen::Vector3d *pos,
                                   Eigen::Quaterniond *quat,
                                   double *stamp_sec)
{
    if (pos == nullptr || quat == nullptr || stamp_sec == nullptr)
    {
        return false;
    }
    if (odomWorld.header.stamp.toSec() > 0.0)
    {
        *pos = Eigen::Vector3d(odomWorld.pose.pose.position.x,
                               odomWorld.pose.pose.position.y,
                               odomWorld.pose.pose.position.z);
        *quat = Eigen::Quaterniond(odomWorld.pose.pose.orientation.w,
                                   odomWorld.pose.pose.orientation.x,
                                   odomWorld.pose.pose.orientation.y,
                                   odomWorld.pose.pose.orientation.z);
        *stamp_sec = odomWorld.header.stamp.toSec();
        return true;
    }
    if (!world_publish_state.initialized)
    {
        return false;
    }
    *pos = world_publish_state.pos;
    *quat = world_publish_state.quat;
    *stamp_sec = lidar_end_time;
    return true;
}

static bool get_world_tum_smooth_pose(Eigen::Vector3d *pos,
                                      Eigen::Quaterniond *quat,
                                      double *stamp_sec)
{
    if (pos == nullptr || quat == nullptr || stamp_sec == nullptr)
    {
        return false;
    }
    if (odomWorldSmooth.header.stamp.toSec() <= 0.0)
    {
        return false;
    }
    *pos = Eigen::Vector3d(odomWorldSmooth.pose.pose.position.x,
                           odomWorldSmooth.pose.pose.position.y,
                           odomWorldSmooth.pose.pose.position.z);
    *quat = Eigen::Quaterniond(odomWorldSmooth.pose.pose.orientation.w,
                               odomWorldSmooth.pose.pose.orientation.x,
                               odomWorldSmooth.pose.pose.orientation.y,
                               odomWorldSmooth.pose.pose.orientation.z);
    *stamp_sec = odomWorldSmooth.header.stamp.toSec();
    return true;
}

static bool ensure_world_tum_stream(WorldTumStreamState *state,
                                    const std::string &file_prefix,
                                    const char *stream_tag)
{
    if (state == nullptr)
    {
        return false;
    }
    if (state->ofs.is_open())
    {
        return true;
    }

    if (state->path.empty())
    {
        const std::string dir = resolve_export_output_dir();
        if (dir.empty())
        {
            return false;
        }
        if (!ensure_directory_exists(dir))
        {
            (void)0;
            return false;
        }
        if (world_tum_session_timestamp.empty())
        {
            world_tum_session_timestamp = build_world_tum_timestamp();
        }

        std::string prefix = file_prefix.empty() ? std::string("world_traj") : file_prefix;
        state->path = dir + prefix + "_" + world_tum_session_timestamp + ".tum";
    }

    const std::ios::openmode mode = state->stream_ready ? (std::ios::out | std::ios::app) : std::ios::out;
    state->ofs.open(state->path.c_str(), mode);
    if (!state->ofs.is_open())
    {
        (void)0;
        return false;
    }
    if (!state->stream_ready)
    {
        (void)0;
    }
    state->stream_ready = true;
    return true;
}

static bool write_world_tum_line(WorldTumStreamState *state,
                                 const Eigen::Vector3d &pos,
                                 Eigen::Quaterniond quat,
                                 double stamp_sec,
                                 const char *stream_tag)
{
    if (state == nullptr || !state->ofs.is_open())
    {
        return false;
    }
    if (!std::isfinite(stamp_sec))
    {
        return false;
    }
    if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z()))
    {
        (void)0;
        return false;
    }
    if (export_world_tum_require_monotonic_stamp && stamp_sec < state->last_stamp)
    {
        (void)0;
        return false;
    }

    const double q_norm = quat.norm();
    if (!std::isfinite(q_norm) || q_norm < 1e-8)
    {
        (void)0;
        return false;
    }
    quat.normalize();

    state->ofs << std::fixed << std::setprecision(9)
               << stamp_sec << " "
               << pos.x() << " " << pos.y() << " " << pos.z() << " "
               << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
    if (!state->ofs.good())
    {
        (void)0;
        return false;
    }

    state->last_stamp = stamp_sec;
    ++state->write_count;
    if (export_world_tum_flush_every_n > 0 &&
        (state->write_count % static_cast<size_t>(export_world_tum_flush_every_n) == 0))
    {
        state->ofs.flush();
    }
    return true;
}

static void close_world_tum_stream(WorldTumStreamState *state, const char *stream_tag)
{
    if (state == nullptr || !state->ofs.is_open())
    {
        return;
    }
    state->ofs.flush();
    state->ofs.close();
    (void)0;
}

static void close_world_tum_loggers()
{
    close_world_tum_stream(&world_tum_raw_stream, "raw");
    close_world_tum_stream(&world_tum_smooth_stream, "smooth");
}

void export_world_tum_trajectories()
{
    if (!export_world_tum_enable)
    {
        return;
    }
    if (!export_world_tum_raw_enable && !export_world_tum_smooth_enable)
    {
        if (!world_tum_stream_warned)
        {
            (void)0;
            world_tum_stream_warned = true;
        }
        return;
    }

    if (export_world_tum_raw_enable)
    {
        Eigen::Vector3d raw_pos;
        Eigen::Quaterniond raw_quat;
        double raw_stamp = 0.0;
        if (get_world_tum_raw_pose(&raw_pos, &raw_quat, &raw_stamp))
        {
            if (export_world_tum_delay_enable && !export_delay_reached(raw_stamp))
            {
                static bool warned_raw_delay = false;
                if (!warned_raw_delay)
                {
                    (void)0;
                    warned_raw_delay = true;
                }
            }
            else if (ensure_world_tum_stream(&world_tum_raw_stream,
                                             export_world_tum_file_prefix_raw,
                                             "raw"))
            {
                write_world_tum_line(&world_tum_raw_stream, raw_pos, raw_quat, raw_stamp, "raw");
            }
        }
    }

    if (export_world_tum_smooth_enable)
    {
        Eigen::Vector3d smooth_pos;
        Eigen::Quaterniond smooth_quat;
        double smooth_stamp = 0.0;
        if (get_world_tum_smooth_pose(&smooth_pos, &smooth_quat, &smooth_stamp))
        {
            if (export_world_tum_delay_enable && !export_delay_reached(smooth_stamp))
            {
                static bool warned_smooth_delay = false;
                if (!warned_smooth_delay)
                {
                    (void)0;
                    warned_smooth_delay = true;
                }
            }
            else if (ensure_world_tum_stream(&world_tum_smooth_stream,
                                             export_world_tum_file_prefix_smooth,
                                             "smooth"))
            {
                write_world_tum_line(&world_tum_smooth_stream, smooth_pos, smooth_quat, smooth_stamp, "smooth");
            }
        }
    }
}

void export_lidar360_ascii_cloud()
{
    if (!export_lidar360_enable || !export_ascii_cloud_enable)
    {
        return;
    }
    if (export_use_smooth_pose)
    {
        if (odomWorldSmooth.header.stamp.toSec() <= 0.0)
        {
            return;
        }
    }
    else
    {
        if (!world_publish_state.initialized)
        {
            return;
        }
    }
    if (!export_delay_reached(lidar_end_time))
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return;
    }
    if (Measures.lidar_beg_time <= 0.0)
    {
        return;
    }

    PointCloudXYZI::Ptr laserCloudSource(dense_pub_en ? feats_undistort : feats_down_body);
    if (!laserCloudSource || laserCloudSource->empty())
    {
        return;
    }

    static std::ofstream cloud_ofs;
    static bool stream_ready = false;
    static std::string cloud_path;
    if (!stream_ready)
    {
        const std::string dir = resolve_export_output_dir();
        if (dir.empty())
        {
            return;
        }
        if (!ensure_directory_exists(dir))
        {
            static bool warned = false;
            if (!warned)
            {
                (void)0;
                warned = true;
            }
            return;
        }
        cloud_path = dir + "points_ascii.txt";
        if (!open_export_stream(cloud_ofs, cloud_path, "ascii_cloud"))
        {
            return;
        }
        stream_ready = true;
    }
    else if (!open_export_stream(cloud_ofs, cloud_path, "ascii_cloud"))
    {
        return;
    }

    const double base_unix_time = Measures.lidar_beg_time;

    PointType world_pt;
    const bool use_smooth_pose = export_use_smooth_pose;
    for (size_t i = 0; i < laserCloudSource->points.size(); ++i)
    {
        const PointType &src_pt = laserCloudSource->points[i];
        const bool converted = use_smooth_pose
                                   ? RGBpointBodyToWorldSmooth(&src_pt, &world_pt)
                                   : RGBpointBodyToWorldState(&src_pt, &world_pt);
        if (!converted)
        {
            continue;
        }
        const double point_unix_time = base_unix_time + src_pt.curvature / 1000.0;
        if (!export_delay_reached(point_unix_time))
        {
            continue;
        }
        double export_time = 0.0;
        if (!export_time_from_unix(point_unix_time, &export_time))
        {
            static bool warned = false;
            if (!warned)
            {
                (void)0;
                warned = true;
            }
            return;
        }
        cloud_ofs << std::fixed << std::setprecision(6)
                  << world_pt.x << "," << world_pt.y << "," << world_pt.z << ","
                  << std::setprecision(9) << export_time << ","
                  << std::setprecision(3) << world_pt.intensity << "\n";
    }
    cloud_ofs.flush();
}

void export_lidar360_trajectory()
{
    if (!export_lidar360_enable || !export_trajectory_enable)
    {
        return;
    }
    Eigen::Vector3d export_pos;
    Eigen::Quaterniond export_quat;
    double pose_time_unix = 0.0;
    if (!get_export_pose(&export_pos, &export_quat, &pose_time_unix))
    {
        return;
    }
    if (!export_delay_reached(pose_time_unix))
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return;
    }
    if (!p_rtk->rtk_isvalid ||
        p_rtk->first_latitude <= 0.0 ||
        p_rtk->first_longitude <= 0.0 ||
        p_rtk->first_altitude <= 0.0)
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return;
    }

    double export_time = 0.0;
    if (!export_time_from_unix(pose_time_unix, &export_time))
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return;
    }

    const Eigen::Vector3d ref_lla(p_rtk->first_latitude,
                                  p_rtk->first_longitude,
                                  p_rtk->first_altitude);
    Eigen::Vector3d lla;
    if (!enu_to_lla(export_pos, ref_lla, &lla))
    {
        return;
    }

    const double grid_x = export_pos.x() + export_offset_x;
    const double grid_y = export_pos.y() + export_offset_y;
    const double grid_z = export_pos.z() + export_offset_z;
    const double height = lla.z() + export_offset_z;

    export_quat.normalize();
    const Eigen::Vector3d euler_rad = RotMtoEuler(export_quat.toRotationMatrix());
    const double roll_deg = euler_rad.x() * 180.0 / M_PI;
    const double pitch_deg = euler_rad.y() * 180.0 / M_PI;
    const double yaw_deg = euler_rad.z() * 180.0 / M_PI;
    const double heading_deg = normalize_angle_deg(90.0 - yaw_deg);

    static std::ofstream traj_ofs;
    static bool stream_ready = false;
    static std::string traj_path;
    if (!stream_ready)
    {
        const std::string dir = resolve_export_output_dir();
        if (dir.empty())
        {
            return;
        }
        if (!ensure_directory_exists(dir))
        {
            static bool warned = false;
            if (!warned)
            {
                (void)0;
                warned = true;
            }
            return;
        }
        traj_path = dir + "trajectory.txt";
        if (!open_export_stream(traj_ofs, traj_path, "trajectory"))
        {
            return;
        }
        stream_ready = true;
    }
    else if (!open_export_stream(traj_ofs, traj_path, "trajectory"))
    {
        return;
    }

    traj_ofs << std::fixed << std::setprecision(9)
             << export_time << ","
             << std::setprecision(8) << lla.x() << "," << lla.y() << ","
             << std::setprecision(3) << height << ","
             << std::setprecision(3) << grid_x << "," << grid_y << "," << grid_z << ","
             << std::setprecision(3) << roll_deg << ","
             << std::setprecision(3) << pitch_deg << ","
             << std::setprecision(3) << heading_deg << "\n";
    traj_ofs.flush();
}

void export_lidar360_pose_json()
{
    if (!export_lidar360_enable || !export_pose_json_enable)
    {
        return;
    }

    Eigen::Vector3d export_pos;
    Eigen::Quaterniond export_quat;
    double pose_time_unix = 0.0;
    if (!get_export_pose(&export_pos, &export_quat, &pose_time_unix))
    {
        return;
    }
    if (!export_delay_reached(pose_time_unix))
    {
        return;
    }

    static std::ofstream pose_ofs;
    static bool stream_ready = false;
    static bool pose_truncated = false;
    static std::string pose_path;
    if (!stream_ready)
    {
        pose_path = resolve_export_pose_json_path();
        if (pose_path.empty())
        {
            return;
        }
        const size_t last_sep = pose_path.find_last_of('/');
        if (last_sep != std::string::npos)
        {
            const std::string dir = pose_path.substr(0, last_sep);
            if (!dir.empty() && !ensure_directory_exists(dir))
            {
                static bool warned = false;
                if (!warned)
                {
                    (void)0;
                    warned = true;
                }
                return;
            }
        }
        if (!open_export_pose_stream(pose_ofs, pose_path, "pose_json", &pose_truncated))
        {
            return;
        }
        stream_ready = true;
    }
    else if (!open_export_pose_stream(pose_ofs, pose_path, "pose_json", &pose_truncated))
    {
        return;
    }

    Eigen::Vector3d pose_pos = export_pos;
    Eigen::Quaterniond pose_quat = export_quat;
    if (export_use_lidar_pose)
    {
        imu_pose_to_lidar_pose(export_pos, export_quat, &pose_pos, &pose_quat);
    }

    pose_quat.normalize();
    pose_ofs << std::fixed << std::setprecision(9)
             << pose_pos.x() << " "
             << pose_pos.y() << " "
             << pose_pos.z() << " "
             << pose_quat.w() << " "
             << pose_quat.x() << " "
             << pose_quat.y() << " "
             << pose_quat.z() << "\n";
    pose_ofs.flush();
}

void export_lidar360_hba_bundle()
{
    if (!export_hba_enable)
    {
        return;
    }

    Eigen::Vector3d export_pos;
    Eigen::Quaterniond export_quat;
    double pose_time_unix = 0.0;
    if (!get_export_world_pose(&export_pos, &export_quat, &pose_time_unix))
    {
        return;
    }
    if (!export_delay_reached(pose_time_unix))
    {
        return;
    }

    PointCloudXYZI::Ptr laserCloudSource(dense_pub_en ? feats_undistort : feats_down_body);
    if (!laserCloudSource || laserCloudSource->empty())
    {
        return;
    }

    static std::ofstream pose_ofs;
    static bool pose_stream_ready = false;
    static bool pose_truncated = false;
    static std::string pose_path;
    if (!pose_stream_ready)
    {
        pose_path = resolve_export_pose_json_path();
        if (pose_path.empty())
        {
            return;
        }
        const size_t last_sep = pose_path.find_last_of('/');
        if (last_sep != std::string::npos)
        {
            const std::string dir = pose_path.substr(0, last_sep);
            if (!dir.empty() && !ensure_directory_exists(dir))
            {
                static bool warned = false;
                if (!warned)
                {
                    (void)0;
                    warned = true;
                }
                return;
            }
        }
        if (!open_export_pose_stream(pose_ofs, pose_path, "pose_json", &pose_truncated))
        {
            return;
        }
        pose_stream_ready = true;
    }
    else if (!open_export_pose_stream(pose_ofs, pose_path, "pose_json", &pose_truncated))
    {
        return;
    }

    static std::string pcd_dir;
    static bool dir_ready = false;
    static int pcd_index = 0;
    if (!dir_ready)
    {
        pcd_dir = resolve_export_output_dir() + "pcd/";
        if (!ensure_directory_exists(pcd_dir))
        {
            static bool warned = false;
            if (!warned)
            {
                (void)0;
                warned = true;
            }
            return;
        }
        dir_ready = true;
    }
    else if (!ensure_directory_exists(pcd_dir))
    {
        return;
    }

    Eigen::Vector3d pose_pos;
    Eigen::Quaterniond pose_quat;
    imu_pose_to_lidar_pose(export_pos, export_quat, &pose_pos, &pose_quat);
    pose_quat.normalize();

    pose_ofs << std::fixed << std::setprecision(9)
             << pose_pos.x() << " "
             << pose_pos.y() << " "
             << pose_pos.z() << " "
             << pose_quat.w() << " "
             << pose_quat.x() << " "
             << pose_quat.y() << " "
             << pose_quat.z() << "\n";
    pose_ofs.flush();

    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
    cloud_xyz.points.resize(laserCloudSource->points.size());
    for (size_t i = 0; i < laserCloudSource->points.size(); ++i)
    {
        cloud_xyz.points[i].x = laserCloudSource->points[i].x;
        cloud_xyz.points[i].y = laserCloudSource->points[i].y;
        cloud_xyz.points[i].z = laserCloudSource->points[i].z;
    }
    cloud_xyz.width = static_cast<uint32_t>(cloud_xyz.points.size());
    cloud_xyz.height = 1;
    cloud_xyz.is_dense = false;

    std::ostringstream oss;
    if (export_pcd_name_fill_num > 0)
    {
        oss << std::setw(export_pcd_name_fill_num)
            << std::setfill('0') << pcd_index;
    }
    else
    {
        oss << pcd_index;
    }
    const std::string pcd_path = pcd_dir + oss.str() + ".pcd";

    pcl::PCDWriter pcd_writer;
    if (pcd_writer.writeBinary(pcd_path, cloud_xyz) != 0)
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return;
    }
    ++pcd_index;
}

void export_lidar360_hba_pcd()
{
    if (!export_lidar360_enable)
    {
        return;
    }

    Eigen::Vector3d export_pos;
    Eigen::Quaterniond export_quat;
    double pose_time_unix = 0.0;
    if (!get_export_pose(&export_pos, &export_quat, &pose_time_unix))
    {
        return;
    }
    if (!export_delay_reached(pose_time_unix))
    {
        return;
    }

    PointCloudXYZI::Ptr laserCloudSource(dense_pub_en ? feats_undistort : feats_down_body);
    if (!laserCloudSource || laserCloudSource->empty())
    {
        return;
    }

    static std::string pcd_dir;
    static bool dir_ready = false;
    static int pcd_index = 0;
    if (!dir_ready)
    {
        pcd_dir = resolve_export_output_dir() + "pcd/";
        if (!ensure_directory_exists(pcd_dir))
        {
            static bool warned = false;
            if (!warned)
            {
                (void)0;
                warned = true;
            }
            return;
        }
        dir_ready = true;
    }
    else if (!ensure_directory_exists(pcd_dir))
    {
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
    cloud_xyz.points.resize(laserCloudSource->points.size());
    for (size_t i = 0; i < laserCloudSource->points.size(); ++i)
    {
        cloud_xyz.points[i].x = laserCloudSource->points[i].x;
        cloud_xyz.points[i].y = laserCloudSource->points[i].y;
        cloud_xyz.points[i].z = laserCloudSource->points[i].z;
    }
    cloud_xyz.width = static_cast<uint32_t>(cloud_xyz.points.size());
    cloud_xyz.height = 1;
    cloud_xyz.is_dense = false;

    std::ostringstream oss;
    if (export_pcd_name_fill_num > 0)
    {
        oss << std::setw(export_pcd_name_fill_num)
            << std::setfill('0') << pcd_index;
    }
    else
    {
        oss << pcd_index;
    }
    const std::string pcd_path = pcd_dir + oss.str() + ".pcd";

    pcl::PCDWriter pcd_writer;
    if (pcd_writer.writeBinary(pcd_path, cloud_xyz) != 0)
    {
        static bool warned = false;
        if (!warned)
        {
            (void)0;
            warned = true;
        }
        return;
    }
    ++pcd_index;
}


void publish_frame_world_pgo(const ros::Publisher &pubLaserCloudWorldPgo)
{
    if (!scan_pub_en)
    {
        return;
    }
    if (!pgo_sw_enable_cloud)
    {
        return;
    }
    if (!pgo_sw_enable)
    {
        return;
    }
    if (odomWorldSmooth.header.stamp.toSec() <= 0.0)
    {
        return;
    }

    PointCloudXYZI::Ptr laserCloudSource(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudSource->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        if (!RGBpointBodyToWorldSmooth(&laserCloudSource->points[i], &laserCloudWorld->points[i]))
        {
            return;
        }
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = odomWorldSmooth.header.stamp;
    laserCloudmsg.header.frame_id = "world";
    pubLaserCloudWorldPgo.publish(laserCloudmsg);
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "map";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "map";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.pose.position.x = state_point.pos(0);
    out.pose.pose.position.y = state_point.pos(1);
    out.pose.pose.position.z = state_point.pos(2);
    out.pose.pose.orientation.x = geoQuat.x;
    out.pose.pose.orientation.y = geoQuat.y;
    out.pose.pose.orientation.z = geoQuat.z;
    out.pose.pose.orientation.w = geoQuat.w;
    out.twist.twist.linear.x = state_point.vel(0);
    out.twist.twist.linear.y = state_point.vel(1);
    out.twist.twist.linear.z = state_point.vel(2);
}


template <typename T>
void set_posestamp_(T &out)
{

    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}
template <typename T>
void set_rtkposestamp_(T &out)
{
    Timepos current_rtk_pos;
    current_rtk_pos = Measures.rtk_pos.back();
    V3D rtk_pos = current_rtk_pos.pos;

    out.pose.position.x = rtk_pos(0);
    out.pose.position.y = rtk_pos(1);
    out.pose.position.z = rtk_pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}
template <typename T>
void set_anyposestamp_(T &out)
{
    V3D lio_pos = state_point.pos;
    Eigen::Matrix3d lio_rot = state_point.rot.toRotationMatrix();
    Eigen::Vector3d world_pos = lio_pos;
    Eigen::Quaterniond world_quat(lio_rot);

    if (Global_GNSS_LIO.valid)
    {
        world_pos = Global_GNSS_LIO.s * (Global_GNSS_LIO.R * lio_pos) + Global_GNSS_LIO.t;
        Eigen::Matrix3d rot_world = Global_GNSS_LIO.R * lio_rot;
        world_quat = Eigen::Quaterniond(rot_world);
    }

    world_quat.normalize();

    out.pose.position.x = world_pos.x();
    out.pose.position.y = world_pos.y();
    out.pose.position.z = world_pos.z();
    out.pose.orientation.x = world_quat.x();
    out.pose.orientation.y = world_quat.y();
    out.pose.orientation.z = world_quat.z();
    out.pose.orientation.w = world_quat.w();
}

template <typename T>
bool set_worldposestamp_(T &out)
{
    if (!world_publish_state.initialized)
    {
        return false;
    }

    Eigen::Quaterniond world_quat = world_publish_state.quat;
    world_quat.normalize();

    out.pose.position.x = world_publish_state.pos.x();
    out.pose.position.y = world_publish_state.pos.y();
    out.pose.position.z = world_publish_state.pos.z();
    out.pose.orientation.x = world_quat.x();
    out.pose.orientation.y = world_quat.y();
    out.pose.orientation.z = world_quat.z();
    out.pose.orientation.w = world_quat.w();
    return true;
}

Eigen::Vector3d transform_world_to_map(const Eigen::Vector3d &world_pos, const Transformation &transform)
{
    if (!transform.valid || std::abs(transform.s) < 1e-6)
    {
        return world_pos;
    }
    Eigen::Matrix3d R_inv = transform.R.transpose();
    Eigen::Vector3d shifted = (world_pos - transform.t) / transform.s;
    return R_inv * shifted;
}

Eigen::Vector3d transform_world_to_map_rotation_only(const Eigen::Vector3d &world_pos,
                                                     const Transformation &transform)
{
    if (!transform.valid || std::abs(transform.s) < 1e-6)
    {
        return world_pos;
    }
    Eigen::Matrix3d R_inv = transform.R.transpose();
    return (1.0 / transform.s) * (R_inv * world_pos);
}

Eigen::Vector3d transform_map_to_world_with_transform(const Eigen::Vector3d &map_pos,
                                                      const Transformation &transform)
{
    if (!transform.valid || std::abs(transform.s) < 1e-6)
    {
        return map_pos;
    }
    return transform.s * (transform.R * map_pos) + transform.t;
}

bool is_rtk_window_ready_for_unified_update()
{
    if (!rtk_fix_active)
    {
        return false;
    }
    return ekftimepos_state_flag && rtktimepos_state_flag &&
           (alignment_latest_window_size >= (rtk_match_at_gnss_epochs ? rtk_alignment_min_pairs : alignment_window_current));
}

bool is_alignment_window_ready_for_tapm()
{
    if (!rtk_initial_alignment_done || !world_map_transform.isTransformValid())
    {
        return false;
    }
    if (!is_rtk_window_ready_for_unified_update())
    {
        return false;
    }
    if (last_rtk_match_ratio < rtk_match_min_ratio)
    {
        return false;
    }
    return true;
}

bool is_pgo_raw_rtk_allowed()
{
    return pgo_raw_rtk_allowed &&
           rtk_fusion_state == RtkFusionState::CORRECTION &&
           rtk_fix_active &&
           world_map_transform.isTransformValid();
}

void apply_alignment_transform(const Transformation &transform, double confidence, double timestamp, bool suppress_log)
{

    Transformation working = transform;
    Transformation blended = working;
    const double alpha = std::max(0.0, std::min(world_map_transform_alpha, 1.0));

    if (working.valid && world_map_transform.isTransformValid())
    {
        const bool skip_jump_check =
            (rtk_fusion_state == RtkFusionState::SMOOTHING) && !smoothing_first_alignment_done;
        const Transformation &prev = world_map_transform.getTransform();
        const Eigen::Vector3d delta_translation_vec = working.t - prev.t;
        const double delta_translation = delta_translation_vec.norm();
        Eigen::Matrix3d R_delta = prev.R.transpose() * working.R;
        Eigen::AngleAxisd aa_delta(R_delta);
        const double delta_rotation_deg = aa_delta.angle() * 180.0 / M_PI;
        Eigen::Vector3d delta_zyx_deg = R_delta.eulerAngles(2, 1, 0) * 180.0 / M_PI;
        const double max_delta_zyx = delta_zyx_deg.cwiseAbs().maxCoeff();

        const bool large_delta = (delta_translation > 1.0);
        if (large_delta)
        {
        }
        else
        {
        }

        if (world_map_transform_max_jump > 0.0 &&
            !skip_jump_check &&
            delta_translation > world_map_transform_max_jump)
        {
            log_measurement_event("[ALIGN]", "reject_update large_jump delta=" + std::to_string(delta_translation));
            log_alignment_event("[ALIGN_REJECT]", transform, "reason=jump delta=" + std::to_string(delta_translation));
            return;
        }

        if (skip_jump_check)
        {
            log_measurement_event("[ALIGN]", "first_smoothing_alignment_skip_jump_check delta=" +
                                             std::to_string(delta_translation));
        }
    }

    if (alpha < 1.0 && working.valid && world_map_transform.isTransformValid())
    {
        const Transformation &prev = world_map_transform.getTransform();

        blended.t = alpha * working.t + (1.0 - alpha) * prev.t;
        blended.s = alpha * working.s + (1.0 - alpha) * prev.s;

        Eigen::Quaterniond q_prev(prev.R);
        Eigen::Quaterniond q_new(working.R);
        q_prev.normalize();
        q_new.normalize();
        Eigen::Quaterniond q_blended = q_prev.slerp(alpha, q_new);
        blended.R = q_blended.toRotationMatrix();
        blended.valid = working.valid;
    }

    Global_GNSS_LIO = blended;
    world_map_transform.updateTransform(blended, confidence, timestamp, suppress_log);
    std::ostringstream oss;
    oss << "confidence=" << confidence
        << " timestamp=" << timestamp
        << " alpha=" << alpha;
    log_alignment_event("[ALIGN_APPLY]", blended, oss.str());

    if (rtk_fusion_state == RtkFusionState::SMOOTHING && !smoothing_first_alignment_done)
    {
        smoothing_first_alignment_done = true;
    }
}

void clear_alignment_buffers(const std::string &reason = "unspecified")
{
    const size_t ekf_size = ekftimepos_buffer.size();
    const size_t rtk_size = rtktimepos_buffer.size();
    const size_t matched_size = matched_ekf_window.size();
    log_measurement_event("[ALIGN]", "clear_buffers reason=" + reason +
                                         " ekf=" + std::to_string(ekf_size) +
                                         " rtk=" + std::to_string(rtk_size) +
                                         " matched=" + std::to_string(matched_size));
    ekftimepos_buffer.clear();
    ekftimepos_state_flag = false;
    rtktimepos_buffer.clear();
    rtktimepos_state_flag = false;
    matched_rtk_window.clear();
    matched_ekf_window.clear();
    last_rtk_match_ratio = 0.0;
    alignment_rotation_stable = false;
    alignment_rotation_stable_enter_time = -1.0;
    alignment_latest_window_size = 0;
}

void reset_alignment_rotation_freeze(const std::string &reason)
{
    if (!alignment_rotation_frozen)
    {
        return;
    }
    alignment_rotation_frozen = false;
    alignment_frozen_rotation.setIdentity();
    tapm_anchor_valid = false;
    tapm_anchor_time = -1.0;
    tapm_recover_time = -1.0;
    tapm_anchor_transform.valid = false;
    tapm_output_active = false;
    tapm_publish_pos.setZero();
    tapm_publish_weight = 0.0;
    log_measurement_event("[ALIGN]", "rotation_unfrozen reason=" + reason);
}

bool get_active_alignment_transform(Transformation &out_transform)
{
    const bool official_valid = world_map_transform.isTransformValid();
    bool has_transform = false;

    if (rtk_fusion_state == RtkFusionState::CORRECTION || !smoothing_alignment_valid)
    {
        if (official_valid)
        {
            out_transform = world_map_transform.getTransform();
            has_transform = true;
        }
        else if (smoothing_alignment_valid)
        {
            out_transform = smoothing_alignment;
            has_transform = true;
        }
    }
    else if (smoothing_alignment_valid)
    {
        out_transform = smoothing_alignment;
        has_transform = true;
    }
    else if (official_valid)
    {
        out_transform = world_map_transform.getTransform();
        has_transform = true;
    }

    if (!has_transform)
    {
        return false;
    }

    return true;
}

bool load_initial_world_transform_from_params(ros::NodeHandle &nh)
{
    bool enable = false;
    nh.param<bool>("rtk/initial_world_transform_enable", enable, false);
    if (!enable)
    {
        return false;
    }

    std::vector<double> r_vec{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    std::vector<double> t_vec{0.0, 0.0, 0.0};
    double scale = 1.0;
    double confidence = 1.0;
    bool freeze_rotation = false;
    nh.param<std::vector<double>>("rtk/initial_world_transform_R", r_vec, r_vec);
    nh.param<std::vector<double>>("rtk/initial_world_transform_t", t_vec, t_vec);
    nh.param<double>("rtk/initial_world_transform_scale", scale, 1.0);
    nh.param<double>("rtk/initial_world_transform_confidence", confidence, 1.0);
    nh.param<bool>("rtk/initial_world_transform_freeze_rotation", freeze_rotation, false);

    if (r_vec.size() != 9 || t_vec.size() != 3 || !std::isfinite(scale) || scale <= 0.0)
    {
        (void)0;
        return false;
    }

    Transformation tf;
    tf.R << r_vec[0], r_vec[1], r_vec[2],
            r_vec[3], r_vec[4], r_vec[5],
            r_vec[6], r_vec[7], r_vec[8];
    tf.t = Eigen::Vector3d(t_vec[0], t_vec[1], t_vec[2]);
    tf.s = scale;
    tf.valid = true;

    if (!tf.R.allFinite() || !tf.t.allFinite())
    {
        (void)0;
        return false;
    }

    const double clamped_confidence = std::max(0.0, std::min(1.0, confidence));
    Global_GNSS_LIO = tf;
    world_map_transform.updateTransform(tf, clamped_confidence, 0.0, true);
    if (freeze_rotation)
    {
        alignment_rotation_frozen = true;
        alignment_frozen_rotation = tf.R;
    }

    std::ostringstream diag;
    diag << std::fixed << std::setprecision(6)
         << "enabled=1 scale=" << tf.s
         << " confidence=" << clamped_confidence
         << " freeze_rotation=" << (freeze_rotation ? 1 : 0)
         << " t=[" << tf.t.transpose() << "]"
         << " R=[" << tf.R(0, 0) << "," << tf.R(0, 1) << "," << tf.R(0, 2)
         << ";" << tf.R(1, 0) << "," << tf.R(1, 1) << "," << tf.R(1, 2)
         << ";" << tf.R(2, 0) << "," << tf.R(2, 1) << "," << tf.R(2, 2)
         << "]";
    write_rtk_diagnostic_log("INIT_WORLD_TF", diag.str());
    return true;
}

bool bootstrap_world_transform_from_latest_rtk(double timestamp, const std::string &reason)
{
    if (!rtk_bootstrap_transform_on_first_fix ||
        world_map_transform.isTransformValid() ||
        Measures.rtk_pos.empty())
    {
        return false;
    }

    const Timepos &latest_rtk = Measures.rtk_pos.back();
    const double rtk_age = std::abs(latest_rtk.timestamp - timestamp);
    if (rtk_bootstrap_transform_max_rtk_age >= 0.0 &&
        rtk_age > rtk_bootstrap_transform_max_rtk_age)
    {
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(6)
             << "skip=1 reason=rtk_age_exceed trigger=" << reason
             << " lidar_time=" << timestamp
             << " rtk_time=" << latest_rtk.timestamp
             << " age=" << rtk_age
             << " max_age=" << rtk_bootstrap_transform_max_rtk_age;
        write_rtk_diagnostic_log("RTK_BOOTSTRAP_TF", diag.str());
        return false;
    }

    Eigen::Vector3d pos_map = state_point.pos;
    if (enable_manual_ekf_rtk_offset)
    {
        pos_map += manual_ekf_rtk_offset;
    }

    Transformation tf;
    tf.R = Eigen::Matrix3d::Identity();
    tf.t = latest_rtk.pos - pos_map;
    tf.s = 1.0;
    tf.valid = true;

    Global_GNSS_LIO = tf;
    world_map_transform.updateTransform(tf, 1.0, timestamp, true);

    std::ostringstream diag;
    diag << std::fixed << std::setprecision(6)
         << "applied=1 trigger=" << reason
         << " lidar_time=" << timestamp
         << " rtk_time=" << latest_rtk.timestamp
         << " age=" << rtk_age
         << " map=[" << pos_map.transpose() << "]"
         << " rtk=[" << latest_rtk.pos.transpose() << "]"
         << " t=[" << tf.t.transpose() << "]";
    write_rtk_diagnostic_log("RTK_BOOTSTRAP_TF", diag.str());
    return true;
}

void log_fusion_transition(RtkFusionState from_state,
                           RtkFusionState to_state,
                           const std::string &reason,
                           double timestamp)
{
    if (from_state == to_state)
    {
        return;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "[FUSION_TRANSITION] time=" << timestamp
        << " from=" << rtk_fusion_state_to_string(from_state)
        << " to=" << rtk_fusion_state_to_string(to_state)
        << " reason=" << reason
        << " fix_active=" << (rtk_fix_active ? 1 : 0)
        << " fix_count=" << rtk_consecutive_fix_count
        << " tapm_state=" << tapm_runtime_state_to_string(get_tapm_runtime_state())
        << " tapm_frame_count=" << tapm_frame_count;
    write_rtk_diagnostic_log("FUSION_TRANSITION", oss.str());
}

void reset_smoothing_completion_request()
{
    smoothing_completion_requested = false;
    smoothing_completion_reason.clear();
}

void request_smoothing_completion(const std::string &reason)
{
    if (rtk_fusion_state != RtkFusionState::SMOOTHING || smoothing_completion_requested)
    {
        return;
    }
    smoothing_completion_requested = true;
    smoothing_completion_reason = reason;
    write_rtk_diagnostic_log("SMOOTHING_COMPLETE_REQUEST", reason);
}

double get_required_rtk_recover_stable_duration()
{
    if (!rtk_recovery_verify_enable)
    {
        return 0.0;
    }
    if (rtk_last_loss_duration >= rtk_recover_long_loss_threshold_sec)
    {
        return rtk_recover_long_stable_duration_sec;
    }
    return rtk_recover_stable_duration_sec;
}

double get_required_pgo_recovery_ramp_duration()
{
    if (rtk_last_loss_duration >= rtk_recover_long_loss_threshold_sec)
    {
        return pgo_sw_recovery_long_ramp_duration_sec;
    }
    return pgo_sw_recovery_ramp_duration_sec;
}

void reset_rtk_recovery_verify_runtime(const std::string &reason)
{
    rtk_recover_stable_start_time = -1.0;
    rtk_recover_prev_valid = false;
    rtk_recover_latest_dt = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_jump_xy = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_jump_z = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_speed_xy = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_speed_z = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_quality_ok = false;
    rtk_recover_latest_continuity_ok = false;
    rtk_recover_latest_reason = reason;
}

void start_pgo_recovery_ramp(double current_time, const std::string &reason)
{
    pgo_sw_recovery_ramp_start_time = current_time;
    pgo_sw_recovery_ramp_duration_active =
        std::max(0.0, get_required_pgo_recovery_ramp_duration());

    std::ostringstream diag;
    diag << std::fixed << std::setprecision(3)
         << "start reason=" << reason
         << " t=" << current_time
         << " loss_duration=" << rtk_last_loss_duration
         << " ramp_duration=" << pgo_sw_recovery_ramp_duration_active
         << " min_weight=" << pgo_sw_recovery_ramp_min_weight
         << " use_rtk_z=" << (pgo_sw_use_rtk_z ? 1 : 0)
         << " use_tapm_z=" << (pgo_sw_use_tapm_z ? 1 : 0)
         << " recovery_z_sigma_scale=" << pgo_sw_recovery_z_sigma_scale;
    write_rtk_diagnostic_log("RTK_RECOVERY_RAMP", diag.str());
}

double get_pgo_recovery_weight(double current_time)
{
    if (pgo_sw_recovery_ramp_start_time < 0.0 ||
        pgo_sw_recovery_ramp_duration_active <= 0.0)
    {
        return 1.0;
    }
    const double tau = current_time - pgo_sw_recovery_ramp_start_time;
    if (tau >= pgo_sw_recovery_ramp_duration_active)
    {
        return 1.0;
    }
    const double u = std::max(0.0, std::min(1.0, tau / pgo_sw_recovery_ramp_duration_active));
    const double smooth = u * u * (3.0 - 2.0 * u);
    return std::max(pgo_sw_recovery_ramp_min_weight,
                    pgo_sw_recovery_ramp_min_weight +
                        (1.0 - pgo_sw_recovery_ramp_min_weight) * smooth);
}

bool is_pgo_recovery_ramp_active(double current_time)
{
    return get_pgo_recovery_weight(current_time) < 1.0 - 1e-6;
}

void update_rtk_recovery_quality(double reference_time, bool has_new_fix_measurement)
{
    if (!rtk_recovery_verify_enable || !rtk_initial_alignment_done ||
        !has_new_fix_measurement || Measures.rtk_pos.empty())
    {
        return;
    }

    const Timepos &sample = Measures.rtk_pos.back();
    bool quality_ok = true;
    bool continuity_ok = true;
    std::string reason = "ok";

    if (rtk_recover_max_h_acc > 0.0 &&
        (!std::isfinite(sample.h_acc) || sample.h_acc > rtk_recover_max_h_acc))
    {
        quality_ok = false;
        reason = "h_acc";
    }
    if (rtk_recover_max_v_acc > 0.0 &&
        (!std::isfinite(sample.v_acc) || sample.v_acc > rtk_recover_max_v_acc))
    {
        quality_ok = false;
        if (reason == "ok")
        {
            reason = "v_acc";
        }
    }

    rtk_recover_latest_dt = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_jump_xy = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_jump_z = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_speed_xy = std::numeric_limits<double>::quiet_NaN();
    rtk_recover_latest_speed_z = std::numeric_limits<double>::quiet_NaN();

    if (rtk_recover_prev_valid)
    {
        const double dt = sample.timestamp - rtk_recover_prev_sample.timestamp;
        rtk_recover_latest_dt = dt;
        if (!std::isfinite(dt) || dt <= 1e-6 ||
            (rtk_recover_max_sample_dt > 0.0 && dt > rtk_recover_max_sample_dt))
        {
            continuity_ok = false;
            if (reason == "ok")
            {
                reason = "sample_dt";
            }
        }
        else
        {
            const Eigen::Vector3d dp = sample.pos - rtk_recover_prev_sample.pos;
            rtk_recover_latest_jump_xy = dp.head<2>().norm();
            rtk_recover_latest_jump_z = std::abs(dp.z());
            rtk_recover_latest_speed_xy = rtk_recover_latest_jump_xy / dt;
            rtk_recover_latest_speed_z = rtk_recover_latest_jump_z / dt;

            if (rtk_recover_max_jump_xy > 0.0 &&
                rtk_recover_latest_jump_xy > rtk_recover_max_jump_xy)
            {
                continuity_ok = false;
                if (reason == "ok")
                {
                    reason = "jump_xy";
                }
            }
            if (rtk_recover_max_jump_z > 0.0 &&
                rtk_recover_latest_jump_z > rtk_recover_max_jump_z)
            {
                continuity_ok = false;
                if (reason == "ok")
                {
                    reason = "jump_z";
                }
            }
            if (rtk_recover_max_speed_xy > 0.0 &&
                rtk_recover_latest_speed_xy > rtk_recover_max_speed_xy)
            {
                continuity_ok = false;
                if (reason == "ok")
                {
                    reason = "speed_xy";
                }
            }
            if (rtk_recover_max_speed_z > 0.0 &&
                rtk_recover_latest_speed_z > rtk_recover_max_speed_z)
            {
                continuity_ok = false;
                if (reason == "ok")
                {
                    reason = "speed_z";
                }
            }
        }
    }

    const bool ok = quality_ok && continuity_ok;
    if (ok)
    {
        if (rtk_recover_stable_start_time < 0.0)
        {
            rtk_recover_stable_start_time = reference_time;
        }
    }
    else
    {
        rtk_recover_stable_start_time = -1.0;
    }

    rtk_recover_prev_sample = sample;
    rtk_recover_prev_valid = true;
    rtk_recover_latest_quality_ok = quality_ok;
    rtk_recover_latest_continuity_ok = continuity_ok;
    rtk_recover_latest_reason = reason;

    const double required = get_required_rtk_recover_stable_duration();
    const double stable_elapsed =
        (rtk_recover_stable_start_time >= 0.0)
            ? (reference_time - rtk_recover_stable_start_time)
            : 0.0;
    const double residual_world =
        world_state.initialized ? (sample.pos - world_state.pos).norm()
                                : std::numeric_limits<double>::quiet_NaN();
    const double residual_publish =
        world_publish_state.initialized ? (sample.pos - world_publish_state.pos).norm()
                                        : std::numeric_limits<double>::quiet_NaN();

    std::ostringstream diag;
    diag << std::fixed << std::setprecision(6)
         << "state=" << rtk_fusion_state_to_string(rtk_fusion_state)
         << " t=" << reference_time
         << " rtk_t=" << sample.timestamp
         << " loss_duration=" << rtk_last_loss_duration
         << " stable_elapsed=" << stable_elapsed
         << " required=" << required
         << " fix_count=" << rtk_consecutive_fix_count
         << " quality_ok=" << (quality_ok ? 1 : 0)
         << " continuity_ok=" << (continuity_ok ? 1 : 0)
         << " reason=" << reason
         << " h_acc=" << sample.h_acc
         << " v_acc=" << sample.v_acc
         << " dt=" << rtk_recover_latest_dt
         << " jump_xy=" << rtk_recover_latest_jump_xy
         << " jump_z=" << rtk_recover_latest_jump_z
         << " speed_xy=" << rtk_recover_latest_speed_xy
         << " speed_z=" << rtk_recover_latest_speed_z
         << " residual_world=" << residual_world
         << " residual_publish=" << residual_publish
         << " rtk=[" << sample.pos.transpose() << "]";
    write_rtk_diagnostic_log("RTK_RECOVERY_CHECK", diag.str());
}

bool is_rtk_recovery_ready(double current_time)
{
    if (!rtk_recovery_verify_enable)
    {
        return true;
    }
    if (rtk_recover_stable_start_time < 0.0 ||
        !rtk_recover_latest_quality_ok ||
        !rtk_recover_latest_continuity_ok)
    {
        return false;
    }
    return current_time - rtk_recover_stable_start_time >=
           get_required_rtk_recover_stable_duration();
}

bool should_apply_world_publish_handoff()
{
    if (!world_publish_handoff_active || !world_state.initialized)
    {
        return false;
    }
    return rtk_fusion_state == RtkFusionState::NO_FIX ||
           rtk_fusion_state == RtkFusionState::WAIT_FIX ||
           rtk_fusion_state == RtkFusionState::SMOOTHING;
}

void start_world_publish_handoff(double timestamp, const std::string &reason)
{
    if (!world_state.initialized || !world_publish_state.initialized)
    {
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(3)
             << "skip_start reason=" << reason
             << " t=" << timestamp
             << " world_init=" << (world_state.initialized ? 1 : 0)
             << " publish_init=" << (world_publish_state.initialized ? 1 : 0);
        write_rtk_diagnostic_log("PUBLISH_HANDOFF", diag.str());
        return;
    }

    world_publish_handoff_active = true;
    world_publish_handoff_time = timestamp;
    world_publish_handoff_raw_pos = world_state.pos;
    world_publish_handoff_raw_quat = world_state.quat;
    world_publish_handoff_raw_quat.normalize();
    world_publish_handoff_pub_pos = world_publish_state.pos;
    world_publish_handoff_pub_quat = world_publish_state.quat;
    world_publish_handoff_pub_quat.normalize();

    std::ostringstream diag;
    diag << std::fixed << std::setprecision(3)
         << "start reason=" << reason
         << " t=" << timestamp
         << " raw=[" << world_publish_handoff_raw_pos.transpose() << "]"
         << " pub=[" << world_publish_handoff_pub_pos.transpose() << "]"
         << " offset=[" << (world_publish_handoff_pub_pos - world_publish_handoff_raw_pos).transpose()
         << "]";
    write_rtk_diagnostic_log("PUBLISH_HANDOFF", diag.str());
}

void stop_world_publish_handoff(double timestamp, const std::string &reason)
{
    if (!world_publish_handoff_active)
    {
        return;
    }

    std::ostringstream diag;
    diag << std::fixed << std::setprecision(3)
         << "stop reason=" << reason
         << " t=" << timestamp
         << " start_t=" << world_publish_handoff_time;
    write_rtk_diagnostic_log("PUBLISH_HANDOFF", diag.str());

    world_publish_handoff_active = false;
    world_publish_handoff_time = -1.0;
}

void apply_world_publish_handoff()
{
    if (!should_apply_world_publish_handoff())
    {
        return;
    }

    const Eigen::Vector3d raw_delta = world_state.pos - world_publish_handoff_raw_pos;
    Eigen::Quaterniond raw_delta_q =
        world_publish_handoff_raw_quat.inverse() * world_state.quat;
    raw_delta_q.normalize();

    world_publish_state.pos = world_publish_handoff_pub_pos + raw_delta;
    world_publish_state.quat = world_publish_handoff_pub_quat * raw_delta_q;
    world_publish_state.quat.normalize();
    world_publish_state.vel = world_state.vel;
    world_publish_state.initialized = world_state.initialized;
    world_publish_state.last_map_pos = world_state.last_map_pos;
    world_publish_state.last_map_quat = world_state.last_map_quat;
}

double compute_tapm_publish_handoff_weight(double tau)
{
    if (tau <= 0.0)
    {
        return 0.0;
    }

    double duration = tapm_nominal_dt *
                      static_cast<double>(tapm_exit_frame_limit > 0
                                              ? tapm_exit_frame_limit
                                              : std::max(1, tapm_grace_frames));
    duration = std::max(duration, tapm_nominal_dt);
    const double u = std::max(0.0, std::min(1.0, tau / duration));
    return u * u * (3.0 - 2.0 * u);
}

Eigen::Vector3d blend_tapm_publish_position(const Eigen::Vector3d &base_pos,
                                            const Eigen::Vector3d &target_pos,
                                            double weight)
{
    const double w = std::max(0.0, std::min(1.0, weight));
    return base_pos + w * (target_pos - base_pos);
}

double smoothstep01(double u)
{
    const double x = std::max(0.0, std::min(1.0, u));
    return x * x * (3.0 - 2.0 * x);
}

void apply_world_output_blend(double timestamp,
                              Eigen::Vector3d &output_pos,
                              Eigen::Quaterniond &output_quat)
{
    output_quat.normalize();
    const bool blend_enabled = world_output_blend_enable &&
                               world_output_blend_duration > 0.0;

    if (blend_enabled &&
        !world_output_blend_active &&
        world_output_last_valid &&
        world_output_blend_max_jump > 0.0)
    {
        const double jump = (output_pos - world_output_last_pos).norm();
        if (jump > world_output_blend_max_jump)
        {
            world_output_blend_active = true;
            world_output_blend_start_time = timestamp;
            world_output_blend_start_pos = world_output_last_pos;
            world_output_blend_start_quat = world_output_last_quat.normalized();

            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "start t=" << timestamp
                 << " jump=" << jump
                 << " max_jump=" << world_output_blend_max_jump
                 << " duration=" << world_output_blend_duration
                 << " from=[" << world_output_blend_start_pos.transpose() << "]"
                 << " target=[" << output_pos.transpose() << "]";
            write_rtk_diagnostic_log("OUTPUT_BLEND", diag.str());
        }
    }

    if (blend_enabled && world_output_blend_active)
    {
        const double tau = timestamp - world_output_blend_start_time;
        const double w = smoothstep01(tau / world_output_blend_duration);
        const Eigen::Vector3d target_pos = output_pos;
        const Eigen::Quaterniond target_quat = output_quat.normalized();
        output_pos = world_output_blend_start_pos +
                     w * (target_pos - world_output_blend_start_pos);
        output_quat = world_output_blend_start_quat.slerp(w, target_quat).normalized();

        if (w >= 1.0 - 1e-6)
        {
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "stop t=" << timestamp
                 << " tau=" << tau;
            write_rtk_diagnostic_log("OUTPUT_BLEND", diag.str());
            world_output_blend_active = false;
            world_output_blend_start_time = -1.0;
        }
    }

    if (world_output_z_guard_enable && world_output_last_valid)
    {
        const double dz = output_pos.z() - world_output_last_pos.z();
        double max_dz = std::numeric_limits<double>::infinity();
        const double dt = timestamp - world_output_last_time;
        if (world_output_z_guard_max_speed > 0.0 && std::isfinite(dt) && dt > 1e-4)
        {
            max_dz = std::min(max_dz, world_output_z_guard_max_speed * dt);
        }
        if (world_output_z_guard_max_step > 0.0)
        {
            max_dz = std::min(max_dz, world_output_z_guard_max_step);
        }

        if (std::isfinite(max_dz) && max_dz > 0.0 && std::abs(dz) > max_dz)
        {
            const double target_z = output_pos.z();
            output_pos.z() = world_output_last_pos.z() +
                             std::copysign(max_dz, dz);

            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "t=" << timestamp
                 << " dt=" << dt
                 << " dz=" << dz
                 << " max_dz=" << max_dz
                 << " from_z=" << world_output_last_pos.z()
                 << " target_z=" << target_z
                 << " guarded_z=" << output_pos.z()
                 << " max_speed=" << world_output_z_guard_max_speed
                 << " max_step=" << world_output_z_guard_max_step;
            write_rtk_diagnostic_log("OUTPUT_Z_GUARD", diag.str());
        }
    }

    world_output_last_valid = true;
    world_output_last_time = timestamp;
    world_output_last_pos = output_pos;
    world_output_last_quat = output_quat.normalized();
}

Eigen::Vector3d get_recovery_history_position()
{
    if (!world_publish_handoff_active || !world_state.initialized)
    {
        return world_state.pos;
    }
    if (rtk_fusion_state != RtkFusionState::NO_FIX &&
        rtk_fusion_state != RtkFusionState::WAIT_FIX &&
           rtk_fusion_state != RtkFusionState::SMOOTHING)
    {
        return world_state.pos;
    }
    return world_publish_handoff_pub_pos +
           (world_state.pos - world_publish_handoff_raw_pos);
}

void record_world_history_sample(double timestamp)
{
    if (!world_state.initialized)
    {
        return;
    }

    Timepos world_record;
    world_record.timestamp = timestamp;
    world_record.pos = get_recovery_history_position();
    world_history_window.push_back(world_record);
    Timepos lio_record = world_record;
    lio_record.pos = state_point.pos;
    tapm_lio_history.push_back(lio_record);
    while (world_history_window.size() >
           static_cast<size_t>(tapm_world_history_size))
    {
        world_history_window.pop_front();
        if (!tapm_lio_history.empty()) tapm_lio_history.pop_front();
    }
}

void start_tapm_smoothing(double current_time)
{
    RUNTIME_TIMER(tapm_entry_timer, "tapm_entry_initialization");
    tapm_active = true;
    tapm_frame_count = 0;
    const double history_start =
        (tapm_recovery_history_start_time >= 0.0)
            ? tapm_recovery_history_start_time
            : current_time;
    tapm_recover_time = current_time;
    tapm_previous_publish_time = current_time;
    tapm_previous_publish_dt = -1.0;
    tapm_previous_publish_correction.setZero();
    tapm_previous_previous_publish_correction.setZero();
    tapm_rotation_handover_start = -1.0;
    tapm_rotation_source = (world_publish_state.quat * Eigen::Quaterniond(state_point.rot).conjugate()).normalized();
    tapm_publish_rotation = tapm_rotation_target = tapm_rotation_source;
    while (!world_history_window.empty() &&
           world_history_window.front().timestamp + 1e-6 < history_start)
    {
        world_history_window.pop_front();
        if (!tapm_lio_history.empty()) tapm_lio_history.pop_front();
    }
    tapm_runtime_debug_last_log_time = -1.0;
    std::ostringstream tapm_enter_oss;
    tapm_enter_oss << std::fixed << std::setprecision(3)
                   << "[TAPM_ENTER] time=" << current_time
                   << " history_start=" << history_start
                   << " blend_start=" << tapm_recover_time
                   << " min_fix_for_smoothing=" << rtk_min_fix_for_smoothing
                   << " tapm_sliding_enable=" << (tapm_sliding_enable ? 1 : 0)
                   << " tapm_optimization_window=" << tapm_optimization_window
                   << " tapm_world_history_size=" << tapm_world_history_size
                   << " recovery_history_size=" << world_history_window.size()
                   << " tapm_exit_mode=frame_count_or_tapm_weight"
                   << " tapm_exit_frame_limit=" << tapm_exit_frame_limit
                   << " tapm_alpha=" << tapm_alpha
                   << " tapm_beta=" << tapm_beta
                   << " tapm_lambda_v=" << tapm_lambda_v
                   << " tapm_lambda_a=" << tapm_lambda_a
                   << " win=" << alignment_latest_window_size << "/" << alignment_window_current
                   << " match_ratio=" << last_rtk_match_ratio;
    write_rtk_diagnostic_log("TAPM_ENTER", tapm_enter_oss.str());
}

void log_tapm_wait_for_alignment()
{
    std::ostringstream oss;
    oss << "tapm_wait reason=alignment_window_not_ready"
        << " tapm_enable=" << (tapm_enable ? 1 : 0)
        << " tapm_sliding_enable=" << (tapm_sliding_enable ? 1 : 0)
        << " rtk_initial_alignment_done=" << (rtk_initial_alignment_done ? 1 : 0)
        << " tf_valid=" << (world_map_transform.isTransformValid() ? 1 : 0)
        << " win=" << alignment_latest_window_size << "/" << alignment_window_current
        << " match_ratio=" << last_rtk_match_ratio
        << " match_threshold=" << rtk_match_min_ratio;
    write_rtk_diagnostic_log("TAPM_WAIT", oss.str());
}

void complete_smoothing_state(double current_time, const std::string &reason)
{
    if (rtk_fusion_state != RtkFusionState::SMOOTHING)
    {
        return;
    }

    RtkFusionState previous_state = rtk_fusion_state;
    rtk_fusion_state = RtkFusionState::CORRECTION;
    log_fusion_transition(previous_state,
                          rtk_fusion_state,
                          reason,
                          current_time);

    clear_alignment_buffers("smoothing_completed");
    alignment_window_current = alignment_window_reset_fix;
    alignment_window_success_counter = 0;
    smoothing_alignment_valid = false;
    smoothing_first_alignment_done = false;
    pgo_raw_rtk_allowed = true;
    clear_pgo_sw_for_transition("smoothing_completed");
    tapm_recovery_history_start_time = -1.0;
    stop_world_publish_handoff(current_time, reason);
    reset_smoothing_completion_request();
}

void enter_wait_fix_state(double current_time)
{
    RtkFusionState previous_state = rtk_fusion_state;
    rtk_fusion_state = RtkFusionState::WAIT_FIX;
    rtk_fix_active = false;
    log_fusion_transition(previous_state,
                          rtk_fusion_state,
                          "rtk_fix_recovered_wait_fix",
                          current_time);

    reset_smoothing_completion_request();
    clear_alignment_buffers("enter_wait_fix");
    alignment_window_current = alignment_window_reset_fix;
    alignment_window_success_counter = 0;
    smoothing_alignment_valid = false;
    smoothing_first_alignment_done = false;
    alignment_stable = false;
    alignment_stable_enter_time = -1.0;
    pgo_raw_rtk_allowed = false;
    clear_pgo_sw_for_transition("enter_wait_fix");
    tapm_recovery_history_start_time =
        (rtk_fix_recover_start_time >= 0.0)
            ? rtk_fix_recover_start_time
            : current_time;
    world_history_window.clear();
    tapm_lio_history.clear();
    std::ostringstream diag;
    diag << std::fixed << std::setprecision(3)
         << "reset_history start=" << tapm_recovery_history_start_time
         << " state=WAIT_FIX";
    write_rtk_diagnostic_log("TAPM_RECOVERY_HISTORY", diag.str());
}

void enter_smoothing_state(double current_time)
{

    if (!world_map_transform.isTransformValid())
    {
        log_measurement_event("[ALIGN]", "initial_world_alignment_required_before_recovery");
        enter_initial_correction_state(current_time);
        return;
    }
    RtkFusionState previous_state = rtk_fusion_state;
    rtk_fusion_state = RtkFusionState::SMOOTHING;
    rtk_fix_active = true;
    log_fusion_transition(previous_state,
                          rtk_fusion_state,
                          "fix_count_reached",
                          current_time);

    reset_smoothing_completion_request();
    clear_alignment_buffers("enter_smoothing");
    alignment_window_current = alignment_window_reset_fix;
    alignment_window_success_counter = 0;
    smoothing_alignment_valid = false;
    smoothing_first_alignment_done = false;
    alignment_stable = false;
    alignment_stable_enter_time = -1.0;
    pgo_raw_rtk_allowed = false;
    clear_pgo_sw_for_transition("enter_smoothing");
    if (tapm_recovery_history_start_time < 0.0)
    {
        tapm_recovery_history_start_time = current_time;
        world_history_window.clear();
    tapm_lio_history.clear();
    }
    start_pgo_recovery_ramp(current_time, "enter_smoothing");
    if (tapm_enable && tapm_sliding_enable)
    {
        start_tapm_smoothing(current_time);
    }
    else
    {
        request_smoothing_completion("tapm_disabled_skip_smoothing");
    }
}

void tick_rtk_fusion_state_with_fix(double current_time)
{
    if (rtk_fusion_state == RtkFusionState::NO_FIX)
    {
        if (!rtk_initial_alignment_done || !world_map_transform.isTransformValid())
        {
            if (rtk_consecutive_fix_count >= rtk_min_fix_for_smoothing)
            {
                enter_initial_correction_state(current_time);
            }
        }
        else if (rtk_consecutive_fix_count > 0)
        {
            enter_wait_fix_state(current_time);
        }
        return;
    }

    if (rtk_fusion_state == RtkFusionState::WAIT_FIX)
    {
        const double wait_elapsed =
            (rtk_fix_recover_start_time >= 0.0)
                ? (current_time - rtk_fix_recover_start_time)
                : 0.0;
        const bool recovery_ready = is_rtk_recovery_ready(current_time);
        if (wait_elapsed >= rtk_wait_fix_duration_sec &&
            rtk_consecutive_fix_count >= rtk_min_fix_for_smoothing &&
            recovery_ready)
        {
            if (tapm_enable && tapm_sliding_enable)
            {
                enter_smoothing_state(current_time);
            }
            else
            {
                enter_recovery_correction_state(current_time);
            }
            return;
        }
        if (rtk_diagnostic_log_enable)
        {
            const double stable_elapsed =
                (rtk_recover_stable_start_time >= 0.0)
                    ? (current_time - rtk_recover_stable_start_time)
                    : 0.0;
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "state=WAIT_FIX"
                 << " t=" << current_time
                 << " wait_elapsed=" << wait_elapsed
                 << " wait_required=" << rtk_wait_fix_duration_sec
                 << " fix_count=" << rtk_consecutive_fix_count
                 << " min_fix=" << rtk_min_fix_for_smoothing
                 << " stable_elapsed=" << stable_elapsed
                 << " stable_required=" << get_required_rtk_recover_stable_duration()
                 << " recovery_ready=" << (recovery_ready ? 1 : 0)
                 << " latest_reason=" << rtk_recover_latest_reason;
            write_rtk_diagnostic_log("RTK_RECOVERY_WAIT", diag.str());
        }
        rtk_fix_active = false;
        return;
    }

    if (rtk_fusion_state == RtkFusionState::SMOOTHING)
    {
        if (smoothing_completion_requested)
        {
            complete_smoothing_state(current_time,
                                     smoothing_completion_reason.empty()
                                         ? "smoothing_completed"
                                         : smoothing_completion_reason);
        }
    }
}

void log_rtk_state_color_if_changed()
{
    const char *color = COLOR_GREEN;
    const char *color_name = "green";
    std::string mode;

    if (!rtk_fix_active)
    {
        color = COLOR_RED;
        color_name = "red";
        mode = "NO_RTK_FIX";
    }
    else if (alignment_rotation_frozen)
    {
        color = COLOR_BLUE;
        color_name = "blue";
        mode = "FIX_FROZEN";
    }
    else
    {
        color = COLOR_GREEN;
        color_name = "green";
        mode = "FIX_NO_FROZEN";
    }

}


void handle_tapm_on_fix_lost(double reference_time)
{
    tapm_output_active = false;
    tapm_frame_count = 0;
    if (tapm_sliding_enable)
    {
        tapm_active = false;
        tapm_recover_time = -1.0;
    }
    if (tapm_enable)
    {
        if (!alignment_rotation_frozen)
        {
            tapm_anchor_valid = false;
            tapm_anchor_transform.valid = false;
        }
        else if (!world_map_transform.isTransformValid())
        {
            tapm_anchor_valid = false;
            tapm_anchor_transform.valid = false;
        }
        else
        {
            tapm_anchor_transform = world_map_transform.getTransform();
            tapm_anchor_transform.R = alignment_frozen_rotation;
            tapm_anchor_transform.valid = true;
            tapm_anchor_time = reference_time;
            tapm_recover_time = -1.0;
            tapm_anchor_valid = true;
        }
    }
}

void enter_no_fix_state(double reference_time)
{
    const RtkFusionState previous_state = rtk_fusion_state;
    if (previous_state != RtkFusionState::NO_FIX)
    {
        start_world_publish_handoff(reference_time, "rtk_fix_lost");
        log_fusion_transition(previous_state,
                              RtkFusionState::NO_FIX,
                              "rtk_fix_lost",
                              reference_time);
        clear_pgo_sw_for_transition("enter_no_fix");
    }
    rtk_fusion_state = RtkFusionState::NO_FIX;
    rtk_consecutive_fix_count = 0;
    if (rtk_loss_start_time < 0.0)
    {
        rtk_loss_start_time = reference_time;
    }
    reset_rtk_recovery_verify_runtime("rtk_fix_lost");
    pgo_raw_rtk_allowed = false;
    reset_smoothing_completion_request();
    clear_alignment_buffers("rtk_fix_lost");
    if (!keep_rotation_frozen_on_loss)
    {
        reset_alignment_rotation_freeze("rtk_fix_lost");
    }
    else if (alignment_rotation_frozen)
    {
        log_measurement_event("[ALIGN]", "rotation_kept_frozen_on_loss");
    }
    smoothing_alignment_valid = false;
    smoothing_first_alignment_done = false;
    alignment_stable = false;
    alignment_stable_enter_time = -1.0;
    alignment_window_gate_released = false;
    alignment_window_current = alignment_window_reset_fix;
    alignment_window_success_counter = 0;
    last_alignment_no_frozen_valid = false;
    tapm_recovery_history_start_time = -1.0;
    world_history_window.clear();
    tapm_lio_history.clear();
}

void enter_initial_correction_state(double reference_time)
{
    bootstrap_world_transform_from_latest_rtk(reference_time, "first_fix_initialization");
    rtk_initial_alignment_done = true;
    pgo_raw_rtk_allowed = true;
    pgo_sw_recovery_ramp_start_time = -1.0;
    pgo_sw_recovery_ramp_duration_active = 0.0;
    clear_pgo_sw_for_transition("first_fix_initialization", true);
    pgo_debug_window_size = 0;
    pgo_debug_rtk_buffer_size = 0;
    pgo_debug_optimized_size = 0;
    RtkFusionState previous_state = rtk_fusion_state;
    rtk_fusion_state = RtkFusionState::CORRECTION;
    rtk_fix_active = true;
    log_fusion_transition(previous_state,
                          rtk_fusion_state,
                          "first_fix_initialization",
                          reference_time);
    stop_world_publish_handoff(reference_time, "first_fix_initialization");
    tapm_recovery_history_start_time = -1.0;
    reset_smoothing_completion_request();
    alignment_window_current = alignment_window_min;
    alignment_window_success_counter = 0;
}

void enter_recovery_correction_state(double reference_time)
{
    RtkFusionState previous_state = rtk_fusion_state;
    rtk_fusion_state = RtkFusionState::CORRECTION;
    rtk_fix_active = true;
    log_fusion_transition(previous_state,
                          rtk_fusion_state,
                          "rtk_fix_recovered_wait_tapm",
                          reference_time);

    reset_smoothing_completion_request();
    clear_alignment_buffers("rtk_fix_recovered");
    alignment_window_current = alignment_window_reset_fix;
    alignment_window_success_counter = 0;
    smoothing_alignment_valid = false;
    smoothing_first_alignment_done = false;
    alignment_stable = false;
    alignment_stable_enter_time = -1.0;
    pgo_raw_rtk_allowed = !(tapm_enable && tapm_sliding_enable);
    clear_pgo_sw_for_transition("enter_recovery_correction");
    tapm_recovery_history_start_time = -1.0;
    start_pgo_recovery_ramp(reference_time, "enter_recovery_correction");
    stop_world_publish_handoff(reference_time, "rtk_fix_recovered_wait_tapm");
}


void process_rtk_fusion_state_machine(double reference_time, bool has_new_fix_measurement)
{
    if (!rtk_fusion_enable)
    {
        if (rtk_fix_active || rtk_fix_signal_active || rtk_fusion_state != RtkFusionState::NO_FIX)
        {
            rtk_fix_active = false;
            rtk_fix_signal_active = false;
            enter_no_fix_state(reference_time);
            write_rtk_diagnostic_log("RTK_FUSION_DISABLED", "force_state=NO_FIX");
        }
        return;
    }

    RtkFusionEvent event;
    event.reference_time = reference_time;
    event.has_new_fix_measurement = has_new_fix_measurement;
    event.was_fix_active = rtk_fix_active;
    const double age = (latest_rtk_timestamp > 0.0)
                           ? std::fabs(reference_time - latest_rtk_timestamp)
                           : std::numeric_limits<double>::infinity();
    const bool signal_has_fix_raw = age <= rtk_fix_timeout;
    const bool signal_was_active = rtk_fix_signal_active;


    if (!rtk_reacquire_enable && rtk_initial_alignment_done &&
        signal_was_active && !signal_has_fix_raw && !rtk_reacquire_blocked)
    {
        rtk_reacquire_blocked = true;
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(3)
             << "reacquire disabled, RTK locked out permanently t=" << reference_time;
        write_rtk_diagnostic_log("RTK_REACQUIRE_BLOCKED", diag.str());
    }


    const bool signal_has_fix = rtk_reacquire_blocked ? false : signal_has_fix_raw;
    rtk_fix_signal_active = signal_has_fix;

    if (signal_was_active && !signal_has_fix)
    {
        rtk_loss_start_time = reference_time;
        rtk_fix_recover_start_time = -1.0;
        reset_rtk_recovery_verify_runtime("signal_lost");
        handle_tapm_on_fix_lost(reference_time);
    }

    if (!signal_has_fix)
    {
        event.has_fix = false;
        rtk_fix_active = false;
        enter_no_fix_state(reference_time);
        log_rtk_state_color_if_changed();
        return;
    }

    if (!signal_was_active && signal_has_fix)
    {
        rtk_fix_recover_start_time = reference_time;
        rtk_last_loss_duration =
            (rtk_loss_start_time >= 0.0)
                ? std::max(0.0, reference_time - rtk_loss_start_time)
                : 0.0;
        rtk_loss_start_time = -1.0;
        rtk_consecutive_fix_count = 0;
        reset_rtk_recovery_verify_runtime("signal_recovered");
        reset_smoothing_completion_request();
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(3)
             << "start wait_fix sec=" << rtk_wait_fix_duration_sec
             << " stable_required=" << get_required_rtk_recover_stable_duration()
             << " loss_duration=" << rtk_last_loss_duration
             << " t=" << reference_time;
        write_rtk_diagnostic_log("RTK_WAIT_FIX", diag.str());
    }

    event.has_fix = true;
    rtk_fix_active = true;

    if (event.was_fix_active && !event.has_fix)
    {
        handle_tapm_on_fix_lost(reference_time);
    }

    if (!event.was_fix_active && event.has_fix)
    {
        if (tapm_enable && tapm_anchor_valid && alignment_rotation_frozen &&
            !tapm_sliding_enable)
        {
            tapm_recover_time = reference_time;
            tapm_frame_count = 0;
        }
    }

    if (!rtk_initial_alignment_done)
    {
        enter_initial_correction_state(reference_time);
        log_rtk_state_color_if_changed();
        return;
    }

    if (event.has_new_fix_measurement)
    {
        rtk_consecutive_fix_count++;
        update_rtk_recovery_quality(reference_time, true);
    }
    tick_rtk_fusion_state_with_fix(reference_time);

    log_rtk_state_color_if_changed();
}

bool selectRtkForTimestamp(const std::deque<Timepos> &rtk_measurements,
                           double query_time,
                           double max_dt,
                           Timepos &selected_rtk,
                           double &selected_dt,
                           std::string &selected_source)
{
    selected_dt = std::numeric_limits<double>::quiet_NaN();
    selected_source = "none";
    if (rtk_measurements.empty() || max_dt < 0.0)
    {
        return false;
    }

    auto upper = std::lower_bound(
        rtk_measurements.begin(), rtk_measurements.end(), query_time,
        [](const Timepos &sample, double ts)
        { return sample.timestamp < ts; });

    if (upper != rtk_measurements.begin() && upper != rtk_measurements.end())
    {
        auto lower = upper;
        --lower;
        const double t0 = lower->timestamp;
        const double t1 = upper->timestamp;
        const double span = t1 - t0;
        const double max_interp_span = std::max(2.5 * max_dt, 0.15);
        if (span > 1e-6 && span <= max_interp_span &&
            query_time >= t0 && query_time <= t1)
        {
            const double ratio = (query_time - t0) / span;
            selected_rtk = *lower;
            selected_rtk.timestamp = query_time;
            selected_rtk.pos = lower->pos + ratio * (upper->pos - lower->pos);
            selected_rtk.h_acc = std::max(lower->h_acc, upper->h_acc);
            selected_rtk.v_acc = std::max(lower->v_acc, upper->v_acc);
            selected_rtk.vel_acc = std::max(lower->vel_acc, upper->vel_acc);
            selected_rtk.fix_type = std::min(lower->fix_type, upper->fix_type);
            if (lower->vel_valid && upper->vel_valid)
            {
                selected_rtk.vel = lower->vel + ratio * (upper->vel - lower->vel);
                selected_rtk.vel_valid = true;
            }
            else
            {
                selected_rtk.vel_valid = false;
            }
            selected_dt = 0.0;
            selected_source = "rtk_interp";
            return true;
        }
    }

    bool found = false;
    double best_dt = max_dt;
    for (const Timepos &sample : rtk_measurements)
    {
        const double dt = std::abs(sample.timestamp - query_time);
        if (dt <= best_dt)
        {
            selected_rtk = sample;
            best_dt = dt;
            found = true;
        }
    }

    if (found)
    {
        selected_dt = best_dt;
        selected_source = "rtk_nearest";
    }
    return found;
}

bool matchRtkWindowWithEkf(const std::deque<Timepos> &ekf_buffer,
                           const std::deque<Timepos> &rtk_measurements,
                           double time_threshold,
                           std::deque<Timepos> &matched_ekf,
                           std::deque<Timepos> &matched_rtk,
                           double &match_ratio,
                           std::string &failure_reason)
{
    matched_ekf.clear();
    matched_rtk.clear();
    match_ratio = 0.0;
    failure_reason.clear();

    if (ekf_buffer.empty() || rtk_measurements.empty())
    {
        std::ostringstream oss;
        oss << "empty_buffers ekf=" << ekf_buffer.size()
            << " rtk=" << rtk_measurements.size();
        failure_reason = oss.str();
        return false;
    }

    if (rtk_match_at_gnss_epochs)
    {
        size_t eligible = 0;
        for (const auto &rtk : rtk_measurements)
        {
            if (rtk.timestamp < ekf_buffer.front().timestamp || rtk.timestamp > ekf_buffer.back().timestamp)
                continue;
            ++eligible;
            auto right = std::lower_bound(ekf_buffer.begin(), ekf_buffer.end(), rtk.timestamp,
                [](const Timepos &p, double stamp) { return p.timestamp < stamp; });
            if (right == ekf_buffer.end()) continue;
            Timepos interpolated = *right;
            if (right->timestamp != rtk.timestamp)
            {
                if (right == ekf_buffer.begin()) continue;
                const auto left = std::prev(right);
                const double dt = right->timestamp - left->timestamp;
                if (dt <= 0.0 || dt > 2.0 * time_threshold) continue;
                const double u = (rtk.timestamp - left->timestamp) / dt;
                interpolated.pos = left->pos + u * (right->pos - left->pos);
            }
            interpolated.timestamp = rtk.timestamp;
            matched_ekf.push_back(interpolated);
            matched_rtk.push_back(rtk);
        }
        match_ratio = eligible ? static_cast<double>(matched_ekf.size()) / eligible : 0.0;
        if (matched_ekf.size() < static_cast<size_t>(rtk_alignment_min_pairs) || match_ratio < rtk_match_min_ratio)
        {
            failure_reason = "insufficient_independent_gnss_epochs";
            return false;
        }
        return true;
    }

    const double search_start = ekf_buffer.front().timestamp - 0.5;
    const double search_end = ekf_buffer.back().timestamp + 0.5;

    auto start_it = std::lower_bound(
        rtk_measurements.begin(), rtk_measurements.end(), search_start,
        [](const Timepos &lhs, double ts)
        { return lhs.timestamp < ts; });
    auto end_it = std::upper_bound(
        rtk_measurements.begin(), rtk_measurements.end(), search_end,
        [](double ts, const Timepos &rhs)
        { return ts < rhs.timestamp; });

    if (start_it == end_it)
    {
        failure_reason = "no_rtk_in_search_window";
        return false;
    }

    auto rtk_it = start_it;
    for (const auto &ekf_pose : ekf_buffer)
    {
        const double lower_bound_time = ekf_pose.timestamp - time_threshold;
        const double upper_bound_time = ekf_pose.timestamp + time_threshold;

        while (rtk_it != end_it && rtk_it->timestamp < lower_bound_time)
        {
            ++rtk_it;
        }

        if (rtk_it == end_it)
        {
            if (failure_reason.empty())
            {
                std::ostringstream oss;
                oss << "rtk_exhausted_after_ts=" << std::fixed << std::setprecision(6) << ekf_pose.timestamp;
                failure_reason = oss.str();
            }
            break;
        }

        auto best_it = rtk_it;
        double best_diff = std::abs(best_it->timestamp - ekf_pose.timestamp);

        auto search_it = rtk_it;
        while (search_it != end_it && search_it->timestamp <= upper_bound_time)
        {
            const double diff = std::abs(search_it->timestamp - ekf_pose.timestamp);
            if (diff < best_diff)
            {
                best_diff = diff;
                best_it = search_it;
                if (diff == 0.0)
                {
                    break;
                }
            }
            ++search_it;
        }

        if (best_diff > time_threshold)
        {
            if (failure_reason.empty())
            {
                std::ostringstream oss;
                oss << "no_match_within_threshold ts=" << std::fixed << std::setprecision(6) << ekf_pose.timestamp
                    << " diff=" << best_diff;
                failure_reason = oss.str();
            }
            continue;
        }

        matched_ekf.push_back(ekf_pose);

        Timepos compensated_rtk = *best_it;
        if (enable_antenna_lever_arm_compensation)
        {
            compensated_rtk.pos = best_it->pos - antenna_offset_T;
        }
        matched_rtk.push_back(compensated_rtk);
        rtk_it = best_it;
    }

    if (matched_ekf.empty())
    {
        if (failure_reason.empty())
        {
            failure_reason = "no_pairs_within_threshold";
        }
        return false;
    }

    match_ratio = static_cast<double>(matched_ekf.size()) / static_cast<double>(ekf_buffer.size());
    if (match_ratio < rtk_match_min_ratio)
    {
        std::ostringstream oss;
        oss << "match_ratio=" << std::fixed << std::setprecision(3) << match_ratio
            << " threshold=" << rtk_match_min_ratio
            << " matched=" << matched_ekf.size()
            << " total=" << ekf_buffer.size();
        if (failure_reason.empty())
        {
            failure_reason = oss.str();
        }
        else
        {
            failure_reason += "; " + oss.str();
        }
        return false;
    }

    return true;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{

    odomAftMapped.header.frame_id = "map";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "map", "body"));
}


void update_world_state()
{

    Eigen::Vector3d pos_map = state_point.pos;
    Eigen::Quaterniond quat_map = state_point.rot;
    Eigen::Vector3d vel_map(state_point.vel(0), state_point.vel(1), state_point.vel(2));

    if (!world_state.initialized)
    {

        if (world_map_transform.isTransformValid())
        {
            world_map_transform.transformPoseToWorld(pos_map, quat_map,
                                                     world_state.pos, world_state.quat);
            world_state.vel = world_map_transform.getTransform().R * vel_map;
            if (enable_manual_ekf_rtk_offset)
            {
                const Transformation &tf = world_map_transform.getTransform();
                world_state.pos += tf.s * (tf.R * manual_ekf_rtk_offset);
            }
        }
        else
        {
            world_state.pos = pos_map;
            world_state.quat = quat_map;
            world_state.vel = vel_map;
        }
        world_state.last_map_pos = pos_map;
        world_state.last_map_quat = quat_map;
        world_state.initialized = true;

        return;
    }


    if (world_map_transform.isTransformValid())
    {

        Eigen::Vector3d pos_world_new;
        Eigen::Quaterniond quat_world_new;
        world_map_transform.transformPoseToWorld(pos_map, quat_map, pos_world_new, quat_world_new);


        Eigen::Matrix3d R_world_map = world_map_transform.getTransform().R;
        Eigen::Vector3d vel_world_new = R_world_map * vel_map;

        if (enable_manual_ekf_rtk_offset)
        {
            const Transformation &tf = world_map_transform.getTransform();
            pos_world_new += tf.s * (tf.R * manual_ekf_rtk_offset);
        }

        const double nan_value = std::numeric_limits<double>::quiet_NaN();
        last_dist_no_frozen = nan_value;

        Eigen::Vector3d pos_world_frozen = pos_world_new;
        Eigen::Vector3d pos_world_no_frozen = pos_world_new;
        Eigen::Vector3d pos_world_adaptive = pos_world_new;
        bool use_adaptive_pos = false;
        double dist_frozen = nan_value;
        double dist_no_frozen = nan_value;
        double dist_diff = nan_value;
        double weight_no_frozen = 0.0;

        const bool rotation_freeze_active = alignment_rotation_frozen;
        const bool freeze_adaptive_inputs_ready =
            rtk_freeze_adaptive_enable && rtk_fix_active && rotation_freeze_active &&
            last_alignment_no_frozen_valid && !Measures.rtk_pos.empty();
        const bool freeze_adaptive_state_allowed =
            rtk_fusion_state == RtkFusionState::CORRECTION;
        if (freeze_adaptive_inputs_ready && freeze_adaptive_state_allowed)
        {
            pos_world_no_frozen = transform_map_to_world_with_transform(pos_map, last_alignment_no_frozen);
            if (enable_manual_ekf_rtk_offset)
            {
                pos_world_no_frozen +=
                    last_alignment_no_frozen.s * (last_alignment_no_frozen.R * manual_ekf_rtk_offset);
            }

            const Eigen::Vector3d rtk_world_pos = Measures.rtk_pos.back().pos;
            dist_frozen = (pos_world_frozen - rtk_world_pos).norm();
            dist_no_frozen = (pos_world_no_frozen - rtk_world_pos).norm();
            dist_diff = std::abs(dist_frozen - dist_no_frozen);

            double adaptive_ratio = 0.0;
            if (rtk_freeze_adaptive_max_dist > 1e-6)
            {
                adaptive_ratio = std::min(1.0, dist_diff / rtk_freeze_adaptive_max_dist);
            }
            weight_no_frozen = rtk_freeze_adaptive_base_weight +
                               (1.0 - rtk_freeze_adaptive_base_weight) * adaptive_ratio;
            weight_no_frozen = std::max(0.0, std::min(1.0, weight_no_frozen));

            pos_world_adaptive = weight_no_frozen * pos_world_no_frozen +
                                 (1.0 - weight_no_frozen) * pos_world_frozen;
            use_adaptive_pos = true;
            last_dist_no_frozen = dist_no_frozen;

            const double dist_adaptive = (pos_world_adaptive - rtk_world_pos).norm();
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(3)
                 << "dist_frozen=" << dist_frozen
                 << " dist_no_frozen=" << dist_no_frozen
                 << " dist_adaptive=" << dist_adaptive
                 << " weight_no_frozen=" << weight_no_frozen
                 << " state=" << rtk_fusion_state_to_string(rtk_fusion_state)
                 << " state_allowed=" << (freeze_adaptive_state_allowed ? 1 : 0)
                 << " max_dist=" << rtk_freeze_adaptive_max_dist
                 << " base_weight=" << rtk_freeze_adaptive_base_weight
                 << " rtk=[" << rtk_world_pos.transpose() << "]"
                 << " frozen=[" << pos_world_frozen.transpose() << "]"
                 << " no_frozen=[" << pos_world_no_frozen.transpose() << "]"
                 << " adaptive=[" << pos_world_adaptive.transpose() << "]";
            write_rtk_diagnostic_log("FREEZE_ADAPT", diag.str());
        }
        else if (freeze_adaptive_inputs_ready && rtk_freeze_adaptive_log_enable)
        {
            static double last_freeze_adapt_skip_log_time = -1.0;
            if (last_freeze_adapt_skip_log_time < 0.0 ||
                lidar_end_time - last_freeze_adapt_skip_log_time >= 0.5)
            {
                last_freeze_adapt_skip_log_time = lidar_end_time;
                std::ostringstream diag;
                diag << std::fixed << std::setprecision(3)
                     << "reason=state_not_allowed"
                     << " state=" << rtk_fusion_state_to_string(rtk_fusion_state)
                     << " rtk_fix_active=" << (rtk_fix_active ? 1 : 0)
                     << " rotation_freeze_active=" << (rotation_freeze_active ? 1 : 0)
                     << " last_no_frozen_valid=" << (last_alignment_no_frozen_valid ? 1 : 0)
                     << " rtk_samples=" << Measures.rtk_pos.size();
                write_rtk_diagnostic_log("FREEZE_ADAPT_SKIP", diag.str());
            }
        }


        world_state.pos = use_adaptive_pos ? pos_world_adaptive : pos_world_new;
        world_state.quat = quat_world_new;
        world_state.vel = vel_world_new;

        if (rtk_freeze_adaptive_log_enable)
        {
            const bool rtk_window_ready = is_rtk_window_ready_for_unified_update();
        }


        static double last_debug_time = 0.0;
        if (lidar_end_time - last_debug_time > 10.0)
        {
            Eigen::Vector3d delta_pos = pos_world_new - pos_map;
            last_debug_time = lidar_end_time;
        }
    }
    else
    {
        const double nan_value = std::numeric_limits<double>::quiet_NaN();
        last_dist_no_frozen = nan_value;


        Eigen::Vector3d delta_pos_map = pos_map - world_state.last_map_pos;


        Eigen::Quaterniond delta_quat_map = world_state.last_map_quat.inverse() * quat_map;


        world_state.pos += delta_pos_map;
        world_state.quat = world_state.quat * delta_quat_map;
        world_state.quat.normalize();
        world_state.vel = vel_map;


        static double last_warn_time = 0.0;
        if (lidar_end_time - last_warn_time > 10.0)
        {
            last_warn_time = lidar_end_time;
        }
    }


    if (rtk_diagnostic_log_enable)
    {
        const bool tf_valid = world_map_transform.isTransformValid();
        const char *branch = tf_valid ? "A_transform" : "B_incremental";
        const Transformation &tf = world_map_transform.getTransform();


        const Eigen::Vector3d ypr = tf.R.eulerAngles(2, 1, 0);
        const double rad2deg = 180.0 / M_PI;
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(4)
             << "branch=" << branch
             << " tf_valid=" << (tf_valid ? 1 : 0)
             << " fusion_state=" << rtk_fusion_state_to_string(rtk_fusion_state)
             << " map=[" << pos_map.transpose() << "]"
             << " world=[" << world_state.pos.transpose() << "]"
             << " dz_world_minus_map=" << (world_state.pos.z() - pos_map.z())
             << " tf_roll_deg=" << ypr.z() * rad2deg
             << " tf_pitch_deg=" << ypr.y() * rad2deg
             << " tf_yaw_deg=" << ypr.x() * rad2deg
             << " tf_s=" << tf.s
             << " tf_t=[" << tf.t.transpose() << "]";
        write_rtk_diagnostic_log("MAP_ODOM", diag.str());
    }


    world_state.last_map_pos = pos_map;
    world_state.last_map_quat = quat_map;
}

void update_world_publish_state()
{
    world_publish_state = world_state;
    apply_world_publish_handoff();
    if (tapm_output_active)
    {
        world_publish_state.pos = tapm_publish_pos;
        if (tapm_rotation_handover_sec > 0.0)
            world_publish_state.quat = (tapm_publish_rotation * Eigen::Quaterniond(state_point.rot)).normalized();
    }
}

void log_tapm_exit_debug(const std::string &reason,
                         double dist_to_alignment,
                         double dist_to_world)
{
    if (!rtk_diagnostic_log_enable)
    {
        return;
    }

    const bool alignment_update_frozen = tapm_active && tapm_sliding_enable;
    const bool map_tf_valid = world_map_transform.isTransformValid();
    const Eigen::Vector3d map_world_pos = map_position_to_world(state_point.pos);
    const Eigen::Vector3d diff_world = tapm_output_pos - world_state.pos;
    const Eigen::Vector3d diff_map = tapm_output_pos - map_world_pos;
    const double rtk_age = (latest_rtk_timestamp > 0.0)
                               ? std::fabs(lidar_end_time - latest_rtk_timestamp)
                               : -1.0;
    const double latest_rtk_time = Measures.rtk_pos.empty()
                                       ? -1.0
                                       : Measures.rtk_pos.back().timestamp;
    const Eigen::Quaterniond map_quat(state_point.rot);
    const Eigen::Quaterniond world_quat(world_state.quat);
    Eigen::Quaterniond map_tf_quat = Eigen::Quaterniond::Identity();
    Eigen::Vector3d map_tf_t = Eigen::Vector3d::Zero();
    double map_tf_s = 1.0;
    double map_tf_conf = -1.0;
    const Eigen::Quaterniond frozen_quat(alignment_frozen_rotation);
    Eigen::Vector3d pos_map_with_offset = state_point.pos;
    if (enable_manual_ekf_rtk_offset)
    {
        pos_map_with_offset += manual_ekf_rtk_offset;
    }
    if (map_tf_valid)
    {
        const Transformation &tf = world_map_transform.getTransform();
        map_tf_quat = Eigen::Quaterniond(tf.R);
        map_tf_t = tf.t;
        map_tf_s = tf.s;
        map_tf_conf = world_map_transform.getConfidence();
    }
    std::ostringstream oss;
    oss << "[TAPM_EXIT_DEBUG] reason=" << reason
        << " frame_count=" << tapm_frame_count
        << " lidar_time=" << std::fixed << std::setprecision(3)
        << lidar_end_time
        << " rtk_time=" << latest_rtk_time
        << " rtk_age=" << rtk_age
        << " rtk_fix_active=" << (rtk_fix_active ? 1 : 0)
        << " rtk_fusion_state=" << rtk_fusion_state_to_string(rtk_fusion_state)
        << " tapm_enable=" << (tapm_enable ? 1 : 0)
        << " tapm_sliding_enable=" << (tapm_sliding_enable ? 1 : 0)
        << " tapm_exit_mode=frame_count_or_tapm_weight"
        << " tapm_exit_frame_limit=" << tapm_exit_frame_limit
        << " dist_to_alignment=" << dist_to_alignment
        << " dist_to_world=" << dist_to_world
        << " alignment_window_current=" << alignment_window_current
        << " alignment_window_reset_fix=" << alignment_window_reset_fix
        << " alignment_latest_window_size=" << alignment_latest_window_size
        << "\n  tapm_output_pos=[" << tapm_output_pos.transpose() << "]"
        << " tapm_publish_pos=[" << tapm_publish_pos.transpose() << "]"
        << " tapm_publish_weight=" << tapm_publish_weight
        << " world_state_pos=[" << world_state.pos.transpose() << "]"
        << " diff_world=[" << diff_world.transpose() << "]"
        << "\n  map_pos=[" << state_point.pos.transpose() << "]"
        << " map_quat=[" << map_quat.x() << "," << map_quat.y() << ","
        << map_quat.z() << "," << map_quat.w() << "]"
        << " pos_map_with_offset=[" << pos_map_with_offset.transpose() << "]"
        << "\n  world_quat=[" << world_quat.x() << "," << world_quat.y() << ","
        << world_quat.z() << "," << world_quat.w() << "]"
        << " map_world_pos=[" << map_world_pos.transpose() << "]"
        << " diff_map=[" << diff_map.transpose() << "]"
        << " map_tf_valid=" << (map_tf_valid ? 1 : 0)
        << " map_tf_s=" << map_tf_s
        << " map_tf_conf=" << map_tf_conf
        << " map_tf_t=[" << map_tf_t.transpose() << "]"
        << " map_tf_quat=[" << map_tf_quat.x() << ","
        << map_tf_quat.y() << "," << map_tf_quat.z() << ","
        << map_tf_quat.w() << "]"
        << "\n  alignment_frozen=" << (alignment_rotation_frozen ? 1 : 0)
        << " alignment_update_frozen=" << (alignment_update_frozen ? 1 : 0)
        << " tapm_active=" << (tapm_active ? 1 : 0)
        << " tapm_output_active=" << (tapm_output_active ? 1 : 0)
        << " tapm_anchor_valid=" << (tapm_anchor_valid ? 1 : 0)
        << " tapm_anchor_time=" << tapm_anchor_time
        << " tapm_recover_time=" << tapm_recover_time
        << "\n  tapm_anchor_t=["
        << tapm_anchor_transform.t.transpose() << "]"
        << " tapm_anchor_s=" << tapm_anchor_transform.s
        << " alignment_frozen_quat=["
        << frozen_quat.x() << ","
        << frozen_quat.y() << ","
        << frozen_quat.z() << ","
        << frozen_quat.w() << "]";
    write_rtk_diagnostic_log("TAPM_EXIT_DEBUG", oss.str());
}


bool tapm_writes_transform = false;

void sync_world_transform_to_tapm(double timestamp)
{

    if (!tapm_writes_transform)
    {
        return;
    }
    if (!world_map_transform.isTransformValid())
    {
        return;
    }

    Transformation tf = world_map_transform.getTransform();
    if (alignment_rotation_frozen)
    {
        tf.R = alignment_frozen_rotation;
    }
    Eigen::Vector3d pos_map_with_offset = state_point.pos;
    if (enable_manual_ekf_rtk_offset)
    {
        pos_map_with_offset += manual_ekf_rtk_offset;
    }
    const Eigen::Vector3d prev_t = tf.t;
    tf.t = tapm_publish_pos - tf.s * (tf.R * pos_map_with_offset);
    tf.valid = true;
    if (rtk_diagnostic_log_enable)
    {
        const Eigen::Vector3d dt = tf.t - prev_t;
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(6)
             << "state=" << rtk_fusion_state_to_string(rtk_fusion_state)
             << " rotation_frozen=" << (alignment_rotation_frozen ? 1 : 0)
             << " tapm_active=" << (tapm_active ? 1 : 0)
             << " publish_weight=" << tapm_publish_weight
             << " dt_norm=" << dt.norm()
             << " prev_t=[" << prev_t.transpose() << "]"
             << " new_t=[" << tf.t.transpose() << "]"
             << " dt=[" << dt.transpose() << "]"
             << " tapm_publish_pos=[" << tapm_publish_pos.transpose() << "]";
        write_rtk_diagnostic_log("TAPM_SYNC_T", diag.str());
    }
    world_map_transform.updateTransform(tf, world_map_transform.getConfidence(), timestamp, true);
    update_world_state();
}

void run_tapm_sliding_state_step()
{
    static double exit_recovery_epoch = -1.0;
    static double exit_observation_epoch = -1.0;
    static int exit_stable_count = 0;
    if (exit_recovery_epoch != tapm_recover_time)
    {
        exit_recovery_epoch = tapm_recover_time;
        exit_observation_epoch = -1.0;
        exit_stable_count = 0;
    }
    if (!tapm_anchor_valid || !tapm_anchor_transform.valid || tapm_anchor_time < 0.0) return;
    const size_t configured_window_size = static_cast<size_t>(tapm_optimization_window);
    const bool has_rtk_sample = !Measures.rtk_pos.empty();
    const double history_start = tapm_recovery_history_start_time;
    const double blend_start = tapm_recover_time;
    const double time_eps = 1e-6;

    std::vector<Timepos> recovery_history;
    recovery_history.reserve(world_history_window.size());
    for (const Timepos &sample : world_history_window)
    {
        if (history_start < 0.0 || sample.timestamp + time_eps >= history_start)
        {
            recovery_history.push_back(sample);
        }
    }

    const bool rtk_after_recovery =
        has_rtk_sample &&
        (history_start < 0.0 ||
         Measures.rtk_pos.back().timestamp + time_eps >= history_start);
    const size_t history_size = recovery_history.size();
    const size_t min_solve_window_size = 3;

    if (history_size >= min_solve_window_size && rtk_after_recovery)
    {
        RUNTIME_TIMER(tapm_timer, "tapm_inclusive");
        const size_t solve_window_size =
            std::min(configured_window_size, history_size);
        std::vector<Timepos> tapm_world_vec;
        tapm_world_vec.reserve(solve_window_size);
        auto window_start = recovery_history.end() -
                            static_cast<std::vector<Timepos>::difference_type>(solve_window_size);
        tapm_world_vec.assign(window_start, recovery_history.end());
        std::vector<Eigen::Vector3d> tapm_lio_reference;
        tapm_lio_reference.reserve(tapm_world_vec.size());
        for (const Timepos& sample : tapm_world_vec)
        {
            const auto found = std::lower_bound(tapm_lio_history.begin(), tapm_lio_history.end(),
                sample.timestamp, [](const Timepos& pose, double epoch) { return pose.timestamp < epoch; });
            if (found == tapm_lio_history.end() || std::abs(found->timestamp - sample.timestamp) > time_eps) return;
            tapm_lio_reference.push_back(tapm_anchor_transform.R * found->pos);
        }
        const Timepos &rtk_current = Measures.rtk_pos.back();

        TapmConfig tapm_config;
        tapm_config.enable = true;
        tapm_config.alpha = tapm_alpha;
        tapm_config.beta = tapm_beta;
        tapm_config.lambda_v = tapm_lambda_v;
        tapm_config.lambda_a = tapm_lambda_a;
        tapm_config.gamma = tapm_gamma;
        tapm_config.nominal_dt = tapm_nominal_dt;
        tapm_config.sigma0_xy = tapm_sigma0_xy;
        tapm_config.sigma0_z = tapm_sigma0_z;
        tapm_config.v_max = tapm_v_max;
        tapm_config.v_max_z = tapm_v_max_z;
        tapm_config.a_max = tapm_a_max;
        tapm_config.a_max_z = tapm_a_max_z;
        tapm_config.rtk_sigma_min = tapm_rtk_sigma_min;
        tapm_config.rtk_match_max_dt = pgo_sw_rtk_match_max_dt;
        tapm_config.grace_frames = tapm_grace_frames;
        tapm_config.max_iterations = tapm_max_iterations;
        tapm_config.use_analytic_jacobian = tapm_use_analytic_jacobian;
        if (tapm_rotation_handover_sec > 0.0 && smoothing_alignment_valid)
        {
            if (tapm_rotation_handover_start < 0.0)
            {
                tapm_rotation_handover_start = lidar_end_time;
                tapm_rotation_target = Eigen::Quaterniond(smoothing_alignment.R).normalized();
            }
            const double rotation_weight = smoothstep01(
                (lidar_end_time - tapm_rotation_handover_start) / tapm_rotation_handover_sec);
            tapm_publish_rotation = tapm_rotation_source.slerp(rotation_weight, tapm_rotation_target).normalized();
        }

        const double sliding_started = omp_get_wtime();
        RUNTIME_TIMER(tapm_solver_timer, "tapm_solver");
        TapmResult tapm_result = solve_tapm_translation_sliding(
            tapm_world_vec,
            tapm_lio_reference,
            rtk_current,
            tapm_anchor_time,
            blend_start,
            tapm_config);
        RUNTIME_STOP(tapm_solver_timer);
        if (!tapm_result.success) return;
        diagnostic_sliding_ms += 1000.0 * (omp_get_wtime() - sliding_started);
        tapm_frame_count++;
        const double publish_weight_for_result = 1.0;
        Eigen::Vector3d publish_pos_for_result =
            blend_tapm_publish_position(tapm_world_vec.back().pos,
                                        tapm_result.position_last,
                                        publish_weight_for_result);
        Eigen::Vector3d output_pos_for_result = tapm_result.position_last;
        const double tapm_raw_output_z = output_pos_for_result.z();
        const double tapm_raw_publish_z = publish_pos_for_result.z();
        if (!tapm_publish_use_z)
        {
            output_pos_for_result.z() = tapm_world_vec.back().pos.z();
            publish_pos_for_result.z() = tapm_world_vec.back().pos.z();
        }
        const double latest_rtk_output_error = tapm_publish_use_z
            ? tapm_result.latest_rtk_error.norm() : tapm_result.latest_rtk_error.head<2>().norm();

        double latest_rtk_publish_error = latest_rtk_output_error;
        if (tapm_runtime_debug_enable && rtk_diagnostic_log_enable)
        {
            const double error_before = (tapm_world_vec.back().pos - rtk_current.pos).norm();
            const double error_after = (tapm_result.position_last - rtk_current.pos).norm();
            const Eigen::Vector3d delta_input_output =
                tapm_result.position_last - tapm_world_vec.back().pos;
            const Eigen::Vector3d delta_input_rtk =
                rtk_current.pos - tapm_world_vec.back().pos;
            const Eigen::Vector3d delta_input_publish =
                publish_pos_for_result - tapm_world_vec.back().pos;
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(3)
                 << "frame=" << tapm_frame_count
                 << " success=" << (tapm_result.success ? 1 : 0)
                 << " used_rtk=" << tapm_result.used_rtk
                 << " latest_rtk_used=" << (tapm_result.latest_rtk_used ? 1 : 0)
                 << " latest_rtk_dt=" << tapm_result.latest_rtk_dt
                 << " latest_rtk_pos=[" << tapm_result.latest_rtk_pos.transpose() << "]"
                 << " latest_rtk_output_error=" << latest_rtk_output_error
                 << " latest_rtk_publish_error=" << latest_rtk_publish_error
                 << " used_tapm=" << tapm_result.used_tapm
                 << " history_start=" << history_start
                 << " blend_start=" << blend_start
                 << " history_size=" << history_size
                 << " solve_window=" << solve_window_size
                 << "/" << configured_window_size
                 << " tau=" << tapm_result.tau_last
                 << " weight_rtk=" << tapm_result.weight_rtk_last
                 << " weight_tapm=" << tapm_result.weight_tapm_last
                 << " error_before=" << error_before
                 << " error_after=" << error_after
                 << " delta=[" << tapm_result.translation.transpose() << "]"
                 << " input_last=[" << tapm_world_vec.back().pos.transpose() << "]"
                 << " rtk_current=[" << rtk_current.pos.transpose() << "]"
                 << " output_last=[" << tapm_result.position_last.transpose() << "]"
                 << " output_used=[" << output_pos_for_result.transpose() << "]"
                 << " publish_weight=" << publish_weight_for_result
                 << " publish_pos=[" << publish_pos_for_result.transpose() << "]"
                 << " publish_use_z=" << (tapm_publish_use_z ? 1 : 0)
                 << " raw_output_z=" << tapm_raw_output_z
                 << " raw_publish_z=" << tapm_raw_publish_z
                 << " delta_input_output=[" << delta_input_output.transpose() << "]"
                 << " delta_input_publish=[" << delta_input_publish.transpose() << "]"
                 << " delta_input_rtk=[" << delta_input_rtk.transpose() << "]"
                 << " exit_limit=" << tapm_exit_frame_limit
                 << " grace_frames=" << tapm_grace_frames
                 << " rtk_match_max_dt=" << tapm_config.rtk_match_max_dt;
            write_rtk_diagnostic_log("TAPM_SLIDING", diag.str());
        }
        if (tapm_result.success)
        {
            const bool tapm_time_valid = tapm_result.tau_last >= -time_eps;
            const bool tapm_has_rtk = tapm_result.used_rtk > 0;
            if (tapm_time_valid && tapm_has_rtk)
            {
                tapm_publish_weight = publish_weight_for_result;
                tapm_output_pos = output_pos_for_result;
                tapm_publish_pos = publish_pos_for_result;
                if (lidar_end_time > tapm_previous_publish_time)
                {
                    tapm_previous_previous_publish_correction = tapm_previous_publish_correction;
                    tapm_previous_publish_correction = tapm_publish_pos - tapm_world_vec.back().pos;
                    tapm_previous_publish_dt = lidar_end_time - tapm_previous_publish_time;
                    tapm_previous_publish_time = lidar_end_time;
                }
                tapm_output_active = true;
                sync_world_transform_to_tapm(lidar_end_time);
            }
            else
            {
                tapm_output_active = false;
                tapm_publish_pos.setZero();
                tapm_publish_weight = 0.0;
                if (rtk_diagnostic_log_enable)
                {
                    std::ostringstream diag;
                    diag << std::fixed << std::setprecision(3)
                         << "skip_output reason="
                         << (!tapm_time_valid ? "negative_tau" : "no_latest_rtk_residual")
                         << " frame=" << tapm_frame_count
                         << " tau=" << tapm_result.tau_last
                         << " used_rtk=" << tapm_result.used_rtk
                         << " latest_rtk_used=" << (tapm_result.latest_rtk_used ? 1 : 0)
                         << " latest_rtk_dt=" << tapm_result.latest_rtk_dt
                         << " used_tapm=" << tapm_result.used_tapm
                         << " history_start=" << history_start
                         << " blend_start=" << blend_start
                         << " history=" << history_size;
                    write_rtk_diagnostic_log("TAPM_SLIDING", diag.str());
                }
            }
        }
        else
        {
            tapm_output_active = false;
            tapm_publish_pos.setZero();
            tapm_publish_weight = 0.0;
        }

        bool tapm_exit = false;
        std::string tapm_exit_reason;
        const bool frame_exit_hit =
            tapm_exit_frame_limit > 0 &&
            tapm_frame_count >= tapm_exit_frame_limit;
        const bool exit_error_ok =
            tapm_result.success &&
            tapm_result.latest_rtk_used &&
            std::isfinite(latest_rtk_publish_error) &&
            (tapm_exit_max_rtk_error <= 0.0 ||
             latest_rtk_publish_error <= tapm_exit_max_rtk_error);
        const bool weight_exit_hit =
            tapm_exit_frame_limit <= 0 &&
            tapm_result.success &&
            tapm_result.used_rtk > 0 &&
            tapm_result.weight_rtk_last >= 0.95 &&
            publish_weight_for_result >= 0.95 &&
            (tapm_rotation_handover_sec <= 0.0 ||
             (tapm_rotation_handover_start >= 0.0 &&
              lidar_end_time - tapm_rotation_handover_start >= tapm_rotation_handover_sec)) &&
            tapm_result.tau_last >= tapm_exit_min_duration &&
            exit_error_ok;
        if (tapm_result.latest_rtk_epoch > exit_observation_epoch)
        {
            exit_observation_epoch = tapm_result.latest_rtk_epoch;
            exit_stable_count = weight_exit_hit ? exit_stable_count + 1 : 0;
        }
        const bool duration_exit_hit = tapm_exit_max_duration > 0.0 &&
            tapm_result.tau_last >= tapm_exit_max_duration;
        if (frame_exit_hit)
        {
            tapm_exit = true;
            tapm_exit_reason = exit_error_ok ? "frame_count" : "frame_count_limit";
        }
        else if (weight_exit_hit && exit_stable_count >= tapm_exit_stable_observations)
        {
            tapm_exit = true;
            tapm_exit_reason = "stable_gnss_observations";
        }
        else if (duration_exit_hit)
        {
            tapm_exit = true;
            tapm_exit_reason = "maximum_duration_not_converged";
        }

        if (tapm_exit)
        {
            log_tapm_exit_debug(tapm_exit_reason, -1.0, -1.0);
            if (tapm_commit_exit_translation && tapm_output_active && world_map_transform.isTransformValid())
            {
                Transformation handover = world_map_transform.getTransform();
                if (tapm_rotation_handover_sec > 0.0) handover.R = tapm_publish_rotation.toRotationMatrix();
                Eigen::Vector3d local = state_point.pos;
                if (enable_manual_ekf_rtk_offset) local += manual_ekf_rtk_offset;
                handover.t = tapm_publish_pos - handover.s * handover.R * local;

                apply_alignment_transform(handover, world_map_transform.getConfidence(), lidar_end_time, true);
                update_world_state();
            }
            if (tapm_output_active)
            {
                if (!Measures.rtk_pos.empty())
                {
                    const Timepos &latest_rtk = Measures.rtk_pos.back();
                    const double world_err = tapm_result.latest_rtk_used
                                                 ? (world_state.pos - tapm_result.latest_rtk_pos).norm()
                                                 : (world_state.pos - latest_rtk.pos).norm();
                    if (rtk_diagnostic_log_enable)
                    {
                        std::ostringstream diag;
                        diag << std::fixed << std::setprecision(3)
                             << "world_err=" << world_err
                             << " lidar_time=" << lidar_end_time
                             << " rtk_time=" << latest_rtk.timestamp
                             << " latest_rtk_used=" << (tapm_result.latest_rtk_used ? 1 : 0)
                             << " latest_rtk_dt=" << tapm_result.latest_rtk_dt
                             << " matched_rtk=[" << tapm_result.latest_rtk_pos.transpose() << "]"
                             << " world=[" << world_state.pos.transpose() << "]"
                             << " rtk=[" << latest_rtk.pos.transpose() << "]";
                        write_rtk_diagnostic_log("TAPM_EXIT_APPLY", diag.str());
                    }
                }
            }
            clear_pgo_sw_for_transition("tapm_exit_before_correction");
            tapm_active = false;
            tapm_anchor_valid = false;
            tapm_anchor_time = -1.0;
            tapm_recover_time = -1.0;
            tapm_anchor_transform.valid = false;
            tapm_frame_count = 0;
            request_smoothing_completion(std::string("tapm_exit_") + tapm_exit_reason);
        }
    }
}

void run_tapm_state_machine_step(bool flg_EKF_inited)
{
    tapm_output_active = false;
    tapm_publish_pos.setZero();
    tapm_publish_weight = 0.0;
    if (tapm_enable && tapm_sliding_enable && tapm_active && rtk_initial_alignment_done && flg_EKF_inited)
    {
        run_tapm_sliding_state_step();
    }
}


void publish_odometry_world(const ros::Publisher &pubOdomWorld)
{

    if (!world_publish_state.initialized)
    {
        return;
    }


    odomWorld.header.frame_id = "world";
    odomWorld.child_frame_id = "body";
    odomWorld.header.stamp = ros::Time().fromSec(lidar_end_time);


    odomWorld.pose.pose.position.x = world_publish_state.pos.x();
    odomWorld.pose.pose.position.y = world_publish_state.pos.y();
    odomWorld.pose.pose.position.z = world_publish_state.pos.z();


    odomWorld.pose.pose.orientation.x = world_publish_state.quat.x();
    odomWorld.pose.pose.orientation.y = world_publish_state.quat.y();
    odomWorld.pose.pose.orientation.z = world_publish_state.quat.z();
    odomWorld.pose.pose.orientation.w = world_publish_state.quat.w();


    odomWorld.twist.twist.linear.x = world_publish_state.vel.x();
    odomWorld.twist.twist.linear.y = world_publish_state.vel.y();
    odomWorld.twist.twist.linear.z = world_publish_state.vel.z();


    RUNTIME_ACTION(runtime_profile::Recorder::get().state = static_cast<int>(rtk_fusion_state));
    RUNTIME_ACTION(runtime_profile::Recorder::get().output());
    pubOdomWorld.publish(odomWorld);


    static tf::TransformBroadcaster br_world;
    tf::Transform transform_world;
    tf::Quaternion q_world;

    transform_world.setOrigin(
        tf::Vector3(world_publish_state.pos.x(), world_publish_state.pos.y(), world_publish_state.pos.z()));
    q_world.setW(world_publish_state.quat.w());
    q_world.setX(world_publish_state.quat.x());
    q_world.setY(world_publish_state.quat.y());
    q_world.setZ(world_publish_state.quat.z());
    transform_world.setRotation(q_world);

    br_world.sendTransform(tf::StampedTransform(transform_world, odomWorld.header.stamp, "world", "body_world"));
}


void publish_rtk_odometry_world(const ros::Publisher &pubRtkOdom)
{
    if (Measures.rtk_pos.empty())
    {
        return;
    }

    const Timepos &latest_rtk = Measures.rtk_pos.back();
    if (latest_rtk.timestamp <= last_published_rtk_odom_timestamp + 1e-6)
    {
        return;
    }

    Eigen::Vector3d rtk_world = latest_rtk.pos;

    Eigen::Quaterniond imu_world_quat = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d rotation_world = Eigen::Matrix3d::Identity();
    if (world_state.initialized)
    {
        imu_world_quat = world_state.quat;
        rotation_world = imu_world_quat.toRotationMatrix();
    }
    else if (world_map_transform.isTransformValid())
    {
        Eigen::Matrix3d R_world_map = world_map_transform.getTransform().R;
        rotation_world = R_world_map * state_point.rot.toRotationMatrix();
        imu_world_quat = Eigen::Quaterniond(rotation_world);
    }
    else
    {
        rotation_world = state_point.rot.toRotationMatrix();
        imu_world_quat = Eigen::Quaterniond(rotation_world);
    }
    imu_world_quat.normalize();

    Eigen::Vector3d imu_world_pos = rtk_world;
    if (enable_antenna_lever_arm_compensation)
    {
        imu_world_pos = compensate_antenna_to_imu(rtk_world, rotation_world);
    }

    odomRtkWorld.header.frame_id = "world";
    odomRtkWorld.child_frame_id = "rtk_body";
    odomRtkWorld.header.stamp = ros::Time().fromSec(latest_rtk.timestamp);

    odomRtkWorld.pose.pose.position.x = imu_world_pos.x();
    odomRtkWorld.pose.pose.position.y = imu_world_pos.y();
    odomRtkWorld.pose.pose.position.z = imu_world_pos.z();

    odomRtkWorld.pose.pose.orientation.x = imu_world_quat.x();
    odomRtkWorld.pose.pose.orientation.y = imu_world_quat.y();
    odomRtkWorld.pose.pose.orientation.z = imu_world_quat.z();
    odomRtkWorld.pose.pose.orientation.w = imu_world_quat.w();

    odomRtkWorld.twist.twist.linear.x = 0.0;
    odomRtkWorld.twist.twist.linear.y = 0.0;
    odomRtkWorld.twist.twist.linear.z = 0.0;

    pubRtkOdom.publish(odomRtkWorld);
    last_published_rtk_odom_timestamp = latest_rtk.timestamp;

    static tf::TransformBroadcaster br_rtk;
    tf::Transform transform_rtk;
    tf::Quaternion q_rtk;
    transform_rtk.setOrigin(tf::Vector3(imu_world_pos.x(), imu_world_pos.y(), imu_world_pos.z()));
    q_rtk.setW(imu_world_quat.w());
    q_rtk.setX(imu_world_quat.x());
    q_rtk.setY(imu_world_quat.y());
    q_rtk.setZ(imu_world_quat.z());
    transform_rtk.setRotation(q_rtk);
    br_rtk.sendTransform(tf::StampedTransform(transform_rtk, odomRtkWorld.header.stamp, "world", "rtk_body"));
}

void publish_odometry_world_smooth(const ros::Publisher &pubOdomWorldSmooth)
{
    if (!pgo_sw_enable)
    {
        pgo_debug_lio_valid = false;
        pgo_debug_lio_time = -1.0;
        pgo_debug_lio_pos.setZero();
        pgo_debug_output_valid = false;
        pgo_debug_output_system_applied = false;
        pgo_debug_output_time = -1.0;
        pgo_debug_output_pos.setZero();
        pgo_debug_output_delta.setZero();
        pgo_debug_output_status = "pgo_disabled";
        pgo_debug_window_size = world_sw_backend.window().size();
        pgo_debug_rtk_buffer_size = world_sw_backend.rtkBufferSize();
        pgo_debug_optimized_size = world_sw_backend.optimizedSize();
        pgo_debug_rtk_factor_count = 0;
        pgo_debug_rtk_sigma_scale = 1.0;
        pgo_debug_constraint_valid = false;
        pgo_debug_constraint_added = false;
        pgo_debug_constraint_time = -1.0;
        pgo_debug_constraint_dt = std::numeric_limits<double>::quiet_NaN();
        pgo_debug_constraint_pos.setZero();
        pgo_debug_constraint_source = "none";
        pgo_debug_constraint_status = "pgo_disabled";
        write_rtk_diagnostic_log("PGO_SW", "skipped reason=disabled");
        return;
    }

    if (!world_state.initialized)
    {
        pgo_debug_lio_valid = false;
        pgo_debug_lio_time = -1.0;
        pgo_debug_lio_pos.setZero();
        pgo_debug_output_valid = false;
        pgo_debug_output_system_applied = false;
        pgo_debug_output_time = -1.0;
        pgo_debug_output_pos.setZero();
        pgo_debug_output_delta.setZero();
        pgo_debug_output_status = "world_not_initialized";
        pgo_debug_window_size = world_sw_backend.window().size();
        pgo_debug_rtk_buffer_size = world_sw_backend.rtkBufferSize();
        pgo_debug_optimized_size = world_sw_backend.optimizedSize();
        pgo_debug_rtk_factor_count = 0;
        pgo_debug_rtk_sigma_scale = 1.0;
        pgo_debug_constraint_valid = false;
        pgo_debug_constraint_added = false;
        pgo_debug_constraint_time = -1.0;
        pgo_debug_constraint_dt = std::numeric_limits<double>::quiet_NaN();
        pgo_debug_constraint_pos.setZero();
        pgo_debug_constraint_source = "none";
        pgo_debug_constraint_status = "world_not_initialized";
        write_rtk_diagnostic_log("PGO_SW", "skipped reason=world_state_not_initialized");
        return;
    }

    const double t_lio = lidar_end_time;
    const bool pgo_log_due =
        rtk_diagnostic_log_enable &&
        (pgo_sw_debug_interval <= 0.0 ||
         pgo_sw_last_debug_log_time < 0.0 ||
         t_lio - pgo_sw_last_debug_log_time >= pgo_sw_debug_interval);
    bool use_opt = false;
    bool pgo_added_rtk = false;
    bool pgo_optimize_run = false;
    bool pgo_optimize_due = false;
    double pgo_optimize_cost_ms = 0.0;
    pgo_sw::Pose opt;

    pgo_sw::Pose lio_pose;
    lio_pose.t = t_lio;
    lio_pose.p = world_state.pos;
    lio_pose.q = world_state.quat;
    world_sw_backend.addLioPose(lio_pose);
    pgo_debug_lio_valid = true;
    pgo_debug_lio_time = lio_pose.t;
    pgo_debug_lio_pos = lio_pose.p;
    pgo_debug_output_valid = false;
    pgo_debug_output_system_applied = false;
    pgo_debug_output_time = -1.0;
    pgo_debug_output_pos.setZero();
    pgo_debug_output_delta.setZero();
    pgo_debug_output_status = "pending";
    pgo_debug_window_size = world_sw_backend.window().size();
    pgo_debug_rtk_buffer_size = world_sw_backend.rtkBufferSize();
    pgo_debug_optimized_size = world_sw_backend.optimizedSize();
    pgo_debug_rtk_factor_count = world_sw_backend.latestRtkFactorCount();
    pgo_debug_rtk_sigma_scale = world_sw_backend.latestRtkSigmaScale();
    pgo_debug_constraint_valid = false;
    pgo_debug_constraint_added = false;
    pgo_debug_constraint_time = -1.0;
    pgo_debug_constraint_dt = std::numeric_limits<double>::quiet_NaN();
    pgo_debug_constraint_pos.setZero();
    pgo_debug_constraint_source = "none";
    pgo_debug_constraint_status = "none";

    const bool use_tapm_virtual_rtk =
        pgo_sw_use_tapm_virtual_rtk &&
        rtk_fusion_state == RtkFusionState::SMOOTHING &&
        tapm_output_active &&
        rtk_fix_active;
    Eigen::Vector3d constraint_pos = Eigen::Vector3d::Zero();
    double constraint_time = -1.0;
    double constraint_sigma_xy = pgo_sw_rtk_sigma_xy;
    double constraint_sigma_z = pgo_sw_rtk_sigma_z;
    std::string constraint_source = "none";
    bool has_position_constraint = false;
    bool position_constraint_rejected = false;
    const double recovery_weight = get_pgo_recovery_weight(t_lio);
    const bool recovery_ramp_active = recovery_weight < 1.0 - 1e-6;
    bool constraint_z_enabled = true;
    bool constraint_z_overridden = false;
    double constraint_original_z = std::numeric_limits<double>::quiet_NaN();

    if (use_tapm_virtual_rtk)
    {
        constraint_pos = tapm_publish_pos;
        constraint_time = t_lio;
        constraint_sigma_xy = pgo_sw_tapm_sigma_xy;
        constraint_sigma_z = pgo_sw_tapm_sigma_z;
        constraint_source = "tapm_virtual";
        has_position_constraint = true;
    }
    else if (is_pgo_raw_rtk_allowed() && !Measures.rtk_pos.empty())
    {
        Timepos matched_rtk;
        double matched_dt = std::numeric_limits<double>::quiet_NaN();
        std::string matched_source;
        if (selectRtkForTimestamp(Measures.rtk_pos,
                                  t_lio,
                                  pgo_sw_rtk_match_max_dt,
                                  matched_rtk,
                                  matched_dt,
                                  matched_source))
        {
            if (matched_source == "rtk_interp" || matched_source == "rtk_nearest")
            {
                constraint_pos = matched_rtk.pos;
                constraint_time = matched_rtk.timestamp;
                constraint_sigma_xy = pgo_sw_rtk_sigma_xy;
                constraint_sigma_z = pgo_sw_rtk_sigma_z;
                if (std::isfinite(matched_rtk.h_acc) && matched_rtk.h_acc > 0.0)
                {
                    constraint_sigma_xy = std::max(constraint_sigma_xy, matched_rtk.h_acc);
                }
                if (std::isfinite(matched_rtk.v_acc) && matched_rtk.v_acc > 0.0)
                {
                    constraint_sigma_z = std::max(constraint_sigma_z, matched_rtk.v_acc);
                }
                constraint_source = matched_source;
                has_position_constraint = true;
            }
            else
            {
                position_constraint_rejected = true;
                pgo_debug_constraint_valid = true;
                pgo_debug_constraint_added = false;
                pgo_debug_constraint_time = matched_rtk.timestamp;
                pgo_debug_constraint_dt = matched_dt;
                pgo_debug_constraint_pos = matched_rtk.pos;
                pgo_debug_constraint_source = matched_source;
                pgo_debug_constraint_status = "nearest_rejected";
                if (pgo_log_due)
                {
                    std::ostringstream diag;
                    diag << std::fixed << std::setprecision(6)
                         << "skip_rtk reason=nearest_rejected"
                         << " source=" << matched_source
                         << " t_lio=" << t_lio
                         << " t_rtk=" << matched_rtk.timestamp
                         << " dt=" << matched_dt
                         << " max_dt=" << pgo_sw_rtk_match_max_dt
                         << " rtk_pos=[" << matched_rtk.pos.transpose() << "]";
                    write_rtk_diagnostic_log("PGO_SW", diag.str());
                }
            }
        }
        else
        {
            position_constraint_rejected = true;
            const Timepos &latest_rtk = Measures.rtk_pos.back();
            pgo_debug_constraint_valid = true;
            pgo_debug_constraint_added = false;
            pgo_debug_constraint_time = latest_rtk.timestamp;
            pgo_debug_constraint_dt = std::abs(latest_rtk.timestamp - t_lio);
            pgo_debug_constraint_pos = latest_rtk.pos;
            pgo_debug_constraint_source = "rtk";
            pgo_debug_constraint_status = "time_diff_exceed";
            if (pgo_log_due)
            {
                std::ostringstream diag;
                diag << std::fixed << std::setprecision(6)
                     << "skip_rtk reason=time_diff_exceed"
                     << " source=rtk"
                     << " t_lio=" << t_lio
                     << " t_rtk=" << latest_rtk.timestamp
                     << " dt=" << std::abs(latest_rtk.timestamp - t_lio)
                     << " max_dt=" << pgo_sw_rtk_match_max_dt
                     << " rtk_pos=[" << latest_rtk.pos.transpose() << "]";
                write_rtk_diagnostic_log("PGO_SW", diag.str());
            }
        }
    }

    if (has_position_constraint)
    {
        constraint_z_enabled =
            (constraint_source == "tapm_virtual") ? pgo_sw_use_tapm_z : pgo_sw_use_rtk_z;
        constraint_original_z = constraint_pos.z();
        if (!constraint_z_enabled)
        {
            constraint_pos.z() = lio_pose.p.z();
            constraint_sigma_z = std::max(constraint_sigma_z, 1.0e6);
            constraint_z_overridden = true;
        }
        else if (recovery_ramp_active && pgo_sw_recovery_z_sigma_scale > 1.0)
        {
            constraint_sigma_z *= pgo_sw_recovery_z_sigma_scale;
        }

        if (recovery_ramp_active)
        {
            const double sigma_scale = 1.0 / std::max(1e-3, recovery_weight);
            constraint_sigma_xy *= sigma_scale;
            constraint_sigma_z *= sigma_scale;
        }
    }

    if (has_position_constraint)
    {
        const double t_rtk = constraint_time;
        const bool new_rtk = (t_rtk > pgo_sw_last_rtk_used + 1e-6);
        const double time_diff = std::abs(t_rtk - t_lio);
        const bool time_ok = (time_diff <= pgo_sw_rtk_match_max_dt);
        pgo_debug_constraint_valid = true;
        pgo_debug_constraint_added = false;
        pgo_debug_constraint_time = t_rtk;
        pgo_debug_constraint_dt = time_diff;
        pgo_debug_constraint_pos = constraint_pos;
        pgo_debug_constraint_source = constraint_source;
        pgo_debug_constraint_status = "candidate";
        if (new_rtk && time_ok)
        {
            pgo_sw::RtkMeas rtk_meas;
            rtk_meas.t = t_rtk;
            rtk_meas.p = constraint_pos;
            rtk_meas.sigma_xy = constraint_sigma_xy;
            rtk_meas.sigma_z = constraint_sigma_z;
            rtk_meas.valid = true;
            world_sw_backend.addRtk(rtk_meas);
            pgo_sw_last_rtk_used = t_rtk;
            pgo_added_rtk = true;
            pgo_debug_rtk_buffer_size = world_sw_backend.rtkBufferSize();
            pgo_debug_constraint_added = true;
            pgo_debug_constraint_status = "added";
            if (pgo_log_due)
            {
                std::ostringstream diag;
                diag << std::fixed << std::setprecision(6)
                     << "add_rtk source=" << constraint_source
                     << " t_lio=" << t_lio
                     << " t_rtk=" << t_rtk
                     << " dt=" << time_diff
                     << " max_dt=" << pgo_sw_rtk_match_max_dt
                     << " sigma_xy=" << constraint_sigma_xy
                     << " sigma_z=" << constraint_sigma_z
                     << " recovery_weight=" << recovery_weight
                     << " recovery_ramp_active=" << (recovery_ramp_active ? 1 : 0)
                     << " z_enabled=" << (constraint_z_enabled ? 1 : 0)
                     << " z_overridden=" << (constraint_z_overridden ? 1 : 0)
                     << " original_z=" << constraint_original_z
                     << " rtk_pos=[" << constraint_pos.transpose() << "]";
                write_rtk_diagnostic_log("PGO_SW", diag.str());
            }
        }
        else
        {
            pgo_debug_constraint_status =
                new_rtk ? "time_diff_exceed" : "duplicate_or_old_rtk";
            if (pgo_log_due)
            {
                std::ostringstream diag;
                diag << std::fixed << std::setprecision(6)
                     << "skip_rtk reason="
                     << (new_rtk ? "time_diff_exceed" : "duplicate_or_old_rtk")
                     << " source=" << constraint_source
                     << " t_lio=" << t_lio
                     << " t_rtk=" << t_rtk
                     << " dt=" << time_diff
                     << " max_dt=" << pgo_sw_rtk_match_max_dt
                     << " last_rtk_used=" << pgo_sw_last_rtk_used;
                write_rtk_diagnostic_log("PGO_SW", diag.str());
            }
        }
    }
    else if (!position_constraint_rejected)
    {
        pgo_debug_constraint_status = "no_constraint";
        if (pgo_log_due)
        {
            std::ostringstream diag;
            diag << std::fixed << std::setprecision(6)
                 << "skip_rtk reason=no_rtk_measurement"
                 << " t_lio=" << t_lio;
            write_rtk_diagnostic_log("PGO_SW", diag.str());
        }
    }

    const bool correction_output_ready =
        rtk_fusion_state == RtkFusionState::CORRECTION &&
        world_map_transform.isTransformValid() &&
        world_sw_backend.rtkBufferSize() > 0;
    const bool pgo_system_output_allowed =
        pgo_sw_use_as_system_output &&
        world_publish_state.initialized &&
        correction_output_ready;

    ++pgo_sw_frame_counter;
    pgo_optimize_due =
        !pgo_sw_last_delta_valid ||
        pgo_sw_optimize_stride <= 1 ||
        (pgo_sw_frame_counter % pgo_sw_optimize_stride == 0);

    if (pgo_optimize_due)
    {
        const double opt_start = omp_get_wtime();
        pgo_optimize_run = true;
        if (world_sw_backend.optimize() && world_sw_backend.getLatestOptimized(&opt))
        {
            pgo_optimize_cost_ms = (omp_get_wtime() - opt_start) * 1000.0;
            pgo_sw_last_opt_cost_ms = pgo_optimize_cost_ms;
            pgo_sw_last_delta = opt.p - lio_pose.p;
            pgo_sw_last_delta_valid = true;
            pgo_sw_last_opt_time = opt.t;
            use_opt = true;
            pgo_debug_output_valid = true;
            pgo_debug_output_system_applied = pgo_system_output_allowed;
            pgo_debug_output_time = opt.t;
            pgo_debug_output_pos = opt.p;
            pgo_debug_output_delta = pgo_sw_last_delta;
            pgo_debug_output_status = "optimized";
        }
        else
        {
            pgo_optimize_cost_ms = (omp_get_wtime() - opt_start) * 1000.0;
            pgo_sw_last_opt_cost_ms = pgo_optimize_cost_ms;
            if (pgo_sw_last_delta_valid)
            {
                use_opt = true;
                pgo_debug_output_valid = true;
                pgo_debug_output_system_applied = pgo_system_output_allowed;
                pgo_debug_output_time = t_lio;
                pgo_debug_output_pos = lio_pose.p + pgo_sw_last_delta;
                pgo_debug_output_delta = pgo_sw_last_delta;
                pgo_debug_output_status = "cached_after_failed_opt";
            }
            else
            {
                pgo_debug_output_valid = false;
                pgo_debug_output_system_applied = false;
                pgo_debug_output_time = -1.0;
                pgo_debug_output_pos.setZero();
                pgo_debug_output_delta.setZero();
                pgo_debug_output_status = "no_output";
            }
        }
    }
    else if (pgo_sw_last_delta_valid)
    {
        use_opt = true;
        pgo_debug_output_valid = true;
        pgo_debug_output_system_applied = pgo_system_output_allowed;
        pgo_debug_output_time = t_lio;
        pgo_debug_output_pos = lio_pose.p + pgo_sw_last_delta;
        pgo_debug_output_delta = pgo_sw_last_delta;
        pgo_debug_output_status = "cached_delta";
    }
    else
    {
        pgo_debug_output_valid = false;
        pgo_debug_output_system_applied = false;
        pgo_debug_output_time = -1.0;
        pgo_debug_output_pos.setZero();
        pgo_debug_output_delta.setZero();
        pgo_debug_output_status = "no_cache";
    }
    pgo_debug_window_size = world_sw_backend.window().size();
    pgo_debug_rtk_buffer_size = world_sw_backend.rtkBufferSize();
    pgo_debug_optimized_size = world_sw_backend.optimizedSize();
    pgo_debug_rtk_factor_count = world_sw_backend.latestRtkFactorCount();
    pgo_debug_rtk_sigma_scale = world_sw_backend.latestRtkSigmaScale();

    if (rtk_diagnostic_log_enable && (pgo_optimize_run || pgo_log_due))
    {
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(6)
             << "run=" << (pgo_optimize_run ? 1 : 0)
             << " due=" << (pgo_optimize_due ? 1 : 0)
             << " stride=" << pgo_sw_optimize_stride
             << " frame=" << pgo_sw_frame_counter
             << " cost_ms=" << pgo_optimize_cost_ms
             << " last_cost_ms=" << pgo_sw_last_opt_cost_ms
             << " status=" << pgo_debug_output_status
             << " system_allowed=" << (pgo_system_output_allowed ? 1 : 0)
             << " added_rtk=" << (pgo_added_rtk ? 1 : 0)
             << " window=" << pgo_debug_window_size
             << " rtk_buf=" << pgo_debug_rtk_buffer_size
             << " opt_buf=" << pgo_debug_optimized_size
             << " rtk_factor=" << pgo_debug_rtk_factor_count
             << " sigma_scale=" << pgo_debug_rtk_sigma_scale
             << " last_delta_valid=" << (pgo_sw_last_delta_valid ? 1 : 0)
             << " last_delta_norm=" << pgo_sw_last_delta.norm()
             << " last_opt_time=" << pgo_sw_last_opt_time
             << " recovery_weight=" << recovery_weight
             << " recovery_ramp_active=" << (recovery_ramp_active ? 1 : 0);
        write_rtk_diagnostic_log("PGO_SW_OPT", diag.str());
    }

    odomWorldSmooth.header.frame_id = "world";
    odomWorldSmooth.child_frame_id = "body_world_smooth";
    odomWorldSmooth.header.stamp = ros::Time().fromSec(t_lio);

    Eigen::Vector3d output_pos = world_publish_state.initialized ? world_publish_state.pos : lio_pose.p;
    Eigen::Quaterniond output_quat = world_publish_state.initialized ? world_publish_state.quat : lio_pose.q;
    output_quat.normalize();
    if (use_opt && pgo_system_output_allowed)
    {
        output_pos = pgo_debug_output_pos;
        pgo_debug_output_system_applied = true;
    }

    if (pgo_system_output_allowed)
    {
        apply_world_output_blend(t_lio, output_pos, output_quat);
        world_publish_state.pos = output_pos;
        world_publish_state.quat = output_quat;
        world_publish_state.quat.normalize();
    }
    else if (use_opt)
    {
        output_pos = pgo_debug_output_pos;
    }

    odomWorldSmooth.pose.pose.position.x = output_pos.x();
    odomWorldSmooth.pose.pose.position.y = output_pos.y();
    odomWorldSmooth.pose.pose.position.z = output_pos.z();
    odomWorldSmooth.pose.pose.orientation.x = output_quat.x();
    odomWorldSmooth.pose.pose.orientation.y = output_quat.y();
    odomWorldSmooth.pose.pose.orientation.z = output_quat.z();
    odomWorldSmooth.pose.pose.orientation.w = output_quat.w();

    if (rtk_diagnostic_log_enable)
    {
        const Eigen::Vector3d out_pos(odomWorldSmooth.pose.pose.position.x,
                                      odomWorldSmooth.pose.pose.position.y,
                                      odomWorldSmooth.pose.pose.position.z);
        Eigen::Quaterniond out_quat(odomWorldSmooth.pose.pose.orientation.w,
                                    odomWorldSmooth.pose.pose.orientation.x,
                                    odomWorldSmooth.pose.pose.orientation.y,
                                    odomWorldSmooth.pose.pose.orientation.z);
        Eigen::Quaterniond lio_quat = lio_pose.q;
        out_quat.normalize();
        lio_quat.normalize();
        const double out_time = odomWorldSmooth.header.stamp.toSec();
        const Eigen::Vector3d out_delta = out_pos - lio_pose.p;
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(6)
             << "state=" << rtk_fusion_state_to_string(rtk_fusion_state)
             << " use_opt=" << (use_opt ? 1 : 0)
             << " system_output=" << (pgo_sw_use_as_system_output ? 1 : 0)
             << " system_applied=" << (pgo_debug_output_system_applied ? 1 : 0)
             << " tf_valid=" << (world_map_transform.isTransformValid() ? 1 : 0)
             << " added_rtk=" << (pgo_added_rtk ? 1 : 0)
             << " window=" << pgo_debug_window_size
             << " rtk_buf=" << pgo_debug_rtk_buffer_size
             << " opt_buf=" << pgo_debug_optimized_size
             << " rtk_factor=" << pgo_debug_rtk_factor_count
             << " sigma_scale=" << pgo_debug_rtk_sigma_scale
             << " has_constraint=" << (has_position_constraint ? 1 : 0)
             << " constraint_source=" << constraint_source
             << " constraint_status=" << pgo_debug_constraint_status
             << " recovery_weight=" << recovery_weight
             << " recovery_ramp_active=" << (recovery_ramp_active ? 1 : 0)
             << " z_enabled=" << (constraint_z_enabled ? 1 : 0)
             << " z_overridden=" << (constraint_z_overridden ? 1 : 0)
             << " original_z=" << constraint_original_z
             << " t_lio=" << t_lio
             << " t_out=" << out_time
             << " optimize_run=" << (pgo_optimize_run ? 1 : 0)
             << " optimize_due=" << (pgo_optimize_due ? 1 : 0)
             << " optimize_cost_ms=" << pgo_optimize_cost_ms
             << " delta_norm=" << out_delta.norm()
             << " lio=[" << lio_pose.p.transpose() << "]"
             << " out=[" << out_pos.transpose() << "]"
             << " delta=[" << out_delta.transpose() << "]"
             << " lio_q=[" << lio_quat.x() << "," << lio_quat.y() << ","
             << lio_quat.z() << "," << lio_quat.w() << "]"
             << " out_q=[" << out_quat.x() << "," << out_quat.y() << ","
             << out_quat.z() << "," << out_quat.w() << "]";
        write_rtk_diagnostic_log("PGO_SW_OUT_DEBUG", diag.str());
    }

    if (has_position_constraint && pgo_log_due)
    {
        const Eigen::Vector3d opt_pos(odomWorldSmooth.pose.pose.position.x,
                                      odomWorldSmooth.pose.pose.position.y,
                                      odomWorldSmooth.pose.pose.position.z);
        const double raw_err = (lio_pose.p - constraint_pos).norm();
        const double opt_err = (opt_pos - constraint_pos).norm();
        const double opt_time = odomWorldSmooth.header.stamp.toSec();
        std::ostringstream diag;
        diag << std::fixed << std::setprecision(3)
             << "use_opt=" << (use_opt ? 1 : 0)
             << " system_output=" << (pgo_sw_use_as_system_output ? 1 : 0)
             << " source=" << constraint_source
             << " window=" << world_sw_backend.window().size()
             << " rtk_factor=" << pgo_debug_rtk_factor_count
             << " sigma_scale=" << pgo_debug_rtk_sigma_scale
             << " t_lio=" << t_lio
             << " t_out=" << opt_time
             << " t_rtk=" << constraint_time
             << " dt_out_rtk=" << std::fabs(opt_time - constraint_time)
             << " raw_err=" << raw_err
             << " opt_err=" << opt_err
             << " raw=[" << lio_pose.p.transpose() << "]"
             << " opt=[" << opt_pos.transpose() << "]"
             << " rtk=[" << constraint_pos.transpose() << "]";
        write_rtk_diagnostic_log("PGO_SW_OUT", diag.str());
    }
    if (pgo_log_due)
    {
        pgo_sw_last_debug_log_time = t_lio;
    }
    odomWorldSmooth.twist.twist.linear.x = world_state.vel.x();
    odomWorldSmooth.twist.twist.linear.y = world_state.vel.y();
    odomWorldSmooth.twist.twist.linear.z = world_state.vel.z();

    pubOdomWorldSmooth.publish(odomWorldSmooth);

    static tf::TransformBroadcaster br_world_smooth;
    tf::Transform transform_world_smooth;
    tf::Quaternion q_world_smooth;
    transform_world_smooth.setOrigin(tf::Vector3(odomWorldSmooth.pose.pose.position.x,
                                                 odomWorldSmooth.pose.pose.position.y,
                                                 odomWorldSmooth.pose.pose.position.z));
    q_world_smooth.setW(odomWorldSmooth.pose.pose.orientation.w);
    q_world_smooth.setX(odomWorldSmooth.pose.pose.orientation.x);
    q_world_smooth.setY(odomWorldSmooth.pose.pose.orientation.y);
    q_world_smooth.setZ(odomWorldSmooth.pose.pose.orientation.z);
    transform_world_smooth.setRotation(q_world_smooth);
    br_world_smooth.sendTransform(tf::StampedTransform(transform_world_smooth,
                                                       odomWorldSmooth.header.stamp,
                                                       "world",
                                                       "body_world_smooth"));
}

void publish_world_navsatfix(const ros::Publisher &pubWorldNavSatFix)
{
    if (odomWorldSmooth.header.stamp.toSec() <= 0.0)
    {
        return;
    }

    if (!p_rtk->rtk_isvalid ||
        p_rtk->first_latitude <= 0.0 ||
        p_rtk->first_longitude <= 0.0 ||
        p_rtk->first_altitude <= 0.0)
    {
        (void)0;
        return;
    }

    const Eigen::Vector3d enu_world(odomWorldSmooth.pose.pose.position.x,
                                    odomWorldSmooth.pose.pose.position.y,
                                    odomWorldSmooth.pose.pose.position.z);
    const Eigen::Vector3d ref_lla(p_rtk->first_latitude,
                                  p_rtk->first_longitude,
                                  p_rtk->first_altitude);
    Eigen::Vector3d lla;
    if (!enu_to_lla(enu_world, ref_lla, &lla))
    {
        (void)0;
        return;
    }

    sensor_msgs::NavSatFix navsat_msg;
    navsat_msg.header.stamp = odomWorldSmooth.header.stamp;
    navsat_msg.header.frame_id = "world";
    navsat_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
    navsat_msg.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;
    navsat_msg.latitude = lla.x();
    navsat_msg.longitude = lla.y();
    navsat_msg.altitude = lla.z();
    navsat_msg.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
    navsat_msg.position_covariance[0] = 25.0;
    navsat_msg.position_covariance[4] = 25.0;
    navsat_msg.position_covariance[8] = 64.0;

    pubWorldNavSatFix.publish(navsat_msg);
}

void publish_rtk_fusion_state(const ros::Publisher &pubRtkFusionState)
{
    if (!rtk_fusion_state_pub_enable)
    {
        return;
    }

    const uint8_t current_state = static_cast<uint8_t>(rtk_fusion_state);
    if (current_state == rtk_fusion_state_last_pub)
    {
        return;
    }

    std_msgs::UInt8 msg;
    msg.data = current_state;
    pubRtkFusionState.publish(msg);
    rtk_fusion_state_last_pub = current_state;
}

bool publish_path(const ros::Publisher pubPath)
{
    set_posestamp_(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "map";
    path.header.frame_id = "map";
    path.header.stamp = msg_body_pose.header.stamp;

    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
    return true;
}

void rtk_publish_path(const ros::Publisher pubPath)
{
    if (Measures.rtk_pos.empty())
        return;
    const Timepos &latest_rtk = Measures.rtk_pos.back();
    if (latest_rtk.timestamp <= last_published_rtk_path_timestamp + 1e-6)
    {
        return;
    }

    set_rtkposestamp_(rtk_msg_body_pose);
    rtk_msg_body_pose.header.stamp = ros::Time().fromSec(latest_rtk.timestamp);
    rtk_msg_body_pose.header.frame_id = "world";
    rtk_path.header.frame_id = "world";
    rtk_path.header.stamp = rtk_msg_body_pose.header.stamp;

    rtk_path.poses.push_back(rtk_msg_body_pose);
    pubPath.publish(rtk_path);
    last_published_rtk_path_timestamp = latest_rtk.timestamp;
}

void any_publish_path(const ros::Publisher pubPath)
{

    set_anyposestamp_(any_msg_body_pose);
    any_msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    any_msg_body_pose.header.frame_id = Global_GNSS_LIO.valid ? "world" : "map";
    any_path.header.frame_id = any_msg_body_pose.header.frame_id;
    any_path.header.stamp = any_msg_body_pose.header.stamp;

    any_path.poses.push_back(any_msg_body_pose);
    pubPath.publish(any_path);
}

void world_publish_path(const ros::Publisher pubPath)
{
    if (!world_publish_state.initialized)
    {
        return;
    }

    if (!set_worldposestamp_(world_msg_body_pose))
    {
        return;
    }

    world_msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    world_msg_body_pose.header.frame_id = "world";
    world_ekf_path.header.frame_id = "world";
    world_ekf_path.header.stamp = world_msg_body_pose.header.stamp;

    world_ekf_path.poses.push_back(world_msg_body_pose);
    const Eigen::Vector3d pos_world(world_msg_body_pose.pose.position.x,
                                    world_msg_body_pose.pose.position.y,
                                    world_msg_body_pose.pose.position.z);
    const Eigen::Quaterniond quat_world(world_msg_body_pose.pose.orientation.w,
                                        world_msg_body_pose.pose.orientation.x,
                                        world_msg_body_pose.pose.orientation.y,
                                        world_msg_body_pose.pose.orientation.z);
    update_world_path_peak_metrics(lidar_end_time, pos_world, quat_world);
    pubPath.publish(world_ekf_path);
}

void tapm_publish_path(const ros::Publisher pubPath)
{
    if (!tapm_enable || !tapm_output_active || !world_publish_state.initialized)
    {
        return;
    }

    Eigen::Quaterniond world_quat = world_publish_state.quat;
    world_quat.normalize();

    tapm_msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    tapm_msg_body_pose.header.frame_id = "world";
    tapm_msg_body_pose.pose.position.x = tapm_publish_pos.x();
    tapm_msg_body_pose.pose.position.y = tapm_publish_pos.y();
    tapm_msg_body_pose.pose.position.z = tapm_publish_pos.z();
    tapm_msg_body_pose.pose.orientation.x = world_quat.x();
    tapm_msg_body_pose.pose.orientation.y = world_quat.y();
    tapm_msg_body_pose.pose.orientation.z = world_quat.z();
    tapm_msg_body_pose.pose.orientation.w = world_quat.w();

    tapm_path.header.frame_id = "world";
    tapm_path.header.stamp = tapm_msg_body_pose.header.stamp;

    tapm_path.poses.push_back(tapm_msg_body_pose);
    pubPath.publish(tapm_path);
}

void world_smooth_publish_path(const ros::Publisher pubPath)
{
    if (!world_path_en || !pgo_sw_enable || !world_state.initialized)
    {
        return;
    }

    world_smooth_msg_body_pose.header.stamp = odomWorldSmooth.header.stamp;
    world_smooth_msg_body_pose.header.frame_id = "world";
    world_smooth_msg_body_pose.pose.position.x = odomWorldSmooth.pose.pose.position.x;
    world_smooth_msg_body_pose.pose.position.y = odomWorldSmooth.pose.pose.position.y;
    world_smooth_msg_body_pose.pose.position.z = odomWorldSmooth.pose.pose.position.z;
    world_smooth_msg_body_pose.pose.orientation.x = odomWorldSmooth.pose.pose.orientation.x;
    world_smooth_msg_body_pose.pose.orientation.y = odomWorldSmooth.pose.pose.orientation.y;
    world_smooth_msg_body_pose.pose.orientation.z = odomWorldSmooth.pose.pose.orientation.z;
    world_smooth_msg_body_pose.pose.orientation.w = odomWorldSmooth.pose.pose.orientation.w;

    world_path.header.frame_id = "world";
    world_path.header.stamp = world_smooth_msg_body_pose.header.stamp;

    world_path.poses.push_back(world_smooth_msg_body_pose);
    pubPath.publish(world_path);
}


void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;


    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];


        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {

            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, plane_fit_threshold))

        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());


            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {

        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }


    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        return;
    }


    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    bool add_rtk_constraint = false;
    std::string rtk_reason = "ok";
    double rtk_time = nan_value;
    double rtk_time_diff = nan_value;
    double rtk_h_acc = nan_value;
    double rtk_v_acc = nan_value;
    double rtk_var_xy = nan_value;
    double rtk_var_z = nan_value;
    Eigen::Vector3d rtk_world(nan_value, nan_value, nan_value);
    Eigen::Vector3d rtk_pos_map(nan_value, nan_value, nan_value);
    Eigen::Vector3d rtk_residual_map(nan_value, nan_value, nan_value);
    Transformation active_tf;
    active_tf.valid = false;
    if (world_map_transform.isTransformValid())
    {
        active_tf = world_map_transform.getTransform();
    }
    if (!rtk_ekf_constraint_enable)
    {
        rtk_reason = "disabled";
    }
    else if (rtk_fusion_state != RtkFusionState::CORRECTION)
    {
        rtk_reason = std::string("state_") + rtk_fusion_state_to_string(rtk_fusion_state);
    }
    else if (!active_tf.valid)
    {
        rtk_reason = "no_transform";
    }
    else if (Measures.rtk_pos.empty())
    {
        rtk_reason = "rtk_empty";
    }
    else
    {
        const Timepos &latest_rtk = Measures.rtk_pos.back();
        rtk_time = latest_rtk.timestamp;
        rtk_time_diff = std::abs(rtk_time - lidar_end_time);
        rtk_h_acc = latest_rtk.h_acc;
        rtk_v_acc = latest_rtk.v_acc;
        if (rtk_time_diff > rtk_lio_time_threshold)
        {
            rtk_reason = "time_diff_exceed";
        }
        else if (rtk_time < last_rtk_constraint_epoch ||
                 (rtk_time == last_rtk_constraint_epoch && lidar_end_time != last_rtk_constraint_lidar_epoch))
        {
            rtk_reason = "already_used_in_previous_ekf_frame";
        }
        else if (!std::isfinite(rtk_v_acc) || rtk_v_acc <= 0.0)
        {
            rtk_reason = "acc_invalid";
        }
        else
        {
            rtk_world = latest_rtk.pos;
            if (enable_antenna_lever_arm_compensation)
            {
                rtk_world -= antenna_offset_T;
            }
            rtk_pos_map = transform_world_to_map(rtk_world, active_tf);
            rtk_var_z = rtk_v_acc * rtk_v_acc;
            rtk_residual_map = s.pos - rtk_pos_map;
            rtk_reason = "ok";
            add_rtk_constraint = true;
        }
    }


    const int state_dim = state_ikfom::DOF;
    const int extra_dim = add_rtk_constraint ? 1 : 0;
    const int meas_dim = effct_feat_num + extra_dim;
    ekfom_data.h_x = MatrixXd::Zero(meas_dim, state_dim);
    ekfom_data.h.resize(meas_dim);
    ekfom_data.z = VectorXd::Zero(meas_dim);
    ekfom_data.R = MatrixXd::Zero(meas_dim, meas_dim);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);


        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);


        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }


        ekfom_data.h(i) = norm_p.intensity;
        double residual_abs = std::fabs(norm_p.intensity);
        double adaptive_var = laser_point_cov + (laser_adaptive_variance ? residual_abs * residual_abs : 0.0);
        ekfom_data.R(i, i) = std::max(laser_point_cov, adaptive_var);
    }

    if (add_rtk_constraint)
    {
        last_rtk_constraint_epoch = rtk_time;
        last_rtk_constraint_lidar_epoch = lidar_end_time;
        const int row = effct_feat_num;
        const auto observation = rtk_height_observation(s.pos, s.vel, active_tf.R,
            active_tf.t, active_tf.s, lidar_end_time - rtk_time, rtk_world.z());
        ekfom_data.h_x.block<1, 3>(row, 0) = observation.jacobian.head<3>();
        ekfom_data.h_x.block<1, 3>(row, 12) = observation.jacobian.tail<3>();
        ekfom_data.h(row) = observation.residual;
        ekfom_data.R(row, row) = rtk_var_z;
    }

    solve_time += omp_get_wtime() - solve_start_;
}


std::vector<TrajectoryPoint> deque_to_vector(const std::deque<TrajectoryPoint> &src)
{
    return std::vector<TrajectoryPoint>(src.begin(), src.end());
}


void h_share_gnss_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{


}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    std::string runtime_profile_path;
    nh.param<std::string>("runtime_profiling/output", runtime_profile_path, "");
    RUNTIME_ACTION(runtime_profile::Recorder::get().configure(runtime_profile_path));

    debug = true;
    path_en = true;
    world_path_en = true;
    world_path_peak_metrics_on_shutdown_enable = false;
    world_path_peak_metrics_start_after_sec = 30.0;
    world_path_peak_metrics_start_after_sec = std::max(0.0, world_path_peak_metrics_start_after_sec);
    scan_pub_en = true;
    dense_pub_en = true;
    scan_body_pub_en = true;
    nh.param<bool>("publish/output_blend_enable", world_output_blend_enable, true);
    nh.param<double>("publish/output_blend_max_jump", world_output_blend_max_jump, 0.8);
    nh.param<double>("publish/output_blend_duration", world_output_blend_duration, 1.0);
    nh.param<bool>("publish/output_z_guard_enable", world_output_z_guard_enable, false);
    nh.param<double>("publish/output_z_guard_max_speed", world_output_z_guard_max_speed, 0.5);
    nh.param<double>("publish/output_z_guard_max_step", world_output_z_guard_max_step, 0.08);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");

    nh.param<string>("common/gnss_topic", gnss_topic, "/ublox_driver/receiver_pvt");
    nh.param<string>("common/gnss_lla_topic", gnss_lla_topic, "/ublox_driver/receiver_lla");


    p_rtk->load_quality_config_from_ros_params();


    nh.param<bool>("rtk/match_at_gnss_epochs", rtk_match_at_gnss_epochs, false);
    nh.param<int>("rtk/alignment_min_pairs", rtk_alignment_min_pairs, 3);
    rtk_alignment_min_pairs = std::max(3, rtk_alignment_min_pairs);
    nh.param<int>("preprocess/s_window", p_rtk->rtk_window_length, 5);
    int param_window_max = std::max(p_rtk->rtk_window_length, 10);
    nh.param<int>("preprocess/s_window_max", param_window_max, param_window_max);
    int param_window_growth = alignment_window_growth_interval;
    nh.param<int>("preprocess/s_window_growth_interval", param_window_growth, param_window_growth);
    alignment_window_min = std::max(3, p_rtk->rtk_window_length);
    alignment_window_max = std::max(alignment_window_min, param_window_max);
    alignment_window_growth_interval = std::max(1, param_window_growth);
    int param_window_reset_fix = alignment_window_min;
    nh.param<int>("preprocess/s_window_reset_fix", param_window_reset_fix, param_window_reset_fix);
    alignment_window_reset_fix = std::max(alignment_window_min,
                                          std::min(param_window_reset_fix, alignment_window_max));
    alignment_stable_window = 5;
    alignment_stable_window = std::max(rtk_match_at_gnss_epochs ? rtk_alignment_min_pairs : alignment_window_min,
                                       std::min(alignment_stable_window, alignment_window_max));
    alignment_window_current = alignment_window_min;
    alignment_window_success_counter = 0;
    nh.param<double>("preprocess/velocity_threshold", velocity_threshold, 0.05);
    bool wahba_log_velocity = false;
    wahba_log_velocity = false;
    double wahba_smoothing_factor = 0.0;
    nh.param<double>("wahba/bspline_smoothing_factor", wahba_smoothing_factor, 0.0);
    BSpline::set_smoothing_factor(wahba_smoothing_factor);


    const bool wahba_python_ready = WahbaPythonBridge::instance().available();
    if (!wahba_python_ready)
    {
        (void)0;
    }
    bool wahba_save_alignment_logs = false;
    wahba_save_alignment_logs = true;
    configure_alignment_logging(wahba_save_alignment_logs, root_dir + "data");
    if (wahba_log_velocity)
    {
        std::string data_dir = root_dir + "data";
        BSpline::configure_debug_logging(true, data_dir);
    }
    else
    {
        BSpline::configure_debug_logging(false);
    }

    nh.param<double>("rtk/fix_timeout", rtk_fix_timeout, 2.0);
    nh.param<string>("rtk/time_source", rtk_time_source, "gpst");
    std::transform(rtk_time_source.begin(),
                   rtk_time_source.end(),
                   rtk_time_source.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    nh.param<int>("rtk/subscriber_queue_size", rtk_subscriber_queue_size, 100);
    nh.param<int>("rtk/input_buffer_max_size", rtk_input_buffer_max_size, 200);
    nh.param<int>("rtk/lla_time_buffer_max_size", rtk_lla_time_buffer_max_size, 200);
    nh.param<int>("rtk/pending_pvt_max_size", rtk_pending_pvt_max_size, 200);
    nh.param<double>("rtk/lla_match_latlon_tolerance", rtk_lla_match_latlon_tolerance, 1e-8);
    nh.param<double>("rtk/lla_match_alt_tolerance", rtk_lla_match_alt_tolerance, 0.05);
    nh.param<bool>("rtk/wait_for_time_compensated_rtk", rtk_wait_for_time_compensated_rtk, false);
    nh.param<double>("rtk/time_compensation_wait_tolerance", rtk_time_compensation_wait_tolerance, 0.05);
    nh.param<double>("rtk/time_compensation_wait_max_staleness", rtk_time_compensation_wait_max_staleness, 1.0);
    nh.param<int>("rtk/min_fix_for_smoothing", rtk_min_fix_for_smoothing, 5);
    nh.param<bool>("rtk/recovery_verify_enable", rtk_recovery_verify_enable, true);
    nh.param<double>("rtk/recover_stable_duration_sec", rtk_recover_stable_duration_sec, 3.0);
    nh.param<double>("rtk/recover_long_loss_threshold_sec", rtk_recover_long_loss_threshold_sec, 30.0);
    nh.param<double>("rtk/recover_long_stable_duration_sec", rtk_recover_long_stable_duration_sec, 8.0);
    nh.param<double>("rtk/recover_max_h_acc", rtk_recover_max_h_acc, 0.5);
    nh.param<double>("rtk/recover_max_v_acc", rtk_recover_max_v_acc, 1.5);
    nh.param<double>("rtk/recover_max_sample_dt", rtk_recover_max_sample_dt, 1.0);
    nh.param<double>("rtk/recover_max_speed_xy", rtk_recover_max_speed_xy, 5.0);
    nh.param<double>("rtk/recover_max_speed_z", rtk_recover_max_speed_z, 2.0);
    nh.param<double>("rtk/recover_max_jump_xy", rtk_recover_max_jump_xy, 2.0);
    nh.param<double>("rtk/recover_max_jump_z", rtk_recover_max_jump_z, 0.8);
    nh.param<bool>("rtk/fusion_enable", rtk_fusion_enable, true);
    nh.param<bool>("rtk/reacquire_enable", rtk_reacquire_enable, true);
    covariance_debug_enable = false;
    nh.param<bool>("rtk/ekf_constraint_enable", rtk_ekf_constraint_enable, true);
    nh.param<bool>("rtk/bootstrap_transform_on_first_fix", rtk_bootstrap_transform_on_first_fix, false);
    nh.param<double>("rtk/bootstrap_transform_max_rtk_age", rtk_bootstrap_transform_max_rtk_age, 1.0);
    nh.param<bool>("rtk/fusion_state_pub_enable", rtk_fusion_state_pub_enable, false);
    nh.param<bool>("rtk/enable_manual_ekf_rtk_offset", enable_manual_ekf_rtk_offset, false);
    std::vector<double> manual_ekf_rtk_offset_vec{0.0, 0.0, 0.0};
    nh.param<std::vector<double>>("rtk/manual_ekf_rtk_offset", manual_ekf_rtk_offset_vec, manual_ekf_rtk_offset_vec);
    if (manual_ekf_rtk_offset_vec.size() >= 3)
    {
        manual_ekf_rtk_offset = Eigen::Vector3d(manual_ekf_rtk_offset_vec[0],
                                                manual_ekf_rtk_offset_vec[1],
                                                manual_ekf_rtk_offset_vec[2]);
    }
    else
    {
        manual_ekf_rtk_offset.setZero();
    }
    nh.param<double>("rtk/match_min_ratio", rtk_match_min_ratio, 0.85);
    rtk_match_min_ratio = std::max(0.0, std::min(1.0, rtk_match_min_ratio));
    world_map_transform_alpha = 1;
    world_map_transform_max_jump = 0;
    alignment_rmse_jump_ratio = 3.0;
    alignment_rotation_jump_deg = 0.0;
    nh.param<bool>("rtk/alignment_continuity_guard_enable", alignment_continuity_guard_enable, true);
    nh.param<double>("rtk/alignment_translation_jump_max", alignment_translation_jump_max, 20.0);
    nh.param<double>("rtk/alignment_pose_jump_max", alignment_pose_jump_max, 1.0);
    nh.param<double>("rtk/alignment_max_fit_rmse", alignment_max_fit_rmse, 0.0);
    nh.param<double>("rtk/lio_time_threshold", rtk_lio_time_threshold, 0.1);
    alignment_stable_rmse_threshold = 0.2;
    alignment_stable_min_duration = 2.5;
    alignment_stable_rotation_deg_threshold = 12.0;
    alignment_rotation_stable_deg_threshold = 1.5;
    alignment_rotation_stable_min_duration = 2.0;
    alignment_yaw_only_enable = false;
    configure_alignment_yaw_only(alignment_yaw_only_enable);
    double neighbor_base_m = 1.0, neighbor_speed_mps = 0.0;
    nh.param<double>("wahba/neighbor_jump_base_m", neighbor_base_m, 1.0);
    nh.param<double>("wahba/neighbor_max_speed_mps", neighbor_speed_mps, 0.0);
    configure_alignment_motion_gate(neighbor_base_m, neighbor_speed_mps);
    alignment_freeze_rotation_at_max_window = true;
    nh.param<bool>("rtk/freeze_rotation_when_stable", alignment_freeze_rotation_when_stable, false);
    alignment_freeze_rotation_yaw_max_delta_deg = 5;
    keep_rotation_frozen_on_loss = true;
    rtk_freeze_adaptive_enable = true;
    rtk_freeze_adaptive_max_dist = 0.1;
    rtk_freeze_adaptive_base_weight = 0.3;
    nh.param<bool>("rtk/freeze_adaptive_log_enable", rtk_freeze_adaptive_log_enable, false);
    nh.param<bool>("rtk/world_pose_log_enable", world_pose_log_enable, false);
    nh.param<bool>("rtk/diagnostic_log_enable", rtk_diagnostic_log_enable, true);
    nh.param<string>("rtk/diagnostic_log_path", rtk_diagnostic_log_path, rtk_diagnostic_log_path);
    nh.param<bool>("rtk/path_log_enable", rtk_path_log_enable, true);
    nh.param<string>("rtk/path_log_path", rtk_path_log_path, rtk_path_log_path);
    nh.param<int>("rtk/path_log_flush_every_n", rtk_path_log_flush_every_n, 50);
    terminal_status_enable = false;
    terminal_status_interval = 0.1;
    alignment_use_gravity_prior = false;
    alignment_gravity_weight = 1000;
    nh.param<bool>("rtk/tapm_enable", tapm_enable, false);
    tapm_sliding_enable = true;
    nh.param<bool>("rtk/tapm_writes_transform", tapm_writes_transform, false);
    tapm_publish_use_z = true;
    nh.param<bool>("rtk/tapm_recovery_clock_weights", tapm_recovery_clock_weights, false);
    nh.param<bool>("rtk/tapm_causal_publish_regularization", tapm_causal_publish_regularization, false);
    nh.param<double>("rtk/tapm_publish_regularization_decay_sec", tapm_publish_regularization_decay_sec, 0.0);
    nh.param<double>("rtk/tapm_rotation_handover_sec", tapm_rotation_handover_sec, 0.0);
    nh.param<double>("rtk/tapm_alpha", tapm_alpha, 0.5);
    nh.param<double>("rtk/tapm_beta", tapm_beta, 1.0);
    nh.param<double>("rtk/tapm_lambda_v", tapm_lambda_v, 10.0);
    nh.param<double>("rtk/tapm_lambda_a", tapm_lambda_a, 100.0);
    nh.param<int>("rtk/tapm_grace_frames", tapm_grace_frames, 15);
    nh.param<double>("rtk/tapm_gamma", tapm_gamma, 0.01);
    nh.param<double>("rtk/tapm_nominal_dt", tapm_nominal_dt, 0.1);
    nh.param<double>("rtk/tapm_sigma0_xy", tapm_sigma0_xy, 0.1);
    nh.param<double>("rtk/tapm_sigma0_z", tapm_sigma0_z, 0.2);
    nh.param<double>("rtk/tapm_v_max", tapm_v_max, 20.0);
    nh.param<double>("rtk/tapm_v_max_z", tapm_v_max_z, 5.0);
    nh.param<double>("rtk/tapm_a_max", tapm_a_max, 8.0);
    nh.param<double>("rtk/tapm_a_max_z", tapm_a_max_z, 3.0);
    nh.param<double>("rtk/tapm_rtk_sigma_min", tapm_rtk_sigma_min, 1e-3);
    nh.param<int>("rtk/tapm_max_iterations", tapm_max_iterations, 30);
    nh.param<int>("rtk/tapm_optimization_window", tapm_optimization_window, 15);
    nh.param<int>("rtk/tapm_world_history_size", tapm_world_history_size, 20);
    nh.param<int>("rtk/tapm_exit_frame_limit", tapm_exit_frame_limit, 0);
    nh.param<double>("rtk/tapm_exit_max_rtk_error", tapm_exit_max_rtk_error, 0.05);
    nh.param<double>("rtk/tapm_exit_min_duration", tapm_exit_min_duration, 0.0);
    nh.param<double>("rtk/tapm_exit_max_duration", tapm_exit_max_duration, 0.0);
    nh.param<int>("rtk/tapm_exit_stable_observations", tapm_exit_stable_observations, 1);
    nh.param<bool>("rtk/tapm_commit_exit_translation", tapm_commit_exit_translation, false);
    nh.param<bool>("rtk/tapm_use_analytic_jacobian", tapm_use_analytic_jacobian, true);
    nh.param<bool>("rtk/tapm_runtime_debug_enable", tapm_runtime_debug_enable, false);
    nh.param<double>("rtk/tapm_runtime_debug_interval", tapm_runtime_debug_interval, 0.2);
    nh.param<bool>("pgo_sw/enable", pgo_sw_enable, false);
    pgo_sw_enable_cloud = true;
    nh.param<bool>("pgo_sw/use_as_system_output", pgo_sw_use_as_system_output, false);
    nh.param<bool>("pgo_sw/use_tapm_virtual_rtk", pgo_sw_use_tapm_virtual_rtk, true);
    nh.param<bool>("pgo_sw/keep_window_through_recovery", pgo_sw_keep_window_through_recovery, true);
    nh.param<int>("pgo_sw/window_size", pgo_sw_window_size, 15);
    nh.param<int>("pgo_sw/optimize_stride", pgo_sw_optimize_stride, 3);
    nh.param<double>("pgo_sw/debug_interval", pgo_sw_debug_interval, 0.5);
    nh.param<double>("pgo_sw/rtk_match_max_dt", pgo_sw_rtk_match_max_dt, rtk_lio_time_threshold);
    nh.param<double>("pgo_sw/odom_sigma_pos", pgo_sw_odom_sigma_pos, 0.1);
    nh.param<double>("pgo_sw/odom_sigma_rot", pgo_sw_odom_sigma_rot, 0.01);
    nh.param<double>("pgo_sw/rtk_sigma_xy", pgo_sw_rtk_sigma_xy, 0.05);
    nh.param<double>("pgo_sw/rtk_sigma_z", pgo_sw_rtk_sigma_z, 0.10);
    nh.param<double>("pgo_sw/tapm_sigma_xy", pgo_sw_tapm_sigma_xy, pgo_sw_rtk_sigma_xy);
    nh.param<double>("pgo_sw/tapm_sigma_z", pgo_sw_tapm_sigma_z, pgo_sw_rtk_sigma_z);
    nh.param<bool>("pgo_sw/use_rtk_z", pgo_sw_use_rtk_z, false);
    nh.param<bool>("pgo_sw/use_tapm_z", pgo_sw_use_tapm_z, false);
    nh.param<double>("pgo_sw/recovery_ramp_duration_sec", pgo_sw_recovery_ramp_duration_sec, 5.0);
    nh.param<double>("pgo_sw/recovery_long_ramp_duration_sec", pgo_sw_recovery_long_ramp_duration_sec, 15.0);
    nh.param<double>("pgo_sw/recovery_ramp_min_weight", pgo_sw_recovery_ramp_min_weight, 0.15);
    nh.param<double>("pgo_sw/recovery_z_sigma_scale", pgo_sw_recovery_z_sigma_scale, 5.0);
    nh.param<float>("rtk/max_point_range", point_max_range, 100.0f);
    nh.param<int>("rtk/min_effective_points", min_effective_points, 5);
    rtk_min_fix_for_smoothing = std::max(1, rtk_min_fix_for_smoothing);
    rtk_recover_stable_duration_sec = std::max(0.0, rtk_recover_stable_duration_sec);
    rtk_recover_long_loss_threshold_sec = std::max(0.0, rtk_recover_long_loss_threshold_sec);
    rtk_recover_long_stable_duration_sec = std::max(rtk_recover_stable_duration_sec,
                                                    rtk_recover_long_stable_duration_sec);
    rtk_recover_max_h_acc = std::max(0.0, rtk_recover_max_h_acc);
    rtk_recover_max_v_acc = std::max(0.0, rtk_recover_max_v_acc);
    rtk_recover_max_sample_dt = std::max(0.0, rtk_recover_max_sample_dt);
    rtk_recover_max_speed_xy = std::max(0.0, rtk_recover_max_speed_xy);
    rtk_recover_max_speed_z = std::max(0.0, rtk_recover_max_speed_z);
    rtk_recover_max_jump_xy = std::max(0.0, rtk_recover_max_jump_xy);
    rtk_recover_max_jump_z = std::max(0.0, rtk_recover_max_jump_z);
    rtk_subscriber_queue_size = std::max(1, rtk_subscriber_queue_size);
    rtk_input_buffer_max_size = std::max(1, rtk_input_buffer_max_size);
    rtk_lla_time_buffer_max_size = std::max(1, rtk_lla_time_buffer_max_size);
    rtk_pending_pvt_max_size = std::max(1, rtk_pending_pvt_max_size);
    rtk_lla_match_latlon_tolerance = std::max(0.0, rtk_lla_match_latlon_tolerance);
    rtk_lla_match_alt_tolerance = std::max(0.0, rtk_lla_match_alt_tolerance);
    rtk_time_compensation_wait_tolerance = std::max(0.0, rtk_time_compensation_wait_tolerance);
    rtk_time_compensation_wait_max_staleness =
        std::max(rtk_time_compensation_wait_tolerance, rtk_time_compensation_wait_max_staleness);
    if (!rtk_use_lla_header_time() && rtk_time_source != "gpst")
    {
        (void)0;
        rtk_time_source = "gpst";
    }
    min_effective_points = std::max(1, min_effective_points);
    world_output_blend_max_jump = std::max(0.0, world_output_blend_max_jump);
    world_output_blend_duration = std::max(0.0, world_output_blend_duration);
    world_output_z_guard_max_speed = std::max(0.0, world_output_z_guard_max_speed);
    world_output_z_guard_max_step = std::max(0.0, world_output_z_guard_max_step);
    pgo_sw_window_size = std::max(2, pgo_sw_window_size);
    pgo_sw_optimize_stride = std::max(1, pgo_sw_optimize_stride);
    pgo_sw_debug_interval = std::max(0.0, pgo_sw_debug_interval);
    pgo_sw_rtk_match_max_dt = std::max(0.0, pgo_sw_rtk_match_max_dt);
    pgo_sw_tapm_sigma_xy = (pgo_sw_tapm_sigma_xy > 0.0) ? pgo_sw_tapm_sigma_xy : pgo_sw_rtk_sigma_xy;
    pgo_sw_tapm_sigma_z = (pgo_sw_tapm_sigma_z > 0.0) ? pgo_sw_tapm_sigma_z : pgo_sw_rtk_sigma_z;
    pgo_sw_recovery_ramp_duration_sec = std::max(0.0, pgo_sw_recovery_ramp_duration_sec);
    pgo_sw_recovery_long_ramp_duration_sec =
        std::max(pgo_sw_recovery_ramp_duration_sec, pgo_sw_recovery_long_ramp_duration_sec);
    pgo_sw_recovery_ramp_min_weight =
        std::max(1e-3, std::min(1.0, pgo_sw_recovery_ramp_min_weight));
    pgo_sw_recovery_z_sigma_scale = std::max(1.0, pgo_sw_recovery_z_sigma_scale);
    world_sw_backend.setWindowSize(static_cast<size_t>(pgo_sw_window_size));
    world_sw_backend.setRtkMatchMaxDt(pgo_sw_rtk_match_max_dt);
    world_sw_backend.setOdomSigmas(pgo_sw_odom_sigma_pos, pgo_sw_odom_sigma_rot);
    world_sw_backend.setDefaultRtkSigmas(pgo_sw_rtk_sigma_xy, pgo_sw_rtk_sigma_z);
    alignment_rmse_jump_ratio = std::max(0.0, alignment_rmse_jump_ratio);
    alignment_rotation_jump_deg = std::max(0.0, alignment_rotation_jump_deg);
    alignment_translation_jump_max = std::max(0.0, alignment_translation_jump_max);
    alignment_pose_jump_max = std::max(0.0, alignment_pose_jump_max);
    alignment_stable_rmse_threshold = std::max(0.0, alignment_stable_rmse_threshold);
    alignment_stable_min_duration = std::max(0.0, alignment_stable_min_duration);
    alignment_freeze_rotation_yaw_max_delta_deg = std::max(0.0, alignment_freeze_rotation_yaw_max_delta_deg);
    rtk_freeze_adaptive_max_dist = std::max(0.0, rtk_freeze_adaptive_max_dist);
    rtk_freeze_adaptive_base_weight = std::max(0.0, std::min(1.0, rtk_freeze_adaptive_base_weight));
    if (!alignment_freeze_rotation_at_max_window &&
        !alignment_freeze_rotation_when_stable)
    {
        alignment_rotation_frozen = false;
    }
    if (covariance_debug_enable)
    {
    }
    alignment_stable_rotation_deg_threshold = std::max(0.0, alignment_stable_rotation_deg_threshold);
    alignment_rotation_stable_deg_threshold = std::max(0.0, alignment_rotation_stable_deg_threshold);
    alignment_rotation_stable_min_duration = std::max(0.0, alignment_rotation_stable_min_duration);
    alignment_gravity_weight = std::max(0.0, alignment_gravity_weight);
    tapm_alpha = std::max(0.0, tapm_alpha);
    tapm_beta = std::max(0.0, tapm_beta);
    tapm_lambda_v = std::max(0.0, tapm_lambda_v);
    tapm_lambda_a = std::max(0.0, tapm_lambda_a);
    tapm_grace_frames = std::max(0, tapm_grace_frames);
    tapm_gamma = std::max(0.0, tapm_gamma);
    tapm_nominal_dt = std::max(1e-3, tapm_nominal_dt);
    tapm_sigma0_xy = std::max(1e-4, tapm_sigma0_xy);
    tapm_sigma0_z = std::max(1e-4, tapm_sigma0_z);
    tapm_v_max = std::max(1e-3, tapm_v_max);
    tapm_v_max_z = std::max(1e-3, tapm_v_max_z);
    tapm_a_max = std::max(1e-3, tapm_a_max);
    tapm_a_max_z = std::max(1e-3, tapm_a_max_z);
    tapm_rtk_sigma_min = std::max(1e-6, tapm_rtk_sigma_min);
    tapm_max_iterations = std::max(1, tapm_max_iterations);
    tapm_optimization_window = std::max(3, tapm_optimization_window);
    tapm_world_history_size = std::max(tapm_optimization_window, tapm_world_history_size);
    tapm_exit_frame_limit = std::max(0, tapm_exit_frame_limit);
    tapm_exit_max_rtk_error = std::max(0.0, tapm_exit_max_rtk_error);
    tapm_exit_min_duration = std::max(0.0, tapm_exit_min_duration);
    tapm_exit_max_duration = std::max(0.0, tapm_exit_max_duration);
    tapm_exit_stable_observations = std::max(1, tapm_exit_stable_observations);
    terminal_status_interval = std::max(0.1, terminal_status_interval);
    rtk_path_log_flush_every_n = std::max(0, rtk_path_log_flush_every_n);
    tapm_runtime_debug_interval = std::max(0.0, tapm_runtime_debug_interval);
    if (tapm_enable && tapm_sliding_enable &&
        tapm_exit_frame_limit > 0 &&
        tapm_grace_frames > 0 &&
        tapm_exit_frame_limit < tapm_grace_frames)
    {
        (void)0;
    }


    point_max_range = std::max(1.0f, point_max_range);
    initialize_rtk_diagnostic_log();
    std::string update_timing_path;
    nh.param<std::string>("diagnostics/update_log", update_timing_path, "");
    if (!update_timing_path.empty())
    {
        update_timing_log.open(update_timing_path);
        update_timing_log << "timestamp,t_vta_ms,t_tapm_ms,t_total_ms,rtk_constraint_epoch,transform_valid,tapm_active,tapm_output_active,history_samples,vta_epoch,sigma1,sigma2,sigma_ratio,vta_history_sec,vta_displacement_m,vta_samples,gravity_used,t_sliding_window_ms,alignment_confidence,tf_x,tf_y,tf_z,tf_qx,tf_qy,tf_qz,tf_qw\n";
    }
    initialize_rtk_path_log();
    {
        std::ostringstream config_diag;
        config_diag << std::fixed << std::setprecision(6)
                    << "lid_topic=" << lid_topic
                    << " imu_topic=" << imu_topic
                    << " gnss_topic=" << gnss_topic
                    << " gnss_lla_topic=" << gnss_lla_topic
                    << " rtk_time_source=" << rtk_time_source
                    << " rtk_subscriber_queue_size=" << rtk_subscriber_queue_size
                    << " rtk_input_buffer_max_size=" << rtk_input_buffer_max_size
                    << " rtk_lio_time_threshold=" << rtk_lio_time_threshold
                    << " rtk_fusion_enable=" << (rtk_fusion_enable ? 1 : 0)
                    << " rtk_wait_for_time_compensated="
                    << (rtk_wait_for_time_compensated_rtk ? 1 : 0)
                    << " rtk_time_comp_wait_tolerance="
                    << rtk_time_compensation_wait_tolerance
                    << " rtk_time_comp_wait_max_staleness="
                    << rtk_time_compensation_wait_max_staleness
                    << " output_blend_enable=" << (world_output_blend_enable ? 1 : 0)
                    << " output_blend_max_jump=" << world_output_blend_max_jump
                    << " output_blend_duration=" << world_output_blend_duration
                    << " output_z_guard_enable=" << (world_output_z_guard_enable ? 1 : 0)
                    << " output_z_guard_max_speed=" << world_output_z_guard_max_speed
                    << " output_z_guard_max_step=" << world_output_z_guard_max_step
                    << " rtk_path_log_enable=" << (rtk_path_log_enable ? 1 : 0)
                    << " rtk_path_log_path=" << rtk_path_log_path
                    << " pgo_use_rtk_z=" << (pgo_sw_use_rtk_z ? 1 : 0)
                    << " pgo_use_tapm_z=" << (pgo_sw_use_tapm_z ? 1 : 0)
                    << " pgo_rtk_sigma_z=" << pgo_sw_rtk_sigma_z
                    << " pgo_tapm_sigma_z=" << pgo_sw_tapm_sigma_z
                    << " pgo_recovery_long_ramp_duration="
                    << pgo_sw_recovery_long_ramp_duration_sec
                    << " pgo_recovery_z_sigma_scale="
                    << pgo_sw_recovery_z_sigma_scale;
        write_rtk_diagnostic_log("CONFIG_SUMMARY", config_diag.str());
    }
    load_initial_world_transform_from_params(nh);

    time_sync_en = false;
    nh.param<double>("common/init_time", init_time, 0.1);
    nh.param<bool>("common/lidar_stamp_is_end", lidar_stamp_is_end, true);
    nh.param<double>("common/imu_sync_offset", imu_sync_offset, 0.1);
    nh.param<double>("mapping/laser_point_cov", laser_point_cov, 0.01);
    nh.param<bool>("mapping/laser_adaptive_variance", laser_adaptive_variance, true);
    nh.param<float>("mapping/mov_threshold", mov_threshold, 1.5f);
    nh.param<float>("mapping/plane_fit_threshold", plane_fit_threshold, 0.1f);
    laser_point_cov = std::max(1e-6, laser_point_cov);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<double>("preprocess/max_scan_duration", p_pre->max_scan_duration, 1.0);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    extrinsic_est_en = true;
    pcd_save_en = false;
    pcd_save_interval = -1;
    export_lidar360_enable = false;
    export_ascii_cloud_enable = false;
    export_trajectory_enable = false;
    export_pose_json_enable = false;
    export_use_smooth_pose = false;
    export_use_lidar_pose = false;
    export_hba_enable = false;
    final_world_body_tf_enable = true;
    nh.param<std::string>("export/output_dir", export_output_dir, "");
    export_pose_json_path = "/tmp/geobridge";
    nh.param<bool>("export/world_tum_enable", export_world_tum_enable, false);
    export_world_tum_raw_enable = true;
    export_world_tum_smooth_enable = true;
    export_world_tum_delay_enable = true;
    export_world_tum_file_prefix_raw = "gt_traj";
    export_world_tum_file_prefix_smooth = "world_smooth_traj";
    export_world_tum_flush_every_n = 5;
    export_world_tum_require_monotonic_stamp = false;
    export_world_tum_use_wall_time_filename = false;
    export_pcd_name_fill_num = 0;
    export_offset_x = 0.0;
    export_offset_y = 0.0;
    export_offset_z = 0.0;
    export_start_after_sec = 0.0;
    std::string export_time_base_param;
    export_time_base_param = "ros";
    bool export_time_base_valid = false;
    export_time_base = parse_export_time_base(export_time_base_param, &export_time_base_valid);
    if (!export_time_base_valid)
    {
        (void)0;
    }
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    if (extrinT.size() != 3 || extrinR.size() != 9 ||
        !std::all_of(extrinT.begin(), extrinT.end(), [](double x) { return std::isfinite(x); }) ||
        !std::all_of(extrinR.begin(), extrinR.end(), [](double x) { return std::isfinite(x); }))
    {
        (void)0;
        return 1;
    }
    (void)0;

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "map";
    rtk_path.header.stamp = ros::Time::now();
    rtk_path.header.frame_id = "world";
    any_path.header.stamp = ros::Time::now();
    any_path.header.frame_id = "map";
    tapm_path.header.stamp = ros::Time::now();
    tapm_path.header.frame_id = "world";


    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));


    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);


    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_rtk = nh.subscribe(gnss_topic, rtk_subscriber_queue_size, rtk_cbk);
    ros::Subscriber sub_rtk_lla;
    if (rtk_use_lla_header_time())
    {
        sub_rtk_lla = nh.subscribe(gnss_lla_topic, rtk_subscriber_queue_size, rtk_lla_cbk);
    }
    if (rtk_diagnostic_log_enable)
    {
        std::ostringstream sub_diag;
        sub_diag << "gnss_topic=" << gnss_topic
                 << " gnss_lla_topic=" << gnss_lla_topic
                 << " time_source=" << rtk_time_source
                 << " pvt_queue=" << rtk_subscriber_queue_size
                 << " lla_subscribed=" << (rtk_use_lla_header_time() ? 1 : 0);
        write_rtk_diagnostic_log("RTK_SUBSCRIBE", sub_diag.str());
    }
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudWorld = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_world", 100000);
    ros::Publisher pubLaserCloudWorldPgo = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_world_pgo", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);

    ros::Publisher pubOdomWorld = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_world", 100000);
    ros::Publisher pubOdomWorldSmooth = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_world_smooth", 100000);
    ros::Publisher pubWorldNavSatFix = nh.advertise<sensor_msgs::NavSatFix>("/world_navsatfix", 10000);
    ros::Publisher pubRtkOdomWorld = nh.advertise<nav_msgs::Odometry>("/rtk_odometry", 100000);
    ros::Publisher pubRtkFusionState = nh.advertise<std_msgs::UInt8>("/rtk_fusion_state", 10);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    ros::Publisher pubPathrtk = nh.advertise<nav_msgs::Path>("/path_rtk", 10000);
    ros::Publisher pubPath_any = nh.advertise<nav_msgs::Path>("/pubPath_any", 10000);
    ros::Publisher pubPath_world = nh.advertise<nav_msgs::Path>("/path_world_ekf", 10000);
    ros::Publisher pubPath_tapm = nh.advertise<nav_msgs::Path>("/path_tapm", 10000);
    ros::Publisher pubPathWorldSmooth = nh.advertise<nav_msgs::Path>("/world_path", 10000);

    signal(SIGINT, SigHandle);

    ros::WallRate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (sync_packages(Measures))
        {
            RUNTIME_ACTION(runtime_profile::Recorder::get().begin(
                Measures.lidar_beg_time, static_cast<int>(rtk_fusion_state), lidar_buffer.size()));
            RUNTIME_TIMER(frame_timer, "frame_service");
            RUNTIME_TIMER(frontend_timer, "integrated_local_frontend");


            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                p_rtk->first_lidar_time = first_lidar_time;

                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time, rtk_process_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();
            double diagnostic_vta_ms = 0.0, diagnostic_tapm_ms = 0.0;
            diagnostic_sliding_ms = 0.0;
            Eigen::Matrix3d cov_debug_P_before_lidar;
            Eigen::Matrix3d cov_debug_P_after_imu;
            double cov_debug_trace_before_lidar = 0.0;
            double cov_debug_trace_after_imu = 0.0;
            bool cov_debug_has_before_lidar = false;


            p_imu->Process(Measures, kf, feats_undistort);
            if (covariance_debug_enable)
            {
                cov_debug_P_after_imu = kf.get_P().block<3, 3>(0, 0);
                cov_debug_trace_after_imu = cov_debug_P_after_imu.trace();
                log_covariance_sample(Measures.lidar_beg_time,
                                      "AfterIMUPred",
                                      cov_debug_P_after_imu);
            }
            const double current_latest_rtk_timestamp =
                Measures.rtk_pos.empty() ? -1.0 : Measures.rtk_pos.back().timestamp;
            const bool has_new_fix_measurement =
                current_latest_rtk_timestamp > latest_rtk_timestamp + 1e-6;
            if (current_latest_rtk_timestamp > 0.0)
            {
                latest_rtk_timestamp = current_latest_rtk_timestamp;
            }
            process_rtk_fusion_state_machine(lidar_end_time, has_new_fix_measurement);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                continue;
            }


            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            if (covariance_debug_enable)
            {
                cov_debug_P_before_lidar = kf.get_P().block<3, 3>(0, 0);
                cov_debug_trace_before_lidar = cov_debug_P_before_lidar.trace();
                cov_debug_has_before_lidar = true;
                std::ostringstream cov_detail;
                cov_detail << std::setprecision(6)
                           << "after_imu_trace=" << cov_debug_trace_after_imu;
                log_covariance_sample(Measures.lidar_beg_time,
                                      "BeforeLiDAR",
                                      cov_debug_P_before_lidar,
                                      cov_detail.str());
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < init_time ? false : true;

            lasermap_fov_segment();


            float max_range_sq = point_max_range * point_max_range;

            pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>);
            filtered->reserve(feats_undistort->size());

            for (const auto &pt : feats_undistort->points)
            {
                if (!pcl::isFinite(pt))
                    continue;

                float d2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
                if (d2 < max_range_sq)
                {
                    filtered->push_back(pt);
                }
            }
            *feats_undistort = *filtered;


            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);

                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();


            if (feats_down_size < min_effective_points)
            {
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);


            if (0)
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true;

            t2 = omp_get_wtime();


            double solve_H_time = 0;

            kf.update_iterated_dyn_share_modified(laser_point_cov, solve_H_time);
            state_point = kf.get_x();
            if (covariance_debug_enable && cov_debug_has_before_lidar)
            {
                const Eigen::Matrix3d P_pos_after_lidar = kf.get_P().block<3, 3>(0, 0);
                const double trace_after = P_pos_after_lidar.trace();
                const double compression_ratio = (trace_after > 1e-12)
                                                     ? (cov_debug_trace_before_lidar / trace_after)
                                                     : 0.0;
                std::ostringstream cov_detail;
                cov_detail << std::setprecision(6) << "compression_ratio=" << compression_ratio
                           << " trace_before=" << cov_debug_trace_before_lidar;
                log_covariance_sample(Measures.lidar_beg_time,
                                      "AfterLiDAR",
                                      P_pos_after_lidar,
                                      cov_detail.str());
            }
            Eigen::Vector3d lidar_pos_map = state_point.pos;

            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            RUNTIME_STOP(frontend_timer);
            RUNTIME_TIMER(transition_timer, "transition_layer_total");


            if (rtk_fix_active)
            {


                Timepos current_ekf_pos;
                current_ekf_pos.timestamp = lidar_end_time;
                current_ekf_pos.pos = state_point.pos;
                const int desired_window_size = std::max(3, alignment_window_current);
                if (static_cast<int>(ekftimepos_buffer.size()) < desired_window_size)
                {
                    ekftimepos_buffer.push_back(current_ekf_pos);
                    ekftimepos_state_flag = (static_cast<int>(ekftimepos_buffer.size()) == desired_window_size);
                }
                else
                {
                    ekftimepos_state_flag = true;
                    ekftimepos_buffer.push_back(current_ekf_pos);
                    while (static_cast<int>(ekftimepos_buffer.size()) > desired_window_size)
                    {
                        ekftimepos_buffer.pop_front();
                    }
                }

                if (!ekftimepos_state_flag)
                {
                    {
                        std::ostringstream oss;
                        oss << "waiting state=" << rtk_fusion_state_to_string(rtk_fusion_state)
                            << " current=" << ekftimepos_buffer.size()
                            << " desired=" << desired_window_size;
                        log_measurement_event("[ALIGN]", oss.str());
                    }
                    rtktimepos_buffer.clear();
                    matched_ekf_window.clear();
                    rtktimepos_state_flag = false;
                }
                else
                {

                    std::deque<Timepos> matched_ekf_candidates;
                    double match_ratio = 0.0;
                    std::string match_failure_reason;
                    if (matchRtkWindowWithEkf(ekftimepos_buffer, Measures.rtk_pos, rtk_lio_time_threshold, matched_ekf_candidates, matched_rtk_window,
                                              match_ratio,
                                              match_failure_reason))
                    {
                        matched_ekf_window = matched_ekf_candidates;
                        rtktimepos_buffer = matched_rtk_window;
                        rtktimepos_state_flag = true;
                        matched_rtk_window.clear();
                        count_rtk_match_success = static_cast<int>(matched_ekf_window.size());
                        alignment_latest_window_size = count_rtk_match_success;
                        last_rtk_match_ratio = match_ratio;
                        {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(3)
                                << "match_success ekf_total=" << ekftimepos_buffer.size()
                                << " matched=" << matched_ekf_window.size()
                                << " ratio=" << match_ratio
                                << " ratio_threshold=" << rtk_match_min_ratio
                                << " time_threshold=" << rtk_lio_time_threshold;
                            log_measurement_event("[ALIGN]", oss.str());
                        }
                    }
                    else
                    {
                        matched_ekf_window.clear();
                        matched_rtk_window.clear();
                        rtktimepos_buffer.clear();
                        rtktimepos_state_flag = false;
                        count_rtk_match_success = 0;
                        count_rtk_match_fail++;
                        last_rtk_match_ratio = 0.0;
                        alignment_latest_window_size = 0;
                        std::string reason = match_failure_reason.empty() ? "unknown_reason" : match_failure_reason;
                        {
                            std::ostringstream oss;
                            oss << std::fixed << std::setprecision(3)
                                << "match_failed ekf_total=" << ekftimepos_buffer.size()
                                << " matched=" << matched_ekf_candidates.size()
                                << " ratio=" << match_ratio
                                << " ratio_threshold=" << rtk_match_min_ratio
                                << " rtk_total=" << Measures.rtk_pos.size()
                                << " time_threshold=" << rtk_lio_time_threshold
                                << " reason=" << reason;
                            log_measurement_event("[ALIGN]", oss.str());
                        }
                    }
                }
            }
            else
            {
                clear_alignment_buffers("rtk_fix_inactive_loop");
            }

            if (rtk_fix_active && ekftimepos_state_flag && rtktimepos_state_flag && !matched_ekf_window.empty())
            {


                if (false)
                {


                }
                else
                {

                    {
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(3)
                            << "compute_alignment start ekf_total=" << ekftimepos_buffer.size()
                            << " ekf_matched=" << matched_ekf_window.size()
                            << " rtk=" << rtktimepos_buffer.size()
                            << " match_ratio=" << last_rtk_match_ratio
                            << " window=" << alignment_window_current
                            << " state=" << rtk_fusion_state_to_string(rtk_fusion_state);
                        log_measurement_event("[ALIGN]", oss.str());
                    }
                    Eigen::Vector3d gravity_map = state_point.grav;
                    Eigen::Vector3d gravity_world(0.0, 0.0, -1.0);
                    double gravity_prior_weight = alignment_use_gravity_prior ? alignment_gravity_weight : 0.0;
                    if (gravity_prior_weight > 0.0 && gravity_map.norm() < 1e-6)
                    {
                        gravity_prior_weight = 0.0;
                    }


                    if (enable_manual_ekf_rtk_offset)
                    {
                        for (auto &ekf_tp : matched_ekf_window)
                        {
                            ekf_tp.pos += manual_ekf_rtk_offset;
                        }
                        std::ostringstream oss;
                        oss << "manual_ekf_rtk_offset applied size=" << matched_ekf_window.size()
                            << " offset=[" << manual_ekf_rtk_offset.transpose() << "]";
                        log_measurement_event("[ALIGN]", oss.str());
                    }

                    const std::vector<TrajectoryPoint> matched_ekf_vec = deque_to_vector(matched_ekf_window);
                    const std::vector<TrajectoryPoint> matched_rtk_vec = deque_to_vector(rtktimepos_buffer);

                    const bool rotation_freeze_active = alignment_rotation_frozen;
                    if (rotation_freeze_active)
                    {
                        std::ostringstream oss;
                        oss << "rotation_freeze_active yaw_only limit_deg="
                            << alignment_freeze_rotation_yaw_max_delta_deg;
                        log_measurement_event("[ALIGN]", oss.str());
                    }

                    const double vta_started = omp_get_wtime();
                    RUNTIME_TIMER(vta_timer, "vta");
                    Transformation new_alignment = compute_full_alignment(
                        matched_ekf_vec,
                        matched_rtk_vec,
                        velocity_threshold,
                        gravity_map,
                        gravity_world,
                        gravity_prior_weight,
                        nullptr);
                    RUNTIME_STOP(vta_timer);
                    diagnostic_vta_ms = 1000.0 * (omp_get_wtime() - vta_started);
                    Transformation new_alignment_no_frozen = new_alignment;

                    if (new_alignment_no_frozen.valid)
                    {
                        last_alignment_no_frozen = new_alignment_no_frozen;
                        last_alignment_no_frozen_valid = true;
                    }
                    else
                    {
                        last_alignment_no_frozen_valid = false;
                    }

                    if (rotation_freeze_active && new_alignment.valid)
                    {
                        const Eigen::Vector3d frozen_zyx =
                            alignment_frozen_rotation.eulerAngles(2, 1, 0);
                        const Eigen::Vector3d new_zyx = new_alignment.R.eulerAngles(2, 1, 0);
                        const double yaw_delta_rad = wrap_angle_rad(new_zyx.x() - frozen_zyx.x());
                        const double yaw_delta_deg = std::abs(yaw_delta_rad) * 180.0 / M_PI;
                        const bool allow_yaw =
                            alignment_freeze_rotation_yaw_max_delta_deg > 0.0 &&
                            yaw_delta_deg <= alignment_freeze_rotation_yaw_max_delta_deg;
                        const double yaw_use = allow_yaw ? new_zyx.x() : frozen_zyx.x();

                        new_alignment.R = compose_zyx_rotation(yaw_use, frozen_zyx.y(), frozen_zyx.z());

                        const size_t sample_count =
                            std::min(matched_ekf_window.size(), rtktimepos_buffer.size());
                        if (sample_count > 0)
                        {
                            Eigen::Vector3d mean_source = Eigen::Vector3d::Zero();
                            Eigen::Vector3d mean_reference = Eigen::Vector3d::Zero();
                            for (size_t i = 0; i < sample_count; ++i)
                            {
                                mean_source += matched_ekf_window[i].pos;
                                mean_reference += rtktimepos_buffer[i].pos;
                            }
                            mean_source /= static_cast<double>(sample_count);
                            mean_reference /= static_cast<double>(sample_count);
                            new_alignment.t =
                                mean_reference - new_alignment.s * (new_alignment.R * mean_source);
                        }
                    }

                    const bool alignment_update_frozen = tapm_active && tapm_sliding_enable;
                    if (rtk_diagnostic_log_enable &&
                        new_alignment.valid &&
                        !matched_ekf_window.empty() &&
                        !rtktimepos_buffer.empty())
                    {
                        const Timepos &latest_ekf_match = matched_ekf_window.back();
                        const Timepos &latest_rtk_match = rtktimepos_buffer.back();
                        const Eigen::Vector3d used_latest =
                            new_alignment.s * (new_alignment.R * latest_ekf_match.pos) + new_alignment.t;
                        const Eigen::Vector3d raw_latest =
                            new_alignment_no_frozen.s *
                                (new_alignment_no_frozen.R * latest_ekf_match.pos) +
                            new_alignment_no_frozen.t;
                        const double latest_err_used = (used_latest - latest_rtk_match.pos).norm();
                        const double latest_err_raw = new_alignment_no_frozen.valid
                                                          ? (raw_latest - latest_rtk_match.pos).norm()
                                                          : -1.0;
                        const double rmse_used =
                            compute_alignment_window_rmse(new_alignment, matched_ekf_window, rtktimepos_buffer);
                        const double rmse_raw = new_alignment_no_frozen.valid
                                                    ? compute_alignment_window_rmse(new_alignment_no_frozen,
                                                                                   matched_ekf_window,
                                                                                   rtktimepos_buffer)
                                                    : -1.0;
                        double rot_delta_deg = -1.0;
                        if (new_alignment_no_frozen.valid)
                        {
                            Eigen::Matrix3d R_delta =
                                new_alignment.R.transpose() * new_alignment_no_frozen.R;
                            Eigen::AngleAxisd aa_delta(R_delta);
                            rot_delta_deg = aa_delta.angle() * 180.0 / M_PI;
                        }
                        std::ostringstream diag;
                        diag << std::fixed << std::setprecision(3)
                             << "state=" << rtk_fusion_state_to_string(rtk_fusion_state)
                             << " update_frozen=" << (alignment_update_frozen ? 1 : 0)
                             << " rotation_frozen=" << (rotation_freeze_active ? 1 : 0)
                             << " window=" << alignment_window_current
                             << " matched=" << matched_ekf_window.size()
                             << " rmse_used=" << rmse_used
                             << " rmse_raw=" << rmse_raw
                             << " latest_err_used=" << latest_err_used
                             << " latest_err_raw=" << latest_err_raw
                             << " rot_delta_deg=" << rot_delta_deg
                             << " used_latest=[" << used_latest.transpose() << "]"
                             << " raw_latest=[" << raw_latest.transpose() << "]"
                             << " rtk_latest=[" << latest_rtk_match.pos.transpose() << "]"
                             << " used_t=[" << new_alignment.t.transpose() << "]"
                             << " raw_t=[" << new_alignment_no_frozen.t.transpose() << "]";
                        write_rtk_diagnostic_log("ALIGN_EVAL", diag.str());
                    }
                    if (!new_alignment.valid)
                    {
                        alignment_window_success_counter = 0;
                        log_measurement_event("[ALIGN]", "compute_alignment invalid_result");
                    }
                    else if (alignment_max_fit_rmse > 0.0 &&
                             compute_alignment_window_rmse(new_alignment, matched_ekf_window, rtktimepos_buffer) > alignment_max_fit_rmse)
                    {
                        write_rtk_diagnostic_log("ALIGN_REJECT", "reason=absolute_fit_rmse");
                    }
                    else if (rtk_fusion_state == RtkFusionState::SMOOTHING)
                    {
                        smoothing_alignment = new_alignment;
                        smoothing_alignment_valid = true;
                        {
                            std::ostringstream oss;
                            oss << "smoothing_alignment_update t=[" << new_alignment.t.transpose()
                                << "] window=" << alignment_window_current
                                << " official_update=0";
                            log_measurement_event("[ALIGN]", oss.str());
                        }
                    }
                    else if (rtk_fusion_state == RtkFusionState::CORRECTION)
                    {

                      do {
                        const double alignment_rmse =
                            compute_alignment_window_rmse(new_alignment, matched_ekf_window, rtktimepos_buffer);


                        alignment_latest_window_size = static_cast<int>(matched_ekf_window.size());
                        bool rmse_ok = true;
                        if (alignment_rmse >= 0.0 && last_alignment_rmse > 0.0 &&
                            alignment_rmse_jump_ratio > 0.0 &&
                            alignment_rmse > last_alignment_rmse * alignment_rmse_jump_ratio)
                        {
                            rmse_ok = false;
                        }

                        const double rotation_delta_before_update = compute_alignment_rotation_delta_deg(new_alignment);
                        const bool rotation_ok = evaluate_alignment_rotation_quality(new_alignment);
                        double alignment_translation_jump = 0.0;
                        double alignment_pose_jump = 0.0;
                        std::string continuity_reject_reason;
                        const bool continuity_ok =
                            evaluate_alignment_continuity_quality(new_alignment,
                                                                  alignment_translation_jump,
                                                                  alignment_pose_jump,
                                                                  continuity_reject_reason);
                        if (!rmse_ok || !rotation_ok || !continuity_ok)
                        {
                            std::ostringstream reject_info;
                            const char *reason =
                                !rmse_ok ? "rmse_jump"
                                         : (!rotation_ok ? "rotation_jump"
                                                         : continuity_reject_reason.c_str());
                            reject_info << "reject_update reason=" << reason
                                        << " rmse=" << alignment_rmse
                                        << " last=" << last_alignment_rmse
                                        << " ratio=" << alignment_rmse_jump_ratio
                                        << " rotation_delta="
                                        << compute_alignment_rotation_delta_deg(new_alignment)
                                        << " translation_jump=" << alignment_translation_jump
                                        << " pose_jump=" << alignment_pose_jump
                                        << " translation_max=" << alignment_translation_jump_max
                                        << " pose_max=" << alignment_pose_jump_max;
                            log_measurement_event("[ALIGN]", reject_info.str());
                            write_rtk_diagnostic_log("ALIGN_REJECT", reject_info.str());
                            break;
                        }


                        double confidence = 1.0;
                        double timestamp = lidar_end_time;
                        if (!alignment_update_frozen)
                        {
                          if (rtk_diagnostic_log_enable && world_map_transform.isTransformValid())
                          {
                              const Eigen::Vector3d prev_t = world_map_transform.getTransform().t;
                              const Eigen::Vector3d dt_t = new_alignment.t - prev_t;
                              std::ostringstream oss;
                              oss << std::fixed << std::setprecision(6)
                                  << "state=" << rtk_fusion_state_to_string(rtk_fusion_state)
                                  << " rotation_frozen=" << (rotation_freeze_active ? 1 : 0)
                                  << " window=" << alignment_window_current
                                  << " dt_norm=" << dt_t.norm()
                                  << " prev_t=[" << prev_t.transpose() << "]"
                                  << " new_t=[" << new_alignment.t.transpose() << "]"
                                  << " dt=[" << dt_t.transpose() << "]";
                              write_rtk_diagnostic_log("ALIGN_T_STEP", oss.str());
                          }
                          apply_alignment_transform(new_alignment, confidence, timestamp, false);
                          update_world_state();
                          smoothing_alignment_valid = false;
                          {
                              std::ostringstream oss;
                              oss << "correction_alignment_update t=[" << new_alignment.t.transpose()
                                  << "] window=" << alignment_window_current;
                              log_measurement_event("[ALIGN]", oss.str());
                          }
                        }

                        if (alignment_rmse >= 0.0)
                        {
                            last_alignment_rmse = alignment_rmse;
                            const bool window_gate_ready = alignment_window_gate_released ||
                                                           (alignment_latest_window_size >= alignment_stable_window);
                            if (!window_gate_ready && !alignment_window_gate_released)
                            {
                                alignment_stable = false;
                                alignment_stable_enter_time = -1.0;
                                std::ostringstream oss;
                                oss << "stable_window_not_ready size=" << alignment_latest_window_size
                                    << " required=" << alignment_stable_window;
                                log_measurement_event("[ALIGN]", oss.str());
                            }

                            const bool rmse_condition_met =
                                (alignment_stable_rmse_threshold <= 0.0) ||
                                (alignment_rmse <= alignment_stable_rmse_threshold);
                            const double rotation_delta_deg = rotation_delta_before_update;
                            const bool rotation_condition_met =
                                (alignment_stable_rotation_deg_threshold <= 0.0) ||
                                (rotation_delta_deg <= alignment_stable_rotation_deg_threshold);

                            if (window_gate_ready && rmse_condition_met && rotation_condition_met)
                            {
                                if (alignment_stable_enter_time < 0.0)
                                {
                                    alignment_stable_enter_time = timestamp;
                                    std::ostringstream oss;
                                    oss << "stable_alignment_timer_start rmse=" << alignment_rmse
                                        << " rot_delta=" << rotation_delta_deg;
                                    log_measurement_event("[ALIGN]", oss.str());
                                }
                                else if (!alignment_stable)
                                {
                                    const double dwell = timestamp - alignment_stable_enter_time;
                                    if (dwell >= alignment_stable_min_duration)
                                    {
                                        alignment_stable = true;
                                        std::ostringstream oss;
                                        oss << "stable_alignment_enter rmse=" << alignment_rmse
                                            << " rot_delta=" << rotation_delta_deg
                                            << " dwell=" << dwell;
                                        log_measurement_event("[ALIGN]", oss.str());
                                        if (!alignment_window_gate_released)
                                        {
                                            alignment_window_gate_released = true;
                                            log_measurement_event("[ALIGN]", "stable_window_released");
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (alignment_stable)
                                {
                                    std::ostringstream oss;
                                    oss << "stable_alignment_exit reason="
                                        << (window_gate_ready ? (rmse_condition_met ? "rotation_unstable" : "rmse_exceed")
                                                             : "window_not_ready")
                                        << " rmse=" << alignment_rmse
                                        << " rot_delta=" << rotation_delta_deg;
                                    log_measurement_event("[ALIGN]", oss.str());
                                }
                                alignment_stable = false;
                                alignment_stable_enter_time = -1.0;
                            }

                            if (alignment_rotation_stable_deg_threshold <= 0.0)
                            {
                                if (!alignment_rotation_stable)
                                {
                                    alignment_rotation_stable = true;
                                    alignment_rotation_stable_enter_time = timestamp;
                                    log_measurement_event("[ALIGN]", "rotation_stable_forced_true");
                                }
                            }
                            else if (rotation_delta_deg <= alignment_rotation_stable_deg_threshold)
                            {
                                if (alignment_rotation_stable_enter_time < 0.0)
                                {
                                    alignment_rotation_stable_enter_time = timestamp;
                                    std::ostringstream oss;
                                    oss << "rotation_stable_timer_start delta=" << rotation_delta_deg;
                                    log_measurement_event("[ALIGN]", oss.str());
                                }
                                else if (!alignment_rotation_stable)
                                {
                                    const double rot_dwell = timestamp - alignment_rotation_stable_enter_time;
                                    if (rot_dwell >= alignment_rotation_stable_min_duration)
                                    {
                                        alignment_rotation_stable = true;
                                        std::ostringstream oss;
                                        oss << "rotation_stable_enter delta=" << rotation_delta_deg
                                            << " dwell=" << rot_dwell;
                                        log_measurement_event("[ALIGN]", oss.str());
                                    }
                                }
                            }
                            else
                            {
                                if (alignment_rotation_stable)
                                {
                                    std::ostringstream oss;
                                    oss << "rotation_stable_exit delta=" << rotation_delta_deg;
                                    log_measurement_event("[ALIGN]", oss.str());
                                }
                                alignment_rotation_stable = false;
                                alignment_rotation_stable_enter_time = -1.0;
                            }
                        }


                        const bool freeze_by_window =
                            alignment_window_current >= alignment_window_max;
                        const bool freeze_by_rotation_stable =
                            alignment_freeze_rotation_when_stable &&
                            alignment_rotation_stable;
                        if (!alignment_rotation_frozen &&
                            world_map_transform.isTransformValid() &&
                            ((alignment_freeze_rotation_at_max_window && freeze_by_window) ||
                             freeze_by_rotation_stable))
                        {
                            alignment_frozen_rotation = world_map_transform.getTransform().R;
                            alignment_rotation_frozen = true;
                            log_measurement_event(
                                "[ALIGN]",
                                freeze_by_rotation_stable
                                    ? "rotation_frozen reason=rotation_stable"
                                    : "rotation_frozen reason=window=max");
                        }

                        if (alignment_window_current < alignment_window_max)
                        {
                            alignment_window_success_counter++;
                            if (alignment_window_success_counter >= alignment_window_growth_interval)
                            {
                                alignment_window_current++;
                                alignment_window_success_counter = 0;
                            }
                        }
                      } while (false);
                    }
                }
            }


            update_world_state();


            const bool rtk_window_ready = is_rtk_window_ready_for_unified_update();
            const bool alignment_update_frozen = tapm_active && tapm_sliding_enable;
            if (rtk_fusion_state == RtkFusionState::CORRECTION &&
                world_state.initialized && world_map_transform.isTransformValid() && rtk_window_ready)
            {
                if (!alignment_update_frozen)
                {
                  Transformation unified_transform = world_map_transform.getTransform();
                  Eigen::Vector3d pos_map_with_offset = state_point.pos;
                  if (enable_manual_ekf_rtk_offset)
                  {
                      pos_map_with_offset += manual_ekf_rtk_offset;
                  }
                  Eigen::Vector3d unified_t = world_state.pos - unified_transform.s * (unified_transform.R * pos_map_with_offset);
                  unified_transform.t = unified_t;
                  unified_transform.valid = true;
                  Global_GNSS_LIO = unified_transform;
                  world_map_transform.updateTransform(unified_transform, world_map_transform.getConfidence(), lidar_end_time, true);
                }
            }

            record_world_history_sample(lidar_end_time);

            const double tapm_started = omp_get_wtime();
            run_tapm_state_machine_step(flg_EKF_inited);
            diagnostic_tapm_ms = 1000.0 * (omp_get_wtime() - tapm_started);


            if (world_pose_log_enable)
            {
                if (!world_state.initialized)
                {
                }
                else
                {
                    const char *world_pose_source = tapm_output_active ? "alignment_tapm_publish" : "alignment";
                    std::ostringstream pose_oss;
                    pose_oss << std::fixed << std::setprecision(3)
                             << "[WORLD_POSE] source=" << world_pose_source
                             << " pos=[" << world_state.pos.transpose()
                             << "] quat_wxyz=[" << world_state.quat.w() << " "
                             << world_state.quat.x() << " "
                             << world_state.quat.y() << " "
                             << world_state.quat.z() << "]";

                    static bool tapm_debug_has_prev = false;
                    static bool tapm_debug_has_prev_prev = false;
                    static Eigen::Vector3d tapm_debug_prev_pos = Eigen::Vector3d::Zero();
                    static Eigen::Vector3d tapm_debug_prev_prev_pos = Eigen::Vector3d::Zero();
                    const Eigen::Vector3d tapm_debug_pos = tapm_output_active ? tapm_publish_pos : world_state.pos;
                    const double tapm_debug_dt =
                        tapm_nominal_dt > 1e-6 ? tapm_nominal_dt : 0.1;
                    const double nan_value = std::numeric_limits<double>::quiet_NaN();
                    Eigen::Vector3d tapm_debug_vel =
                        Eigen::Vector3d(nan_value, nan_value, nan_value);
                    Eigen::Vector3d tapm_debug_acc =
                        Eigen::Vector3d(nan_value, nan_value, nan_value);
                    bool tapm_debug_vel_valid = false;
                    bool tapm_debug_acc_valid = false;

                    if (tapm_debug_has_prev)
                    {
                        tapm_debug_vel =
                            (tapm_debug_pos - tapm_debug_prev_pos) / tapm_debug_dt;
                        tapm_debug_vel_valid = true;
                    }
                    if (tapm_debug_has_prev && tapm_debug_has_prev_prev)
                    {
                        tapm_debug_acc =
                            (tapm_debug_pos - 2.0 * tapm_debug_prev_pos + tapm_debug_prev_prev_pos) /
                            (tapm_debug_dt * tapm_debug_dt);
                        tapm_debug_acc_valid = true;
                    }

                    if (tapm_debug_vel_valid)
                    {
                        update_tapm_top5(tapm_top_speed,
                                         tapm_debug_vel.norm(),
                                         tapm_debug_vel,
                                         lidar_end_time);
                    }
                    if (tapm_debug_acc_valid)
                    {
                        update_tapm_top5(tapm_top_accel,
                                         tapm_debug_acc.norm(),
                                         tapm_debug_acc,
                                         lidar_end_time);
                    }

                    std::ostringstream kin_oss;
                    kin_oss << std::fixed << std::setprecision(3)
                            << "[TAPM_KIN] source=" << (tapm_output_active ? "tapm_publish" : "alignment")
                            << " dt=" << tapm_debug_dt
                            << " vel=[" << tapm_debug_vel.transpose() << "]"
                            << " accel=[" << tapm_debug_acc.transpose() << "]"
                            << " vel_valid=" << (tapm_debug_vel_valid ? 1 : 0)
                            << " accel_valid=" << (tapm_debug_acc_valid ? 1 : 0);

                    tapm_debug_prev_prev_pos = tapm_debug_prev_pos;
                    tapm_debug_prev_pos = tapm_debug_pos;
                    tapm_debug_has_prev_prev = tapm_debug_has_prev;
                    tapm_debug_has_prev = true;
                }

                if (world_map_transform.isTransformValid())
                {
                    const Transformation &tf = world_map_transform.getTransform();
                    std::ostringstream tf_oss;
                    tf_oss << std::fixed << std::setprecision(3)
                           << "[WORLD_TF] valid=1 s=" << tf.s
                           << " t=[" << tf.t.transpose() << "]"
                           << " R:\n"
                           << tf.R;
                }
                else
                {
                }
            }


            update_world_publish_state();
            publish_odometry(pubOdomAftMapped);
            publish_odometry_world_smooth(pubOdomWorldSmooth);
            RUNTIME_STOP(transition_timer);


            publish_odometry_world(pubOdomWorld);
            publish_rtk_odometry_world(pubRtkOdomWorld);
            log_rtk_path_samples();
            publish_rtk_fusion_state(pubRtkFusionState);
            publish_world_navsatfix(pubWorldNavSatFix);
            export_world_tum_trajectories();
            export_lidar360_ascii_cloud();
            export_lidar360_trajectory();
            if (export_hba_enable)
            {
                export_lidar360_hba_bundle();
            }
            else
            {
                export_lidar360_pose_json();
                export_lidar360_hba_pcd();
            }

            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            if (update_timing_log.is_open())
            {
                const auto& vta = alignment_vta_diagnostics();
                const auto& diagnostic_tf = world_map_transform.getTransform();
                const Eigen::Quaterniond diagnostic_quat(diagnostic_tf.R);
                update_timing_log << std::fixed << std::setprecision(9) << lidar_end_time << ","
                    << diagnostic_vta_ms << "," << diagnostic_tapm_ms << "," << 1000.0*(t5-t0) << ","
                    << last_rtk_constraint_epoch << "," << world_map_transform.isTransformValid() << ","
                    << tapm_active << "," << tapm_output_active << "," << matched_ekf_window.size() << ","
                    << vta.timestamp << "," << vta.sigma1 << "," << vta.sigma2 << "," << vta.ratio << ","
                    << vta.history_sec << "," << vta.displacement_m << "," << vta.samples << "," << vta.gravity_used << ","
                    << diagnostic_sliding_ms << "," << world_map_transform.getConfidence() << ","
                    << diagnostic_tf.t.x() << "," << diagnostic_tf.t.y() << "," << diagnostic_tf.t.z() << ","
                    << diagnostic_quat.x() << "," << diagnostic_quat.y() << "," << diagnostic_quat.z() << "," << diagnostic_quat.w() << "\n";
                update_timing_log.flush();
            }


            if (path_en)
            {
                const bool path_published = publish_path(pubPath);
                rtk_publish_path(pubPathrtk);
                any_publish_path(pubPath_any);
                world_publish_path(pubPath_world);
                tapm_publish_path(pubPath_tapm);
                world_smooth_publish_path(pubPathWorldSmooth);
            }
            publish_terminal_status_panel(lidar_end_time);
            if (scan_pub_en)
                publish_frame_world_aligned(pubLaserCloudWorld);
            if (scan_pub_en)
                publish_frame_world_pgo(pubLaserCloudWorldPgo);
            if (scan_pub_en || pcd_save_en)
                publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);
        }

        status = ros::ok();
        rate.sleep();
    }

    RUNTIME_ACTION(runtime_profile::Recorder::get().flush());
    (void)0;
    print_tapm_top5("speed", tapm_top_speed);
    print_tapm_top5("accel", tapm_top_accel);
    print_world_path_peak_metrics_on_shutdown();
    close_rtk_path_logger();
    close_world_tum_loggers();
    print_final_world_body_transform();


    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        (void)0;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    return 0;
}
