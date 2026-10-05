
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <utility>

#include <Eigen/Eigen>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <common_lib.h>
#include <gnss_comm/GnssPVTSolnMsg.h>
#include <gnss_comm/gnss_utility.hpp>
#include <so3_math.h>

#include "use-ikfom.hpp"

#define MAX_INI_COUNT (20)


class RTKQualityAnalyzer {
public:
    enum class QualityLevel {
        EXCELLENT = 5,
        GOOD = 4,
        ACCEPTABLE = 3,
        POOR = 2,
        BAD = 1,
        INVALID = 0
    };

    struct CarrierPhaseAnalysis {
        QualityLevel level;
        double score;
        double expected_accuracy_cm;
        std::string description;
    };

    struct SatelliteGeometryAnalysis {
        QualityLevel satellite_count_level;
        QualityLevel pdop_level;
        double satellite_score;
        double pdop_score;
        double combined_score;
        std::string geometry_assessment;
    };

    struct AccuracyReportAnalysis {
        bool is_reasonable;
        double credibility_score;
        double effective_h_acc;
        double effective_v_acc;
        std::string credibility_reason;
    };

    struct QualityMetrics {

        uint8_t carr_soln;
        uint8_t num_sv;
        double p_dop;
        double h_acc;
        double v_acc;
        bool valid_fix;
        bool diff_soln;


        QualityLevel carrier_quality;
        QualityLevel geometric_quality;
        QualityLevel satellite_quality;
        QualityLevel accuracy_quality;


        double overall_score;
        QualityLevel overall_level;
        std::string analysis_summary;


        CarrierPhaseAnalysis carrier_analysis;
        SatelliteGeometryAnalysis geometry_analysis;
        AccuracyReportAnalysis accuracy_analysis;
    };


    RTKQualityAnalyzer()
        : excellent_threshold_(90.0)
        , good_threshold_(75.0)
        , acceptable_threshold_(60.0)
        , poor_threshold_(40.0)
        , bad_threshold_(20.0)
        , analysis_count_(0)
        , avg_score_(0.0)
    {}

    ~RTKQualityAnalyzer() = default;


    QualityMetrics analyze_rtk_quality(const gnss_comm::GnssPVTSolnMsg& rtk_msg) {
        QualityMetrics metrics;


        metrics.carr_soln = rtk_msg.carr_soln;
        metrics.num_sv = rtk_msg.num_sv;
        metrics.p_dop = rtk_msg.p_dop;
        metrics.h_acc = rtk_msg.h_acc;
        metrics.v_acc = rtk_msg.v_acc;
        metrics.valid_fix = rtk_msg.valid_fix;
        metrics.diff_soln = rtk_msg.diff_soln;


        if (!metrics.valid_fix || !metrics.diff_soln) {
            metrics.overall_score = 0.0;
            metrics.overall_level = QualityLevel::INVALID;
            metrics.analysis_summary = "RTK解算无效或未应用差分修正";
            return metrics;
        }


        metrics.carrier_analysis = analyze_carrier_phase(metrics.carr_soln);
        metrics.geometry_analysis = analyze_satellite_geometry(metrics.num_sv, metrics.p_dop);
        metrics.accuracy_analysis = analyze_accuracy_report(metrics.h_acc, metrics.v_acc,
                                                           metrics.carr_soln, metrics.num_sv);


        metrics.overall_score = metrics.carrier_analysis.score * 0.5 +
                               metrics.geometry_analysis.combined_score * 0.35 +
                               metrics.accuracy_analysis.credibility_score * 100 * 0.15;

        metrics.overall_level = score_to_level(metrics.overall_score);


        metrics.carrier_quality = score_to_level(metrics.carrier_analysis.score);
        metrics.geometric_quality = score_to_level(metrics.geometry_analysis.combined_score);
        metrics.satellite_quality = score_to_level(metrics.geometry_analysis.satellite_score);
        metrics.accuracy_quality = score_to_level(metrics.accuracy_analysis.credibility_score * 100);


        metrics.analysis_summary = generate_analysis_summary(metrics.carrier_analysis,
                                                            metrics.geometry_analysis,
                                                            metrics.accuracy_analysis,
                                                            metrics.overall_score);


        analysis_count_++;
        avg_score_ = (avg_score_ * (analysis_count_ - 1) + metrics.overall_score) / analysis_count_;

        return metrics;
    }


    void set_quality_thresholds(double excellent_thresh = 90.0,
                               double good_thresh = 75.0,
                               double acceptable_thresh = 60.0,
                               double poor_thresh = 40.0,
                               double bad_thresh = 20.0) {
        excellent_threshold_ = excellent_thresh;
        good_threshold_ = good_thresh;
        acceptable_threshold_ = acceptable_thresh;
        poor_threshold_ = poor_thresh;
        bad_threshold_ = bad_thresh;
    }


    double get_average_score() const { return avg_score_; }
    size_t get_analysis_count() const { return analysis_count_; }


    static std::string quality_level_to_string(QualityLevel level) {
        switch (level) {
            case QualityLevel::EXCELLENT:  return "Excellent";
            case QualityLevel::GOOD:       return "Good";
            case QualityLevel::ACCEPTABLE: return "Acceptable";
            case QualityLevel::POOR:       return "Poor";
            case QualityLevel::BAD:        return "Bad";
            case QualityLevel::INVALID:    return "Invalid";
            default:                       return "Unknown";
        }
    }

    static QualityLevel score_to_level(double score) {
        if (score >= 90.0) return QualityLevel::EXCELLENT;
        if (score >= 75.0) return QualityLevel::GOOD;
        if (score >= 60.0) return QualityLevel::ACCEPTABLE;
        if (score >= 40.0) return QualityLevel::POOR;
        if (score >= 20.0) return QualityLevel::BAD;
        return QualityLevel::INVALID;
    }

private:

    CarrierPhaseAnalysis analyze_carrier_phase(uint8_t carr_soln) {
        CarrierPhaseAnalysis analysis;

        switch (carr_soln) {
            case 2:
                analysis.level = QualityLevel::EXCELLENT;
                analysis.score = 95.0;
                analysis.expected_accuracy_cm = 2.0;
                analysis.description = "RTK固定解：载波相位模糊度已固定，最高精度";
                break;

            case 1:
                analysis.level = QualityLevel::GOOD;
                analysis.score = 70.0;
                analysis.expected_accuracy_cm = 15.0;
                analysis.description = "RTK浮点解：载波相位可用，但模糊度未固定";
                break;

            case 0:
                analysis.level = QualityLevel::POOR;
                analysis.score = 30.0;
                analysis.expected_accuracy_cm = 100.0;
                analysis.description = "无载波相位解：仅使用伪距测量";
                break;

            default:
                analysis.level = QualityLevel::INVALID;
                analysis.score = 0.0;
                analysis.expected_accuracy_cm = 1000.0;
                analysis.description = "未知载波相位状态";
                break;
        }

        return analysis;
    }

    SatelliteGeometryAnalysis analyze_satellite_geometry(uint8_t num_sv, double p_dop) {
        SatelliteGeometryAnalysis analysis;


        analysis.satellite_score = analyze_satellite_count(num_sv);
        analysis.satellite_count_level = score_to_level(analysis.satellite_score);


        analysis.pdop_score = analyze_pdop_quality(p_dop);
        analysis.pdop_level = score_to_level(analysis.pdop_score);


        analysis.combined_score = (analysis.satellite_score * 0.6 + analysis.pdop_score * 0.4);


        analysis.geometry_assessment = generate_geometry_description(num_sv, p_dop);

        return analysis;
    }

    AccuracyReportAnalysis analyze_accuracy_report(double h_acc, double v_acc,
                                                   uint8_t carr_soln, uint8_t num_sv) {
        AccuracyReportAnalysis analysis;


        bool h_acc_reasonable = (h_acc > 0.005 && h_acc < 20.0);
        bool v_acc_reasonable = (v_acc > 0.008 && v_acc < 30.0);
        analysis.is_reasonable = h_acc_reasonable && v_acc_reasonable;

        if (!analysis.is_reasonable) {
            analysis.credibility_score = 0.0;
            analysis.effective_h_acc = estimate_accuracy_from_context(carr_soln, num_sv, true);
            analysis.effective_v_acc = estimate_accuracy_from_context(carr_soln, num_sv, false);
            analysis.credibility_reason = "设备报告精度超出合理范围，使用估计值";
            return analysis;
        }


        double expected_h_acc = estimate_accuracy_from_context(carr_soln, num_sv, true);
        double expected_v_acc = estimate_accuracy_from_context(carr_soln, num_sv, false);

        double h_consistency = calculate_consistency(h_acc, expected_h_acc);
        double v_consistency = calculate_consistency(v_acc, expected_v_acc);

        analysis.credibility_score = (h_consistency + v_consistency) / 2.0;


        if (analysis.credibility_score > 0.7) {
            analysis.effective_h_acc = h_acc;
            analysis.effective_v_acc = v_acc;
            analysis.credibility_reason = "设备报告精度与预期一致，直接使用";
        } else {

            double blend_factor = analysis.credibility_score;
            analysis.effective_h_acc = blend_factor * h_acc + (1-blend_factor) * expected_h_acc;
            analysis.effective_v_acc = blend_factor * v_acc + (1-blend_factor) * expected_v_acc;
            analysis.credibility_reason = "设备报告精度可信度中等，混合使用估计值";
        }

        return analysis;
    }


    double analyze_satellite_count(uint8_t num_sv) {

        if (num_sv >= 15) return 100.0;
        if (num_sv >= 12) return 90.0;
        if (num_sv >= 10) return 80.0;
        if (num_sv >= 8)  return 70.0;
        if (num_sv >= 6)  return 55.0;
        if (num_sv >= 4)  return 35.0;
        return 10.0;
    }

    double analyze_pdop_quality(double p_dop) {

        if (p_dop <= 1.0) return 100.0;
        if (p_dop <= 1.5) return 95.0;
        if (p_dop <= 2.0) return 85.0;
        if (p_dop <= 3.0) return 75.0;
        if (p_dop <= 4.0) return 60.0;
        if (p_dop <= 6.0) return 45.0;
        if (p_dop <= 8.0) return 25.0;
        return 5.0;
    }

    double estimate_accuracy_from_context(uint8_t carr_soln, uint8_t num_sv, bool horizontal) {

        double base_acc = horizontal ? 0.02 : 0.04;


        double carr_factor = (carr_soln == 2) ? 1.0 :
                            (carr_soln == 1) ? 5.0 : 20.0;


        double sat_factor = (num_sv >= 10) ? 1.0 :
                           (num_sv >= 8) ? 1.2 :
                           (num_sv >= 6) ? 1.8 : 3.0;

        return base_acc * carr_factor * sat_factor;
    }

    double calculate_consistency(double reported, double expected) {
        double ratio = reported / expected;

        if (ratio >= 0.5 && ratio <= 2.0) {
            return 1.0 - std::abs(1.0 - ratio);
        } else if (ratio < 0.5) {
            return 0.3;
        } else {
            return 0.1;
        }
    }

    std::string generate_geometry_description(uint8_t num_sv, double p_dop) {
        std::stringstream ss;
        ss << "卫星数: " << static_cast<int>(num_sv);

        if (num_sv >= 12) ss << "(冗余度高)";
        else if (num_sv >= 8) ss << "(冗余度足够)";
        else if (num_sv >= 6) ss << "(冗余度低)";
        else ss << "(冗余度不足)";

        ss << ", PDOP: " << std::fixed << std::setprecision(1) << p_dop;

        if (p_dop <= 2.0) ss << "(几何优秀)";
        else if (p_dop <= 3.0) ss << "(几何良好)";
        else if (p_dop <= 5.0) ss << "(几何可接受)";
        else ss << "(几何较差)";

        return ss.str();
    }

    std::string generate_analysis_summary(const CarrierPhaseAnalysis& carrier,
                                         const SatelliteGeometryAnalysis& geometry,
                                         const AccuracyReportAnalysis& accuracy,
                                         double overall_score) {
        std::stringstream ss;
        ss << carrier.description << "; ";
        ss << geometry.geometry_assessment << "; ";
        ss << accuracy.credibility_reason << "; ";
        ss << "综合评分: " << std::fixed << std::setprecision(1) << overall_score << "/100";
        return ss.str();
    }


    double excellent_threshold_;
    double good_threshold_;
    double acceptable_threshold_;
    double poor_threshold_;
    double bad_threshold_;


    size_t analysis_count_;
    double avg_score_;
};


class RTKAntennaCalibrator {
public:
    struct CalibrationConfig {
        bool enable_lever_arm_compensation = true;
        bool enable_online_estimation = false;
        bool enable_attitude_compensation = true;
        bool debug_mode = false;

        Eigen::Vector3d antenna_offset_T = Eigen::Vector3d(0.0, 0.0, 0.3);
        double estimation_convergence_threshold = 0.01;
        double estimation_min_motion_threshold = 2.0;
        int min_samples_for_estimation = 50;
    };

    struct CalibrationSample {
        double timestamp;
        Eigen::Vector3d pos_rtk_world;
        Eigen::Vector3d pos_imu_world;
        Eigen::Matrix3d rotation_world;
        Eigen::Vector3d velocity;
        double quality_score;
    };

    struct CompensationStats {
        double mean_compensation_magnitude = 0.0;
        double max_compensation_magnitude = 0.0;
        Eigen::Vector3d mean_compensation_vector = Eigen::Vector3d::Zero();
        int total_compensations = 0;
        double improvement_percentage = 0.0;
    };

    RTKAntennaCalibrator()
        : calibration_converged_(false)
        , last_estimation_error_(std::numeric_limits<double>::max())
        , estimation_iterations_(0)
    {
        estimated_offset_ = config_.antenna_offset_T;
        resetStats();
    }

    ~RTKAntennaCalibrator() = default;

    void loadConfigFromROS(ros::NodeHandle& nh) {

        config_.enable_lever_arm_compensation = false;
        config_.enable_online_estimation = false;
        config_.enable_attitude_compensation = false;
        config_.debug_mode = false;
    }

    void setAntennaOffset(const Eigen::Vector3d& offset) {
        config_.antenna_offset_T = offset;
        estimated_offset_ = offset;

        if (config_.debug_mode) {
            (void)0;
        }
    }

    Eigen::Vector3d getAntennaOffset() const {
        return config_.antenna_offset_T;
    }

    Eigen::Vector3d compensateAntennaToIMU(const Eigen::Vector3d& pos_antenna_world,
                                           const Eigen::Matrix3d& rotation_world) const {

        (void)rotation_world;
        return pos_antenna_world;
    }

    Eigen::Vector3d compensateIMUToAntenna(const Eigen::Vector3d& pos_imu_world,
                                           const Eigen::Matrix3d& rotation_world) const {

        (void)rotation_world;
        return pos_imu_world;
    }

    void addCalibrationSample(const CalibrationSample& sample) {
        if (!config_.enable_online_estimation) {
            return;
        }

        calibration_samples_.push_back(sample);

        const size_t max_samples = 1000;
        if (calibration_samples_.size() > max_samples) {
            calibration_samples_.pop_front();
        }

        cleanOldSamples(sample.timestamp);
    }

    bool performOnlineEstimation() {
        if (!config_.enable_online_estimation ||
            calibration_samples_.size() < static_cast<size_t>(config_.min_samples_for_estimation)) {
            return false;
        }

        if (!hasEnoughMotionForEstimation()) {
            if (config_.debug_mode) {
                (void)0;
            }
            return false;
        }

        std::vector<CalibrationSample> valid_samples;
        valid_samples.reserve(calibration_samples_.size());
        for (const auto& sample : calibration_samples_) {
            if (sample.velocity.norm() > config_.estimation_min_motion_threshold &&
                sample.quality_score > 0.7) {
                valid_samples.push_back(sample);
            }
        }

        if (valid_samples.size() < 20) {
            return false;
        }

        Eigen::Vector3d new_offset = computeWeightedLeastSquares(valid_samples);
        double estimation_error = computeEstimationError(valid_samples, new_offset);

        bool converged = false;
        if (estimation_iterations_ > 0) {
            double error_improvement = last_estimation_error_ - estimation_error;
            converged = (error_improvement < config_.estimation_convergence_threshold);
        }

        estimated_offset_ = new_offset;
        last_estimation_error_ = estimation_error;
        estimation_iterations_++;
        estimation_error_history_.push_back(estimation_error);

        if (converged) {
            calibration_converged_ = true;
            config_.antenna_offset_T = estimated_offset_;

            if (config_.debug_mode) {
                (void)0;
                (void)0;
            }
        }

        return converged;
    }

    Eigen::Vector3d performStaticCalibration(const std::vector<Eigen::Vector3d>& rtk_positions,
                                             const std::vector<Eigen::Vector3d>& imu_positions,
                                             const std::vector<Eigen::Matrix3d>& rotations) {
        if (rtk_positions.size() != imu_positions.size() ||
            rtk_positions.size() != rotations.size() ||
            rtk_positions.size() < 3) {
            (void)0;
            return config_.antenna_offset_T;
        }

        const int n_equations = static_cast<int>(rtk_positions.size());
        Eigen::MatrixXd A(3 * n_equations, 3);
        Eigen::VectorXd b(3 * n_equations);

        for (int i = 0; i < n_equations; ++i) {
            A.block<3,3>(3 * i, 0) = rotations[i];
            b.segment<3>(3 * i) = rtk_positions[i] - imu_positions[i];
        }

        Eigen::Vector3d estimated_offset = A.colPivHouseholderQr().solve(b);

        double rmse = 0.0;
        for (int i = 0; i < n_equations; ++i) {
            Eigen::Vector3d predicted_rtk = imu_positions[i] + rotations[i] * estimated_offset;
            Eigen::Vector3d error = rtk_positions[i] - predicted_rtk;
            rmse += error.squaredNorm();
        }
        rmse = std::sqrt(rmse / n_equations);

        (void)0;
        (void)0;
        (void)0;

        return estimated_offset;
    }

    CompensationStats getCompensationStats() const {
        return stats_;
    }

    double validateCalibration(const std::vector<CalibrationSample>& test_samples) const {
        if (test_samples.empty()) {
            return 0.0;
        }

        double total_error = 0.0;
        for (const auto& sample : test_samples) {
            Eigen::Vector3d compensated_imu =
                compensateAntennaToIMU(sample.pos_rtk_world, sample.rotation_world);
            Eigen::Vector3d error = compensated_imu - sample.pos_imu_world;
            total_error += error.squaredNorm();
        }

        return std::sqrt(total_error / test_samples.size());
    }

    void resetStats() {
        stats_.mean_compensation_magnitude = 0.0;
        stats_.max_compensation_magnitude = 0.0;
        stats_.mean_compensation_vector.setZero();
        stats_.total_compensations = 0;
        stats_.improvement_percentage = 0.0;
    }

    void printCalibrationInfo() const {
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;

        if (stats_.total_compensations > 0) {
            (void)0;
            (void)0;
            (void)0;
            (void)0;
            (void)0;
        }
        (void)0;
    }

    bool isCalibrationConverged() const {
        return calibration_converged_;
    }

    const CalibrationConfig& getConfig() const {
        return config_;
    }

private:
    void updateCompensationStats(const Eigen::Vector3d& compensation_vector) {
        stats_.total_compensations++;

        double magnitude = compensation_vector.norm();
        stats_.mean_compensation_magnitude =
            (stats_.mean_compensation_magnitude * (stats_.total_compensations - 1) + magnitude) /
            stats_.total_compensations;

        if (magnitude > stats_.max_compensation_magnitude) {
            stats_.max_compensation_magnitude = magnitude;
        }

        stats_.mean_compensation_vector =
            (stats_.mean_compensation_vector * (stats_.total_compensations - 1) + compensation_vector) /
            stats_.total_compensations;
    }

    bool hasEnoughMotionForEstimation() const {
        if (calibration_samples_.size() < 10) {
            return false;
        }

        double max_velocity = 0.0;
        double total_displacement = 0.0;

        for (size_t i = 1; i < calibration_samples_.size(); ++i) {
            double velocity = calibration_samples_[i].velocity.norm();
            max_velocity = std::max(max_velocity, velocity);

            Eigen::Vector3d displacement =
                calibration_samples_[i].pos_rtk_world - calibration_samples_[i - 1].pos_rtk_world;
            total_displacement += displacement.norm();
        }

        return (max_velocity > config_.estimation_min_motion_threshold && total_displacement > 10.0);
    }

    Eigen::Vector3d computeWeightedLeastSquares(const std::vector<CalibrationSample>& samples) const {
        const int n = static_cast<int>(samples.size());
        Eigen::MatrixXd A(3 * n, 3);
        Eigen::VectorXd b(3 * n);
        Eigen::VectorXd weights(3 * n);

        for (int i = 0; i < n; ++i) {
            const auto& sample = samples[i];
            A.block<3,3>(3 * i, 0) = sample.rotation_world;
            b.segment<3>(3 * i) = sample.pos_rtk_world - sample.pos_imu_world;

            double weight = sample.quality_score * std::min(sample.velocity.norm() / 5.0, 1.0);
            weights.segment<3>(3 * i).setConstant(weight);
        }

        Eigen::MatrixXd W = weights.asDiagonal();
        Eigen::MatrixXd AtWA = A.transpose() * W * A;
        Eigen::VectorXd AtWb = A.transpose() * W * b;

        return AtWA.colPivHouseholderQr().solve(AtWb);
    }

    double computeEstimationError(const std::vector<CalibrationSample>& samples,
                                  const Eigen::Vector3d& offset) const {
        if (samples.empty()) {
            return 0.0;
        }

        double total_error = 0.0;
        for (const auto& sample : samples) {
            Eigen::Vector3d predicted_rtk = sample.pos_imu_world + sample.rotation_world * offset;
            Eigen::Vector3d error = sample.pos_rtk_world - predicted_rtk;
            total_error += error.squaredNorm();
        }

        return std::sqrt(total_error / samples.size());
    }

    void cleanOldSamples(double current_time, double max_age = 300.0) {
        while (!calibration_samples_.empty()) {
            if (current_time - calibration_samples_.front().timestamp > max_age) {
                calibration_samples_.pop_front();
            } else {
                break;
            }
        }
    }

    CalibrationConfig config_;
    std::deque<CalibrationSample> calibration_samples_;
    CompensationStats stats_;
    bool calibration_converged_;
    double last_estimation_error_;
    Eigen::Vector3d estimated_offset_;
    int estimation_iterations_;
    std::vector<double> estimation_error_history_;
};

inline RTKAntennaCalibrator& GetGlobalRtkAntennaCalibrator() {
    static RTKAntennaCalibrator calibrator;
    return calibrator;
}


class RTKConstraintValidator {
public:
    struct TrajectoryPoint {
        double timestamp = 0.0;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
        double velocity_norm = 0.0;
        bool is_rtk_fixed = false;
        double rtk_quality_score = 0.0;
    };

    struct ValidationMetrics {
        double mean_error_before_constraint = 0.0;
        double mean_error_after_constraint = 0.0;
        double error_reduction_percentage = 0.0;
        double max_error_reduction = 0.0;

        int total_constraints_attempted = 0;
        int successful_constraints_applied = 0;
        double constraint_success_rate = 0.0;
        double average_correction_magnitude = 0.0;
        double max_single_correction = 0.0;

        bool has_trajectory_jumps = false;
        double max_velocity_spike = 0.0;
        double pose_continuity_score = 1.0;
        double covariance_growth_rate = 0.0;

        double analysis_duration = 0.0;
        double total_distance_traveled = 0.0;
        double average_constraint_interval = 0.0;
    };

    RTKConstraintValidator(ros::NodeHandle& nh)
        : nh_(nh)
        , max_trajectory_history_time_(300.0)
        , min_error_reduction_threshold_(0.1)
        , max_acceptable_position_jump_(0.5)
        , stability_analysis_window_(60.0)
        , real_time_monitoring_enabled_(false)
    {
        path_map_pub_ = nh_.advertise<nav_msgs::Path>("/rtk_validator/path_map", 10);
        path_world_pub_ = nh_.advertise<nav_msgs::Path>("/rtk_validator/path_world", 10);
        path_rtk_pub_ = nh_.advertise<nav_msgs::Path>("/rtk_validator/path_rtk", 10);
        error_pub_ = nh_.advertise<std_msgs::Float64>("/rtk_validator/current_error", 10);
        constraint_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/rtk_validator/constraint_markers", 10);

        log_directory_ = "/tmp/rtk_validation_logs/";
        std::system(("mkdir -p " + log_directory_).c_str());

        std::string log_filename = log_directory_ + "validation_" +
                                   std::to_string(ros::Time::now().toSec()) + ".log";
        validation_log_.open(log_filename);

        if (validation_log_.is_open()) {
            validation_log_ << "RTK Constraint Validation Log Started at "
                            << ros::Time::now() << std::endl;
            validation_log_ << "Columns: timestamp,event_type,error_before,error_after,"
                            << "correction_magnitude,rtk_quality" << std::endl;
        }

        (void)0;
    }

    ~RTKConstraintValidator() {
        if (validation_log_.is_open()) {
            validation_log_.close();
        }
    }

    void addTrajectoryPoint(const std::string& trajectory_name, const TrajectoryPoint& point) {
        auto& trajectory = trajectories_[trajectory_name];
        trajectory.push_back(point);

        double current_time = point.timestamp;
        while (!trajectory.empty() &&
               (current_time - trajectory.front().timestamp) > max_trajectory_history_time_) {
            trajectory.pop_front();
        }

        if (trajectory_name == "world") {
            detectTrajectoryAnomalies(trajectory);
        }

        cleanOldData(current_time);

        if (real_time_monitoring_enabled_ && trajectory_name == "world") {
            publishValidationVisualization();
            if (trajectories_.count("rtk") > 0) {
                double current_error = computeInterpolatedError(trajectories_["world"],
                                                                trajectories_["rtk"],
                                                                current_time);
                std_msgs::Float64 error_msg;
                error_msg.data = current_error;
                error_pub_.publish(error_msg);
            }
        }
    }

    void recordConstraintEvent(double timestamp,
                               const Eigen::Vector3d& correction_vector,
                               bool constraint_accepted,
                               double rtk_quality) {
        ConstraintEvent event;
        event.timestamp = timestamp;
        event.correction = correction_vector;
        event.accepted = constraint_accepted;
        event.rtk_quality = rtk_quality;

        if (trajectories_.count("world") > 0 && trajectories_.count("rtk") > 0) {
            event.pre_constraint_error =
                computeInterpolatedError(trajectories_["world"], trajectories_["rtk"], timestamp - 0.01);
            event.post_constraint_error =
                computeInterpolatedError(trajectories_["world"], trajectories_["rtk"], timestamp + 0.01);
        } else {
            event.pre_constraint_error = -1.0;
            event.post_constraint_error = -1.0;
        }

        constraint_events_.push_back(event);

        while (!constraint_events_.empty() &&
               (timestamp - constraint_events_.front().timestamp) > max_trajectory_history_time_) {
            constraint_events_.pop_front();
        }

        cleanOldData(timestamp);

        logValidationEvent("CONSTRAINT," + std::to_string(timestamp) + "," +
                           std::to_string(event.pre_constraint_error) + "," +
                           std::to_string(event.post_constraint_error) + "," +
                           std::to_string(correction_vector.norm()) + "," +
                           std::to_string(rtk_quality) + "," +
                           (constraint_accepted ? "ACCEPTED" : "REJECTED"));

        (void)0;
    }

    void recordEKFState(double timestamp,
                        const Eigen::Vector3d& position,
                        const Eigen::Matrix3d& position_covariance) {
        EKFStateRecord record;
        record.timestamp = timestamp;
        record.position = position;
        record.covariance_trace = position_covariance.trace();
        ekf_states_.push_back(record);

        while (!ekf_states_.empty() &&
               (timestamp - ekf_states_.front().timestamp) > max_trajectory_history_time_) {
            ekf_states_.pop_front();
        }
    }

    ValidationMetrics computeValidationMetrics(double time_window = -1.0) {
        ValidationMetrics metrics = {};

        if (constraint_events_.empty() || trajectories_.count("world") == 0 || trajectories_.count("rtk") == 0) {
            (void)0;
            return metrics;
        }

        double current_time = ros::Time::now().toSec();
        double start_time = (time_window > 0) ? (current_time - time_window)
                                              : constraint_events_.front().timestamp;

        std::vector<ConstraintEvent> recent_events;
        for (const auto& event : constraint_events_) {
            if (event.timestamp >= start_time) {
                recent_events.push_back(event);
            }
        }

        if (recent_events.empty()) {
            return metrics;
        }

        metrics.analysis_duration = current_time - start_time;
        metrics.total_constraints_attempted = recent_events.size();

        std::vector<double> corrections;
        std::vector<double> errors_before, errors_after;
        int successful_constraints = 0;

        for (const auto& event : recent_events) {
            if (event.accepted) {
                successful_constraints++;
                corrections.push_back(event.correction.norm());

                if (event.pre_constraint_error >= 0 && event.post_constraint_error >= 0) {
                    errors_before.push_back(event.pre_constraint_error);
                    errors_after.push_back(event.post_constraint_error);
                }
            }
        }

        metrics.successful_constraints_applied = successful_constraints;
        metrics.constraint_success_rate =
            recent_events.empty() ? 0.0 : static_cast<double>(successful_constraints) / recent_events.size();

        if (!corrections.empty()) {
            metrics.average_correction_magnitude =
                std::accumulate(corrections.begin(), corrections.end(), 0.0) / corrections.size();
            metrics.max_single_correction = *std::max_element(corrections.begin(), corrections.end());
        }

        if (!errors_before.empty() && !errors_after.empty() && errors_before.size() == errors_after.size()) {
            double mean_before = std::accumulate(errors_before.begin(), errors_before.end(), 0.0) / errors_before.size();
            double mean_after = std::accumulate(errors_after.begin(), errors_after.end(), 0.0) / errors_after.size();

            metrics.mean_error_before_constraint = mean_before;
            metrics.mean_error_after_constraint = mean_after;

            if (mean_before > 1e-6) {
                metrics.error_reduction_percentage = ((mean_before - mean_after) / mean_before) * 100.0;
            }

            double max_reduction = 0.0;
            for (size_t i = 0; i < errors_before.size(); ++i) {
                double reduction = errors_before[i] - errors_after[i];
                max_reduction = std::max(max_reduction, reduction);
            }
            metrics.max_error_reduction = max_reduction;
        }

        if (trajectories_.count("world") > 0) {
            const auto& world_traj = trajectories_["world"];
            metrics.pose_continuity_score = computePoseContinuityScore(world_traj, stability_analysis_window_);

            double max_jump = 0.0;
            for (size_t i = 1; i < world_traj.size(); ++i) {
                if (world_traj[i].timestamp - world_traj[i - 1].timestamp < 1.0) {
                    double jump = (world_traj[i].position - world_traj[i - 1].position).norm();
                    max_jump = std::max(max_jump, jump);
                }
            }
            metrics.max_velocity_spike = max_jump;
            metrics.has_trajectory_jumps = (max_jump > max_acceptable_position_jump_);
        }

        if (recent_events.size() > 1) {
            double total_time = recent_events.back().timestamp - recent_events.front().timestamp;
            metrics.average_constraint_interval = total_time / (recent_events.size() - 1);
        }

        if (!ekf_states_.empty()) {
            std::vector<double> cov_traces;
            for (const auto& state : ekf_states_) {
                if (state.timestamp >= start_time) {
                    cov_traces.push_back(state.covariance_trace);
                }
            }

            if (cov_traces.size() > 1) {
                double cov_start = cov_traces.front();
                double cov_end = cov_traces.back();
                double time_span = metrics.analysis_duration;

                if (time_span > 1e-6 && cov_start > 1e-6) {
                    metrics.covariance_growth_rate = (cov_end - cov_start) / (cov_start * time_span);
                }
            }
        }

        return metrics;
    }

    bool validateTrajectoryAlignment(double max_acceptable_error = 1.0) {
        if (trajectories_.count("world") == 0 || trajectories_.count("rtk") == 0) {
            return false;
        }

        double current_time = ros::Time::now().toSec();
        double rmse = computeTrajectoryRMSE("world", "rtk", current_time - 60.0, current_time);
        bool is_aligned = (rmse >= 0.0 && rmse <= max_acceptable_error);

        logValidationEvent("ALIGNMENT_CHECK,rmse=" + std::to_string(rmse) +
                           ",threshold=" + std::to_string(max_acceptable_error) +
                           ",result=" + (is_aligned ? "PASS" : "FAIL"));

        return is_aligned;
    }

    double computeTrajectoryRMSE(const std::string& trajectory_a,
                                 const std::string& trajectory_b,
                                 double time_start = 0.0,
                                 double time_end = -1.0) {
        if (trajectories_.count(trajectory_a) == 0 || trajectories_.count(trajectory_b) == 0) {
            return -1.0;
        }

        const auto& traj_a = trajectories_[trajectory_a];
        const auto& traj_b = trajectories_[trajectory_b];

        if (traj_a.empty() || traj_b.empty()) {
            return -1.0;
        }

        if (time_end < 0) {
            time_end = ros::Time::now().toSec();
        }

        std::vector<double> errors;
        double dt = 0.1;
        for (double t = time_start; t <= time_end; t += dt) {
            double error = computeInterpolatedError(traj_a, traj_b, t);
            if (error >= 0) {
                errors.push_back(error);
            }
        }

        if (errors.empty()) {
            return -1.0;
        }

        double sum_sq = 0.0;
        for (double error : errors) {
            sum_sq += error * error;
        }
        return std::sqrt(sum_sq / errors.size());
    }

    void enableRealTimeMonitoring(bool enable = true) {
        real_time_monitoring_enabled_ = enable;
        (void)0;
    }

    bool isConstraintEffective(double recent_time_window = 30.0) {
        ValidationMetrics metrics = computeValidationMetrics(recent_time_window);
        return (metrics.max_error_reduction >= min_error_reduction_threshold_) &&
               (metrics.constraint_success_rate >= 0.5);
    }

    double getCurrentErrorReductionRate() {
        ValidationMetrics metrics = computeValidationMetrics(30.0);
        return metrics.error_reduction_percentage;
    }

    void publishValidationVisualization() {
        auto publish_path = [this](ros::Publisher& pub, const std::string& traj_name,
                                   const std::string& frame_id) {
            if (trajectories_.count(traj_name) == 0) return;

            nav_msgs::Path path;
            path.header.frame_id = frame_id;
            path.header.stamp = ros::Time::now();

            for (const auto& point : trajectories_[traj_name]) {
                geometry_msgs::PoseStamped pose;
                pose.header = path.header;
                pose.header.stamp = ros::Time(point.timestamp);
                pose.pose.position.x = point.position.x();
                pose.pose.position.y = point.position.y();
                pose.pose.position.z = point.position.z();
                pose.pose.orientation.x = point.orientation.x();
                pose.pose.orientation.y = point.orientation.y();
                pose.pose.orientation.z = point.orientation.z();
                pose.pose.orientation.w = point.orientation.w();
                path.poses.push_back(pose);
            }

            pub.publish(path);
        };

        publish_path(path_map_pub_, "map", "map");
        publish_path(path_world_pub_, "world", "world");
        publish_path(path_rtk_pub_, "rtk", "world");

        publishVisualizationMarkers();
    }

    void saveValidationReport(const std::string& filepath) {
        ValidationMetrics metrics = computeValidationMetrics();

        std::ofstream report(filepath);
        if (!report.is_open()) {
            (void)0;
            return;
        }

        report << "RTK Constraint Validation Report" << std::endl;
        report << "Generated: " << ros::Time::now() << std::endl;
        report << "================================" << std::endl << std::endl;

        report << "ERROR REDUCTION ANALYSIS:" << std::endl;
        report << "  Mean error before constraint: " << std::fixed << std::setprecision(4)
               << metrics.mean_error_before_constraint << " m" << std::endl;
        report << "  Mean error after constraint:  " << std::fixed << std::setprecision(4)
               << metrics.mean_error_after_constraint << " m" << std::endl;
        report << "  Error reduction percentage:   " << std::fixed << std::setprecision(2)
               << metrics.error_reduction_percentage << "%" << std::endl;
        report << "  Maximum error reduction:      " << std::fixed << std::setprecision(4)
               << metrics.max_error_reduction << " m" << std::endl << std::endl;

        report << "CONSTRAINT EFFECTIVENESS:" << std::endl;
        report << "  Total constraints attempted:  " << metrics.total_constraints_attempted << std::endl;
        report << "  Successful constraints:       " << metrics.successful_constraints_applied << std::endl;
        report << "  Success rate:                 " << std::fixed << std::setprecision(1)
               << metrics.constraint_success_rate * 100 << "%" << std::endl;
        report << "  Average correction magnitude: " << std::fixed << std::setprecision(4)
               << metrics.average_correction_magnitude << " m" << std::endl;
        report << "  Maximum single correction:    " << std::fixed << std::setprecision(4)
               << metrics.max_single_correction << " m" << std::endl;
        report << "  Average constraint interval:  " << std::fixed << std::setprecision(2)
               << metrics.average_constraint_interval << " s" << std::endl << std::endl;

        report << "SYSTEM STABILITY:" << std::endl;
        report << "  Trajectory jumps detected:    " << (metrics.has_trajectory_jumps ? "YES" : "NO") << std::endl;
        report << "  Maximum velocity spike:       " << std::fixed << std::setprecision(4)
               << metrics.max_velocity_spike << " m/s" << std::endl;
        report << "  Pose continuity score:        " << std::fixed << std::setprecision(3)
               << metrics.pose_continuity_score << " (0-1)" << std::endl;
        report << "  Covariance growth rate:       " << std::fixed << std::setprecision(6)
               << metrics.covariance_growth_rate << " /s" << std::endl << std::endl;

        report << "ANALYSIS SUMMARY:" << std::endl;
        report << "  Analysis duration:            " << std::fixed << std::setprecision(1)
               << metrics.analysis_duration << " s" << std::endl;

        report << std::endl << "OVERALL ASSESSMENT:" << std::endl;
        if (metrics.error_reduction_percentage > 20.0 && metrics.constraint_success_rate > 0.8 &&
            !metrics.has_trajectory_jumps && metrics.pose_continuity_score > 0.8) {
            report << "  RTK constraint system is HIGHLY EFFECTIVE" << std::endl;
        } else if (metrics.error_reduction_percentage > 10.0 && metrics.constraint_success_rate > 0.6) {
            report << "  RTK constraint system is MODERATELY EFFECTIVE" << std::endl;
        } else {
            report << "  RTK constraint system effectiveness is LIMITED" << std::endl;
        }

        report.close();
        (void)0;
    }

    void exportTrajectoryComparison(const std::string& csv_path) {
        if (trajectories_.count("world") == 0 || trajectories_.count("rtk") == 0) {
            (void)0;
            return;
        }

        const auto& world_traj = trajectories_["world"];
        const auto& rtk_traj = trajectories_["rtk"];

        double start_time = std::max(world_traj.front().timestamp, rtk_traj.front().timestamp);
        double end_time = std::min(world_traj.back().timestamp, rtk_traj.back().timestamp);

        std::vector<double> errors = getErrorTimeSeries("world", "rtk", start_time, end_time);

        std::ofstream csv(csv_path);
        if (!csv.is_open()) {
            (void)0;
            return;
        }

        csv << "time,error_m\n";
        double t = start_time;
        for (double error : errors) {
            csv << std::fixed << std::setprecision(3) << t << "," << std::setprecision(4) << error << "\n";
            t += 0.1;
        }
        csv.close();
        (void)0;
    }

    void setValidationParameters(double min_error_reduction = 0.1,
                                 double max_acceptable_jump = 0.5,
                                 double stability_window = 60.0) {
        min_error_reduction_threshold_ = min_error_reduction;
        max_acceptable_position_jump_ = max_acceptable_jump;
        stability_analysis_window_ = stability_window;
    }

private:
    struct ConstraintEvent {
        double timestamp = 0.0;
        Eigen::Vector3d correction = Eigen::Vector3d::Zero();
        bool accepted = false;
        double rtk_quality = 0.0;
        double pre_constraint_error = -1.0;
        double post_constraint_error = -1.0;
    };

    struct EKFStateRecord {
        double timestamp = 0.0;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        double covariance_trace = 0.0;
    };

    double computeInterpolatedError(const std::deque<TrajectoryPoint>& traj_a,
                                    const std::deque<TrajectoryPoint>& traj_b,
                                    double timestamp) {
        auto find_closest = [](const std::deque<TrajectoryPoint>& traj, double t) -> std::pair<int, int> {
            if (traj.empty()) return {-1, -1};

            int best_idx = 0;
            double best_diff = std::abs(traj[0].timestamp - t);

            for (size_t i = 1; i < traj.size(); ++i) {
                double diff = std::abs(traj[i].timestamp - t);
                if (diff < best_diff) {
                    best_diff = diff;
                    best_idx = static_cast<int>(i);
                }
            }

            int before_idx = best_idx, after_idx = best_idx;

            if (best_idx > 0 && traj[best_idx].timestamp > t) {
                before_idx = best_idx - 1;
                after_idx = best_idx;
            } else if (best_idx < static_cast<int>(traj.size()) - 1 && traj[best_idx].timestamp < t) {
                before_idx = best_idx;
                after_idx = best_idx + 1;
            }

            return {before_idx, after_idx};
        };

        auto [a_before, a_after] = find_closest(traj_a, timestamp);
        auto [b_before, b_after] = find_closest(traj_b, timestamp);

        if (a_before < 0 || b_before < 0) {
            return -1.0;
        }

        Eigen::Vector3d pos_a, pos_b;

        if (a_before == a_after) {
            pos_a = traj_a[a_before].position;
        } else {
            double t_before = traj_a[a_before].timestamp;
            double t_after = traj_a[a_after].timestamp;
            double alpha = (timestamp - t_before) / (t_after - t_before);
            pos_a = (1.0 - alpha) * traj_a[a_before].position + alpha * traj_a[a_after].position;
        }

        if (b_before == b_after) {
            pos_b = traj_b[b_before].position;
        } else {
            double t_before = traj_b[b_before].timestamp;
            double t_after = traj_b[b_after].timestamp;
            double alpha = (timestamp - t_before) / (t_after - t_before);
            pos_b = (1.0 - alpha) * traj_b[b_before].position + alpha * traj_b[b_after].position;
        }

        return (pos_a - pos_b).norm();
    }

    std::vector<double> getErrorTimeSeries(const std::string& reference_traj,
                                           const std::string& test_traj,
                                           double time_start,
                                           double time_end) {
        std::vector<double> errors;
        if (trajectories_.count(reference_traj) == 0 || trajectories_.count(test_traj) == 0) {
            return errors;
        }

        const auto& ref = trajectories_[reference_traj];
        const auto& test = trajectories_[test_traj];

        if (ref.empty() || test.empty()) {
            return errors;
        }

        for (double t = time_start; t <= time_end; t += 0.1) {
            double error = computeInterpolatedError(ref, test, t);
            if (error >= 0.0) {
                errors.push_back(error);
            }
        }
        return errors;
    }

    void detectTrajectoryAnomalies(const std::deque<TrajectoryPoint>& trajectory) {
        if (trajectory.size() < 2) {
            return;
        }

        const auto& latest = trajectory.back();
        const auto& previous = trajectory[trajectory.size() - 2];

        double dt = latest.timestamp - previous.timestamp;
        if (dt <= 0.0 || dt > 1.0) {
            return;
        }

        double jump = (latest.position - previous.position).norm();
        if (jump > max_acceptable_position_jump_ * 2.0) {
            (void)0;
        }
    }

    double computePoseContinuityScore(const std::deque<TrajectoryPoint>& trajectory,
                                      double time_window) {
        if (trajectory.size() < 2) {
            return 1.0;
        }

        double current_time = trajectory.back().timestamp;
        double start_time = current_time - time_window;

        std::vector<double> velocity_changes;
        std::vector<double> position_jumps;

        for (size_t i = 1; i < trajectory.size(); ++i) {
            if (trajectory[i].timestamp < start_time) continue;

            double dt = trajectory[i].timestamp - trajectory[i - 1].timestamp;
            if (dt > 0.001 && dt < 1.0) {
                double pos_jump = (trajectory[i].position - trajectory[i - 1].position).norm();
                double expected_displacement = trajectory[i - 1].velocity_norm * dt;

                if (expected_displacement > 0.01) {
                    double jump_ratio = pos_jump / expected_displacement;
                    position_jumps.push_back(std::min(jump_ratio, 3.0));
                }

                double vel_change = std::abs(trajectory[i].velocity_norm - trajectory[i - 1].velocity_norm);
                velocity_changes.push_back(vel_change);
            }
        }

        double continuity_score = 1.0;

        if (!position_jumps.empty()) {
            double avg_jump_ratio = std::accumulate(position_jumps.begin(), position_jumps.end(), 0.0) /
                                    position_jumps.size();
            continuity_score *= std::max(0.0, 1.0 - (avg_jump_ratio - 1.0) * 0.5);
        }

        if (!velocity_changes.empty()) {
            double max_vel_change = *std::max_element(velocity_changes.begin(), velocity_changes.end());
            continuity_score *= std::max(0.0, 1.0 - max_vel_change * 0.2);
        }

        return std::max(0.0, std::min(1.0, continuity_score));
    }

    void publishVisualizationMarkers() {
        visualization_msgs::MarkerArray marker_array;

        for (size_t i = 0; i < constraint_events_.size(); ++i) {
            const auto& event = constraint_events_[i];

            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time(event.timestamp);
            marker.ns = "constraint_events";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::Marker::ARROW;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = 0.0;
            marker.pose.position.y = 0.0;
            marker.pose.position.z = 0.0;

            marker.scale.x = event.correction.norm();
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;

            if (event.accepted) {
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.color.a = std::min(1.0, event.rtk_quality / 100.0);
            } else {
                marker.color.r = 1.0;
                marker.color.g = 0.0;
                marker.color.b = 0.0;
                marker.color.a = 0.5;
            }

            marker.lifetime = ros::Duration(60.0);
            marker_array.markers.push_back(marker);
        }

        constraint_marker_pub_.publish(marker_array);
    }

    void cleanOldData(double current_time) {
        for (auto& kv : trajectories_) {
            auto& traj = kv.second;
            while (!traj.empty() && current_time - traj.front().timestamp > max_trajectory_history_time_) {
                traj.pop_front();
            }
        }

        while (!constraint_events_.empty() &&
               current_time - constraint_events_.front().timestamp > max_trajectory_history_time_) {
            constraint_events_.pop_front();
        }

        while (!ekf_states_.empty() &&
               current_time - ekf_states_.front().timestamp > max_trajectory_history_time_) {
            ekf_states_.pop_front();
        }
    }

    void logValidationEvent(const std::string& message) {
        if (validation_log_.is_open()) {
            validation_log_ << ros::Time::now().toSec() << "," << message << std::endl;
            validation_log_.flush();
        }
    }

    ros::NodeHandle nh_;
    std::map<std::string, std::deque<TrajectoryPoint>> trajectories_;
    std::deque<ConstraintEvent> constraint_events_;
    std::deque<EKFStateRecord> ekf_states_;

    ros::Publisher path_map_pub_;
    ros::Publisher path_world_pub_;
    ros::Publisher path_rtk_pub_;
    ros::Publisher error_pub_;
    ros::Publisher constraint_marker_pub_;

    double max_trajectory_history_time_;
    double min_error_reduction_threshold_;
    double max_acceptable_position_jump_;
    double stability_analysis_window_;
    bool real_time_monitoring_enabled_;

    std::ofstream validation_log_;
    std::string log_directory_;
};

class RtkProcess
{

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW


        RtkProcess();
        ~RtkProcess();
        double gpst2time(uint32_t week, double tow);
        bool unix2gpst(double unix_timestamp, uint32_t &week, double &tow) const;
        void Process(MeasureGroup &meas,
                     deque<gnss_comm::GnssPVTSolnMsg::ConstPtr> &rtk_buffer,
                     double max_timestamp = std::numeric_limits<double>::infinity());


        RTKQualityAnalyzer::QualityMetrics analyze_rtk_message_quality(const gnss_comm::GnssPVTSolnMsg::ConstPtr& rtk_msg);
        bool should_accept_rtk_message(const RTKQualityAnalyzer::QualityMetrics& quality);
        void log_quality_statistics();
        void load_quality_config_from_ros_params();


        double first_lidar_time;
        double first_latitude;
        double first_longitude;
        double first_altitude;
        double origin_lat_sum;
        double origin_lon_sum;
        double origin_alt_sum;
        int origin_sample_count;
        int origin_sample_target;
        double origin_hacc_max;
        double origin_vacc_max;
        double last_rtk_time_stamp;

        int rtk_window_length;

        double gpst_leap_seconds_offset_;
        double gpst_time_bias_;

        std::deque<gnss_comm::GnssPVTSolnMsg::ConstPtr> valid_rtk_window;

        bool rtk_need_init;
        bool rtk_isvalid;


        RTKQualityAnalyzer quality_analyzer_;
        bool enable_quality_analysis_;
        int quality_analysis_count_;
        double min_acceptable_score_;

    private:
        void Rtk_init(const gnss_comm::GnssPVTSolnMsg::ConstPtr first_rtk_msg);
        void lla2xyz(double lat, double lon, double alt, V3D &xyz);

};

RtkProcess::RtkProcess()
{
    rtk_need_init = true;
    rtk_isvalid = false;
    first_lidar_time = -1.0;
    first_latitude = 0.0;
    first_longitude = 0.0;
    first_altitude = 0.0;
    origin_lat_sum = 0.0;
    origin_lon_sum = 0.0;
    origin_alt_sum = 0.0;
    origin_sample_count = 0;
    origin_sample_target = 20;
    origin_hacc_max = 1.0;
    origin_vacc_max = 2.0;
    rtk_window_length = 5;
    gpst_leap_seconds_offset_ = 17.84;
    gpst_time_bias_ = 0.0029;


    enable_quality_analysis_ = true;
    quality_analysis_count_ = 0;
    min_acceptable_score_ = 60.0;


}

RtkProcess::~RtkProcess() {}


double RtkProcess::gpst2time(uint32_t week, double tow)
{

    gnss_comm::gtime_t gps_time = gnss_comm::gpst2time(week, tow);

    double unix_timestamp = gnss_comm::time2sec(gps_time) - gpst_leap_seconds_offset_ + gpst_time_bias_;
    return unix_timestamp;
}

bool RtkProcess::unix2gpst(double unix_timestamp, uint32_t &week, double &tow) const
{
    if (!std::isfinite(unix_timestamp))
    {
        return false;
    }

    const double gps_sec = unix_timestamp + gpst_leap_seconds_offset_ - gpst_time_bias_;
    gnss_comm::gtime_t gps_time = gnss_comm::sec2time(gps_sec);
    week = 0;
    tow = gnss_comm::time2gpst(gps_time, &week);
    return std::isfinite(tow);
}


void RtkProcess::Rtk_init(const gnss_comm::GnssPVTSolnMsg::ConstPtr first_rtk_msg)
{
    if (first_rtk_msg->carr_soln != 2 || !first_rtk_msg->valid_fix || !first_rtk_msg->diff_soln ||
        !std::isfinite(first_rtk_msg->latitude) || !std::isfinite(first_rtk_msg->longitude) ||
        !std::isfinite(first_rtk_msg->altitude) || std::abs(first_rtk_msg->latitude) > 90.0 ||
        std::abs(first_rtk_msg->longitude) > 180.0)
    {
        return;
    }
    if (origin_hacc_max > 0.0 &&
        (!std::isfinite(first_rtk_msg->h_acc) || first_rtk_msg->h_acc > origin_hacc_max))
    {
        return;
    }
    if (origin_vacc_max > 0.0 &&
        (!std::isfinite(first_rtk_msg->v_acc) || first_rtk_msg->v_acc > origin_vacc_max))
    {
        return;
    }

    origin_lat_sum += first_rtk_msg->latitude;
    origin_lon_sum += first_rtk_msg->longitude;
    origin_alt_sum += first_rtk_msg->altitude;
    origin_sample_count++;

    if (origin_sample_target > 0 && origin_sample_count >= origin_sample_target)
    {
        first_latitude = origin_lat_sum / static_cast<double>(origin_sample_count);
        first_longitude = origin_lon_sum / static_cast<double>(origin_sample_count);
        first_altitude = origin_alt_sum / static_cast<double>(origin_sample_count);
        last_rtk_time_stamp = gpst2time(first_rtk_msg->time.week, first_rtk_msg->time.tow);

        if (!std::isfinite(first_altitude) || std::abs(first_latitude) > 90.0 || std::abs(first_longitude) > 180.0)
        {
            (void)0;
            return;
        }
        rtk_need_init = false;
        rtk_isvalid = true;
        (void)0;
    }
}


void RtkProcess::lla2xyz(double lat, double lon, double alt, V3D &xyz)
{


    Eigen::Vector3d lla_cur(lat, lon, alt);
    Eigen::Vector3d lla_first(first_latitude, first_longitude, first_altitude);

    Eigen::Vector3d ecef_cur = gnss_comm::geo2ecef(lla_cur);
    Eigen::Vector3d ecef_first = gnss_comm::geo2ecef(lla_first);

    Eigen::Vector3d ecef_diff = ecef_cur - ecef_first;

    xyz = gnss_comm::ecef2enu(lla_first, ecef_diff);
}

void RtkProcess::Process(MeasureGroup &meas,
                         deque<gnss_comm::GnssPVTSolnMsg::ConstPtr> &rtk_buffer,
                         double max_timestamp)
{
    gnss_comm::GnssPVTSolnMsg::ConstPtr msg;
    double unix_timestamp;

    if (rtk_buffer.empty()) return;


    if (rtk_need_init)
    {
        while (!rtk_buffer.empty())
        {
            msg = rtk_buffer.front();
            unix_timestamp = gpst2time(msg->time.week, msg->time.tow);
            if (unix_timestamp > max_timestamp)
            {
                break;
            }
            Rtk_init(msg);
            rtk_buffer.pop_front();
            if (!rtk_need_init)
            {
                break;
            }

        }

        if (rtk_need_init) return;
    }


    while (!rtk_buffer.empty())
    {
        msg = rtk_buffer.front();
        unix_timestamp = gpst2time(msg->time.week, msg->time.tow);
        if (unix_timestamp > max_timestamp)
        {
            break;
        }

        if (unix_timestamp < last_rtk_time_stamp)
        {
            (void)0;
            rtk_buffer.clear();
            valid_rtk_window.clear();
            last_rtk_time_stamp = 0.0;
            rtk_need_init = true;
            rtk_isvalid = false;
            origin_lat_sum = 0.0;
            origin_lon_sum = 0.0;
            origin_alt_sum = 0.0;
            origin_sample_count = 0;
            return;
        }

        bool should_accept = msg->carr_soln == 2 && msg->valid_fix && msg->diff_soln &&
            std::isfinite(msg->latitude) && std::isfinite(msg->longitude) && std::isfinite(msg->altitude) &&
            std::abs(msg->latitude) <= 90.0 && std::abs(msg->longitude) <= 180.0 &&
            std::isfinite(msg->h_acc) && msg->h_acc > 0.0 && std::isfinite(msg->v_acc) && msg->v_acc > 0.0;

        if (should_accept) {
            last_rtk_time_stamp = unix_timestamp;
            valid_rtk_window.push_back(msg);
        } else {
            valid_rtk_window.clear();
        }

        rtk_buffer.pop_front();
    }


    while(!valid_rtk_window.empty())
    {
        msg = valid_rtk_window.front();
        unix_timestamp = gpst2time(msg->time.week, msg->time.tow);

        V3D xyz;
        Timepos rtk_time_pos;
        lla2xyz(msg->latitude, msg->longitude, msg->altitude, xyz);
        rtk_time_pos.pos = xyz;
        rtk_time_pos.timestamp = gpst2time(msg->time.week, msg->time.tow);
        rtk_time_pos.fix_type = static_cast<int>(msg->fix_type);
        rtk_time_pos.h_acc = msg->h_acc;
        rtk_time_pos.v_acc = msg->v_acc;
        rtk_time_pos.vel_valid = false;
        if (std::isfinite(msg->vel_n) && std::isfinite(msg->vel_e) && std::isfinite(msg->vel_d))
        {
            Eigen::Vector3d vel_enu(msg->vel_e, msg->vel_n, -msg->vel_d);
            rtk_time_pos.vel = vel_enu;
            rtk_time_pos.vel_acc = msg->vel_acc;
            rtk_time_pos.vel_valid = true;
        }
        meas.rtk_pos.push_back(rtk_time_pos);


        valid_rtk_window.pop_front();
    }

    return;
}


RTKQualityAnalyzer::QualityMetrics RtkProcess::analyze_rtk_message_quality(const gnss_comm::GnssPVTSolnMsg::ConstPtr& rtk_msg)
{
    auto quality = quality_analyzer_.analyze_rtk_quality(*rtk_msg);


    quality_analysis_count_++;


    if (quality_analysis_count_ % 10 == 0) {
        (void)0;
    }

    return quality;
}

bool RtkProcess::should_accept_rtk_message(const RTKQualityAnalyzer::QualityMetrics& quality)
{

    if (quality.overall_level == RTKQualityAnalyzer::QualityLevel::INVALID) {
        (void)0;
        return false;
    }


    if (quality.overall_score < min_acceptable_score_) {
        (void)0;
        return false;
    }


    if (quality.carrier_quality == RTKQualityAnalyzer::QualityLevel::INVALID ||
        quality.carrier_quality == RTKQualityAnalyzer::QualityLevel::BAD) {
        (void)0;
        return false;
    }


    if (quality.geometric_quality == RTKQualityAnalyzer::QualityLevel::INVALID) {
        (void)0;
        return false;
    }


    if (quality.carr_soln == 2) {
        (void)0;
        return true;
    }


    (void)0;
    return false;
}

void RtkProcess::log_quality_statistics()
{
    if (quality_analysis_count_ == 0) return;

    static int last_log_count = 0;
    int analyzed_since_last = quality_analysis_count_ - last_log_count;


    last_log_count = quality_analysis_count_;
}

void RtkProcess::load_quality_config_from_ros_params()
{
    ros::NodeHandle nh;

    nh.param("rtk/gpst_leap_seconds", gpst_leap_seconds_offset_, gpst_leap_seconds_offset_);
    nh.param("rtk/gpst_time_bias", gpst_time_bias_, gpst_time_bias_);
    (void)0;


    nh.param("rtk_quality/enable_quality_analysis", enable_quality_analysis_, true);
    nh.param("rtk_quality/min_acceptable_score", min_acceptable_score_, 60.0);


    double excellent_thresh = 90.0, good_thresh = 75.0, acceptable_thresh = 60.0;
    double poor_thresh = 40.0, bad_thresh = 20.0;

    nh.param("rtk_quality/excellent_threshold", excellent_thresh, 90.0);
    nh.param("rtk_quality/good_threshold", good_thresh, 75.0);
    nh.param("rtk_quality/acceptable_threshold", acceptable_thresh, 60.0);
    nh.param("rtk_quality/poor_threshold", poor_thresh, 40.0);
    nh.param("rtk_quality/bad_threshold", bad_thresh, 20.0);


    quality_analyzer_.set_quality_thresholds(excellent_thresh, good_thresh,
                                             acceptable_thresh, poor_thresh, bad_thresh);


    if (enable_quality_analysis_) {
        (void)0;
        (void)0;
        (void)0;


        bool accept_rtk_fixed = true;
        bool accept_rtk_float_high_quality = true;
        double rtk_float_threshold = 75.0;
        int min_sat_count = 6;
        double max_pdop = 8.0;

        nh.param("rtk_quality/accept_rtk_fixed", accept_rtk_fixed, true);
        nh.param("rtk_quality/accept_rtk_float_high_quality", accept_rtk_float_high_quality, true);
        nh.param("rtk_quality/accept_rtk_float_threshold", rtk_float_threshold, 75.0);
        nh.param("rtk_quality/min_satellite_count", min_sat_count, 6);
        nh.param("rtk_quality/max_pdop_threshold", max_pdop, 8.0);

        (void)0;
        (void)0;
    } else {
        (void)0;
    }
}
