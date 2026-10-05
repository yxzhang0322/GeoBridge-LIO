#pragma once

#include <deque>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pgo_sw {

struct Pose {
    double t = 0.0;
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct RtkMeas {
    double t = 0.0;
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    double sigma_xy = 0.05;
    double sigma_z = 0.10;
    bool valid = true;
};

class SlidingWindowBackend {
public:
    SlidingWindowBackend();

    void setWindowSize(size_t size);
    void setRtkMatchMaxDt(double dt);
    void setOdomSigmas(double sigma_pos, double sigma_rot);
    void setDefaultRtkSigmas(double sigma_xy, double sigma_z);

    void addLioPose(const Pose& pose);
    void addRtk(const RtkMeas& meas);

    bool optimize();
    bool getLatestOptimized(Pose* out) const;
    void clearRtk();
    void clear();

    const std::deque<Pose>& window() const { return window_; }
    size_t rtkBufferSize() const { return rtk_buffer_.size(); }
    size_t optimizedSize() const { return optimized_.size(); }
    size_t latestRtkFactorCount() const { return latest_rtk_factor_count_; }
    double latestRtkSigmaScale() const { return latest_rtk_sigma_scale_; }

private:
    size_t window_size_ = 15;
    double rtk_match_max_dt_ = 0.1;
    double odom_sigma_pos_ = 0.1;
    double odom_sigma_rot_ = 0.01;
    double rtk_sigma_xy_default_ = 0.05;
    double rtk_sigma_z_default_ = 0.10;

    std::deque<Pose> window_;
    std::deque<Pose> optimized_;
    std::deque<RtkMeas> rtk_buffer_;
    size_t latest_rtk_factor_count_ = 0;
    double latest_rtk_sigma_scale_ = 1.0;
};

}
