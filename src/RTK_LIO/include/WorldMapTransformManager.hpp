#ifndef WORLD_MAP_TRANSFORM_MANAGER_HPP
#define WORLD_MAP_TRANSFORM_MANAGER_HPP

#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <common_lib.h>
#include <wahba/alignment.h>
#include <ros/ros.h>


class WorldMapTransformManager
{
public:
    WorldMapTransformManager();
    ~WorldMapTransformManager();


    void updateTransform(const Transformation& transform,
                         double confidence = 1.0,
                         double timestamp = 0.0,
                         bool suppress_log = false);


    Eigen::Vector3d transformPositionToWorld(const Eigen::Vector3d& pos_map) const;


    Eigen::Matrix3d transformRotationToWorld(const Eigen::Matrix3d& rot_map) const;


    Eigen::Quaterniond transformQuaternionToWorld(const Eigen::Quaterniond& quat_map) const;


    void transformPoseToWorld(const Eigen::Vector3d& pos_map, const Eigen::Quaterniond& quat_map,
                             Eigen::Vector3d& pos_world, Eigen::Quaterniond& quat_world) const;


    bool isTransformValid() const { return transform_.valid && confidence_ > min_confidence_; }


    double getConfidence() const { return confidence_; }


    double getLastUpdateTime() const { return last_update_time_; }


    std::string getTransformInfo() const;


    void setMinConfidence(double min_conf) { min_confidence_ = min_conf; }


    const Transformation& getTransform() const { return transform_; }

private:
    Transformation transform_;
    double confidence_;
    double last_update_time_;
    double min_confidence_;


    Eigen::Matrix4d getTransformMatrix() const;
};

#endif
