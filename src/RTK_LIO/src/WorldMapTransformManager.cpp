#include "WorldMapTransformManager.hpp"
#include <sstream>
#include <iomanip>

WorldMapTransformManager::WorldMapTransformManager()
    : confidence_(0.0)
    , last_update_time_(0.0)
    , min_confidence_(0.5)
{

    transform_.R = Eigen::Matrix3d::Identity();
    transform_.t = Eigen::Vector3d::Zero();
    transform_.s = 1.0;
    transform_.valid = false;
}

WorldMapTransformManager::~WorldMapTransformManager()
{

}

void WorldMapTransformManager::updateTransform(const Transformation& transform,
                                              double confidence,
                                              double timestamp,
                                              bool suppress_log)
{

    transform_ = transform;
    confidence_ = confidence;
    last_update_time_ = (timestamp > 0.0) ? timestamp : ros::Time::now().toSec();

    if (!suppress_log)
    {


    }
}

Eigen::Vector3d WorldMapTransformManager::transformPositionToWorld(const Eigen::Vector3d& pos_map) const
{
    if (!isTransformValid()) {
        (void)0;
        return pos_map;
    }


    return transform_.s * transform_.R * pos_map + transform_.t;
}

Eigen::Matrix3d WorldMapTransformManager::transformRotationToWorld(const Eigen::Matrix3d& rot_map) const
{
    if (!isTransformValid()) {
        (void)0;
        return rot_map;
    }


    return transform_.R * rot_map;
}

Eigen::Quaterniond WorldMapTransformManager::transformQuaternionToWorld(const Eigen::Quaterniond& quat_map) const
{
    if (!isTransformValid()) {
        (void)0;
        return quat_map;
    }


    Eigen::Matrix3d rot_map = quat_map.toRotationMatrix();
    Eigen::Matrix3d rot_world = transformRotationToWorld(rot_map);
    return Eigen::Quaterniond(rot_world);
}

void WorldMapTransformManager::transformPoseToWorld(const Eigen::Vector3d& pos_map,
                                                   const Eigen::Quaterniond& quat_map,
                                                   Eigen::Vector3d& pos_world,
                                                   Eigen::Quaterniond& quat_world) const
{
    pos_world = transformPositionToWorld(pos_map);
    quat_world = transformQuaternionToWorld(quat_map);
}

std::string WorldMapTransformManager::getTransformInfo() const
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "WorldMapTransform Info:\n";
    ss << "  Valid: " << (transform_.valid ? "true" : "false") << "\n";
    ss << "  Confidence: " << confidence_ << "\n";
    ss << "  Scale: " << transform_.s << "\n";
    ss << "  Translation: [" << transform_.t.transpose() << "]\n";
    ss << "  Rotation Matrix:\n";
    for (int i = 0; i < 3; ++i) {
        ss << "    [";
        for (int j = 0; j < 3; ++j) {
            ss << std::setw(8) << transform_.R(i, j);
            if (j < 2) ss << ", ";
        }
        ss << "]\n";
    }
    ss << "  Last Update: " << std::setprecision(6) << last_update_time_ << "s";
    return ss.str();
}

Eigen::Matrix4d WorldMapTransformManager::getTransformMatrix() const
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    if (isTransformValid()) {


        T.block<3,3>(0,0) = transform_.s * transform_.R;
        T.block<3,1>(0,3) = transform_.t;
    }

    return T;
}
