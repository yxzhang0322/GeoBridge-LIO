#pragma once
#include <Eigen/Core>

struct RtkHeightObservation
{
    double residual;
    Eigen::Matrix<double, 1, 6> jacobian;
};

inline RtkHeightObservation rtk_height_observation(const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity, const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation, double scale, double age, double height)
{
    RtkHeightObservation observation;

    observation.residual = (scale * rotation * (position - age * velocity) + translation).z() - height;
    observation.jacobian.head<3>() = scale * rotation.row(2);
    observation.jacobian.tail<3>() = -age * scale * rotation.row(2);
    return observation;
}
