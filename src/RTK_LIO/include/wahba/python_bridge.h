#pragma once

#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <common_lib.h>
#include <pybind11/embed.h>

class WahbaPythonBridge {
public:
    static WahbaPythonBridge& instance();

    bool available() const { return available_; }

    bool compute_bspline_velocities(const std::vector<Timepos>& points,
                                    std::vector<Eigen::Vector3d>& velocities,
                                    std::vector<Eigen::Vector3d>* spline_positions,
                                    const std::string& label,
                                    double smoothing_factor,
                                    std::string* error = nullptr);

    bool tls_refine(const std::vector<Eigen::Vector3d>& lio_points,
                    const std::vector<Eigen::Vector3d>& rtk_points,
                    const Eigen::Matrix3d& initial_rotation,
                    double sigma,
                    double c_bar_squared,
                    int max_iter,
                    bool use_adaptive_threshold,
                    double adaptive_scale,
                    Eigen::Matrix3d& refined_rotation,
                    std::vector<bool>& inlier_mask,
                    std::map<std::string, double>* stats = nullptr,
                    std::string* error = nullptr);

private:
    WahbaPythonBridge();
    bool initialize();
    bool ensure_module();
    bool append_module_path();
    bool fetch_attributes();

    pybind11::scoped_interpreter interpreter_;
    pybind11::object module_;
    pybind11::object bspline_func_;
    pybind11::object tls_func_;
    bool available_ = false;
    bool attempted_init_ = false;
};
