#ifndef BSPLINE_H
#define BSPLINE_H

#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <Eigen/Dense>
#include <iostream>
#include "alignment.h"

class BSpline {
public:
    BSpline();
    ~BSpline();

    static constexpr uint64_t INVALID_RECORD_ID = std::numeric_limits<uint64_t>::max();


    static void configure_debug_logging(bool enabled, const std::string& directory = "");


    static void set_smoothing_factor(double factor);


    static bool debug_logging_enabled();


    static double get_smoothing_factor();


    static std::vector<Eigen::Vector3d> compute_velocities(
        const std::vector<TrajectoryPoint>& points,
        const std::string& debug_label = "",
        uint64_t* debug_record_id = nullptr);


    static void finalize_debug_record(uint64_t record_id, bool keep_record);

private:

    static double F03(double t);
    static double F13(double t);
    static double F23(double t);
    static double F33(double t);


    static double dF03(double t);
    static double dF13(double t);
    static double dF23(double t);
    static double dF33(double t);

    static double clamp_smoothing_factor(double factor);
};

#endif
