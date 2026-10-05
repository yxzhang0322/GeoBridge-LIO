#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>


class RTKAntennaCalibrationTool {
public:
    struct CalibrationSample {
        double timestamp;
        Eigen::Vector3d pos_rtk_world;
        Eigen::Vector3d pos_imu_world;
        Eigen::Matrix3d rotation_world;
        bool is_valid;
    };

    RTKAntennaCalibrationTool(ros::NodeHandle& nh) : nh_(nh), sample_count_(0), calibration_mode_(false) {

        rtk_sub_ = nh_.subscribe("/aft_mapped_to_world", 10,
                                &RTKAntennaCalibrationTool::rtkOdometryCallback, this);
        imu_sub_ = nh_.subscribe("/Odometry", 10,
                                &RTKAntennaCalibrationTool::imuOdometryCallback, this);


        calibration_pub_ = nh_.advertise<std_msgs::String>("/rtk_antenna_calibration_result", 1);

        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;

        printStatus();
    }

    void rtkOdometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        rtk_odom_ = *msg;
        rtk_received_ = true;
    }

    void imuOdometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        imu_odom_ = *msg;
        imu_received_ = true;
    }

    void run() {
        ros::Rate rate(10);
        std::string input;

        while (ros::ok()) {
            ros::spinOnce();


            if (std::cin.rdbuf()->in_avail() > 0) {
                std::getline(std::cin, input);
                processCommand(input);
            }

            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber rtk_sub_;
    ros::Subscriber imu_sub_;
    ros::Publisher calibration_pub_;

    nav_msgs::Odometry rtk_odom_;
    nav_msgs::Odometry imu_odom_;
    bool rtk_received_ = false;
    bool imu_received_ = false;

    std::vector<CalibrationSample> samples_;
    int sample_count_;
    bool calibration_mode_;

    void processCommand(const std::string& cmd) {
        if (cmd == "s" || cmd == "start") {
            startCalibration();
        } else if (cmd == "c" || cmd == "capture") {
            captureSample();
        } else if (cmd == "f" || cmd == "finish") {
            finishCalibration();
        } else if (cmd == "r" || cmd == "reset") {
            resetSamples();
        } else if (cmd == "q" || cmd == "quit") {
            (void)0;
            ros::shutdown();
        } else if (cmd == "h" || cmd == "help") {
            printHelp();
        } else {
            (void)0;
        }
    }

    void startCalibration() {
        calibration_mode_ = true;
        (void)0;
        (void)0;
    }

    void captureSample() {
        if (!calibration_mode_) {
            (void)0;
            return;
        }

        if (!rtk_received_ || !imu_received_) {
            (void)0;
            return;
        }

        CalibrationSample sample;
        sample.timestamp = ros::Time::now().toSec();
        sample.is_valid = true;


        sample.pos_rtk_world = Eigen::Vector3d(
            rtk_odom_.pose.pose.position.x,
            rtk_odom_.pose.pose.position.y,
            rtk_odom_.pose.pose.position.z
        );


        sample.pos_imu_world = Eigen::Vector3d(
            imu_odom_.pose.pose.position.x,
            imu_odom_.pose.pose.position.y,
            imu_odom_.pose.pose.position.z
        );


        Eigen::Quaterniond q(
            imu_odom_.pose.pose.orientation.w,
            imu_odom_.pose.pose.orientation.x,
            imu_odom_.pose.pose.orientation.y,
            imu_odom_.pose.pose.orientation.z
        );
        sample.rotation_world = q.toRotationMatrix();

        samples_.push_back(sample);
        sample_count_++;

        (void)0;
        (void)0;
        (void)0;

        Eigen::Vector3d diff = sample.pos_rtk_world - sample.pos_imu_world;
        (void)0;

        printStatus();
    }

    void finishCalibration() {
        if (samples_.size() < 3) {
            (void)0;
            return;
        }

        (void)0;

        Eigen::Vector3d antenna_offset = performStaticCalibration();


        (void)0;
        (void)0;
        (void)0;
        (void)0;


        double rmse = validateCalibration(antenna_offset);
        (void)0;

        if (rmse < 0.05) {
            (void)0;
        } else if (rmse < 0.10) {
            (void)0;
        } else if (rmse < 0.20) {
            (void)0;
        } else {
            (void)0;
            (void)0;
        }


        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;


        saveCalibrationResults(antenna_offset, rmse);


        std_msgs::String result_msg;
        result_msg.data = "Calibration completed. Antenna offset: [" +
                         std::to_string(antenna_offset.x()) + ", " +
                         std::to_string(antenna_offset.y()) + ", " +
                         std::to_string(antenna_offset.z()) + "] m, RMSE: " +
                         std::to_string(rmse) + " m";
        calibration_pub_.publish(result_msg);

        calibration_mode_ = false;
    }

    Eigen::Vector3d performStaticCalibration() {
        const int n = samples_.size();
        Eigen::MatrixXd A(3 * n, 3);
        Eigen::VectorXd b(3 * n);

        for (int i = 0; i < n; ++i) {
            const auto& sample = samples_[i];
            A.block<3,3>(3*i, 0) = sample.rotation_world;
            b.segment<3>(3*i) = sample.pos_rtk_world - sample.pos_imu_world;
        }


        return A.colPivHouseholderQr().solve(b);
    }

    double validateCalibration(const Eigen::Vector3d& offset) {
        if (samples_.empty()) return 0.0;

        double total_error = 0.0;
        for (const auto& sample : samples_) {
            Eigen::Vector3d predicted_rtk = sample.pos_imu_world + sample.rotation_world * offset;
            Eigen::Vector3d error = sample.pos_rtk_world - predicted_rtk;
            total_error += error.squaredNorm();
        }

        return std::sqrt(total_error / samples_.size());
    }

    void resetSamples() {
        samples_.clear();
        sample_count_ = 0;
        calibration_mode_ = false;
        (void)0;
        printStatus();
    }

    void saveCalibrationResults(const Eigen::Vector3d& offset, double rmse) {
        (void)offset;
        (void)rmse;
    }

    void printStatus() {
        (void)0;
        (void)0;
        (void)0;
        (void)0;

        if (sample_count_ >= 3) {
            (void)0;
        } else {
            (void)0;
        }
        (void)0;
    }

    void printHelp() {
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
        (void)0;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "rtk_antenna_calibration_tool");
    ros::NodeHandle nh;

    RTKAntennaCalibrationTool tool(nh);
    tool.run();

    return 0;
}
