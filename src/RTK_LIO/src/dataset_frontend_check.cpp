#include "preprocess.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>


int main(int argc, char** argv)
{
  if (argc != 8) {
    (void)0;
    return 2;
  }
  ros::Time::init();
  Preprocess pre;
  pre.lidar_type = std::stoi(argv[2]);
  pre.N_SCANS = std::stoi(argv[3]);
  pre.SCAN_RATE = std::stoi(argv[4]);
  pre.max_scan_duration = std::stod(argv[5]);
  pre.point_filter_num = std::stoi(argv[6]);
  pre.blind = std::stod(argv[7]);
  pre.feature_enabled = false;
  size_t scans = 0, nonempty = 0, imu_count = 0, points = 0;
  double last_imu = -1, last_scan = -1, max_scan = 0;
  bool finite = true, monotonic = true, bounded = true;
  std::vector<double> acceleration_norms;
  try {
    rosbag::Bag bag(argv[1], rosbag::bagmode::Read);
    rosbag::View view(bag);
    for (const auto& record : view) {
      if (record.getTopic() == "/dataset/imu") {
        auto imu = record.instantiate<sensor_msgs::Imu>();
        if (!imu) throw std::runtime_error("Wrong IMU message type");
        const auto& a = imu->linear_acceleration;
        const auto& w = imu->angular_velocity;
        double norm = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
        finite &= std::isfinite(norm) && std::isfinite(w.x) && std::isfinite(w.y) && std::isfinite(w.z);
        const double stamp = imu->header.stamp.toSec();
        monotonic &= stamp > last_imu;
        last_imu = stamp;
        acceleration_norms.push_back(norm);
        ++imu_count;
      } else if (record.getTopic() == "/dataset/lidar") {
        PointCloudXYZI::Ptr output(new PointCloudXYZI);
        double stamp = 0;
        if (pre.lidar_type == AVIA) {
          auto cloud = record.instantiate<livox_ros_driver::CustomMsg>();
          if (!cloud) throw std::runtime_error("Wrong CustomMsg type");
          stamp = cloud->header.stamp.toSec();
          pre.process(cloud, output);
        } else {
          auto cloud = record.instantiate<sensor_msgs::PointCloud2>();
          if (!cloud) throw std::runtime_error("Wrong PointCloud2 type");
          stamp = cloud->header.stamp.toSec();
          pre.process(cloud, output);
        }
        monotonic &= stamp > last_scan;
        last_scan = stamp;
        ++scans;
        nonempty += !output->empty();
        points += output->size();
        for (const auto& p : *output) {
          finite &= std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::isfinite(p.curvature);
          bounded &= p.curvature >= 0 && p.curvature <= pre.max_scan_duration * 1000;
          max_scan = std::max(max_scan, p.curvature * 0.001);
        }
      } else {
        throw std::runtime_error("Unexpected topic in isolated sensor bag: " + record.getTopic());
      }
    }
  } catch (const std::exception& e) {
    (void)0;
    return 1;
  }
  std::sort(acceleration_norms.begin(), acceleration_norms.end());
  const double median_acc = acceleration_norms.empty() ? 0 : acceleration_norms[acceleration_norms.size()/2];
  const bool passed = finite && monotonic && bounded && scans > 2 && nonempty == scans && imu_count > 20;
  (void)0;
  return passed ? 0 : 1;
}
