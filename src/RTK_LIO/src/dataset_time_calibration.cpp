#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <fstream>
#include <iomanip>

int main(int argc, char** argv)
{
    if (argc != 6) return 2;
    if (std::ifstream(argv[4]).good() || std::ifstream(argv[5]).good()) return 2;
    ros::Time::init();
    const double first = std::stod(argv[2]), last = std::stod(argv[3]);
    std::ofstream rotations(argv[4]), imu(argv[5]);
    rotations << "start,finish,omega_z,fitness,converged,rx,ry,rz,tx,ty,tz\n" << std::setprecision(15);
    imu << "timestamp,omega_z,omega_x,omega_y,acc_x,acc_y,acc_z\n" << std::setprecision(15);
    using Cloud = pcl::PointCloud<pcl::PointXYZ>;
    Cloud::Ptr previous;
    double previous_stamp = 0.0;
    rosbag::Bag bag(argv[1],rosbag::bagmode::Read);
    rosbag::View view(bag,rosbag::TopicQuery(std::vector<std::string>{"/dataset/lidar","/dataset/imu"}),
        ros::Time(first-1),ros::Time(last+3));
    for (const auto& record : view)
    {
        if (auto msg = record.instantiate<sensor_msgs::Imu>())
        {
            imu << msg->header.stamp.toSec() << "," << msg->angular_velocity.z << ","
                << msg->angular_velocity.x << "," << msg->angular_velocity.y << ","
                << msg->linear_acceleration.x << "," << msg->linear_acceleration.y << ","
                << msg->linear_acceleration.z << "\n";
            continue;
        }
        auto msg = record.instantiate<sensor_msgs::PointCloud2>();
        if (!msg || msg->header.stamp.toSec()<first || msg->header.stamp.toSec()>last) continue;
        Cloud::Ptr raw(new Cloud), filtered(new Cloud);
        pcl::fromROSMsg(*msg,*raw);
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setLeafSize(.8f,.8f,.8f); voxel.setInputCloud(raw); voxel.filter(*filtered);
        const double stamp = msg->header.stamp.toSec()+.05;
        if (previous)
        {
            pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
            icp.setMaximumIterations(30); icp.setMaxCorrespondenceDistance(3.0);
            icp.setTransformationEpsilon(1e-7); icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setInputSource(filtered); icp.setInputTarget(previous);
            Cloud aligned; icp.align(aligned);
            const auto tf = icp.getFinalTransformation();
            const Eigen::AngleAxisf aa(tf.block<3,3>(0,0));
            const Eigen::Vector3f rv=aa.angle()*aa.axis();
            rotations << previous_stamp << "," << stamp << ","
                << std::atan2(tf(1,0),tf(0,0))/(stamp-previous_stamp) << ","
                << icp.getFitnessScore() << "," << icp.hasConverged() << ","
                << rv.x() << "," << rv.y() << "," << rv.z() << ","
                << tf(0,3) << "," << tf(1,3) << "," << tf(2,3) << "\n";
        }
        previous = filtered; previous_stamp = stamp;
    }
    return 0;
}
