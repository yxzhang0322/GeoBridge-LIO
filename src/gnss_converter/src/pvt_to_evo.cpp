
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/NavSatFix.h>
#include <gnss_comm/GnssPVTSolnMsg.h>
#include <gnss_comm/GnssTimeMsg.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <ros/package.h>


#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>


const double GPS_EPOCH_OFFSET_FROM_UNIX_EPOCH = 315964800.0;

const int LEAP_SECONDS = 18;


const double EARTH_RADIUS_METERS = 6371000.0;
const double DEG_TO_RAD = M_PI / 180.0;


ros::Time gpsTimeToRosTime(uint32_t week, double tow) {
    double total_gps_seconds_from_gps_epoch = static_cast<double>(week) * 7.0 * 24.0 * 60.0 * 60.0 + tow;
    double total_gps_seconds_from_unix_epoch = total_gps_seconds_from_gps_epoch + GPS_EPOCH_OFFSET_FROM_UNIX_EPOCH;
    double unix_seconds_utc_double = total_gps_seconds_from_unix_epoch - static_cast<double>(LEAP_SECONDS);

    if (unix_seconds_utc_double < 0) {
        (void)0;
        return ros::Time(0,0);
    }
    uint32_t sec = static_cast<uint32_t>(unix_seconds_utc_double);
    uint32_t nsec = static_cast<uint32_t>((unix_seconds_utc_double - static_cast<double>(sec)) * 1e9);
    return ros::Time(sec, nsec);
}


class GnssConverterNode {
public:
    GnssConverterNode() : nh_("~"), origin_set_(false) {

        nh_.param<std::string>("input_topic", input_topic_name_, "/your_gnss_pvt_topic");
        nh_.param<std::string>("output_navsatfix_topic", output_navsatfix_topic_name_, "/gnss_fix");
        nh_.param<std::string>("output_txt_filename", output_txt_filename_, "gnss_trajectory_relative.txt");
        nh_.param<double>("min_distance", min_distance_, 0.1);
        nh_.param<std::string>("output_path_topic_name", output_path_topic_name_, "/gnss_converter/path");

        nh_.param<std::string>("fixed_frame_id", fixed_frame_id_, "camera_init");


        std::string package_path = ros::package::getPath("gnss_converter");
        if (package_path.empty()) {
            (void)0;
            ros::shutdown();
            return;
        }
        output_txt_filepath_ = package_path + "/data/" + output_txt_filename_;


        pvt_sub_ = nh_.subscribe(input_topic_name_, 10, &GnssConverterNode::pvtCallback, this);
        navsat_pub_ = nh_.advertise<sensor_msgs::NavSatFix>(output_navsatfix_topic_name_, 10);
        path_pub_ = nh_.advertise<nav_msgs::Path>(output_path_topic_name_, 10);
        vision_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gnss_converter/odom", 100);


        path_msg_.header.frame_id = fixed_frame_id_;


        txt_file_out_.open(output_txt_filepath_, std::ios::out | std::ios::trunc);
        if (!txt_file_out_.is_open()) {
            (void)0;
        } else {
            (void)0;
            txt_file_out_ << "#timestamp rel_east(x) rel_north(y) rel_up(z) qx qy qz qw" << std::endl;
        }

        (void)0;
    }

    ~GnssConverterNode() {
        if (txt_file_out_.is_open()) {
            txt_file_out_.close();
        }
    }

    void pvtCallback(const gnss_comm::GnssPVTSolnMsg::ConstPtr& msg) {
        ros::Time current_gnss_ros_time;

        current_gnss_ros_time = gpsTimeToRosTime(msg->time.week, msg->time.tow);

        if (current_gnss_ros_time < ros::Time(1.0) && msg->valid_fix) {
             (void)0;
        }


        if (!msg->valid_fix) {
            return;
        }


        sensor_msgs::NavSatFix navsat_msg;
        navsat_msg.header.stamp = ros::Time::now();
        navsat_msg.header.frame_id = "gnss_link";

        if (msg->fix_type == 0 || msg->fix_type == 5) {
            navsat_msg.status.status = sensor_msgs::NavSatStatus::STATUS_NO_FIX;
        } else {
            if (msg->diff_soln || msg->carr_soln > 0) {
                navsat_msg.status.status = sensor_msgs::NavSatStatus::STATUS_GBAS_FIX;
            } else {
                navsat_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
            }
        }
        navsat_msg.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;

        navsat_msg.latitude = msg->latitude;
        navsat_msg.longitude = msg->longitude;
        navsat_msg.altitude = msg->altitude;

        navsat_msg.position_covariance_type = sensor_msgs::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
        navsat_msg.position_covariance[0] = msg->h_acc * msg->h_acc;
        navsat_msg.position_covariance[4] = msg->h_acc * msg->h_acc;
        navsat_msg.position_covariance[8] = msg->v_acc * msg->v_acc;

        navsat_pub_.publish(navsat_msg);


        double current_lon_deg = msg->longitude;
        double current_lat_deg = msg->latitude;
        double current_alt_m = msg->altitude;

        double last_pose[3];
        double current_pose[3];

        if (!origin_set_) {
            origin_lon_deg_ = current_lon_deg;
            origin_lat_deg_ = current_lat_deg;
            origin_alt_m_ = current_alt_m;
            origin_lat_rad_ = origin_lat_deg_ * DEG_TO_RAD;
            origin_set_ = true;
            (void)0;
            last_pose[0] = 0.0;
            last_pose[1] = 0.0;
            last_pose[2] = 0.0;
        }


        double current_lon_rad = current_lon_deg * DEG_TO_RAD;
        double current_lat_rad = current_lat_deg * DEG_TO_RAD;
        double delta_lon_rad = current_lon_rad - (origin_lon_deg_ * DEG_TO_RAD);
        double delta_lat_rad = current_lat_rad - origin_lat_rad_;

        double rel_east_m = delta_lon_rad * EARTH_RADIUS_METERS * std::cos(origin_lat_rad_);
        double rel_north_m = delta_lat_rad * EARTH_RADIUS_METERS;
        double rel_up_m = current_alt_m - origin_alt_m_;


        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header.stamp = current_gnss_ros_time;
        transform_stamped.header.frame_id = fixed_frame_id_;
        transform_stamped.child_frame_id = "gnss_link";
        transform_stamped.transform.translation.x = rel_east_m;
        transform_stamped.transform.translation.y = rel_north_m;
        transform_stamped.transform.translation.z = rel_up_m;

        tf2::Quaternion q;
        q.setRPY(0, 0, 0);
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();
        tf_broadcaster_.sendTransform(transform_stamped);


        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = current_gnss_ros_time;
        pose_stamped.header.frame_id = fixed_frame_id_;
        pose_stamped.pose.position.x = rel_east_m;
        pose_stamped.pose.position.y = rel_north_m;
        pose_stamped.pose.position.z = rel_up_m;
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();


        path_msg_.poses.push_back(pose_stamped);
        path_msg_.header.stamp = current_gnss_ros_time;
        path_pub_.publish(path_msg_);


        vision_.pose.position.x = rel_east_m;
        vision_.pose.position.y = rel_north_m;
        vision_.pose.position.z = rel_up_m;

        vision_.pose.orientation.x = 0;
        vision_.pose.orientation.y = 0;
        vision_.pose.orientation.z = 0;
        vision_.pose.orientation.w = 1;

        vision_.header.stamp = ros::Time::now();
        vision_pub_.publish(vision_);


        current_pose [0] = rel_east_m;
        current_pose [1] = rel_north_m;
        current_pose [2] = rel_up_m;

        double range = sqrt(pow((current_pose[0] - last_pose[0]), 2) + pow((current_pose[1] - last_pose[1]), 2) + pow((current_pose[3] - last_pose[3]), 2));
        if (txt_file_out_.is_open() && range >= min_distance_) {
            txt_file_out_ << std::fixed << std::setprecision(9) << current_gnss_ros_time.toSec() << " "
                          << std::fixed << std::setprecision(4) << rel_east_m << " "
                          << std::fixed << std::setprecision(4) << rel_north_m << " "
                          << std::fixed << std::setprecision(4) << rel_up_m << " "
                          << std::fixed << std::setprecision(4) << q.x() << " "
                          << std::fixed << std::setprecision(4) << q.y() << " "
                          << std::fixed << std::setprecision(4) << q.z() << " "
                          << std::fixed << std::setprecision(4) << q.w()
                          << std::endl;
            last_pose[0] = current_pose[0];
            last_pose[1] = current_pose[1];
            last_pose[2] = current_pose[2];
        }

    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber pvt_sub_;
    ros::Publisher navsat_pub_;
    ros::Publisher path_pub_;
    ros::Publisher vision_pub_;
    std::ofstream txt_file_out_;
    double min_distance_;


    tf2_ros::TransformBroadcaster tf_broadcaster_;

    nav_msgs::Path path_msg_;
    geometry_msgs::PoseStamped vision_;

    std::string input_topic_name_;
    std::string output_navsatfix_topic_name_;
    std::string output_txt_filename_;
    std::string output_txt_filepath_;
    std::string output_path_topic_name_;

    std::string fixed_frame_id_;

    bool origin_set_;
    double origin_lon_deg_;
    double origin_lat_deg_;
    double origin_alt_m_;
    double origin_lat_rad_;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "gnss_pvt_converter_node");
    GnssConverterNode converter_node;
    ros::spin();
    return 0;
}
