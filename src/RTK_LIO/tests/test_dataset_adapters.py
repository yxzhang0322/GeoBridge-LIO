import math
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Imu, PointField
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from datasets.adapters import azimuth_cloud, cloud_array, increment_imu, normalized_imu, shifted_stamp
from datasets.reference import classify_pvt, pseudo_position
from public_datasets import config_for, ros_process_environment, smoke_log_status
from argparse import Namespace
from datasets.novatel import bestpos_pvt, epoch, ecef, enu_rotation

def config(profile='velodyne32'):
    return config_for(Namespace(sensor_profile=profile, dataset_config=None, start=0, duration=30))

class AdaptersTest(unittest.TestCase):

    def test_novatel_measurement_contract(self):
        msg = Namespace(header=Namespace(gps_week=2036, gps_week_seconds=376657000), position_type=50, solution_status=0, latitude=22.3, longitude=114.1, altitude=16.0, undulation=-2.2, latitude_std=0.03, longitude_std=0.04, altitude_std=0.1, sol_svs=10, diff_age=2.0)
        cfg = dict(fixed_types=[50, 56], max_diff_age_sec=10, max_h_std_m=1, max_v_std_m=2)
        pvt = bestpos_pvt(msg, cfg)
        self.assertEqual(pvt.time.tow, 376657.0)
        self.assertAlmostEqual(pvt.altitude, 13.8)
        self.assertAlmostEqual(pvt.h_acc, 0.05)
        self.assertEqual(pvt.carr_soln, 2)
        self.assertTrue(math.isnan(pvt.vel_n))
        self.assertEqual(epoch(msg), 1547714239.0)
        msg.position_type = 19
        self.assertEqual(bestpos_pvt(msg, cfg).carr_soln, 0)
        msg.position_type = 50
        msg.solution_status = 1
        self.assertFalse(bestpos_pvt(msg, cfg).valid_fix)
        msg.solution_status = 0
        msg.latitude_std = -0.03
        self.assertFalse(bestpos_pvt(msg, cfg).valid_fix)

    def test_reference_ecef_enu_axes(self):
        np.testing.assert_allclose(ecef(0, 0, 0), [6378137, 0, 0], atol=1e-08)
        np.testing.assert_allclose(enu_rotation(0, 0) @ [0, 1, 0], [1, 0, 0], atol=1e-08)

    def test_rosout_initialization_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'runtime.log').write_text('buffered stdout\n')
            (out / 'ros_logs/run').mkdir(parents=True)
            (out / 'ros_logs/run/rosout.log').write_text('10.0 INFO /laserMapping [source] IMU Initial Done\n10.1 ERROR /laserMapping [source] invalid scan\n')
            result = smoke_log_status(out)
            self.assertTrue(result['imu_initialized'])
            self.assertEqual(len(result['runtime_errors']), 1)

    def test_empty_dataset_override_does_not_load_urbanloco(self):
        import roslaunch
        package = Path(__file__).resolve().parents[1]
        launch = roslaunch.config.ROSLaunchConfig()
        roslaunch.xmlloader.XmlLoader().load(str(package / 'launch/public_dataset.launch'), launch, argv=['output_dir:=/tmp/unused_launch_test', 'dataset_config:=', 'sensor_config:=' + str(package / 'config/sensors/livox_avia.yaml')], verbose=False)
        self.assertEqual(launch.params['/mapping/extrinsic_T'].value, [0.0, 0.0, 0.0])
        self.assertNotIn('/dataset/name', launch.params)

    def test_ros_environment_does_not_inherit_unrelated_credentials(self):
        with patch.dict('os.environ', {'UNRELATED_API_KEY': 'test-placeholder', 'PATH': '/test'}):
            env = ros_process_environment()
            self.assertNotIn('UNRELATED_API_KEY', env)
            self.assertEqual(env['PATH'], '/test')

    def test_stamp_nanosecond_precision(self):
        stamp = rospy.Time(1547714239, 999999999)
        self.assertEqual(shifted_stamp(stamp, 2e-09).to_nsec(), stamp.to_nsec() + 2)

    def test_launch_config_xmlrpc_serializable(self):
        import xmlrpc.client
        for profile in ['livox_avia', 'livox_mid360', 'velodyne32']:
            xmlrpc.client.dumps((config(profile),), allow_none=False)

    def cloud(self, ring=31):
        fields = [PointField(name, offset, kind, 1) for name, offset, kind in [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 12, 7), ('ring', 16, 4)]]
        return point_cloud2.create_cloud(Header(stamp=rospy.Time(100, 0)), fields, [(1, 0, 0, 1, ring), (0, -1, 0, 1, ring), (-1, 0, 0, 1, ring)])

    def test_scan_end_to_start_and_seconds(self):
        cloud = azimuth_cloud(self.cloud(), config())
        self.assertEqual(cloud.header.stamp.to_nsec(), 99900000000)
        times = [row[0] for row in point_cloud2.read_points(cloud, field_names=['time'])]
        np.testing.assert_allclose(times, [0, 0.025, 0.05])
        self.assertEqual(cloud.fields[-1].datatype, PointField.FLOAT32)

    def test_ring_out_of_range(self):
        with self.assertRaises(ValueError):
            azimuth_cloud(self.cloud(32), config())

    def test_missing_field(self):
        cloud = self.cloud()
        cloud.fields.pop()
        with self.assertRaises(ValueError):
            cloud_array(cloud)

    def test_truncated_cloud(self):
        cloud = self.cloud()
        cloud.data = cloud.data[:-1]
        with self.assertRaises(ValueError):
            cloud_array(cloud)

    def test_increment_rates(self):
        q = Rotation.from_rotvec([0, 0, 0.002]).as_quat()
        raw = {'Acceleration': {'frame': 'ENU', 'Delta v.x': 0.01, 'Delta v.y': 0, 'Delta v.z': 0.0981}, 'Angular Velocity': dict({'frame': 'ENU'}, **{'Delta q%d' % i: float(q[j]) for i, j in [(0, 3), (1, 0), (2, 1), (3, 2)]})}
        imu = increment_imu(raw, Header(stamp=rospy.Time(100)), 0.01, config())
        self.assertAlmostEqual(imu.linear_acceleration.z, 9.81)
        self.assertAlmostEqual(imu.angular_velocity.z, 0.2)
        with self.assertRaises(ValueError):
            increment_imu(raw, Header(), 0.02, config())

    def test_mid360_units_and_covariance(self):
        msg = Imu()
        msg.linear_acceleration.z = 1
        msg.linear_acceleration_covariance[0] = 0.01
        out = normalized_imu(msg, config('livox_mid360'))
        self.assertAlmostEqual(out.linear_acceleration.z, 9.81)
        self.assertAlmostEqual(out.linear_acceleration_covariance[0], 0.01 * 9.81 ** 2)
        self.assertEqual(out.orientation_covariance[0], -1)
        self.assertEqual(msg.linear_acceleration.z, 1)

    def test_quality_not_promoted(self):
        self.assertEqual(classify_pvt(True, 0), 'SINGLE')
        self.assertEqual(classify_pvt(False, 2), 'INVALID')

    def test_reference_causality_and_half_open_mask(self):
        self.assertIsNone(pseudo_position(10, [0, 0, 0], 10, [(10, 20)]))
        self.assertIsNotNone(pseudo_position(20, [0, 0, 0], 20, [(10, 20)]))
        with self.assertRaises(ValueError):
            pseudo_position(21, [0, 0, 0], 20, [])
if __name__ == '__main__':
    unittest.main()
