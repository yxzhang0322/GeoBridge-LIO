import ast
import copy
import math
import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, PointCloud2, PointField

def shifted_stamp(stamp, offset):
    ns = stamp.to_nsec() + int(round(offset * 1000000000.0))
    if ns < 0:
        raise ValueError('Negative measurement timestamp')
    return rospy.Time(ns // 1000000000, ns % 1000000000)

def cloud_array(msg):
    types = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}
    if msg.is_bigendian:
        raise ValueError('Big-endian input is not supported by this adapter')
    if not msg.width or not msg.height or msg.point_step <= 0:
        raise ValueError('Empty cloud')
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
        raise ValueError('Truncated cloud payload')
    fields = {f.name: f for f in msg.fields}
    names = ['x', 'y', 'z', 'intensity', 'ring']
    if len(fields) != len(msg.fields):
        raise ValueError('Duplicate cloud fields')
    for name in names:
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in types:
            raise ValueError('Missing/incompatible cloud field: ' + name)
        if f.offset + np.dtype(types[f.datatype]).itemsize > msg.point_step:
            raise ValueError('Cloud field exceeds point_step: ' + name)
    dtype = np.dtype({'names': names, 'formats': [types[fields[n].datatype] for n in names], 'offsets': [fields[n].offset for n in names], 'itemsize': msg.point_step})
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data, strides=(msg.row_step, msg.point_step)).reshape(-1)

def azimuth_cloud(msg, config):
    adapter, pre = (config['adapter'], config['preprocess'])
    points = cloud_array(msg)
    valid = np.ones(len(points), dtype=bool)
    for name in ['x', 'y', 'z', 'intensity', 'ring']:
        valid &= np.isfinite(points[name])
    valid &= (points['ring'] >= 0) & (points['ring'] < pre['scan_line'])
    if not np.all(valid):
        raise ValueError('Nonfinite cloud fields or out-of-range ring')
    period = 1.0 / pre['scan_rate']
    if period > pre['max_scan_duration']:
        raise ValueError('scan_rate conflicts with max_scan_duration')
    yaw = np.arctan2(points['y'], points['x'])
    phase = np.remainder(adapter['rotation_direction'] * (yaw - yaw[0]), 2 * np.pi)
    out = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'), ('ring', '<u2'), ('time', '<f4')])
    for name in ['x', 'y', 'z', 'intensity', 'ring']:
        out[name] = points[name]
    out['time'] = phase * period / (2 * np.pi)
    out.sort(order='time')
    result = PointCloud2()
    result.header = copy.deepcopy(msg.header)
    offset = adapter['lidar_offset_sec']
    if adapter['scan_stamp'] == 'end':
        offset -= period
    elif adapter['scan_stamp'] != 'start':
        raise ValueError('scan_stamp must be start or end')
    result.header.stamp = shifted_stamp(msg.header.stamp, offset)
    result.header.frame_id = adapter['lidar_frame']
    result.height, result.width = (1, len(out))
    result.fields = [PointField(n, out.dtype.fields[n][1], PointField.UINT16 if n == 'ring' else PointField.FLOAT32, 1) for n in out.dtype.names]
    result.point_step = out.dtype.itemsize
    result.row_step = result.width * result.point_step
    result.is_dense, result.is_bigendian = (True, False)
    result.data = out.tobytes()
    return result

def livox_cloud(msg, config):
    import io
    from livox_ros_driver.msg import CustomMsg
    if msg._md5sum != CustomMsg._md5sum:
        raise ValueError('Unsupported Livox CustomMsg wire schema')
    if msg.point_num != len(msg.points) or not msg.points:
        raise ValueError('Invalid Livox point count')
    limit = config['preprocess']['max_scan_duration'] * 1000000000.0
    if any((p.line >= config['preprocess']['scan_line'] or p.offset_time > limit for p in msg.points)):
        raise ValueError('Invalid Livox line or point offset')
    data = io.BytesIO()
    msg.serialize(data)
    out = CustomMsg().deserialize(data.getvalue())
    out.header.stamp = shifted_stamp(msg.header.stamp, config['adapter']['lidar_offset_sec'])
    out.header.frame_id = config['adapter']['lidar_frame']
    return out

def normalized_imu(msg, config):
    out = copy.deepcopy(msg)
    a = config['adapter']
    out.header.stamp = shifted_stamp(msg.header.stamp, a['imu_offset_sec'])
    out.header.frame_id = a['imu_frame']
    for field, scale in [('linear_acceleration', a['acceleration_scale']), ('angular_velocity', a['gyro_scale'])]:
        vector = getattr(out, field)
        for axis in ['x', 'y', 'z']:
            value = getattr(vector, axis) * scale
            if not math.isfinite(value):
                raise ValueError('Nonfinite IMU value')
            setattr(vector, axis, value)
        covariance = field + '_covariance'
        setattr(out, covariance, [v * scale * scale for v in getattr(out, covariance)])
    out.orientation.x = out.orientation.y = out.orientation.z = 0.0
    out.orientation.w = 1.0
    out.orientation_covariance = [-1.0] + [0.0] * 8
    return out

def parse_increment(message):
    data = ast.literal_eval(message.data)
    if not isinstance(data, dict):
        raise ValueError('Xsens payload must be a dictionary')
    return data

def increment_imu(data, header, dt, config):
    a = config['adapter']
    nominal = a['nominal_imu_interval_sec']
    if not math.isfinite(dt) or dt <= 0 or dt > a['max_imu_interval_sec'] or (abs(dt - nominal) > nominal * 0.1):
        raise ValueError('Dropped or irregular Xsens sample; cannot scale one delta-v across a gap')
    accel = data['Acceleration']
    gyro = data['Angular Velocity']
    if accel['frame'] != 'ENU' or gyro['frame'] != 'ENU':
        raise ValueError('Unexpected Xsens coordinate convention')
    dv = np.array([accel['Delta v.' + axis] for axis in ['x', 'y', 'z']], dtype=float)
    dq = np.array([gyro['Delta q' + str(i)] for i in [1, 2, 3, 0]], dtype=float)
    if not np.all(np.isfinite(dv)) or not np.all(np.isfinite(dq)) or abs(np.linalg.norm(dq) - 1) > 0.01:
        raise ValueError('Invalid inertial increment')
    omega = Rotation.from_quat(dq).as_rotvec() / dt
    result = Imu()
    result.header = copy.deepcopy(header)
    result.header.stamp = shifted_stamp(header.stamp, a['imu_offset_sec'])
    result.header.frame_id = a['imu_frame']
    result.orientation.w = 1.0
    result.orientation_covariance[0] = -1.0
    for i, axis in enumerate(['x', 'y', 'z']):
        setattr(result.linear_acceleration, axis, float(dv[i] / dt))
        setattr(result.angular_velocity, axis, float(omega[i]))
    return result
