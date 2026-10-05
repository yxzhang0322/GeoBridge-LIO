import math
import numpy as np
GPS_EPOCH_UNIX = 315964800

def epoch(msg, leap_seconds=18):
    return GPS_EPOCH_UNIX + int(msg.header.gps_week) * 604800 + int(msg.header.gps_week_seconds) * 0.001 - leap_seconds

def quality(msg):
    if msg.solution_status != 0:
        return 'INVALID'
    return {50: 'FIXED', 56: 'INS_FIXED', 34: 'FLOAT', 55: 'INS_FLOAT', 54: 'INS_DIFFERENTIAL', 19: 'PROPAGATED'}.get(int(msg.position_type), 'OTHER')

def bestpos_pvt(msg, cfg):
    from gnss_comm.msg import GnssPVTSolnMsg
    out = GnssPVTSolnMsg()
    out.time.week = msg.header.gps_week
    out.time.tow = msg.header.gps_week_seconds * 0.001
    out.fix_type = 4 if int(msg.position_type) in [54, 55, 56] else 3
    out.latitude, out.longitude = (msg.latitude, msg.longitude)
    out.height_msl = msg.altitude
    out.altitude = msg.altitude + msg.undulation
    out.h_acc = math.hypot(msg.latitude_std, msg.longitude_std)
    out.v_acc = msg.altitude_std
    out.num_sv = msg.sol_svs
    finite = all((math.isfinite(v) for v in [out.latitude, out.longitude, out.altitude, out.h_acc, out.v_acc]))
    out.valid_fix = bool(finite and msg.solution_status == 0 and (abs(out.latitude) <= 90) and (abs(out.longitude) <= 180) and (msg.latitude_std > 0) and (msg.longitude_std > 0) and (0 <= msg.diff_age <= cfg['max_diff_age_sec']) and (0 < out.h_acc <= cfg['max_h_std_m']) and (0 < out.v_acc <= cfg['max_v_std_m']))
    fixed = int(msg.position_type) in cfg['fixed_types']
    floating = int(msg.position_type) in [34, 55]
    out.diff_soln = bool(out.valid_fix and (fixed or floating or msg.position_type == 54))
    out.carr_soln = 2 if out.valid_fix and fixed else 1 if out.valid_fix and floating else 0
    out.p_dop = float('nan')
    out.vel_n = out.vel_e = out.vel_d = out.vel_acc = float('nan')
    return out

def ecef(lat, lon, height):
    lat, lon = np.deg2rad([lat, lon])
    e2 = 0.0066943799901413165
    n = 6378137.0 / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + height) * np.cos(lat) * np.cos(lon), (n + height) * np.cos(lat) * np.sin(lon), (n * (1 - e2) + height) * np.sin(lat)])

def enu_rotation(lat, lon):
    lat, lon = np.deg2rad([lat, lon])
    return np.array([[-np.sin(lon), np.cos(lon), 0], [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
