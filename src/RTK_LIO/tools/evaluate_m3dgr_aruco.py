import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from trajectory_eval import validate

def read_gt(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) != 7 or not lines[-1].startswith('bag_time:'):
        raise ValueError('Expected three rotation rows, three translation rows and bag_time')
    r = np.array([[float(v) for v in line.split()] for line in lines[:3]])
    t = np.array([float(v) for v in lines[3:6]])
    duration = float(lines[-1].split(':', 1)[1].strip().rstrip('s'))
    if r.shape != (3, 3) or not np.isfinite(r).all() or (not np.isfinite(t).all()) or (duration <= 0):
        raise ValueError('Invalid ArUco reference')
    if np.linalg.det(r) < 0 or np.linalg.norm(r.T @ r - np.eye(3)) > 0.001:
        raise ValueError('Reference rotation is not SO3 within text rounding tolerance')
    return (r, t, duration)

def evaluate_relative(trajectory, reference, expected_start=None, expected_end=None):
    a = validate(trajectory)
    rg, tg, duration = read_gt(reference)
    rotation = Rotation.from_quat(a[:, 4:])
    matrices = rotation.as_matrix() if hasattr(rotation, 'as_matrix') else rotation.as_dcm()
    re = matrices[0].T @ matrices[-1]
    te = matrices[0].T @ (a[-1, 1:4] - a[0, 1:4])
    from_matrix = getattr(Rotation, 'from_matrix', None) or Rotation.from_dcm
    angle = float(np.linalg.norm(from_matrix(rg.T @ re).as_rotvec()))
    complete = (expected_start is None or a[0, 0] - expected_start <= 2) and (expected_end is None or expected_end - a[-1, 0] <= 0.2)
    return dict(status='success' if complete else 'failed', evaluation='ArUco start-end relative pose', translation_error_m=float(np.linalg.norm(te - tg)), rotation_error_deg=float(np.degrees(angle)), rotation_error_rad=angle, official_rotation_frobenius=float(np.linalg.norm(rg - re)), estimated_relative_rotation=re.tolist(), estimated_relative_translation=te.tolist(), reference_rotation=rg.tolist(), reference_translation=tg.tolist(), scale=1.0, alignment='none', trajectory_samples=len(a), start_time=float(a[0, 0]), end_time=float(a[-1, 0]), tracking_duration_sec=float(a[-1, 0] - a[0, 0]), tracking_rate=min(1.0, float((a[-1, 0] - a[0, 0]) / duration)), complete=bool(complete), max_timestamp_gap_sec=float(np.diff(a[:, 0]).max()), frame_protocol='Official evaluator direct first/last estimator body poses; published reference does not explicitly identify its sensor frame', failure_reason=None if complete else 'Trajectory does not cover required endpoints')
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', required=True)
    p.add_argument('--est', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--expected-start', type=float)
    p.add_argument('--expected-end', type=float)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        report = evaluate_relative(np.loadtxt(a.est), a.gt, a.expected_start, a.expected_end)
    except Exception as e:
        report = dict(status='failed', failure_reason=str(e), failure_stage='endpoint_evaluation')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
