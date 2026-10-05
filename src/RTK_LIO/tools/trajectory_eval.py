import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def validate(a):
    a = np.array(a, dtype=float, copy=True, ndmin=2)
    if a.shape[1] != 8 or len(a) < 2 or (not np.isfinite(a).all()):
        raise ValueError('TUM requires at least two finite eight-column poses')
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError('Timestamps must be strictly increasing')
    norms = np.linalg.norm(a[:, 4:], axis=1)
    if np.any(norms < 1e-10) or np.any(abs(norms - 1) > 0.01):
        raise ValueError('Invalid quaternion norm')
    a[:, 4:] /= norms[:, None]
    return a

def associate(gt, est, tolerance=0.15, max_gap=0.25):
    gt, est = (validate(gt), validate(est))
    t = gt[:, 0]
    j = np.searchsorted(est[:, 0], t, side='left')
    left, right = (np.clip(j - 1, 0, len(est) - 1), np.clip(j, 0, len(est) - 1))
    nearest = np.minimum(abs(t - est[left, 0]), abs(est[right, 0] - t))
    gap = est[right, 0] - est[left, 0]
    valid = (t >= est[0, 0]) & (t <= est[-1, 0]) & (nearest <= tolerance) & ((gap <= max_gap) | (nearest < 1e-08))
    t = t[valid]
    if not len(t):
        raise ValueError('No timestamp matches')
    positions = np.column_stack([np.interp(t, est[:, 0], est[:, k]) for k in (1, 2, 3)])
    rotations = Slerp(est[:, 0], Rotation.from_quat(est[:, 4:]))(t)
    return (gt[valid], positions, rotations, valid)

def rigid_fit(source, target):
    a, b = (source.mean(axis=0), target.mean(axis=0))
    u, singular, vt = np.linalg.svd((source - a).T @ (target - b))
    if singular[0] < 1e-10 or singular[1] < 1e-10:
        raise ValueError('SE3 alignment requires non-collinear position support')
    d = np.eye(3)
    d[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ d @ u.T
    return (r, b - r @ a, singular)

def evaluate(gt, est, tolerance=0.15, max_gap=0.25, alignment_interval=None, evaluation_start=None, position_only=False):
    gt, est = (validate(gt), validate(est))
    matched, pos, rot, valid = associate(gt, est, tolerance, max_gap)
    times = matched[:, 0]
    fit = np.ones(len(times), dtype=bool) if alignment_interval is None else (times >= alignment_interval[0]) & (times <= alignment_interval[1])
    use = np.ones(len(times), dtype=bool) if evaluation_start is None else times >= evaluation_start
    if fit.sum() < 3 or use.sum() < 3:
        raise ValueError('Insufficient alignment or evaluation samples')
    r, t, s = rigid_fit(pos[fit], matched[fit, 1:4])
    aligned = pos @ r.T + t
    from_matrix = getattr(Rotation, 'from_matrix', None) or Rotation.from_dcm
    aligned_rot = from_matrix(r) * rot
    trans = np.linalg.norm(aligned - matched[:, 1:4], axis=1)
    angles = np.linalg.norm((Rotation.from_quat(matched[:, 4:]).inv() * aligned_rot).as_rotvec(), axis=1)
    eligible = np.ones(len(gt), dtype=bool) if evaluation_start is None else gt[:, 0] >= evaluation_start
    report = dict(status='success', T_RMSE_m=float(np.sqrt(np.mean(trans[use] ** 2))), T_Max_m=float(trans[use].max()), R_RMSE_rad=float(np.sqrt(np.mean(angles[use] ** 2))), R_RMSE_deg=float(np.rad2deg(np.sqrt(np.mean(angles[use] ** 2)))), matched_count=int(use.sum()), total_estimate_count=len(est), total_gt_count=len(gt), coverage=float(use.sum() / eligible.sum()), dropped_reference_count=int(eligible.sum() - use.sum()), association_tolerance_sec=tolerance, max_interpolation_gap_sec=max_gap, interpolation='estimate translation linear, quaternion SLERP at reference timestamps; no extrapolation', start_time=float(times[use][0]), end_time=float(times[use][-1]), failure_reason=None, alignment=dict(rotation=r.tolist(), translation=t.tolist(), scale=1.0, method='SE3 position Kabsch', sample_count=int(fit.sum()), interval=[float(times[fit][0]), float(times[fit][-1])], singular_values=s.tolist()))
    errors = np.column_stack([times, trans, np.rad2deg(angles), use.astype(int)])
    full_rotation = from_matrix(r) * Rotation.from_quat(est[:, 4:])
    tum = np.column_stack([est[:, 0], est[:, 1:4] @ r.T + t, full_rotation.as_quat()])
    if position_only:
        report.update(R_RMSE_rad=None, R_RMSE_deg=None, rotation_status='UNAVAILABLE: reference has no measured orientation')
        errors[:, 2] = np.nan
    return (report, errors, tum)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', required=True)
    p.add_argument('--est', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tolerance', type=float, default=0.15)
    p.add_argument('--max-gap', type=float, default=0.25)
    p.add_argument('--alignment-interval', type=float, nargs=2)
    p.add_argument('--evaluation-start', type=float)
    p.add_argument('--position-only', action='store_true', help='Reference quaternion is a placeholder, not measured attitude')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        report, errors, aligned = evaluate(np.loadtxt(a.gt), np.loadtxt(a.est), a.tolerance, a.max_gap, a.alignment_interval, a.evaluation_start, a.position_only)
        np.savetxt(out / 'errors.csv', errors, delimiter=',', header='timestamp,translation_m,rotation_deg,evaluated', comments='')
        np.savetxt(out / 'aligned.tum', aligned, fmt='%.10f')
        (out / 'alignment.json').write_text(json.dumps(report['alignment'], indent=2) + '\n')
    except Exception as e:
        report = dict(status='failed', failure_stage='evaluation', failure_reason=str(e))
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    if report['status'] == 'failed':
        raise SystemExit(1)
if __name__ == '__main__':
    main()
