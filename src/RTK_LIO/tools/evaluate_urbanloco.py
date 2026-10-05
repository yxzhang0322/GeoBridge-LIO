import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.novatel import ecef, enu_rotation
from public_datasets import new_dir, write_json, sha256

def native_outages(native):
    events = []
    loss = None
    for row in native:
        if row['accepted_carrier'] != 2:
            if loss is None:
                loss = float(row['timestamp'])
        elif loss is not None:
            if row['timestamp'] - loss >= 2.0:
                events.append((loss, float(row['timestamp'])))
            loss = None
    return events

def recovery_metrics(trajectory, native, ref_times, errors, valid, origin, out, name):
    events = native_outages(native)
    grid = np.arange(trajectory[0, 0], trajectory[-1, 0], 0.1)
    pos = np.column_stack([np.interp(grid, trajectory[:, 0], trajectory[:, i]) for i in [1, 2, 3]])
    right = np.clip(np.searchsorted(trajectory[:, 0], grid), 1, len(trajectory) - 1)
    supported = trajectory[right, 0] - trajectory[right - 1, 0] <= 0.25
    velocity = np.diff(pos, axis=0) / 0.1
    acceleration = np.diff(velocity, axis=0) / 0.1
    jerk = np.diff(acceleration, axis=0) / 0.1
    jerktime = grid[3:]
    jerkvalid = supported[3:] & supported[2:-1] & supported[1:-2] & supported[:-3]
    errornorm = np.linalg.norm(errors, axis=1)
    reports = []
    for loss, recovery in events:
        report = dict(outage_start=loss, recovery=recovery, duration_sec=recovery - loss)
        for label, epoch in [('pre_outage', loss - 1), ('end_outage', recovery - 1), ('post_1s', recovery + 1), ('post_2s', recovery + 2), ('post_5s', recovery + 5)]:
            idx = int(np.argmin(np.abs(ref_times - epoch)))
            report[label + '_error_m'] = float(errornorm[idx]) if valid[idx] and abs(ref_times[idx] - epoch) < 0.51 else None
        transient = (grid[1:] >= recovery) & (grid[1:] <= recovery + 10) & supported[1:] & supported[:-1]
        jerkwindow = (jerktime >= recovery) & (jerktime <= recovery + 10) & jerkvalid
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)[transient]
        report['max_position_step_10s_m'] = float(steps.max()) if len(steps) else None
        correction = np.linalg.norm(np.diff(pos, n=2, axis=0), axis=1)
        cmask = (grid[2:] >= recovery) & (grid[2:] <= recovery + 10) & supported[2:] & supported[1:-1] & supported[:-2]
        report['max_constant_velocity_innovation_10s_m'] = float(correction[cmask].max()) if cmask.any() else None
        report['peak_jerk_10s_mps3'] = float(np.linalg.norm(jerk[jerkwindow], axis=1).max()) if jerkwindow.any() else None
        longer = (grid[1:] >= recovery) & (grid[1:] <= recovery + 30) & supported[1:] & supported[:-1]
        longer_jerk = (jerktime >= recovery) & (jerktime <= recovery + 30) & jerkvalid
        longer_correction = (grid[2:] >= recovery) & (grid[2:] <= recovery + 30) & supported[2:] & supported[1:-1] & supported[:-2]
        report['max_position_step_30s_m'] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1)[longer].max()) if longer.any() else None
        report['peak_jerk_30s_mps3'] = float(np.linalg.norm(jerk[longer_jerk], axis=1).max()) if longer_jerk.any() else None
        report['max_constant_velocity_innovation_30s_m'] = float(correction[longer_correction].max()) if longer_correction.any() else None
        late = valid & (ref_times >= recovery + 10) & (ref_times <= recovery + 30)
        report['late_recovery_10_30s'] = summarize(errors[late])
        convergence = None
        for i in np.where(valid & (ref_times >= recovery) & (ref_times <= recovery + 30))[0]:
            window = (ref_times >= ref_times[i]) & (ref_times < ref_times[i] + 5)
            if window.sum() >= 5 and valid[window].all() and (errornorm[window] <= 0.5).all():
                convergence = float(ref_times[i] - recovery)
                break
        report['convergence_0p5m_5s_sec'] = convergence
        reports.append(report)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(grid[1:] - origin, np.linalg.norm(velocity, axis=1))
    axes[0].set_ylabel('Speed (m/s)')
    axes[1].plot(jerktime[jerkvalid] - origin, np.linalg.norm(jerk[jerkvalid], axis=1))
    axes[1].set_ylabel('Jerk (m/s^3)')
    for ax in axes:
        for loss, recovery in events:
            ax.axvspan(loss - origin, recovery - origin, color='red', alpha=0.1)
        ax.grid(alpha=0.2)
    axes[1].set_xlabel('Bag-relative time (s)')
    fig.suptitle(name + '; fixed 10 Hz differentiation')
    fig.tight_layout()
    fig.savefig(out / (name + '_smoothness.png'), dpi=150)
    plt.close(fig)
    return reports

def rigid_fit(source, target):
    a, b = (source.mean(axis=0), target.mean(axis=0))
    u, s, vt = np.linalg.svd((source - a).T @ (target - b))
    d = np.eye(3)
    d[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ d @ u.T
    return (r, b - r @ a, s)

def summarize(errors):
    if not len(errors):
        return {'count': 0}
    norm = np.linalg.norm(errors, axis=1)
    return dict(count=len(errors), rmse_3d_m=float(np.sqrt(np.mean(norm ** 2))), rmse_xy_m=float(np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=1)))), rmse_z_m=float(np.sqrt(np.mean(errors[:, 2] ** 2))), median_m=float(np.median(norm)), p95_m=float(np.percentile(norm, 95)), max_m=float(norm.max()), final_m=float(norm[-1]))

def evaluate(args):
    out = new_dir(args.out)
    prepared = Path(args.prepared)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    with open(args.config) as f:
        protocol = yaml.safe_load(f)['experiment']
    reference = np.genfromtxt(prepared / 'reference.csv', delimiter=',', names=True)
    origin = [reference['lat'][0], reference['lon'][0], reference['ellipsoid_height'][0]]
    xyz = np.array([ecef(row['lat'], row['lon'], row['ellipsoid_height']) for row in reference])
    truth = (xyz - ecef(*origin)) @ enu_rotation(*origin[:2]).T
    times = reference['timestamp']
    relative = times - manifest['record_interval'][0]
    good = (reference['ins_status'] == 3) & (reference['position_type'] == 56) & (np.maximum.reduce([reference['north_std'], reference['east_std'], reference['up_std']]) <= protocol['reference_max_std_m'])
    outputs, matches = ({}, {})
    for spec in args.trajectory:
        name, filename = spec.split('=', 1)
        trajectory = np.loadtxt(filename, ndmin=2)
        if trajectory.shape[1] != 8 or not np.isfinite(trajectory).all() or np.any(np.diff(trajectory[:, 0]) <= 0):
            raise ValueError('Invalid trajectory: ' + filename)
        indices = np.searchsorted(trajectory[:, 0], times)
        bounded = (indices > 0) & (indices < len(trajectory))
        clipped = np.clip(indices, 1, len(trajectory) - 1)
        gap = trajectory[clipped, 0] - trajectory[clipped - 1, 0]
        valid = bounded & (gap <= 0.25)
        position = np.column_stack([np.interp(times, trajectory[:, 0], trajectory[:, i]) for i in [1, 2, 3]])
        matches[name] = (position, valid, trajectory, filename)
    common = np.logical_and.reduce([row[1] for row in matches.values()])
    calibration = common & (relative >= protocol['calibration_interval_sec'][0]) & (relative <= protocol['calibration_interval_sec'][1])
    evaluation = common & (relative >= protocol['evaluation_start_sec'])
    if calibration.sum() < 10 or evaluation.sum() < 10:
        raise ValueError('Insufficient common calibration/evaluation coverage; do not silently change windows')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(truth[:, 0], truth[:, 1], color='black', label='SPAN reference')
    native = np.genfromtxt(prepared / 'native_rtk.csv', delimiter=',', names=True)
    axes[1, 1].step(native['timestamp'] - manifest['record_interval'][0], native['accepted_carrier'], where='post', label='Native RTK carrier')
    axes[1, 1].step(relative, good.astype(int), where='post', label='Reference high quality')
    rows = []
    aligned_outputs = {}
    native_outputs = {}
    for name, (position, valid, trajectory, filename) in matches.items():
        r, t, s = rigid_fit(position[calibration], truth[calibration])
        aligned = position @ r.T + t
        aligned_outputs[name] = aligned
        errors = aligned - truth
        global_r, global_t, global_s = rigid_fit(position[common], truth[common])
        aligned_global = position @ global_r.T + global_t
        rpe_valid = evaluation[1:] & evaluation[:-1] & (np.diff(times) <= protocol['reference_max_gap_sec'])
        rpe = np.diff(aligned, axis=0) - np.diff(truth, axis=0)
        report = dict(initial_se3_evaluation=summarize(errors[evaluation]), initial_se3_high_quality=summarize(errors[evaluation & good]), initial_se3_degraded_reference=summarize(errors[evaluation & ~good]), full_se3_ate=summarize((aligned_global - truth)[common]), rpe_adjacent_reference=summarize(rpe[rpe_valid]), alignment_rotation=r.tolist(), alignment_translation=t.tolist(), alignment_singular_values=s.tolist(), trajectory=filename, sha256=sha256(filename), total_poses=len(trajectory), unmatched_reference_count=int((~valid).sum()), common_reference_count=int(common.sum()), reference_independent=False, orientation_error_reported=False)
        report['native_recoveries'] = recovery_metrics(trajectory, native, times, errors, common, manifest['record_interval'][0], out, name)
        if name in args.world_enu:
            accepted = native[native['accepted_carrier'] == 2][:20]
            if len(accepted) != 20:
                raise ValueError('Cannot reconstruct the configured 20-fix RTK origin')
            native_origin = np.array([accepted[key].mean() for key in ['lat', 'lon', 'ellipsoid_height']])
            r_native = enu_rotation(*native_origin[:2])
            r_ref = enu_rotation(*origin[:2])
            native_aligned = position @ r_native @ r_ref.T + (ecef(*native_origin) - ecef(*origin)) @ r_ref.T
            native_errors = native_aligned - truth
            native_outputs[name] = (native_aligned, native_errors)
            report['native_enu_no_fit'] = summarize(native_errors[evaluation])
            report['native_enu_high_quality'] = summarize(native_errors[evaluation & good])
            report['native_enu_origin'] = native_origin.tolist()
            report['native_enu_origin_source'] = 'Same first 20 admitted native fixed measurements as RtkProcess; no GT fit'
            report['native_enu_recoveries'] = recovery_metrics(trajectory, native, times, native_errors, common, manifest['record_interval'][0], out, name + '_ENU')
            with (out / (name + '_native_enu_errors.csv')).open('x') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'relative', 'ex', 'ey', 'ez', 'evaluation', 'reference_high_quality'])
                for i in np.where(common)[0]:
                    writer.writerow([times[i], relative[i], *native_errors[i], int(evaluation[i]), int(good[i])])
        update_file = Path(filename).parent / 'updates.csv'
        if update_file.exists():
            updates = np.genfromtxt(update_file, delimiter=',', names=True)
            report['runtime_ms'] = {key: dict(mean=float(np.mean(updates[key])), p95=float(np.percentile(updates[key], 95)), maximum=float(np.max(updates[key]))) for key in ['t_vta_ms', 't_tapm_ms', 't_sliding_window_ms', 't_total_ms'] if key in updates.dtype.names}
            if 'sigma_ratio' in updates.dtype.names:
                unique = np.r_[True, np.diff(updates['vta_epoch']) > 0] & (updates['vta_samples'] >= 3)
                ratios = updates['sigma_ratio'][unique]
                report['vta'] = dict(updates=int(unique.sum()), ratio_min=float(ratios.min()) if len(ratios) else None, ratio_max=float(ratios.max()) if len(ratios) else None, gravity_updates=int(updates['gravity_used'][unique].sum()))
                epochs = updates['timestamp'][unique]
                previous = np.column_stack([np.interp(epochs - 1, trajectory[:, 0], trajectory[:, axis]) for axis in [1, 2]])
                middle = np.column_stack([np.interp(epochs - 0.5, trajectory[:, 0], trajectory[:, axis]) for axis in [1, 2]])
                current = np.column_stack([np.interp(epochs, trajectory[:, 0], trajectory[:, axis]) for axis in [1, 2]])
                a, b = (middle - previous, current - middle)
                speed = np.linalg.norm(current - previous, axis=1)
                heading_change = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1)))
                motion = np.where(speed < 0.2, 'low_speed', np.where(heading_change > 0.05, 'turning', 'straight'))
                report['vta']['motion_ratio_median'] = {state: float(np.median(ratios[motion == state])) for state in np.unique(motion)}
                report['vta']['independent_contribution_established'] = False
                with (out / (name + '_vta.csv')).open('x') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'sigma1', 'sigma2', 'sigma_ratio', 'history_sec', 'displacement_m', 'samples', 'gravity_used', 'motion'])
                    for row, state in zip(updates[unique], motion):
                        writer.writerow([row[key] for key in ['timestamp', 'sigma1', 'sigma2', 'sigma_ratio', 'vta_history_sec', 'vta_displacement_m', 'vta_samples', 'gravity_used']] + [state])
                fig_v, ax = plt.subplots(figsize=(12, 3))
                ax.plot(updates['timestamp'][unique] - manifest['record_interval'][0], ratios, label='Velocity sigma2/sigma1 (before gravity)')
                for state, marker in [('turning', 'o'), ('straight', '.'), ('low_speed', 'x')]:
                    selected = motion == state
                    ax.scatter(epochs[selected] - manifest['record_interval'][0], ratios[selected], s=10, marker=marker, label=state)
                for loss, recovery in native_outages(native):
                    ax.axvspan(loss - manifest['record_interval'][0], recovery - manifest['record_interval'][0], color='red', alpha=0.08)
                    ax.axvline(recovery - manifest['record_interval'][0], color='black', linewidth=0.7, linestyle='--')
                ax.fill_between(updates['timestamp'] - manifest['record_interval'][0], 0, 1, where=updates['tapm_active'] > 0, alpha=0.15, label='TAPM active')
                ax.set(xlabel='Bag-relative time (s)', ylabel='Singular value ratio', ylim=(0, 1))
                ax.legend()
                ax.grid(alpha=0.2)
                fig_v.tight_layout()
                fig_v.savefig(out / (name + '_vta.png'), dpi=150)
                plt.close(fig_v)
            if name == 'TAPM':
                for ax in [axes[0, 1], axes[1, 0]]:
                    ax.fill_between(updates['timestamp'] - manifest['record_interval'][0], 0, 1, where=updates['tapm_active'] > 0, transform=ax.get_xaxis_transform(), color='green', alpha=0.08)
        for interval_name, mask in [('pre_degradation', evaluation & (relative < 180)), ('late', evaluation & (relative >= 240))]:
            report[interval_name] = summarize(errors[mask])
        outputs[name] = report
        axes[0, 0].plot(aligned[common, 0], aligned[common, 1], label=name)
        axes[0, 1].plot(relative[common], np.linalg.norm(errors[common], axis=1), label=name)
        axes[1, 0].plot(relative[common], errors[common, 2], label=name)
        for i in np.where(common)[0]:
            rows.append([name, times[i], relative[i], *aligned[i], *truth[i], *errors[i], int(good[i]), int(evaluation[i])])
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.2)
    for loss, recovery in native_outages(native):
        for ax in [axes[0, 1], axes[1, 0]]:
            ax.axvspan(loss - manifest['record_interval'][0], recovery - manifest['record_interval'][0], color='red', alpha=0.08)
            ax.axvline(recovery - manifest['record_interval'][0], color='black', linewidth=0.7, linestyle='--')
    axes[0, 0].set(xlabel='East (m)', ylabel='North (m)', title='Fixed initial SE(3) alignment, no scale')
    axes[0, 0].axis('equal')
    axes[0, 1].set(xlabel='Bag-relative measurement time (s)', ylabel='Position error (m)')
    axes[1, 0].set(xlabel='Bag-relative measurement time (s)', ylabel='Vertical error (m)')
    axes[1, 1].set(xlabel='Bag-relative measurement time (s)', ylabel='Status')
    fig.suptitle('UrbanLoco: native RTK and correlated SPAN reference (provisional calibration)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)
    if native_outputs:
        fig, axes_native = plt.subplots(1, 2, figsize=(13, 5))
        axes_native[0].plot(truth[evaluation, 0], truth[evaluation, 1], color='black', label='SPAN reference')
        for name, (positions, native_errors) in native_outputs.items():
            axes_native[0].plot(positions[evaluation, 0], positions[evaluation, 1], label=name)
            axes_native[1].plot(relative[evaluation], np.linalg.norm(native_errors[evaluation], axis=1), label=name)
        for loss, recovery in native_outages(native):
            axes_native[1].axvspan(loss - manifest['record_interval'][0], recovery - manifest['record_interval'][0], color='red', alpha=0.08)
            axes_native[1].axvline(recovery - manifest['record_interval'][0], color='black', linewidth=0.7, linestyle='--')
        if 'TAPM' in matches:
            update_file = Path(matches['TAPM'][3]).parent / 'updates.csv'
            if update_file.exists():
                updates = np.genfromtxt(update_file, delimiter=',', names=True)
                axes_native[1].fill_between(updates['timestamp'] - manifest['record_interval'][0], 0, 1, where=updates['tapm_active'] > 0, transform=axes_native[1].get_xaxis_transform(), color='green', alpha=0.08)
        axes_native[0].set(xlabel='East (m)', ylabel='North (m)')
        axes_native[0].axis('equal')
        axes_native[1].set(xlabel='Bag-relative time (s)', ylabel='Position error (m)', xlim=(protocol['evaluation_start_sec'], relative[evaluation][-1]))
        for ax in axes_native:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle('Native ENU origin conversion only; correlated reference; fixed evaluation interval')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / 'native_enu_comparison.png', dpi=160)
        plt.close(fig)
    for event, (loss, recovery) in enumerate(native_outages(native)):
        mask = common & (times >= recovery - 5) & (times <= recovery + 30)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(truth[mask, 0], truth[mask, 1], color='black', label='SPAN reference')
        for name, aligned in aligned_outputs.items():
            ax.plot(aligned[mask, 0], aligned[mask, 1], label=name)
        ax.set(xlabel='East (m)', ylabel='North (m)', title='Recovery %d: -5 to +30 s' % (event + 1))
        ax.axis('equal')
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / ('recovery_%02d.png' % event), dpi=160)
        plt.close(fig)
    with (out / 'errors.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'timestamp', 'relative', 'x', 'y', 'z', 'ref_x', 'ref_y', 'ref_z', 'ex', 'ey', 'ez', 'reference_high_quality', 'evaluation'])
        writer.writerows(rows)
    write_json(out / 'metrics.json', dict(protocol=protocol, results=outputs, common_evaluation_count=int(evaluation.sum()), reference='SPAN-CPT INSPVAX; correlated with BESTPOS input; not independent GT', frame_limitations='Position points not fully verified; no attitude GT/lever arm assumed. Initial SE3 absorbs constant frame offset.', recovery_definitions='All native non-fixed intervals >=2 s; 10 Hz linear resampling; both 10 s and 30 s transients are reported to include late exits. Position step includes physical motion; constant-velocity innovation is second position difference; jerk third difference/dt^3; convergence <=0.5 m for 5 reference epochs within 30 s. Null means not established.', reference_sha256=sha256(prepared / 'reference.csv'), prepared_manifest=str(prepared / 'manifest.json')))
    print(json.dumps(outputs, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--trajectory', action='append', required=True, help='NAME=/absolute/path.tum')
    parser.add_argument('--world-enu', action='append', default=[], help='Explicitly identify a native world-ENU output; adds origin conversion without GT fitting')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'config/datasets/urbanloco_rtk.yaml'))
    parser.add_argument('--out', required=True)
    evaluate(parser.parse_args())
