import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from scipy.interpolate import splprep, splev
import matplotlib.animation as animation
SHOW_FIGURES: bool = True
SAVE_FIGURES: bool = True
DISPLAY_DPI: int = 150
SAVE_DPI: int = 300
SLIDING_WINDOW_SIZE: int = 5
SLIDING_WINDOW_STEP: int = 1
TIME_MATCH_THRESHOLD: float = 0.1
VELOCITY_THRESHOLD: float = 0.1
VELOCITY_SCALE_FACTOR: float = 0.2
ARROW_HEAD_WIDTH: float = 0.001
ARROW_HEAD_LENGTH: float = 0.001
ARROW_SKIP_POINTS: int = 1
plt.rcParams['figure.dpi'] = DISPLAY_DPI
plt.rcParams['savefig.dpi'] = SAVE_DPI
plt.rcParams['axes.unicode_minus'] = False
USE_COARSE_ALIGNMENT: bool = True
USE_TLS_REFINEMENT: bool = True
SIGMA_POSITION: float = 0.1
P_CONFIDENCE: float = 0.99
C_BAR_POSITION_SQ: float = chi2.ppf(P_CONFIDENCE, df=3) if 'chi2' in globals() else 11.345
TLS_MAX_ITER: int = 10
USE_ADAPTIVE_THRESHOLD: bool = True
ADAPTIVE_THRESHOLD_SCALE: float = 2.5
USE_BSPLINE_SMOOTHING: bool = True
BSPLINE_SMOOTHING_FACTOR: float = 0.1
PRINT_ROTATION_DEBUG: bool = True
ANSI_RED = '\x1b[31m'
ANSI_GREEN = '\x1b[32m'
ANSI_RESET = '\x1b[0m'
ANSI_BLUE = '\x1b[34m'
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
ALIGNED_RESULTS_DIR = PROJECT_ROOT / 'aligned_results'
PICTURE_DIR = PROJECT_ROOT / 'picture'
DATA_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PICTURE_DIR.mkdir(parents=True, exist_ok=True)

def print_rotation_matrix(label: str, matrix: np.ndarray, color: str):
    if PRINT_ROTATION_DEBUG:
        print(f'{color}{label}:\n{matrix}{ANSI_RESET}')

def invert_similarity(R: np.ndarray, t: np.ndarray, s: float):
    if abs(s) < 1e-09:
        s = 1.0
    R_inv = R.T
    s_inv = 1.0 / s
    t_inv = -s_inv * (R_inv @ t)
    return (R_inv, t_inv, s_inv)

def _maybe_save_and_show(filename):
    if SAVE_FIGURES and filename:
        output_path = Path(filename)
        if not output_path.is_absolute():
            output_path = PICTURE_DIR / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=SAVE_DPI, bbox_inches='tight')
        print(f'  ✓ 图像已保存: {output_path}')

def load_trajectory(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f'文件不存在: {filename}')
    data: np.ndarray = np.loadtxt(filename, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f'文件列数不足(需要: timestamp x y z): {filename}')
    timestamps: np.ndarray = data[:, 0]
    positions: np.ndarray = data[:, 1:4]
    return (timestamps, positions)

def select_corresponding_points_by_time(ts1: np.ndarray, pos1: np.ndarray, ts2: np.ndarray, pos2: np.ndarray, time_window: float=0.1):
    indices1: list = []
    indices2: list = []
    time_diffs_list: list = []
    for i in range(len(ts1)):
        time_diffs: np.ndarray = np.abs(ts2 - ts1[i])
        min_idx: int = np.argmin(time_diffs)
        min_time_diff: float = time_diffs[min_idx]
        if min_time_diff <= time_window:
            indices1.append(i)
            indices2.append(min_idx)
            time_diffs_list.append(min_time_diff)
    return (np.array(indices1), np.array(indices2), np.array(time_diffs_list))

def build_snapshot(pos_lio: np.ndarray, mapped_points: np.ndarray, counts: np.ndarray):
    snapshot = pos_lio.copy()
    mask = counts > 0
    snapshot[mask] = mapped_points[mask]
    return snapshot

def compute_velocities(positions: np.ndarray, timestamps: np.ndarray):
    velocities: np.ndarray = np.zeros_like(positions)
    if len(positions) < 2:
        return velocities
    for i in range(len(positions) - 1):
        dt: float = timestamps[i + 1] - timestamps[i]
        if dt > 1e-06:
            velocities[i] = (positions[i + 1] - positions[i]) / dt
        else:
            velocities[i] = np.zeros(3)
    velocities[-1] = velocities[-2] if len(velocities) > 1 else np.zeros(3)
    return velocities

def compute_velocities_bspline_full(positions: np.ndarray, timestamps: np.ndarray, smoothing_factor: float=0.1):
    base_velocities = compute_velocities(positions, timestamps)
    if len(positions) < 4:
        return (base_velocities, None)
    try:
        sort_indices: np.ndarray = np.argsort(timestamps)
        timestamps_sorted: np.ndarray = timestamps[sort_indices]
        positions_sorted: np.ndarray = positions[sort_indices]
        unique_indices, unique_inverse = np.unique(timestamps_sorted, return_inverse=True)
        if len(unique_indices) < len(timestamps_sorted):
            positions_unique: np.ndarray = np.zeros((len(unique_indices), 3))
            for i in range(len(unique_indices)):
                positions_unique[i] = np.mean(positions_sorted[unique_inverse == i], axis=0)
            timestamps_final: np.ndarray = unique_indices
            positions_final: np.ndarray = positions_unique
        else:
            timestamps_final = timestamps_sorted
            positions_final = positions_sorted
        if len(positions_final) < 4:
            return (base_velocities, None)
        tck, u = splprep([positions_final[:, 0], positions_final[:, 1], positions_final[:, 2]], u=timestamps_final, s=smoothing_factor, k=3)
        velocities_list: list = splev(timestamps, tck, der=1)
        spline_positions_list: list = splev(timestamps, tck, der=0)
        velocities = np.vstack(velocities_list).T
        spline_positions = np.vstack(spline_positions_list).T
        return (velocities, spline_positions)
    except Exception:
        return (base_velocities, None)

def compute_velocities_bspline(positions: np.ndarray, timestamps: np.ndarray, smoothing_factor: float=0.1):
    velocities, _ = compute_velocities_bspline_full(positions, timestamps, smoothing_factor)
    return velocities

def wahba_svd(A: np.ndarray, B: np.ndarray, weights: np.ndarray=None):
    if len(A) == 0 or len(B) == 0:
        raise ValueError('输入向量数量为0')
    if weights is None:
        weights = np.ones(len(A))
    weights = np.asarray(weights, dtype=float)
    wsum: float = np.sum(weights)
    if wsum <= 0:
        raise ValueError('权重之和必须为正')
    weights = weights / wsum
    A_mean: np.ndarray = np.average(A, axis=0, weights=weights)
    B_mean: np.ndarray = np.average(B, axis=0, weights=weights)
    A_centered: np.ndarray = A - A_mean
    B_centered: np.ndarray = B - B_mean
    H: np.ndarray = np.zeros((3, 3))
    for i in range(len(A)):
        H += weights[i] * np.outer(B_centered[i], A_centered[i])
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    U, S, Vt = np.linalg.svd(H)
    Rm: np.ndarray = U @ Vt
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = U @ Vt
    return Rm

def umeyama_alignment(A: np.ndarray, B: np.ndarray):
    assert A.shape == B.shape
    n, _ = A.shape
    if n < 3:
        raise ValueError('Umeyama 需要至少3对对应点')
    A_mean: np.ndarray = np.mean(A, axis=0)
    B_mean: np.ndarray = np.mean(B, axis=0)
    A_centered: np.ndarray = A - A_mean
    B_centered: np.ndarray = B - B_mean
    var_A: float = np.mean(np.sum(A_centered ** 2, axis=1))
    var_A = max(var_A, 1e-12)
    H: np.ndarray = A_centered.T @ B_centered / n
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    U, S, Vt = np.linalg.svd(H)
    d: float = np.linalg.det(Vt.T @ U.T)
    S_matrix: np.ndarray = np.diag([1.0, 1.0, d])
    Rm: np.ndarray = Vt.T @ S_matrix @ U.T
    s: float = np.trace(np.diag(S) @ S_matrix) / var_A
    t: np.ndarray = B_mean - s * (Rm @ A_mean)
    return (Rm, t, s)

def apply_transformation(points: np.ndarray, Rm: np.ndarray, t: np.ndarray=None, s: float=1.0):
    if t is None:
        t = np.zeros(3)
    return s * (points @ Rm.T) + t

def compute_alignment_metrics(aligned: np.ndarray, reference: np.ndarray):
    errors: np.ndarray = np.linalg.norm(aligned - reference, axis=1)
    return {'mean': float(np.mean(errors)), 'std': float(np.std(errors)), 'max': float(np.max(errors)), 'min': float(np.min(errors)), 'rmse': float(np.sqrt(np.mean(errors ** 2))), 'errors': errors}

def tls_position_refinement(lio_points: np.ndarray, rtk_points: np.ndarray, R_init: np.ndarray, sigma: float, c_bar_squared: float, max_iter: int=10, use_adaptive_threshold: bool=True, adaptive_scale: float=2.5, verbose: bool=False):
    if verbose:
        print('\n' + '=' * 20 + ' TLS位置优化 ' + '=' * 20)
    R: np.ndarray = R_init.copy()
    N: int = len(lio_points)
    tls_stats = {'iterations': 0, 'initial_rmse': 0.0, 'final_rmse': 0.0, 'initial_inliers': N, 'final_inliers': N, 'rmse_improvement': 0.0, 'outliers_removed': 0}
    if N < 3:
        if verbose:
            print('  ✗ 点数不足 (<3)，跳过TLS优化。')
        return (R, np.ones(N, dtype=bool), tls_stats)
    lio_rotated_init: np.ndarray = apply_transformation(lio_points, R)
    residuals_init: np.ndarray = rtk_points - lio_rotated_init
    initial_rmse: float = np.sqrt(np.mean(np.sum(residuals_init ** 2, axis=1)))
    tls_stats['initial_rmse'] = initial_rmse
    if use_adaptive_threshold:
        initial_residuals: np.ndarray = rtk_points - apply_transformation(lio_points, R)
        initial_res_sq_norms: np.ndarray = np.sum(initial_residuals ** 2, axis=1)
        median_res_sq: float = np.median(initial_res_sq_norms)
        c_bar_squared_adaptive: float = median_res_sq * adaptive_scale
        initial_inliers: int = int(np.sum(initial_res_sq_norms <= c_bar_squared_adaptive))
        tls_stats['initial_inliers'] = initial_inliers
        if verbose:
            print(f'  使用自适应阈值: c_bar^2 = {c_bar_squared_adaptive:.4f}')
        if verbose:
            print(f'  初始RMSE: {initial_rmse:.4f} m, 初始内点: {initial_inliers}/{N} ({initial_inliers / N * 100:.1f}%)')
    else:
        c_bar_squared_adaptive: float = c_bar_squared * sigma ** 2
        initial_res_sq_norms: np.ndarray = np.sum(residuals_init ** 2, axis=1)
        initial_inliers: int = int(np.sum(initial_res_sq_norms <= c_bar_squared_adaptive))
        tls_stats['initial_inliers'] = initial_inliers
        if verbose:
            print(f'  使用固定统计阈值: c_bar^2 = {c_bar_squared_adaptive:.4f}')
        if verbose:
            print(f'  初始RMSE: {initial_rmse:.4f} m, 初始内点: {initial_inliers}/{N} ({initial_inliers / N * 100:.1f}%)')
    for iteration in range(max_iter):
        lio_rotated: np.ndarray = apply_transformation(lio_points, R)
        residuals: np.ndarray = rtk_points - lio_rotated
        residual_norms_sq: np.ndarray = np.sum(residuals ** 2, axis=1)
        weights: np.ndarray = (residual_norms_sq <= c_bar_squared_adaptive).astype(float)
        n_inliers: int = int(np.sum(weights))
        if verbose:
            print(f'  迭代 {iteration}: {n_inliers}/{N} 个inliers ({n_inliers / N * 100:.1f}%)')
        if n_inliers < 3:
            if verbose:
                print(f'  警告: 迭代 {iteration} 仅有 {n_inliers} 个inliers，优化中止。')
            break
        R_new: np.ndarray = wahba_svd(lio_points, rtk_points, weights)
        rotation_change: float = np.linalg.norm(R_new - R, 'fro')
        R = R_new
        tls_stats['iterations'] = iteration + 1
        if rotation_change < 1e-06:
            if verbose:
                print(f'  ✓ 收敛于迭代 {iteration}')
            break
    lio_rotated_final: np.ndarray = apply_transformation(lio_points, R)
    residuals_final: np.ndarray = rtk_points - lio_rotated_final
    final_weights_mask: np.ndarray = np.sum(residuals_final ** 2, axis=1) / sigma ** 2 <= c_bar_squared
    final_rmse: float = np.sqrt(np.mean(np.sum(residuals_final ** 2, axis=1)))
    final_inliers: int = int(np.sum(final_weights_mask))
    tls_stats['final_rmse'] = final_rmse
    tls_stats['final_inliers'] = final_inliers
    tls_stats['rmse_improvement'] = initial_rmse - final_rmse
    tls_stats['outliers_removed'] = N - final_inliers
    if verbose:
        print(f'  最终RMSE: {final_rmse:.4f} m, 最终内点: {final_inliers}/{N} ({final_inliers / N * 100:.1f}%)')
        print(f"  RMSE改善: {tls_stats['rmse_improvement']:.4f} m, 剔除外点: {tls_stats['outliers_removed']}个")
    return (R, final_weights_mask, tls_stats)

def bspline_velocities_for_cpp(positions, timestamps, smoothing_factor=None):
    if smoothing_factor is None:
        smoothing_factor = BSPLINE_SMOOTHING_FACTOR
    positions_np = np.asarray(positions, dtype=float)
    timestamps_np = np.asarray(timestamps, dtype=float)
    velocities, spline_positions = compute_velocities_bspline_full(positions_np, timestamps_np, smoothing_factor=smoothing_factor)
    return {'velocities': velocities.tolist(), 'spline': None if spline_positions is None else spline_positions.tolist()}

def tls_position_refinement_for_cpp(lio_points, rtk_points, R_init, sigma, c_bar_squared, max_iter=10, use_adaptive_threshold=True, adaptive_scale=2.5, verbose=False):
    lio_np = np.asarray(lio_points, dtype=float)
    rtk_np = np.asarray(rtk_points, dtype=float)
    R_init_np = np.asarray(R_init, dtype=float)
    R, mask, stats = tls_position_refinement(lio_np, rtk_np, R_init_np, sigma, c_bar_squared, max_iter=max_iter, use_adaptive_threshold=use_adaptive_threshold, adaptive_scale=adaptive_scale, verbose=verbose)
    return (R.tolist(), mask.astype(bool).tolist(), stats)

def visualize_full_trajectory_alignment(lio_full: np.ndarray, rtk_full: np.ndarray, lio_mapped_full: np.ndarray, filename: str):
    fig: plt.Figure = plt.figure(figsize=(15, 10), dpi=DISPLAY_DPI)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.plot(lio_full[:, 0], lio_full[:, 1], 'b-', lw=2, alpha=0.8, label='LIO Trajectory (Full)')
    ax.plot(rtk_full[:, 0], rtk_full[:, 1], 'r--', lw=1.5, alpha=0.6, label='RTK Trajectory (Reference)')
    ax.plot(lio_mapped_full[:, 0], lio_mapped_full[:, 1], 'g-', lw=2, alpha=0.8, label='LIO Trajectory (Mapped to RTK)')
    ax.set_title('Full Trajectory Alignment Comparison (XY Plane)', fontweight='bold', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    _maybe_save_and_show(filename)

def process_single_window(lio_points: np.ndarray, rtk_points: np.ndarray, ts_lio_sel: np.ndarray, ts_rtk_sel: np.ndarray, R_initial_guess: np.ndarray):
    stats: dict = {}
    metrics_original: dict = compute_alignment_metrics(rtk_points, lio_points)
    stats['initial_rmse'] = metrics_original['rmse']
    if USE_COARSE_ALIGNMENT:
        if USE_BSPLINE_SMOOTHING:
            vel_lio: np.ndarray = compute_velocities_bspline(lio_points, ts_lio_sel, smoothing_factor=BSPLINE_SMOOTHING_FACTOR)
            vel_rtk: np.ndarray = compute_velocities_bspline(rtk_points, ts_rtk_sel, smoothing_factor=BSPLINE_SMOOTHING_FACTOR)
        else:
            vel_lio: np.ndarray = compute_velocities(lio_points, ts_lio_sel)
            vel_rtk: np.ndarray = compute_velocities(rtk_points, ts_rtk_sel)
        vel_norm_lio: np.ndarray = np.linalg.norm(vel_lio, axis=1)
        vel_norm_rtk: np.ndarray = np.linalg.norm(vel_rtk, axis=1)
        valid_mask: np.ndarray = (vel_norm_lio >= VELOCITY_THRESHOLD) & (vel_norm_rtk >= VELOCITY_THRESHOLD) & (vel_norm_lio > 1e-06) & (vel_norm_rtk > 1e-06)
        if np.sum(valid_mask) < 3:
            return (None, None, None, {'error': '有效速度向量不足'})
        vel_lio_valid: np.ndarray = vel_lio[valid_mask]
        vel_rtk_valid: np.ndarray = vel_rtk[valid_mask]
        lio_points_for_umeyama: np.ndarray = lio_points[valid_mask]
        rtk_points_for_umeyama: np.ndarray = rtk_points[valid_mask]
        weights: np.ndarray = vel_norm_lio[valid_mask] * vel_norm_rtk[valid_mask]
        R_coarse: np.ndarray = wahba_svd(vel_lio_valid, vel_rtk_valid, weights)
        R_tls_init: np.ndarray = R_initial_guess @ R_coarse
        lio_coarse_aligned: np.ndarray = apply_transformation(lio_points_for_umeyama, R_coarse)
        rmse_before_tls: float = np.sqrt(np.mean(np.linalg.norm(lio_coarse_aligned - rtk_points_for_umeyama, axis=1) ** 2))
        stats['rmse_before_tls'] = rmse_before_tls
        R_coarse_final: np.ndarray = R_coarse
        if USE_TLS_REFINEMENT:
            R_tls, _, tls_stats = tls_position_refinement(lio_points=lio_points_for_umeyama, rtk_points=rtk_points_for_umeyama, R_init=R_tls_init, sigma=SIGMA_POSITION, c_bar_squared=C_BAR_POSITION_SQ, max_iter=TLS_MAX_ITER, use_adaptive_threshold=USE_ADAPTIVE_THRESHOLD, adaptive_scale=ADAPTIVE_THRESHOLD_SCALE, verbose=False)
            R_coarse_final = R_tls
            lio_tls_aligned: np.ndarray = apply_transformation(lio_points_for_umeyama, R_tls)
            rmse_after_tls: float = np.sqrt(np.mean(np.linalg.norm(lio_tls_aligned - rtk_points_for_umeyama, axis=1) ** 2))
            stats['rmse_after_tls'] = rmse_after_tls
            stats['tls_improvement'] = rmse_before_tls - rmse_after_tls
            stats['tls_improvement_percent'] = stats['tls_improvement'] / rmse_before_tls * 100 if rmse_before_tls > 0 else 0
            stats['tls_iterations'] = tls_stats['iterations']
            stats['tls_outliers_removed'] = tls_stats['outliers_removed']
            stats['tls_final_inliers'] = tls_stats['final_inliers']
            ANSI_YELLOW = '\x1b[33m'
            print(f"{ANSI_YELLOW}  [TLS] 迭代{tls_stats['iterations']}次, RMSE: {rmse_before_tls:.4f}→{rmse_after_tls:.4f} m (改善{stats['tls_improvement']:.4f} m, {stats['tls_improvement_percent']:.2f}%), 剔除外点{tls_stats['outliers_removed']}个{ANSI_RESET}")
        else:
            stats['rmse_after_tls'] = rmse_before_tls
            stats['tls_improvement'] = 0.0
            stats['tls_improvement_percent'] = 0.0
            stats['tls_iterations'] = 0
            stats['tls_outliers_removed'] = 0
            stats['tls_final_inliers'] = len(lio_points_for_umeyama)
        print_rotation_matrix('粗匹配旋转矩阵 (LIO→RTK)', R_coarse_final, ANSI_RED)
    else:
        lio_points_for_umeyama: np.ndarray = lio_points
        rtk_points_for_umeyama: np.ndarray = rtk_points
        R_coarse_final: np.ndarray = np.eye(3)
        valid_mask: np.ndarray = np.ones(len(lio_points), dtype=bool)
        if len(lio_points_for_umeyama) < 3:
            return (None, None, None, {'error': '匹配点不足'})
        print_rotation_matrix('粗匹配旋转矩阵 (LIO→RTK)', R_coarse_final, ANSI_RED)
    lio_points_for_umeyama_rot: np.ndarray = apply_transformation(lio_points_for_umeyama, R_coarse_final)
    R_umeyama, t_umeyama, s_umeyama = umeyama_alignment(lio_points_for_umeyama_rot, rtk_points_for_umeyama)
    R_fine: np.ndarray = R_umeyama @ R_coarse_final
    t_fine: np.ndarray = t_umeyama
    s_fine: float = s_umeyama
    lio_fine_aligned_eval: np.ndarray = apply_transformation(lio_points[valid_mask], R_fine, t_fine, s_fine)
    rtk_eval: np.ndarray = rtk_points[valid_mask]
    metrics_fine: dict = compute_alignment_metrics(lio_fine_aligned_eval, rtk_eval)
    stats['final_rmse'] = metrics_fine['rmse']
    print_rotation_matrix('细匹配旋转矩阵 (LIO→RTK)', R_fine, ANSI_GREEN)
    delta_norm = np.linalg.norm(R_umeyama - np.eye(3))
    print(f'{ANSI_BLUE}[细匹配] R_umeyama:\n{R_umeyama}\nΔR Fro norm = {delta_norm:.4e}{ANSI_RESET}')
    return (R_fine, t_fine, s_fine, stats)

def create_dynamic_visualization(lio_full, rtk_full, lio_mapped_history, filename):
    fig, ax = plt.subplots(figsize=(15, 10), dpi=DISPLAY_DPI)
    ax.plot(lio_full[:, 0], lio_full[:, 1], 'b-', lw=2, alpha=0.8, label='LIO Trajectory (Full)')
    ax.plot(rtk_full[:, 0], rtk_full[:, 1], 'r--', lw=1.5, alpha=0.4, label='RTK Trajectory (Original)')
    aligned_line, = ax.plot([], [], 'g-', lw=2, alpha=0.8, label='LIO Trajectory (Mapped)')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    ax.set_title('Dynamic Trajectory Alignment Process (XY Plane)', fontweight='bold', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    all_points = np.vstack([lio_full, rtk_full])
    x_min, y_min, _ = np.min(all_points, axis=0)
    x_max, y_max, _ = np.max(all_points, axis=0)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    plt.tight_layout()

    def animate(i):
        aligned_points = lio_mapped_history[i]
        aligned_line.set_data(aligned_points[:, 0], aligned_points[:, 1])
        time_text.set_text(f'Window: {i}/{len(lio_mapped_history) - 1}')
        return (aligned_line, time_text)
    anim = animation.FuncAnimation(fig, animate, frames=len(lio_mapped_history), interval=200, blit=True, repeat_delay=2000)
    try:
        anim.save(filename, writer='pillow', fps=5)
        print(f'  ✓ 动画已保存: {filename}')
    except Exception as e:
        print(f'  ✗ 保存动画失败: {e}')
        print("  请确保已安装 'Pillow' 库 (pip install Pillow)")
    plt.close(fig)

def main():
    print('=' * 70)
    print('轨迹对齐算法 - 基于Wahba问题的旋转对齐 (全流程滑窗法)')
    print('=' * 70)
    print('\n[步骤1] 加载轨迹数据...')
    lio_file: str = 'orign_path/lio_trajectory.txt'
    rtk_file: str = 'orign_path/rtk_trajectory.txt'
    try:
        ts_lio: np.ndarray
        pos_lio: np.ndarray
        ts_lio, pos_lio = load_trajectory(lio_file)
        ts_rtk: np.ndarray
        pos_rtk: np.ndarray
        ts_rtk, pos_rtk = load_trajectory(rtk_file)
        print(f'  ✓ LIO轨迹: {len(pos_lio)} 个点 (基准)')
        print(f'  ✓ RTK轨迹: {len(pos_rtk)} 个点 (待对齐)')
    except Exception as e:
        print(f'  ✗ 加载失败: {e}')
        return
    print('\n[步骤2] 全流程滑窗对齐...')
    print(f'  滑窗大小: {SLIDING_WINDOW_SIZE} 点, 步长: {SLIDING_WINDOW_STEP} 点')
    current_lio_start_idx: int = 0
    R_last_window: np.ndarray = np.eye(3)
    t_last_window: np.ndarray = np.zeros(3)
    s_last_window: float = 1.0
    all_window_stats: list = []
    total_windows: int = 0
    lio_full_mapped: np.ndarray = np.zeros_like(pos_lio)
    blend_counts: np.ndarray = np.zeros(len(pos_lio), dtype=float)
    lio_aligned_history = []
    lio_aligned_history.append(build_snapshot(pos_lio, lio_full_mapped, blend_counts))
    while current_lio_start_idx + SLIDING_WINDOW_SIZE <= len(ts_lio):
        window_num: int = total_windows + 1
        print('-' * 70)
        print(f'处理滑窗 #{window_num} (LIO索引: {current_lio_start_idx} -> {current_lio_start_idx + SLIDING_WINDOW_SIZE - 1})')
        start_time: float = time.time()
        lio_window_indices: np.ndarray = np.arange(current_lio_start_idx, current_lio_start_idx + SLIDING_WINDOW_SIZE)
        ts_lio_window: np.ndarray = ts_lio[lio_window_indices]
        pos_lio_window: np.ndarray = pos_lio[lio_window_indices]
        idx_lio_in_window: np.ndarray
        idx_rtk_matched: np.ndarray
        idx_lio_in_window, idx_rtk_matched, _ = select_corresponding_points_by_time(ts_lio_window, pos_lio_window, ts_rtk, pos_rtk, TIME_MATCH_THRESHOLD)
        if len(idx_lio_in_window) < 4:
            print(f'  ✗ 匹配点不足 ({len(idx_lio_in_window)}个)，跳过此窗口。')
            lio_aligned_history.append(lio_full_mapped.copy())
            current_lio_start_idx += SLIDING_WINDOW_STEP
            continue
        lio_points: np.ndarray = pos_lio_window[idx_lio_in_window]
        rtk_points: np.ndarray = pos_rtk[idx_rtk_matched]
        ts_lio_sel: np.ndarray = ts_lio_window[idx_lio_in_window]
        ts_rtk_sel: np.ndarray = ts_rtk[idx_rtk_matched]
        R_fine: np.ndarray
        t_fine: np.ndarray
        s_fine: float
        stats: dict
        R_fine, t_fine, s_fine, stats = process_single_window(lio_points, rtk_points, ts_lio_sel, ts_rtk_sel, R_last_window)
        end_time: float = time.time()
        computation_time: float = end_time - start_time
        stats['computation_time'] = computation_time
        stats['window_index'] = window_num
        if R_fine is None:
            print(f"  ✗ 窗口对齐失败: {stats.get('error', '未知错误')}")
            R_fine, t_fine, s_fine = (R_last_window, t_last_window, s_last_window)
            stats['improvement_rmse'] = 0
            stats['improvement_percent'] = 0
            print(f'  ✓ 计算时间: {computation_time:.4f} 秒')
        else:
            R_last_window, t_last_window, s_last_window = (R_fine, t_fine, s_fine)
            initial_rmse: float = stats.get('initial_rmse', 0)
            final_rmse: float = stats.get('final_rmse', 0)
            improvement: float = initial_rmse - final_rmse
            improvement_percent: float = improvement / initial_rmse * 100 if initial_rmse > 0 else 0
            stats['improvement_rmse'] = improvement
            stats['improvement_percent'] = improvement_percent
            print(f'  ✓ 初始RMSE: {initial_rmse:.4f} m -> 最终RMSE: {final_rmse:.4f} m')
            print(f'  ✓ 提升: {improvement:.4f} m ({improvement_percent:.2f}%)')
            print(f'  ✓ 计算时间: {computation_time:.4f} 秒')
        all_window_stats.append(stats)
        min_lio_ts: float = np.min(ts_lio_sel)
        max_lio_ts: float = np.max(ts_lio_sel)
        lio_segment_mask: np.ndarray = (ts_lio >= min_lio_ts) & (ts_lio <= max_lio_ts)
        if np.any(lio_segment_mask):
            new_points = apply_transformation(pos_lio[lio_segment_mask], R_fine, t_fine, s_fine)
            counts_col = blend_counts[lio_segment_mask][:, None]
            lio_full_mapped[lio_segment_mask] = (lio_full_mapped[lio_segment_mask] * counts_col + new_points) / (counts_col + 1.0)
            blend_counts[lio_segment_mask] += 1.0
            print(f'  ✓ 已更新 {np.sum(lio_segment_mask)} 个LIO点的RTK映射')
        lio_aligned_history.append(build_snapshot(pos_lio, lio_full_mapped, blend_counts))
        total_windows += 1
        current_lio_start_idx += SLIDING_WINDOW_STEP
    print('\n' + '=' * 70)
    print('全流程对齐完成 - 总结')
    print('=' * 70)
    if not all_window_stats:
        print('未处理任何窗口，程序退出。')
        return
    print(f'总共处理了 {total_windows} 个滑窗。')
    avg_time: float = np.mean([s['computation_time'] for s in all_window_stats])
    avg_improvement: float = np.mean([s['improvement_percent'] for s in all_window_stats])
    print(f'\n平均计算时间: {avg_time:.4f} 秒/窗')
    print(f'平均RMSE提升: {avg_improvement:.2f} %/窗')
    if USE_TLS_REFINEMENT:
        tls_improvements = [s.get('tls_improvement', 0) for s in all_window_stats if 'tls_improvement' in s]
        tls_improvement_percents = [s.get('tls_improvement_percent', 0) for s in all_window_stats if 'tls_improvement_percent' in s]
        tls_outliers = [s.get('tls_outliers_removed', 0) for s in all_window_stats if 'tls_outliers_removed' in s]
        if tls_improvements:
            avg_tls_improvement = np.mean(tls_improvements)
            avg_tls_improvement_percent = np.mean(tls_improvement_percents)
            total_outliers = sum(tls_outliers)
            print(f'\n{ANSI_GREEN}========== TLS 效果统计 =========={ANSI_RESET}')
            print(f'{ANSI_GREEN}平均TLS RMSE改善: {avg_tls_improvement:.4f} m ({avg_tls_improvement_percent:.2f}%){ANSI_RESET}')
            print(f'{ANSI_GREEN}总共剔除外点: {total_outliers} 个{ANSI_RESET}')
            print(f'{ANSI_GREEN}TLS 对最终结果的贡献占比: {(avg_tls_improvement_percent / avg_improvement * 100 if avg_improvement > 0 else 0):.1f}%{ANSI_RESET}')
    print('\n详细统计:')
    print('-' * 110)
    if USE_TLS_REFINEMENT:
        print(f"{'窗口 #':<8} {'时间 (s)':<10} {'总RMSE提升 (%)':<16} {'最终RMSE (m)':<14} {'TLS改善 (m)':<14} {'TLS剔除外点':<12}")
    else:
        print(f"{'窗口 #':<10} {'时间 (s)':<15} {'RMSE提升 (%)':<20} {'最终RMSE (m)':<15}")
    print('-' * 110)
    for stats in all_window_stats:
        if USE_TLS_REFINEMENT:
            print(f"{stats['window_index']:<8} {stats['computation_time']:<10.4f} {stats['improvement_percent']:<16.2f} {stats.get('final_rmse', 0):<14.4f} {stats.get('tls_improvement', 0):<14.4f} {stats.get('tls_outliers_removed', 0):<12}")
        else:
            print(f"{stats['window_index']:<10} {stats['computation_time']:<15.4f} {stats['improvement_percent']:<20.2f} {stats.get('final_rmse', 0):<15.4f}")
    print('-' * 110)
    lio_mapped_final = build_snapshot(pos_lio, lio_full_mapped, blend_counts)
    print('\n[步骤4] 生成最终静态可视化...')
    visualize_full_trajectory_alignment(lio_full=pos_lio, rtk_full=pos_rtk, lio_mapped_full=lio_mapped_final, filename=str(PICTURE_DIR / 'full_trajectory_comparison_sliding_window.png'))
    print('✓ 最终对齐效果图已保存。')
    if SHOW_FIGURES:
        print('\n[步骤5] 显示所有图像...')
        print('  请注意：关闭所有图像窗口后程序将最终退出。')
        plt.show()
    print('\n' + '=' * 70)
    print('程序执行完毕。')
    print('=' * 70)
    if False:
        print('\n[步骤final] 正在生成动态对齐过程可视化 (GIF)...')
        create_dynamic_visualization(pos_lio, pos_rtk, lio_aligned_history, filename=str(PICTURE_DIR / 'dynamic_alignment_process.gif'))
        print('✓ 动态对齐过程图已保存。已经退出程序')
    return
if __name__ == '__main__':
    main()
