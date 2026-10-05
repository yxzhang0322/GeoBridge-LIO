import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob
from datetime import datetime
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import json

class RTKConstraintAnalyzer:

    def __init__(self, log_directory='/tmp/rtk_validation_logs/'):
        self.log_directory = log_directory
        self.trajectory_data = {}
        self.constraint_events = []
        self.ekf_states = []
        self.validation_metrics = {}
        plt.style.use('seaborn-v0_8')
        sns.set_palette('husl')

    def load_validation_logs(self):
        log_files = glob.glob(os.path.join(self.log_directory, 'validation_*.log'))
        if not log_files:
            print(f'No validation log files found in {self.log_directory}')
            return False
        print(f'Loading {len(log_files)} validation log files...')
        for log_file in log_files:
            self._parse_log_file(log_file)
        return True

    def _parse_log_file(self, log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('RTK Constraint Validation Log'):
                    continue
                if line.startswith('Columns:'):
                    continue
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                timestamp = float(parts[0])
                event_type = parts[1]
                if event_type == 'CONSTRAINT':
                    self._parse_constraint_event(timestamp, parts[2:])
                elif event_type == 'TRAJECTORY':
                    self._parse_trajectory_point(timestamp, parts[2:])
                elif event_type == 'EKF_STATE':
                    self._parse_ekf_state(timestamp, parts[2:])
        except Exception as e:
            print(f'Error parsing {log_file}: {e}')

    def _parse_constraint_event(self, timestamp, data):
        try:
            event = {'timestamp': timestamp, 'error_before': float(data[0]) if data[0] != '-1' else None, 'error_after': float(data[1]) if data[1] != '-1' else None, 'correction_magnitude': float(data[2]), 'rtk_quality': float(data[3]), 'accepted': data[4] == 'ACCEPTED' if len(data) > 4 else True}
            self.constraint_events.append(event)
        except:
            pass

    def load_trajectory_csv(self, csv_file, trajectory_name):
        try:
            df = pd.read_csv(csv_file)
            trajectory = []
            for _, row in df.iterrows():
                point = {'timestamp': row['timestamp'], 'position': np.array([row['x'], row['y'], row['z']]), 'orientation': np.array([row['qx'], row['qy'], row['qz'], row['qw']]) if 'qx' in row else None}
                trajectory.append(point)
            self.trajectory_data[trajectory_name] = trajectory
            print(f"Loaded {len(trajectory)} points for trajectory '{trajectory_name}'")
            return True
        except Exception as e:
            print(f'Error loading trajectory CSV {csv_file}: {e}')
            return False

    def compute_trajectory_alignment_metrics(self, ref_traj='rtk', test_traj='world'):
        if ref_traj not in self.trajectory_data or test_traj not in self.trajectory_data:
            print(f'Trajectories {ref_traj} or {test_traj} not found')
            return None
        ref_data = self.trajectory_data[ref_traj]
        test_data = self.trajectory_data[test_traj]
        ref_times = [p['timestamp'] for p in ref_data]
        test_times = [p['timestamp'] for p in test_data]
        t_start = max(min(ref_times), min(test_times))
        t_end = min(max(ref_times), max(test_times))
        if t_start >= t_end:
            print('No overlapping time range found')
            return None
        dt = 0.1
        common_times = np.arange(t_start, t_end, dt)
        ref_positions = self._interpolate_trajectory(ref_data, common_times)
        test_positions = self._interpolate_trajectory(test_data, common_times)
        errors = np.linalg.norm(ref_positions - test_positions, axis=1)
        metrics = {'rmse': np.sqrt(np.mean(errors ** 2)), 'mean_error': np.mean(errors), 'median_error': np.median(errors), 'max_error': np.max(errors), 'std_error': np.std(errors), 'error_95_percentile': np.percentile(errors, 95), 'total_distance': np.sum(np.linalg.norm(np.diff(ref_positions, axis=0), axis=1)), 'analysis_duration': t_end - t_start, 'num_samples': len(common_times)}
        return {'metrics': metrics, 'times': common_times, 'errors': errors, 'ref_positions': ref_positions, 'test_positions': test_positions}

    def _interpolate_trajectory(self, trajectory, target_times):
        times = np.array([p['timestamp'] for p in trajectory])
        positions = np.array([p['position'] for p in trajectory])
        valid_mask = (target_times >= times[0]) & (target_times <= times[-1])
        interpolated = np.zeros((len(target_times), 3))
        if np.any(valid_mask):
            for i in range(3):
                f = interp1d(times, positions[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
                interpolated[valid_mask, i] = f(target_times[valid_mask])
        return interpolated

    def analyze_constraint_effectiveness(self):
        if not self.constraint_events:
            print('No constraint events found')
            return None
        events = pd.DataFrame(self.constraint_events)
        valid_events = events.dropna(subset=['error_before', 'error_after'])
        if valid_events.empty:
            print('No valid constraint events with error data found')
            return None
        error_reductions = valid_events['error_before'] - valid_events['error_after']
        relative_reductions = error_reductions / valid_events['error_before'] * 100
        effectiveness = {'total_constraints': len(events), 'valid_constraints': len(valid_events), 'accepted_constraints': len(events[events['accepted']]), 'acceptance_rate': events['accepted'].mean() * 100, 'mean_error_reduction': error_reductions.mean(), 'median_error_reduction': error_reductions.median(), 'max_error_reduction': error_reductions.max(), 'mean_relative_reduction': relative_reductions.mean(), 'successful_reductions': (error_reductions > 0.05).sum(), 'success_rate': (error_reductions > 0.05).mean() * 100, 'average_correction_magnitude': events['correction_magnitude'].mean(), 'average_rtk_quality': events['rtk_quality'].mean()}
        return {'effectiveness': effectiveness, 'events_df': valid_events, 'error_reductions': error_reductions, 'relative_reductions': relative_reductions}

    def create_comprehensive_visualization(self, save_dir='./validation_plots/'):
        os.makedirs(save_dir, exist_ok=True)
        self._plot_trajectory_3d_comparison(save_dir)
        self._plot_error_time_series(save_dir)
        self._plot_constraint_effectiveness(save_dir)
        self._plot_error_distributions(save_dir)
        self._plot_constraint_timing(save_dir)
        self._create_performance_dashboard(save_dir)
        print(f'Validation plots saved to {save_dir}')

    def _plot_trajectory_3d_comparison(self, save_dir):
        fig = plt.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        colors = {'rtk': 'red', 'world': 'blue', 'map': 'green'}
        labels = {'rtk': 'RTK GPS', 'world': 'World (EKF)', 'map': 'Map (EKF)'}
        for traj_name, traj_data in self.trajectory_data.items():
            if traj_name in colors:
                positions = np.array([p['position'] for p in traj_data])
                ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=colors[traj_name], label=labels[traj_name], alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend()
        ax2 = fig.add_subplot(222)
        for traj_name, traj_data in self.trajectory_data.items():
            if traj_name in colors:
                positions = np.array([p['position'] for p in traj_data])
                ax2.plot(positions[:, 0], positions[:, 1], color=colors[traj_name], label=labels[traj_name], alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Trajectory XY View')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        ax3 = fig.add_subplot(223)
        for traj_name, traj_data in self.trajectory_data.items():
            if traj_name in colors:
                times = [p['timestamp'] for p in traj_data]
                altitudes = [p['position'][2] for p in traj_data]
                ax3.plot(times, altitudes, color=colors[traj_name], label=labels[traj_name])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude Profile')
        ax3.legend()
        ax3.grid(True)
        ax4 = fig.add_subplot(224)
        alignment = self.compute_trajectory_alignment_metrics('rtk', 'world')
        if alignment:
            distances = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(alignment['ref_positions'], axis=0), axis=1)]))
            ax4.plot(distances, alignment['errors'], 'b-', alpha=0.7)
            ax4.set_xlabel('Distance Traveled (m)')
            ax4.set_ylabel('Position Error (m)')
            ax4.set_title('Error vs Distance')
            ax4.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_time_series(self, save_dir):
        alignment = self.compute_trajectory_alignment_metrics('rtk', 'world')
        if not alignment:
            return
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        times = alignment['times']
        errors = alignment['errors']
        axes[0].plot(times, errors, 'b-', alpha=0.7, linewidth=1)
        axes[0].axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}m')
        axes[0].fill_between(times, 0, errors, alpha=0.3)
        axes[0].set_ylabel('Position Error (m)')
        axes[0].set_title('RTK-World Trajectory Error Over Time')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(times, errors, 'b-', alpha=0.7, linewidth=1)
        if self.constraint_events:
            constraint_times = [e['timestamp'] for e in self.constraint_events if e['accepted']]
            constraint_errors = [np.interp(t, times, errors) for t in constraint_times]
            axes[1].scatter(constraint_times, constraint_errors, color='red', s=50, zorder=5, alpha=0.8, label='RTK Constraints')
        axes[1].set_ylabel('Position Error (m)')
        axes[1].set_title('Error with RTK Constraint Events')
        axes[1].legend()
        axes[1].grid(True)
        window_size = 100
        if len(errors) > window_size:
            sliding_mean = np.convolve(errors, np.ones(window_size) / window_size, mode='valid')
            sliding_std = np.array([np.std(errors[i:i + window_size]) for i in range(len(errors) - window_size + 1)])
            sliding_times = times[window_size - 1:]
            axes[2].plot(sliding_times, sliding_mean, 'g-', label='Mean Error (sliding)')
            axes[2].fill_between(sliding_times, sliding_mean - sliding_std, sliding_mean + sliding_std, alpha=0.3, color='green', label='±1σ')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Position Error (m)')
        axes[2].set_title('Sliding Window Error Statistics')
        axes[2].legend()
        axes[2].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_time_series.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_constraint_effectiveness(self, save_dir):
        effectiveness_data = self.analyze_constraint_effectiveness()
        if not effectiveness_data:
            return
        events_df = effectiveness_data['events_df']
        error_reductions = effectiveness_data['error_reductions']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].scatter(events_df['error_before'], events_df['error_after'], c=events_df['rtk_quality'], s=60, alpha=0.7, cmap='viridis')
        axes[0, 0].plot([0, events_df['error_before'].max()], [0, events_df['error_before'].max()], 'r--', alpha=0.5, label='No improvement line')
        axes[0, 0].set_xlabel('Error Before Constraint (m)')
        axes[0, 0].set_ylabel('Error After Constraint (m)')
        axes[0, 0].set_title('Constraint Effectiveness')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
        cbar.set_label('RTK Quality Score')
        axes[0, 1].hist(error_reductions, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', label='No improvement')
        axes[0, 1].axvline(x=np.mean(error_reductions), color='green', linestyle='--', label=f'Mean: {np.mean(error_reductions):.3f}m')
        axes[0, 1].set_xlabel('Error Reduction (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Reduction Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[1, 0].scatter(events_df['timestamp'], events_df['correction_magnitude'], c=events_df['rtk_quality'], s=60, alpha=0.7, cmap='plasma')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Correction Magnitude (m)')
        axes[1, 0].set_title('Constraint Magnitude Over Time')
        axes[1, 0].grid(True)
        cbar2 = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar2.set_label('RTK Quality Score')
        axes[1, 1].scatter(events_df['rtk_quality'], error_reductions, c=events_df['correction_magnitude'], s=60, alpha=0.7, cmap='coolwarm')
        axes[1, 1].set_xlabel('RTK Quality Score')
        axes[1, 1].set_ylabel('Error Reduction (m)')
        axes[1, 1].set_title('Quality vs Effectiveness')
        axes[1, 1].grid(True)
        cbar3 = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar3.set_label('Correction Magnitude (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'constraint_effectiveness.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_dashboard(self, save_dir):
        fig = plt.figure(figsize=(20, 12))
        alignment = self.compute_trajectory_alignment_metrics('rtk', 'world')
        effectiveness = self.analyze_constraint_effectiveness()
        if not alignment or not effectiveness:
            return
        metrics = alignment['metrics']
        eff_metrics = effectiveness['effectiveness']
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        ax_metrics = fig.add_subplot(gs[0, :2])
        ax_metrics.axis('off')
        metrics_text = f"\n        TRAJECTORY ALIGNMENT METRICS:\n        • RMSE: {metrics['rmse']:.4f} m\n        • Mean Error: {metrics['mean_error']:.4f} m\n        • Max Error: {metrics['max_error']:.4f} m\n        • 95th Percentile: {metrics['error_95_percentile']:.4f} m\n\n        CONSTRAINT EFFECTIVENESS:\n        • Success Rate: {eff_metrics['success_rate']:.1f}%\n        • Mean Reduction: {eff_metrics['mean_error_reduction']:.4f} m\n        • Acceptance Rate: {eff_metrics['acceptance_rate']:.1f}%\n        • Average RTK Quality: {eff_metrics['average_rtk_quality']:.1f}\n        "
        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax_indicators = fig.add_subplot(gs[0, 2:])
        error_score = max(0, 1 - metrics['rmse'] / 2.0)
        effectiveness_score = eff_metrics['success_rate'] / 100.0
        quality_score = min(1.0, eff_metrics['average_rtk_quality'] / 80.0)
        overall_score = (error_score + effectiveness_score + quality_score) / 3.0
        scores = [error_score, effectiveness_score, quality_score, overall_score]
        labels = ['Error\nPerformance', 'Constraint\nEffectiveness', 'RTK\nQuality', 'Overall\nScore']
        colors = ['red' if s < 0.5 else 'yellow' if s < 0.8 else 'green' for s in scores]
        bars = ax_indicators.bar(labels, scores, color=colors, alpha=0.7)
        ax_indicators.set_ylim(0, 1)
        ax_indicators.set_ylabel('Performance Score')
        ax_indicators.set_title('System Performance Indicators')
        ax_indicators.grid(True, alpha=0.3)
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax_indicators.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        ax_trend = fig.add_subplot(gs[1, :])
        times = alignment['times']
        errors = alignment['errors']
        window = min(100, len(errors) // 10)
        if window > 1:
            smooth_errors = np.convolve(errors, np.ones(window) / window, mode='same')
            ax_trend.plot(times, smooth_errors, 'b-', linewidth=2, label='Smoothed Error')
        ax_trend.plot(times, errors, 'lightblue', alpha=0.5, linewidth=0.5, label='Raw Error')
        if self.constraint_events:
            constraint_times = [e['timestamp'] for e in self.constraint_events if e['accepted']]
            if constraint_times:
                constraint_errors = [np.interp(t, times, errors) for t in constraint_times]
                ax_trend.scatter(constraint_times, constraint_errors, color='red', s=30, alpha=0.8, label='RTK Constraints', zorder=5)
        ax_trend.set_xlabel('Time (s)')
        ax_trend.set_ylabel('Position Error (m)')
        ax_trend.set_title('Error Trend with RTK Constraints')
        ax_trend.legend()
        ax_trend.grid(True, alpha=0.3)
        ax_dist = fig.add_subplot(gs[2, :2])
        ax_dist.hist(errors, bins=50, alpha=0.7, color='skyblue', density=True, edgecolor='black')
        ax_dist.axvline(x=metrics['mean_error'], color='red', linestyle='--', label=f"Mean: {metrics['mean_error']:.3f}m")
        ax_dist.axvline(x=metrics['median_error'], color='green', linestyle='--', label=f"Median: {metrics['median_error']:.3f}m")
        ax_dist.set_xlabel('Position Error (m)')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title('Error Distribution')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        ax_summary = fig.add_subplot(gs[2, 2:])
        if effectiveness and 'error_reductions' in effectiveness:
            reductions = effectiveness['error_reductions']
            positive_reductions = reductions[reductions > 0]
            negative_reductions = reductions[reductions <= 0]
            ax_summary.bar(['Improvements', 'No Effect/Worse'], [len(positive_reductions), len(negative_reductions)], color=['green', 'red'], alpha=0.7)
            ax_summary.set_ylabel('Number of Constraints')
            ax_summary.set_title('Constraint Outcome Summary')
            ax_summary.grid(True, alpha=0.3)
            total = len(reductions)
            for i, (count, label) in enumerate([(len(positive_reductions), 'Improvements'), (len(negative_reductions), 'No Effect/Worse')]):
                percentage = count / total * 100 if total > 0 else 0
                ax_summary.text(i, count + max(count) * 0.01, f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        assessment_text = ''
        if overall_score >= 0.8:
            assessment_text = '🟢 EXCELLENT: RTK constraints are highly effective'
        elif overall_score >= 0.6:
            assessment_text = '🟡 GOOD: RTK constraints show moderate effectiveness'
        elif overall_score >= 0.4:
            assessment_text = '🟠 FAIR: RTK constraints have limited effectiveness'
        else:
            assessment_text = '🔴 POOR: RTK constraints are not working effectively'
        fig.suptitle(f'RTK Constraint Validation Dashboard\n{assessment_text}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(save_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_validation_report(self, output_file='rtk_validation_report.json'):
        alignment = self.compute_trajectory_alignment_metrics('rtk', 'world')
        effectiveness = self.analyze_constraint_effectiveness()
        report = {'analysis_timestamp': datetime.now().isoformat(), 'log_directory': self.log_directory, 'data_summary': {'trajectories_loaded': list(self.trajectory_data.keys()), 'constraint_events': len(self.constraint_events), 'trajectory_points': {name: len(traj) for name, traj in self.trajectory_data.items()}}}
        if alignment:
            report['trajectory_alignment'] = alignment['metrics']
        if effectiveness:
            report['constraint_effectiveness'] = effectiveness['effectiveness']
        if alignment and effectiveness:
            error_score = max(0, 1 - alignment['metrics']['rmse'] / 2.0)
            effectiveness_score = effectiveness['effectiveness']['success_rate'] / 100.0
            overall_score = (error_score + effectiveness_score) / 2.0
            report['overall_assessment'] = {'error_performance_score': error_score, 'constraint_effectiveness_score': effectiveness_score, 'overall_score': overall_score, 'recommendation': self._get_recommendation(overall_score)}
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'Validation report saved to {output_file}')
        return report

    def _get_recommendation(self, overall_score):
        if overall_score >= 0.8:
            return 'RTK constraint system is performing excellently. Continue current configuration.'
        elif overall_score >= 0.6:
            return 'RTK constraint system shows good performance. Consider fine-tuning parameters for better results.'
        elif overall_score >= 0.4:
            return 'RTK constraint system has limited effectiveness. Review RTK quality thresholds and constraint acceptance criteria.'
        else:
            return 'RTK constraint system is not working effectively. Check RTK signal quality, timing synchronization, and constraint implementation.'

def main():
    parser = argparse.ArgumentParser(description='RTK Constraint Validation Analysis')
    parser.add_argument('--log-dir', default='/tmp/rtk_validation_logs/', help='Directory containing validation log files')
    parser.add_argument('--output-dir', default='./validation_analysis/', help='Output directory for plots and reports')
    parser.add_argument('--rtk-csv', help='RTK trajectory CSV file (optional)')
    parser.add_argument('--world-csv', help='World trajectory CSV file (optional)')
    parser.add_argument('--map-csv', help='Map trajectory CSV file (optional)')
    args = parser.parse_args()
    analyzer = RTKConstraintAnalyzer(args.log_dir)
    print('Loading validation data...')
    if not analyzer.load_validation_logs():
        print('Failed to load validation logs')
        return
    if args.rtk_csv:
        analyzer.load_trajectory_csv(args.rtk_csv, 'rtk')
    if args.world_csv:
        analyzer.load_trajectory_csv(args.world_csv, 'world')
    if args.map_csv:
        analyzer.load_trajectory_csv(args.map_csv, 'map')
    os.makedirs(args.output_dir, exist_ok=True)
    print('Computing trajectory alignment metrics...')
    alignment = analyzer.compute_trajectory_alignment_metrics('rtk', 'world')
    if alignment:
        print(f"RMSE: {alignment['metrics']['rmse']:.4f} m")
        print(f"Mean Error: {alignment['metrics']['mean_error']:.4f} m")
        print(f"Max Error: {alignment['metrics']['max_error']:.4f} m")
    print('Analyzing constraint effectiveness...')
    effectiveness = analyzer.analyze_constraint_effectiveness()
    if effectiveness:
        eff = effectiveness['effectiveness']
        print(f"Constraint Success Rate: {eff['success_rate']:.1f}%")
        print(f"Mean Error Reduction: {eff['mean_error_reduction']:.4f} m")
        print(f"Acceptance Rate: {eff['acceptance_rate']:.1f}%")
    print('Creating visualizations...')
    analyzer.create_comprehensive_visualization(args.output_dir)
    report_file = os.path.join(args.output_dir, 'validation_report.json')
    analyzer.generate_validation_report(report_file)
    print(f'Analysis complete. Results saved to {args.output_dir}')
if __name__ == '__main__':
    main()
