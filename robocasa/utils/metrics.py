import numpy as np


def compute_jerk(positions, dt):
    """Compute smoothness metrics from a position trajectory.

    Jerk = 3rd derivative of position w.r.t. time (finite differences).

    Args:
        positions: (T, D) array of positions.
        dt: time step in seconds (1/control_freq).

    Returns:
        dict with jerk_mean, jerk_max, jerk_rms, num_steps.
    """
    if len(positions) < 4:
        return {'jerk_mean': 0.0, 'jerk_max': 0.0, 'jerk_rms': 0.0, 'num_steps': len(positions)}

    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    jerk_norms = np.linalg.norm(jerk, axis=1)

    return {
        'jerk_mean': float(jerk_norms.mean()),
        'jerk_max': float(jerk_norms.max()),
        'jerk_rms': float(np.sqrt((jerk_norms ** 2).mean())),
        'num_steps': len(positions),
    }


def compute_path_length(positions):
    """Total Euclidean path length."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def compute_safe_boundary_violations(robot_positions, person_pos, safe_radius=1.0):
    """Count steps where robot enters the human's safe boundary.

    Args:
        robot_positions: (T, D) array of robot positions.
        person_pos: (D,) fixed person position.
        safe_radius: safe boundary radius in meters.

    Returns:
        dict with violation_count, violation_ratio, min_distance.
    """
    robot_positions = np.asarray(robot_positions)
    person_pos = np.asarray(person_pos)
    if len(robot_positions) == 0:
        return {'violation_count': 0, 'violation_ratio': 0.0, 'min_distance': float('inf')}

    distances = np.linalg.norm(robot_positions[:, :2] - person_pos[:2], axis=1)
    violations = distances < safe_radius
    return {
        'violation_count': int(violations.sum()),
        'violation_ratio': float(violations.mean()),
        'min_distance': float(distances.min()),
    }


def compute_all_metrics(positions, dt, person_pos=None, safe_radius=1.0):
    """Compute all evaluation metrics. Extension point for future metrics."""
    positions = np.asarray(positions)
    metrics = {}
    metrics.update(compute_jerk(positions, dt))
    metrics['path_length'] = compute_path_length(positions)
    if person_pos is not None:
        metrics.update(compute_safe_boundary_violations(positions, person_pos, safe_radius))
    return metrics
