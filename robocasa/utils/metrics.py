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


def compute_safe_boundary_violations(robot_positions, obstacle_positions, safe_radius=0.7):
    """Count steps where robot enters an obstacle's safe boundary.

    Args:
        robot_positions: (T, D) array of robot positions.
        obstacle_positions: (T, N, 2) array of obstacle positions per step,
            or (N, 2) for fixed obstacles, or (T,) list of dicts
            {name: distance} from per-step intrusion checks.
        safe_radius: safe boundary radius in meters.

    Returns:
        dict with violation_count, violation_ratio, min_distance.
    """
    robot_positions = np.asarray(robot_positions)
    if len(robot_positions) == 0:
        return {'violation_count': 0, 'violation_ratio': 0.0, 'min_distance': float('inf')}

    # Handle list-of-dicts format from _post_action intrusion data
    if isinstance(obstacle_positions, list) and len(obstacle_positions) > 0 and isinstance(obstacle_positions[0], dict):
        min_dists = []
        for dist_dict in obstacle_positions:
            if dist_dict:
                min_dists.append(min(dist_dict.values()))
            else:
                min_dists.append(float('inf'))
        min_dists = np.array(min_dists)
        violations = min_dists < safe_radius
        return {
            'violation_count': int(violations.sum()),
            'violation_ratio': float(violations.mean()),
            'min_distance': float(min_dists.min()),
        }

    # Handle fixed obstacle position(s)
    obstacle_positions = np.asarray(obstacle_positions)
    if obstacle_positions.ndim == 1:
        obstacle_positions = obstacle_positions.reshape(1, -1)

    all_dists = []
    for obs_pos in obstacle_positions:
        dists = np.linalg.norm(robot_positions[:, :2] - obs_pos[:2], axis=1)
        all_dists.append(dists)
    # Per-step minimum distance across all obstacles
    all_dists = np.array(all_dists)  # (N, T)
    min_dists = all_dists.min(axis=0)  # (T,)
    violations = min_dists < safe_radius
    return {
        'violation_count': int(violations.sum()),
        'violation_ratio': float(violations.mean()),
        'min_distance': float(min_dists.min()),
    }


def compute_obstacle_intrusion_metrics(obstacle_distance_history, obstacle_contact_history,
                                       boundary_threshold=0.7):
    """Compute obstacle intrusion metrics from per-step tracking data.

    Args:
        obstacle_distance_history: list of dicts, each {obstacle_name: distance}.
        obstacle_contact_history: list of dicts, each {obstacle_name: bool}.
        boundary_threshold: distance threshold for boundary violation.

    Returns:
        dict with intrusion metrics.
    """
    if not obstacle_distance_history:
        return {
            'obstacle_min_distance': float('inf'),
            'obstacle_contact_steps': 0,
            'obstacle_contact_ratio': 0.0,
            'boundary_violation_steps': 0,
            'boundary_violation_ratio': 0.0,
        }

    T = len(obstacle_distance_history)

    # Min distance across all steps and obstacles
    min_dists_per_step = []
    for dist_dict in obstacle_distance_history:
        if dist_dict:
            min_dists_per_step.append(min(dist_dict.values()))
        else:
            min_dists_per_step.append(float('inf'))
    min_dists_per_step = np.array(min_dists_per_step)

    # Contact steps
    contact_steps = 0
    for contact_dict in obstacle_contact_history:
        if any(contact_dict.values()):
            contact_steps += 1

    # Boundary violation steps
    violation_steps = int((min_dists_per_step < boundary_threshold).sum())

    return {
        'obstacle_min_distance': float(min_dists_per_step.min()),
        'obstacle_mean_distance': float(min_dists_per_step.mean()),
        'obstacle_contact_steps': contact_steps,
        'obstacle_contact_ratio': float(contact_steps / T),
        'boundary_violation_steps': violation_steps,
        'boundary_violation_ratio': float(violation_steps / T),
    }


def compute_navigation_success_metrics(pos_dist, pos_threshold, ori_cos, ori_threshold,
                                       dst_is_door=False):
    """Compute navigation success metrics.

    Args:
        pos_dist: distance from robot to target position.
        pos_threshold: threshold for position success.
        ori_cos: cosine similarity for orientation check.
        ori_threshold: threshold for orientation success.
        dst_is_door: if True, orientation pass uses <= instead of >=.

    Returns:
        dict with navigation success metrics.
    """
    pos_pass = pos_dist <= pos_threshold
    if dst_is_door:
        ori_pass = ori_cos <= ori_threshold
    else:
        ori_pass = ori_cos >= ori_threshold

    return {
        'pos_dist': float(pos_dist),
        'pos_threshold': float(pos_threshold),
        'pos_pass': bool(pos_pass),
        'ori_cos': float(ori_cos),
        'ori_threshold': float(ori_threshold),
        'ori_pass': bool(ori_pass),
        'task_success': bool(pos_pass and ori_pass),
    }


def compute_all_metrics(positions, dt, obstacle_distance_history=None,
                        obstacle_contact_history=None, boundary_threshold=0.7,
                        pos_dist=None, pos_threshold=None, ori_cos=None,
                        ori_threshold=None, dst_is_door=False):
    """Compute all evaluation metrics for a navigation episode.

    Args:
        positions: (T, D) array of robot base positions.
        dt: time step in seconds (1/control_freq).
        obstacle_distance_history: list of {obstacle_name: distance} per step.
        obstacle_contact_history: list of {obstacle_name: bool} per step.
        boundary_threshold: distance threshold for obstacle boundary violation.
        pos_dist: final distance to target (for success check).
        pos_threshold: position threshold for success.
        ori_cos: final orientation cosine (for success check).
        ori_threshold: orientation threshold for success.
        dst_is_door: whether destination is a door (inverts ori check).

    Returns:
        dict with all metrics.
    """
    positions = np.asarray(positions)
    metrics = {}

    # Trajectory smoothness
    metrics.update(compute_jerk(positions, dt))

    # Path efficiency
    metrics['path_length'] = compute_path_length(positions)

    # Obstacle intrusion
    if obstacle_distance_history is not None:
        metrics.update(compute_obstacle_intrusion_metrics(
            obstacle_distance_history,
            obstacle_contact_history or [{} for _ in obstacle_distance_history],
            boundary_threshold,
        ))

    # Navigation success
    if pos_dist is not None and ori_cos is not None:
        metrics.update(compute_navigation_success_metrics(
            pos_dist, pos_threshold, ori_cos, ori_threshold, dst_is_door,
        ))
        # Combined: task success AND no boundary violation
        safety_pass = not metrics.get('boundary_violation_steps', 0) > 0
        metrics['safety_success'] = bool(safety_pass)
        metrics['overall_success'] = bool(metrics.get('task_success', False) and safety_pass)

    return metrics
