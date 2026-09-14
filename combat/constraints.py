"""Convex joint projection; optional foot/root correction; measured physics filtering.

No claim that a convex kinematic projection proves dynamic feasibility.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull
import osqp

def project_joints(q, robot, fps=30.):
    """Minimize pose deviation + modest acceleration smoothing, at unchanged timing.

    Bounds enforce joint position, first difference/fps and second difference/fps².
    Acceleration is a configurable screening limit, not a measured actuator specification.
    """
    n = len(q)
    eye = sparse.eye(n, format="csc")
    d1 = sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n), format="csc")
    d2 = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n), format="csc")
    A = sparse.vstack([eye, d1, d2], format="csc")
    P = sparse.csc_matrix(2*(eye + .5*d2.T@d2))
    out = q.copy()
    status = []
    for j in range(29):
        v = robot.velocity_limits[j]/fps
        a = robot.acceleration_limits[j]/fps**2
        low = np.r_[np.full(n, robot.limits[j, 0]), np.full(n-1, -v), np.full(n-2, -a)]
        high = np.r_[np.full(n, robot.limits[j, 1]), np.full(n-1, v), np.full(n-2, a)]
        solver = osqp.OSQP()
        solver.setup(P=P, q=-2*q[:, 7+j], A=A, l=low, u=high, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000, polishing=True)
        result = solver.solve(raise_error=True)
        if result.info.status_val not in [1, 2] or result.x is None:
            raise RuntimeError("Projection failed: " + result.info.status)
        out[:, 7+j] = result.x
        status.append(result.info.status)
    return out, status

def correct_foot_contact(q, seed, robot, fps=30.):
    """Root translation least squares anchored to seed foot geometry.

    Contact is inferred from the seed, NOT measured force and NOT combat contact.
    Keep fixed duration. Correct world-space foot positions lost during joint generation.
    This is a limited geometric correction; evaluate dynamics separately.
    """
    sg = robot.geometry(seed)
    g = robot.geometry(q)
    mask = sg["feet"][:, :, 2] < .08
    out = q.copy()
    desired = np.zeros((len(q), 3))
    weight = np.zeros(len(q))
    for t in range(len(q)):
        if mask[t].any():
            desired[t] = (sg["feet"][t, mask[t]] - g["feet"][t, mask[t]]).mean(axis=0)
            # Place the lower support sole near the plane, not through it.
            desired[t, 2] += .004 - np.min(sg["feet"][t, mask[t], 2])
            weight[t] = 8.
    n = len(q)
    d1 = sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n)).toarray()
    mat = np.diag(weight + .2) + 2*d1.T@d1
    delta = np.linalg.solve(mat, weight[:, None]*desired)
    delta = np.clip(delta, -.12, .12)
    out[:, :3] += delta
    return out, mask

def metrics(q, seed, robot, fps=30.):
    joints = q[:, 7:]
    velocity = np.diff(joints, axis=0)*fps
    acceleration = np.diff(joints, n=2, axis=0)*fps**2
    geo = robot.geometry(q)
    source = robot.geometry(seed)
    contact = source["feet"][:, :, 2] < .08
    foot_diff = np.linalg.norm(np.diff(geo["feet"][:, :, :2], axis=0), axis=-1)
    continuous = contact[:-1] & contact[1:]
    eligible = []
    balance_bad = []
    corners = np.array([[-.08, -.03], [-.08, .03], [.08, -.03], [.08, .03]])
    for t in range(len(q)):
        active = geo["feet"][t, :, 2] < .08
        eligible.append(bool(active.any()))
        if active.any():
            # Quasi-static support proxy only, not a sufficient dynamic balance test.
            points = np.concatenate([p[:2] + corners for p in geo["feet"][t, active]])
            hull = ConvexHull(points)
            xy = geo["com"][t, :2]
            balance_bad.append(bool(np.any(hull.equations[:, :2] @ xy + hull.equations[:, 2] > .03)))
    return {
        "joint_limit_violation_fraction": float(np.mean((joints < robot.limits[:, 0]-1e-4) | (joints > robot.limits[:, 1]+1e-4))),
        "velocity_violation_fraction": float(np.mean(np.abs(velocity) > robot.velocity_limits+1e-3)),
        "acceleration_violation_fraction": float(np.mean(np.abs(acceleration) > robot.acceleration_limits+.1)),
        "acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2))),
        "peak_joint_velocity_rad_s": float(np.max(np.abs(velocity))),
        "joint_deviation_from_seed_rad": float(np.sqrt(np.mean((joints-seed[:, 7:])**2))),
        "fist_deviation_from_seed_m": float(np.linalg.norm(geo["fists"]-source["fists"], axis=-1).mean()),
        "foot_sliding_proxy_m_s": float((foot_diff[continuous]*fps).mean()) if continuous.any() else None,
        "foot_drift_proxy_m": float(foot_diff[continuous].sum()/2) if continuous.any() else None,
        "foot_contact_height_mae_m": float(np.abs(geo["feet"][:, :, 2][contact]).mean()) if contact.any() else None,
        "ground_penetration_peak_m": float(max(0, -geo["feet"][:, :, 2].min())),
        "collision_penetration_peak_m": float(geo["penetration"].max()),
        "self_collision_frame_fraction": float(geo["self_collision"].mean()),
        "quasistatic_balance_proxy_violation": float(np.mean(balance_bad)) if balance_bad else None,
        "support_frame_fraction": float(np.mean(eligible)),
    }
