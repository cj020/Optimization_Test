"""Simultaneous dwell-location loss on fixed catheters.

Each catheter is the ordered needle mesh from ``needle_mesh.mesh_needle_in_ptv``.
A candidate chooses an arc length along that mesh and a dwell time for every
dwell. Catheter trajectories are fixed. This module evaluates the objective
only.

The objective is

    L(s, t) = J_dose(D(s, t)) + w_l2 * ||t||_2

``J_dose`` is the clinical dose penalty in ``cost_function.cost_function``
(CTV under-dose and over-dose, OAR over-dose). ``D`` is the TG-43 dose, in Gy,
of the active dwells.

``t`` is the dwell-position vector: one component per active dwell position,
equal to that position's dwell time in seconds. Its Euclidean norm is

    ||t||_2 = sqrt(sum_i t_i^2)

A longer dwell time increases the norm. Another dwell with positive time adds
a component and also increases the norm. A non-positive dwell time is inactive,
is omitted from ``t``, and adds nothing. Further penalties can be added later.
"""

import numpy as np

from cost_function import cost_function, print_cost_report
from dose_contribution import dose_contribution


# =====================================================================
# Loss parameters (adjust as needed)
# =====================================================================

SIMULTANEOUS_LOSS_PARAMS = {
    # Weight on ||t||_2, with t in seconds. Starting value, not yet balanced
    # against J_dose.
    "w_l2": 1.0,
}

_LENGTH_EPS_MM = 1e-6


def copy_simultaneous_loss_params(params=None):
    """Return a copy of the loss parameters."""
    src = SIMULTANEOUS_LOSS_PARAMS if params is None else params
    return dict(src)


def _resolve_loss_params(params):
    resolved = dict(SIMULTANEOUS_LOSS_PARAMS)
    if params:
        resolved.update(params)
    return resolved


def _channel_sort_key(channel_id):
    """Sort numeric channel ids numerically and other ids after them."""
    try:
        return (0, int(channel_id))
    except (TypeError, ValueError):
        return (1, str(channel_id))


def mesh_arc_lengths(mesh_points):
    """Cumulative arc length (mm) of an ordered catheter mesh.

    The first point is 0. A dwell on mesh vertex ``k`` has arc length
    ``mesh_arc_lengths(mesh)[k]``.
    """
    pts = np.asarray(mesh_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("mesh_points must have shape (K, 3).")
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    segment = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment)))


def arc_lengths_at_indices(mesh_points, indices):
    """Arc length (mm) of selected mesh vertices."""
    arc = mesh_arc_lengths(mesh_points)
    idx = np.asarray(indices, dtype=int)
    if idx.size and (np.any(idx < 0) or np.any(idx >= len(arc))):
        raise IndexError("Mesh index is outside the catheter mesh.")
    return arc[idx]


def sample_mesh(mesh_points, arc_mm):
    """Place dwells on one catheter mesh.

    Parameters
    ----------
    mesh_points : (K, 3) world coordinates, ordered along the catheter
    arc_mm : (N,) arc length in mm from the first mesh point

    Returns
    -------
    positions : (N, 3)
        World coordinates of the dwells.
    directions : (N, 3)
        Unit tangent of the mesh segment that contains each dwell.

    Raises
    ------
    ValueError
        If an arc length lies outside the meshed segment.
    """
    pts = np.asarray(mesh_points, dtype=np.float64)
    requested = np.asarray(arc_mm, dtype=np.float64).reshape(-1)
    n = len(requested)
    positions = np.zeros((n, 3), dtype=np.float64)
    directions = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return positions, directions
    if len(pts) < 2:
        raise ValueError("Catheter mesh needs at least two points to place a dwell.")

    arc = mesh_arc_lengths(pts)
    segment = np.diff(pts, axis=0)
    segment_length = np.linalg.norm(segment, axis=1)
    if not np.any(segment_length > _LENGTH_EPS_MM):
        raise ValueError("Catheter mesh has zero length.")

    s_lo = float(arc[0])
    s_hi = float(arc[-1])
    outside = (requested < s_lo - _LENGTH_EPS_MM) | (requested > s_hi + _LENGTH_EPS_MM)
    if np.any(outside):
        raise ValueError(
            "Dwell arc length is outside the catheter mesh "
            f"[{s_lo:.4f}, {s_hi:.4f}] mm: {requested[outside]}."
        )
    clamped = np.clip(requested, s_lo, s_hi)

    raw_index = np.searchsorted(arc, clamped, side="right") - 1
    raw_index = np.clip(raw_index, 0, len(pts) - 2)

    for i in range(n):
        seg = int(raw_index[i])
        while seg < len(segment_length) - 1 and segment_length[seg] <= _LENGTH_EPS_MM:
            seg += 1
        while seg > 0 and segment_length[seg] <= _LENGTH_EPS_MM:
            seg -= 1
        length = float(segment_length[seg])
        if length <= _LENGTH_EPS_MM:
            raise ValueError("Catheter mesh has zero length.")
        alpha = float(np.clip((clamped[i] - arc[seg]) / length, 0.0, 1.0))
        positions[i] = (1.0 - alpha) * pts[seg] + alpha * pts[seg + 1]
        directions[i] = segment[seg] / length

    return positions, directions


def _as_float_vector(values, channel_id, name):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Channel {channel_id}: {name} contains a non-finite value.")
    return arr


def _mesh_for(needle_meshes, channel_id):
    if channel_id in needle_meshes:
        return needle_meshes[channel_id]
    key = int(channel_id)
    if key in needle_meshes:
        return needle_meshes[key]
    raise KeyError(f"Channel {channel_id} has dwell variables but no needle mesh.")


def assemble_active_dwells(needle_meshes, dwell_arc_mm, dwell_times):
    """Map per-catheter arc lengths and times onto dose-calculation arrays.

    A dwell with time <= 0 is inactive and is omitted. Channels are emitted
    in sorted channel-id order. Within a channel, the given order is kept.

    Returns
    -------
    dwells : dict
        ``positions`` (N, 3), ``directions`` (N, 3), ``times`` (N,),
        ``channel_ids`` (N,), ``arc_mm`` (N,).
    """
    if set(dwell_arc_mm) != set(dwell_times):
        raise ValueError("dwell_arc_mm and dwell_times must use the same channel ids.")

    positions = []
    directions = []
    times = []
    channel_ids = []
    arc_mm = []

    for channel_id in sorted(dwell_arc_mm.keys(), key=_channel_sort_key):
        requested = _as_float_vector(dwell_arc_mm[channel_id], channel_id, "arc length")
        dwell_time = _as_float_vector(dwell_times[channel_id], channel_id, "dwell time")
        if len(requested) != len(dwell_time):
            raise ValueError(
                f"Channel {channel_id}: {len(requested)} arc lengths and "
                f"{len(dwell_time)} dwell times."
            )
        if np.any(dwell_time < 0.0):
            raise ValueError(f"Channel {channel_id}: dwell time is negative.")
        active = dwell_time > 0.0
        if not np.any(active):
            continue

        pos, direction = sample_mesh(
            _mesh_for(needle_meshes, channel_id), requested[active]
        )
        positions.append(pos)
        directions.append(direction)
        times.append(dwell_time[active])
        channel_ids.append(np.full(int(active.sum()), channel_id))
        arc_mm.append(requested[active])

    if not positions:
        return {
            "positions": np.zeros((0, 3), dtype=np.float64),
            "directions": np.zeros((0, 3), dtype=np.float64),
            "times": np.zeros(0, dtype=np.float64),
            "channel_ids": np.zeros(0, dtype=int),
            "arc_mm": np.zeros(0, dtype=np.float64),
        }

    return {
        "positions": np.vstack(positions),
        "directions": np.vstack(directions),
        "times": np.concatenate(times),
        "channel_ids": np.concatenate(channel_ids),
        "arc_mm": np.concatenate(arc_mm),
    }


def dwell_position_l2(times):
    """L2 norm of the dwell-position vector, in seconds.

    Each component is the dwell time at one active position.
    ``||t||_2 = sqrt(sum_i t_i^2)``. An empty set has norm 0.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.size == 0:
        return 0.0
    return float(np.linalg.norm(t))


def simultaneous_loss(
    needle_meshes,
    dwell_arc_mm,
    dwell_times,
    masks,
    ctv_names,
    oar_names,
    volume,
    spacing,
    origin,
    L,
    S_k,
    Lambda,
    rx_params=None,
    loss_params=None,
    dose_gy=None,
    n_workers=None,
):
    """Evaluate ``L = J_dose + w_l2 * ||t||_2``.

    Parameters
    ----------
    needle_meshes : dict[channel_id, (K, 3)]
        Ordered mesh points for each fixed catheter, in world mm.
    dwell_arc_mm : dict[channel_id, array]
        Arc length of each dwell from the first mesh point, in mm.
        Vertex ``k`` is ``mesh_arc_lengths(mesh)[k]``. Values must lie on
        that mesh.
    dwell_times : dict[channel_id, array]
        Dwell time in seconds, aligned with ``dwell_arc_mm``. Time <= 0 is
        inactive and is omitted from both the dose and ``||t||_2``. Negative
        time is rejected.
    masks, ctv_names, oar_names
        Same inputs as ``cost_function``.
    volume, spacing, origin, L, S_k, Lambda
        Same inputs as ``dose_contribution``. ``volume`` supplies the dose
        grid shape and is used when ``dose_gy`` is omitted.
    rx_params : dict or None
        Overrides for the clinical dose penalty in ``cost_function``.
    loss_params : dict or None
        Overrides for ``SIMULTANEOUS_LOSS_PARAMS``. The used key is ``w_l2``.
    dose_gy : ndarray or None
        Dose in Gy on the ``volume`` grid. When omitted, the TG-43 dose of
        these dwells is computed. When supplied, it must already be the dose
        of this same dwell set; ``||t||_2`` is still computed from the times.
    n_workers : int or None
        Passed to ``dose_contribution`` when this function computes the dose.

    Returns
    -------
    L : float
        Total objective.
    breakdown : dict
        ``J_dose``, ``P_l2``, ``dwell_position_l2``, ``w_l2``, and the active
        dwell arrays.
    dose_gy : ndarray
        Dose grid in Gy used for ``J_dose``.
    """
    params = _resolve_loss_params(loss_params)
    w_l2 = float(params["w_l2"])
    if w_l2 < 0.0:
        raise ValueError("w_l2 must be >= 0.")

    dwells = assemble_active_dwells(needle_meshes, dwell_arc_mm, dwell_times)
    t_l2 = dwell_position_l2(dwells["times"])
    p_l2 = w_l2 * t_l2

    if dose_gy is None:
        if len(dwells["times"]) == 0:
            dose_gy = np.zeros(volume.shape, dtype=np.float64)
        else:
            dose_cgy = dose_contribution(
                dwell_pos=dwells["positions"],
                norm_dwell_dir=dwells["directions"],
                dwell_times=dwells["times"],
                volume=volume,
                spacing=spacing,
                origin=origin,
                L=L,
                S_k=S_k,
                Lambda=Lambda,
                n_workers=n_workers,
                verbose=False,
            )
            dose_gy = np.asarray(dose_cgy, dtype=np.float64) / 100.0
    else:
        dose_gy = np.asarray(dose_gy, dtype=np.float64)
        if dose_gy.shape != volume.shape:
            raise ValueError(
                f"dose_gy shape {dose_gy.shape} does not match volume shape {volume.shape}."
            )

    j_dose, dose_breakdown = cost_function(
        dose_gy, masks, ctv_names, oar_names, rx_params
    )
    total = float(j_dose) + float(p_l2)

    breakdown = {
        "L": total,
        "J_dose": float(j_dose),
        "P_l2": float(p_l2),
        "dwell_position_l2": t_l2,
        "w_l2": w_l2,
        "n_active": int(len(dwells["times"])),
        "total_time_s": float(dwells["times"].sum()) if len(dwells["times"]) else 0.0,
        "dose": dose_breakdown,
        "positions": dwells["positions"],
        "directions": dwells["directions"],
        "times": dwells["times"],
        "channel_ids": dwells["channel_ids"],
        "arc_mm": dwells["arc_mm"],
    }
    return total, breakdown, dose_gy


def print_simultaneous_loss(L, breakdown, rx_params=None):
    """Pretty-print one simultaneous-loss evaluation.

    ``rx_params`` should be the prescription overrides passed to
    ``simultaneous_loss``, so the printed dose limits match that evaluation.
    """
    print(f"\n{'='*70}")
    print("SIMULTANEOUS DWELL-LOCATION LOSS")
    print(f"{'='*70}")
    print("  L = J_dose + w_l2 * ||t||_2")
    print(f"  Active dwell positions : {breakdown['n_active']}")
    print(f"  Total dwell time       : {breakdown['total_time_s']:.4f} s")
    print(f"  ||t||_2                : {breakdown['dwell_position_l2']:.4f} s")
    print(f"  w_l2                   : {breakdown['w_l2']:.4g}")
    print(f"  J_dose                 : {breakdown['J_dose']:.2f}")
    print(f"  P_l2                   : {breakdown['P_l2']:.2f}")
    print(f"  TOTAL LOSS L           : {L:.2f}")

    print_cost_report(breakdown["J_dose"], breakdown["dose"], rx_params)
