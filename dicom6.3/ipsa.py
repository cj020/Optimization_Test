import numpy as np
import time
from dose_contribution import dose_contribution
from simultaneous_loss import evaluate_simultaneous_loss, copy_simultaneous_loss_params

# =====================================================================
# IPSA hyper-parameters (adjust as needed)
# =====================================================================

IPSA_DEFAULTS = {
    "n_iterations": 500,      # maximum iterations
    "T_0": None,              # initial temperature (auto-set from initial cost)
    "alpha": 0.985,           # geometric cooling rate per iteration
    "perturbation_std": None, # std of dwell-time perturbation in seconds (auto-set)
    "t_min": 0.0,             # minimum dwell time (s)
    "t_max": None,            # maximum dwell time (auto-set to 3× max initial)
    "seed": 42,
    # Probability of proposing a mesh-index move instead of a time change.
    # Position moves require channel_ids. 0 disables them.
    "move_probability_position": 0.5,
    # Std of the mesh-index step for a position move (vertices).
    "position_step_std": 3.0,
    # Stop when the cost over the last min_iterations proposals satisfies
    # |J_k / J_{k-min_iterations} - 1| <= cost_ratio_tol.
    # One dwell update changes a voxel-sum cost only slightly, so this is
    # not a comparison of two successive updates. None disables the test.
    "cost_ratio_tol": 1e-4,
    "min_iterations": 50,
}


# =====================================================================
# Convergence: ratio of two successive costs
# =====================================================================

def sequential_cost_ratio(J_current, J_previous):
    """Return J_current / J_previous for two successive cost values.

    Both costs equal to zero give ratio 1. A zero previous cost with a
    nonzero current cost gives infinity.
    """
    J_current = float(J_current)
    J_previous = float(J_previous)
    if abs(J_previous) <= 1e-12:
        if abs(J_current) <= 1e-12:
            return 1.0
        return float("inf")
    return J_current / J_previous


def cost_ratio_converged(J_current, J_previous, tol):
    """Test |J_current / J_previous - 1| <= tol.

    Returns (converged, ratio). ``tol is None`` leaves the test inactive.
    """
    ratio = sequential_cost_ratio(J_current, J_previous)
    if tol is None or not np.isfinite(ratio):
        return False, ratio
    return abs(ratio - 1.0) <= float(tol), ratio


# =====================================================================
# Geometry helpers
# =====================================================================

def _as_channel_key(channel_id):
    try:
        return int(channel_id)
    except (TypeError, ValueError):
        return channel_id


def _channel_groups(channel_ids):
    """Map channel id -> global indices of dwells on that catheter."""
    groups = {}
    for i, ch in enumerate(np.asarray(channel_ids)):
        groups.setdefault(_as_channel_key(ch), []).append(i)
    return {ch: np.asarray(ix, dtype=int) for ch, ix in groups.items()}


def _one_dwell_dose_gy(pos, direction, time_s, volume, spacing, origin,
                       L, S_k, Lambda):
    """TG-43 dose (Gy) of a single dwell."""
    dt = float(time_s)
    if dt <= 0.0:
        return np.zeros(volume.shape, dtype=np.float64)
    dose_cgy = dose_contribution(
        dwell_pos=np.asarray(pos, dtype=np.float64).reshape(1, 3),
        norm_dwell_dir=np.asarray(direction, dtype=np.float64).reshape(1, 3),
        dwell_times=np.array([dt], dtype=np.float64),
        volume=volume,
        spacing=spacing,
        origin=origin,
        L=L, S_k=S_k, Lambda=Lambda,
        verbose=False,
    )
    return np.asarray(dose_cgy, dtype=np.float64) / 100.0


def _propose_mesh_index(local, n_local, step_std, rng):
    """Neighboring mesh vertex, clipped to the catheter."""
    step = int(np.round(rng.normal(0.0, step_std)))
    if step == 0:
        step = 1 if rng.random() < 0.5 else -1
    return int(np.clip(local + step, 0, n_local - 1))


# =====================================================================
# IPSA optimisation loop
# =====================================================================

def ipsa_optimize(dwell_positions, dwell_directions, current_times,
                  current_dose_gy, masks, ctv_names, oar_names,
                  volume, spacing, origin, L, S_k, Lambda,
                  rx_params=None, ipsa_params=None, channel_ids=None,
                  loss_params=None):
    """Run IPSA on dwell times and, when ``channel_ids`` is given, mesh positions.

    Candidate dwells are the mesh vertices.  Each iteration proposes either a
    time change at one vertex or a transfer of that vertex's time onto a
    nearby vertex of the same catheter.  Dose is updated from the change of
    that one dwell via ``dose_contribution()``.  The objective is the
    simultaneous loss ``L = J_dose + w_l2 * ||t||_2``.

    Parameters
    ----------
    dwell_positions   : (N, 3) world coords of candidate dwells (mesh vertices)
    dwell_directions  : (N, 3) unit direction vectors
    current_times     : (N,) initial dwell times (s)
    current_dose_gy   : (nz, ny, nx) initial dose grid in Gy
    masks             : dict[str, bool array] — structure masks on the image grid
    ctv_names, oar_names : lists of structure names
    volume            : image volume array (used by dose_contribution for shape)
    spacing, origin   : image-grid geometry
    L, S_k, Lambda    : TG-43 source parameters
    rx_params         : overrides for PRESCRIPTION_PARAMS
    ipsa_params       : overrides for IPSA_DEFAULTS
    channel_ids       : (N,) catheter id of each candidate. Required for
                        position moves. Time-only search if omitted.
    loss_params       : overrides for SIMULTANEOUS_LOSS_PARAMS

    Returns
    -------
    optimized_times : (N,) dwell-time array on the candidate mesh
    history : dict with loss / temperature / acceptance / cost-ratio history.
    The run stops early when the loss over the last ``min_iterations``
    proposals satisfies ``|L_k / L_{k-min_iterations} - 1| <= cost_ratio_tol``.
    """
    ip = dict(IPSA_DEFAULTS)
    if ipsa_params:
        ip.update(ipsa_params)

    t = current_times.astype(np.float64).copy()
    dose_gy = current_dose_gy.astype(np.float64).copy()
    positions = np.asarray(dwell_positions, dtype=np.float64)
    directions = np.asarray(dwell_directions, dtype=np.float64)
    n_dwells = len(t)
    if n_dwells == 0:
        raise ValueError("IPSA needs at least one candidate dwell.")
    
    channel_ids = None if channel_ids is None else np.asarray(channel_ids)
    if channel_ids is not None and len(channel_ids) != n_dwells:
        raise ValueError("channel_ids must have one entry per dwell.")

    groups = _channel_groups(channel_ids) if channel_ids is not None else {}
    local_index = np.zeros(n_dwells, dtype=int)
    for ix in groups.values():
        local_index[ix] = np.arange(len(ix), dtype=int)

    positive = t[t > 0]
    if ip["t_max"] is None:
        ip["t_max"] = float(3.0 * np.max(t)) if np.max(t) > 0 else 1.0
    if ip["perturbation_std"] is None:
        ip["perturbation_std"] = (
            float(0.3 * np.mean(positive)) if len(positive) else 1.0
        )

    t_min = ip["t_min"]
    t_max = ip["t_max"]
    sigma_t = ip["perturbation_std"]
    n_iter = ip["n_iterations"]
    alpha = ip["alpha"]
    p_pos = float(ip["move_probability_position"])
    if channel_ids is None:
        p_pos = 0.0
    step_std = float(ip["position_step_std"])
    cost_ratio_tol = ip["cost_ratio_tol"]
    min_iterations = int(ip["min_iterations"])
    rng = np.random.default_rng(ip["seed"])
    loss_params = copy_simultaneous_loss_params(loss_params)

    L, breakdown = evaluate_simultaneous_loss(
        dose_gy, t, masks, ctv_names, oar_names,
        rx_params=rx_params, loss_params=loss_params,
    )

    if ip["T_0"] is None:
        ip["T_0"] = max(L * 0.01, 1.0)
    T = ip["T_0"]

    best_L = L
    best_t = t.copy()

    history = {
        "cost": [L],
        "j_dose": [breakdown["J_dose"]],
        "p_l2": [breakdown["P_l2"]],
        "temperature": [T],
        "accept_rate": [],
        "cost_ratio": [],
        "move": [],
        "n_active": [int(np.sum(t > 0))],
        "w_l2": breakdown["w_l2"],
        "converged": False,
        "stop_reason": "max_iterations",
    }

    print(f"\n  IPSA configuration:")
    print(f"    Grid               : {volume.shape}  spacing {tuple(float(v) for v in spacing)} mm")
    print(f"    Iterations         : {n_iter}")
    print(f"    Candidate dwells   : {n_dwells}")
    print(f"    Initially active   : {int(np.sum(t > 0))}")
    print(f"    Position moves     : "
          f"{'on  p=' + f'{p_pos:.2f}' if p_pos > 0 else 'off (time only)'}")
    if p_pos > 0:
        print(f"    Position step std  : {step_std:.2f} mesh vertices")
    print(f"    Initial T          : {T:.4f}")
    print(f"    Cooling rate alpha : {alpha}")
    print(f"    Perturbation std   : {sigma_t:.4f} s")
    print(f"    Dwell time bounds  : [{t_min:.2f}, {t_max:.2f}] s")
    print(f"    Loss               : L = J_dose + w_l2 * ||t||_2")
    print(f"    w_l2               : {breakdown['w_l2']:.4g}")
    print(f"    Initial J_dose     : {breakdown['J_dose']:.2f}")
    print(f"    Initial P_l2       : {breakdown['P_l2']:.2f}")
    if cost_ratio_tol is None:
        print(f"    Cost-ratio stop    : off")
    else:
        print(f"    Cost-ratio stop    : |L_k / L_(k-{min_iterations}) - 1|"
              f" <= {cost_ratio_tol:.3g}")
    print(f"    Initial loss L     : {L:.2f}")

    t0 = time.time()

    for iteration in range(n_iter):
        accepted = False
        move_name = "none"
        changed = False
        proposed_dose = dose_gy
        t_trial = t

        want_position = (p_pos > 0.0) and (rng.random() < p_pos)
        active_indices = np.flatnonzero(t > 0)

        if want_position and len(active_indices) > 0:
            i = int(rng.choice(active_indices))
            ix = groups[_as_channel_key(channel_ids[i])]
            n_local = len(ix)
            new_local = _propose_mesh_index(
                int(local_index[i]), n_local, step_std, rng
            )
            j = int(ix[new_local])
            if j != i and (t[j] + t[i]) <= t_max + 1e-12:
                old_dose = _one_dwell_dose_gy(
                    positions[i], directions[i], t[i],
                    volume, spacing, origin, L, S_k, Lambda,
                )
                new_dose = _one_dwell_dose_gy(
                    positions[j], directions[j], t[i],
                    volume, spacing, origin, L, S_k, Lambda,
                )
                proposed_dose = dose_gy - old_dose + new_dose
                t_trial = t.copy()
                t_trial[j] = t[j] + t[i]
                t_trial[i] = 0.0
                changed = True
                move_name = "position"

        if not changed:
            i = int(rng.integers(n_dwells))
            delta_t = rng.normal(0.0, sigma_t)
            t_new = float(np.clip(t[i] + delta_t, t_min, t_max))
            actual_dt = t_new - t[i]
            if abs(actual_dt) >= 1e-12:
                delta_dose = _one_dwell_dose_gy(
                    positions[i], directions[i], abs(actual_dt),
                    volume, spacing, origin, L, S_k, Lambda,
                )
                sign = 1.0 if actual_dt > 0.0 else -1.0
                proposed_dose = dose_gy + sign * delta_dose
                t_trial = t.copy()
                t_trial[i] = t_new
                changed = True
                move_name = "time"

        if changed:
            L_new, trial_breakdown = evaluate_simultaneous_loss(
                proposed_dose, t_trial, masks, ctv_names, oar_names,
                rx_params=rx_params, loss_params=loss_params,
            )
            dL = L_new - L
            if dL < 0.0 or rng.random() < np.exp(-dL / max(T, 1e-30)):
                t = t_trial
                dose_gy = proposed_dose
                L = L_new
                breakdown = trial_breakdown
                accepted = True
                if L < best_L:
                    best_L = L
                    best_t = t.copy()

        T *= alpha
        history["cost"].append(L)
        history["j_dose"].append(breakdown["J_dose"])
        history["p_l2"].append(breakdown["P_l2"])
        history["temperature"].append(T)
        history["accept_rate"].append(1.0 if accepted else 0.0)
        history["move"].append(move_name)
        history["n_active"].append(int(np.sum(t > 0)))

        ratio = np.nan
        ratio_met = False
        if cost_ratio_tol is not None and (iteration + 1) >= min_iterations:
            L_ref = history["cost"][-(min_iterations + 1)]
            ratio_met, ratio = cost_ratio_converged(L, L_ref, cost_ratio_tol)
        history["cost_ratio"].append(ratio)

        converged_now = bool(ratio_met)
        if (iteration + 1) % 10 == 0 or iteration == 0 or converged_now or iteration == n_iter - 1:
            recent_accept = np.mean(history["accept_rate"][-min(10, iteration + 1):])
            elapsed = time.time() - t0
            ratio_text = f"{ratio:.6f}" if np.isfinite(ratio) else "n/a"
            print(f"    Iter {iteration + 1:>4}/{n_iter}  "
                  f"L={L:>12.2f}  best={best_L:>12.2f}  "
                  f"J={breakdown['J_dose']:>10.2f}  "
                  f"P_l2={breakdown['P_l2']:>8.2f}  "
                  f"active={int(np.sum(t > 0)):>4}  "
                  f"ratio={ratio_text}  "
                  f"T={T:.4f}  accept={recent_accept:.0%}  "
                  f"[{elapsed:.1f}s]")

        if converged_now:
            history["converged"] = True
            history["stop_reason"] = "cost_ratio"
            break

    elapsed = time.time() - t0
    n_run = len(history["accept_rate"])
    history["n_iterations_run"] = n_run
    J0 = history["cost"][0]
    moves = np.asarray(history["move"])
    n_time = int(np.sum(moves == "time"))
    n_pos = int(np.sum(moves == "position"))    
    print(f"\n  IPSA completed in {elapsed:.1f}s  ({n_run} iterations)")
    if history["converged"]:
        last_ratio = history["cost_ratio"][-1]
        print(f"  Stopped: |L_k / L_(k-{min_iterations}) - 1| = {abs(last_ratio - 1.0):.3g}"
              f"  <= {cost_ratio_tol:.3g}")
    else:
        print(f"  Stopped: reached the iteration limit ({n_iter})")
    print(f"  Proposals : {n_time} time, {n_pos} position")
    print(f"  Best loss L : {best_L:.2f}  (initial: {J0:.2f})")
    if abs(J0) > 1e-12:
        print(f"  Reduction : {(1 - best_L / J0) * 100:.1f}%")
    print(f"  Active dwells : {int(np.sum(best_t > 0))} / {n_dwells} mesh vertices")

    return best_t, history


# =====================================================================
# High-level convenience wrapper
# =====================================================================

def run_ipsa(dwell_positions, dwell_directions, dwell_times,
             initial_dose_gy,
             masks, ctv_names, oar_names,
             volume, spacing, origin,
             L, S_k, Lambda,
             rx_params=None, ipsa_params=None, channel_ids=None,
             loss_params=None):
    """Prepare data and run IPSA on mesh candidate dwells.

    Returns
    -------
    optimized_dwell_times : (N,) — times on the candidate mesh
    history : dict
    """
    print(f"\n{'='*70}")
    print("IPSA OPTIMISATION (time + mesh position, simultaneous loss)")
    print(f"{'='*70}")

    n_active = int(np.sum(np.asarray(dwell_times) > 0))
    print(f"\n  Candidate mesh vertices : {len(dwell_times)}")
    print(f"  Initially active        : {n_active}")

    optimized_times, history = ipsa_optimize(
        dwell_positions, dwell_directions, dwell_times,
        initial_dose_gy, masks, ctv_names, oar_names,
        volume, spacing, origin, L, S_k, Lambda,
        rx_params=rx_params, ipsa_params=ipsa_params,
        channel_ids=channel_ids, loss_params=loss_params,
    )

    opt_active = optimized_times[optimized_times > 0]
    print(f"\n  Optimised dwell time stats:")
    if len(opt_active) == 0:
        print("    No active dwells remaining.")
    else:
        print(f"    Active : {len(opt_active)}")
        print(f"    Mean   : {opt_active.mean():.4f} s")
        print(f"    Std    : {opt_active.std():.4f} s")
        print(f"    Min    : {opt_active.min():.4f} s")
        print(f"    Max    : {opt_active.max():.4f} s")
        print(f"    Total  : {opt_active.sum():.2f} s")

    if channel_ids is not None and len(opt_active) > 0:
        print(f"\n  Active dwells per channel:")
        channel_ids = np.asarray(channel_ids)
        for ch in sorted(set(channel_ids.tolist()),
                         key=lambda c: (0, int(c)) if str(c).lstrip('-').isdigit()
                         else (1, str(c))):
            mask = channel_ids == ch
            n_ch = int(np.sum(optimized_times[mask] > 0))
            tot_ch = float(optimized_times[mask].sum())
            print(f"    Channel {ch}: {n_ch:>4} / {int(mask.sum())} vertices  "
                  f"({tot_ch:.2f} s)")
            
    return optimized_times, history
