import numpy as np
import time
from dose_contribution import dose_contribution, print_progress_bar
from cost_function import cost_function, PRESCRIPTION_PARAMS

# =====================================================================
# IPSA hyper-parameters (adjust as needed)
# =====================================================================

IPSA_DEFAULTS = {
    "n_iterations": 500,
    "T_0": None,              # initial temperature (auto-set from initial cost)
    "alpha": 0.985,           # geometric cooling rate per full sweep
    "perturbation_std": None, # std of dwell-time perturbation in seconds (auto-set)
    "t_min": 0.0,             # minimum dwell time (s)
    "t_max": None,            # maximum dwell time (auto-set to 3× max initial)
    "seed": 42,
}


# =====================================================================
# IPSA optimisation loop
# =====================================================================

def ipsa_optimize(dwell_positions, dwell_directions, current_times,
                  current_dose_gy, masks, ctv_names, oar_names,
                  volume, spacing, origin, L, S_k, Lambda,
                  rx_params=None, ipsa_params=None):
    """Run IPSA using dose_contribution() for dose updates and cost_function()
    for evaluation.

    In each iteration one active dwell is randomly selected and its time
    perturbed.  The dose *change* from that single dwell is computed via
    ``dose_contribution()`` (only one dwell active, so it is fast) and
    added / subtracted from the running dose grid.  The cost is then
    evaluated on the full 3-D grid using the CTV/OAR masks.

    Parameters
    ----------
    dwell_positions   : (N, 3) world coords
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

    Returns
    -------
    optimized_times : (N,) full dwell-time array
    history : dict with cost / temperature / acceptance history
    """
    ip = dict(IPSA_DEFAULTS)
    if ipsa_params:
        ip.update(ipsa_params)

    t = current_times.astype(np.float64).copy()
    dose_gy = current_dose_gy.astype(np.float64).copy()

    active_indices = np.flatnonzero(t > 0)
    n_active = len(active_indices)

    if ip["t_max"] is None:
        ip["t_max"] = float(3.0 * np.max(t))
    if ip["perturbation_std"] is None:
        ip["perturbation_std"] = float(0.3 * np.mean(t[active_indices]))

    t_min   = ip["t_min"]
    t_max   = ip["t_max"]
    sigma_t = ip["perturbation_std"]
    n_iter  = ip["n_iterations"]
    alpha   = ip["alpha"]
    rng     = np.random.default_rng(ip["seed"])

    # ---- initial cost ----
    J, _ = cost_function(dose_gy, masks, ctv_names, oar_names, rx_params)

    if ip["T_0"] is None:
        ip["T_0"] = max(J * 0.01, 1.0)
    T = ip["T_0"]

    best_J = J
    best_t = t.copy()

    history = {
        "cost": [J],
        "temperature": [T],
        "accept_rate": [],
    }

    print(f"\n  IPSA configuration:")
    print(f"    Grid               : {volume.shape}  spacing {tuple(float(v) for v in spacing)} mm")
    print(f"    Iterations         : {n_iter}")
    print(f"    Active dwells      : {n_active}")
    print(f"    Initial T          : {T:.4f}")
    print(f"    Cooling rate alpha : {alpha}")
    print(f"    Perturbation std   : {sigma_t:.4f} s")
    print(f"    Dwell time bounds  : [{t_min:.2f}, {t_max:.2f}] s")
    print(f"    Initial cost       : {J:.2f}")

    t0 = time.time()
    n_dwells = len(t)

    for iteration in range(n_iter):
        # --- pick a random active dwell and propose a new time ---
        i = rng.choice(active_indices)
        delta_t = rng.normal(0, sigma_t)
        t_new = np.clip(t[i] + delta_t, t_min, t_max)
        actual_dt = t_new - t[i]

        if abs(actual_dt) < 1e-12:
            history["cost"].append(J)
            history["temperature"].append(T)
            history["accept_rate"].append(0.0)
            T *= alpha
            continue

        # --- compute dose change for this single dwell via dose_contribution ---
        single_times = np.zeros(n_dwells)
        single_times[i] = abs(actual_dt)

        delta_dose_cgy = dose_contribution(
            dwell_pos=dwell_positions,
            norm_dwell_dir=dwell_directions,
            dwell_times=single_times,
            volume=volume,
            spacing=spacing,
            origin=origin,
            L=L, S_k=S_k, Lambda=Lambda,
            verbose=False,
        )
        delta_dose_gy = delta_dose_cgy / 100.0

        sign = 1.0 if actual_dt > 0 else -1.0
        proposed_dose = dose_gy + sign * delta_dose_gy

        # --- evaluate cost on the proposed dose grid ---
        J_new, _ = cost_function(proposed_dose, masks, ctv_names, oar_names, rx_params)
        dJ = J_new - J

        # --- simulated annealing acceptance ---
        accepted = False
        if dJ < 0 or rng.random() < np.exp(-dJ / max(T, 1e-30)):
            t[i] = t_new
            dose_gy = proposed_dose
            J = J_new
            accepted = True

            if J < best_J:
                best_J = J
                best_t = t.copy()

        T *= alpha
        history["cost"].append(J)
        history["temperature"].append(T)
        history["accept_rate"].append(1.0 if accepted else 0.0)

        if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == n_iter - 1:
            recent_accept = np.mean(history["accept_rate"][-min(10, iteration + 1):])
            elapsed = time.time() - t0
            print(f"    Iter {iteration + 1:>4}/{n_iter}  "
                  f"J={J:>12.2f}  best={best_J:>12.2f}  "
                  f"T={T:.4f}  accept={recent_accept:.0%}  "
                  f"[{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"\n  IPSA completed in {elapsed:.1f}s")
    print(f"  Best cost : {best_J:.2f}  (initial: {history['cost'][0]:.2f})")
    print(f"  Reduction : {(1 - best_J / history['cost'][0]) * 100:.1f}%")

    return best_t, history


# =====================================================================
# High-level convenience wrapper
# =====================================================================

def run_ipsa(dwell_positions, dwell_directions, dwell_times,
             initial_dose_gy,
             masks, ctv_names, oar_names,
             volume, spacing, origin,
             L, S_k, Lambda,
             rx_params=None, ipsa_params=None):
    """Prepare data and run IPSA optimisation.

    Returns
    -------
    optimized_dwell_times : (N,) — full dwell-time array with optimised active times
    history : dict
    """
    print(f"\n{'='*70}")
    print("IPSA DOSE OPTIMISATION")
    print(f"{'='*70}")

    active_mask = dwell_times > 0
    n_active = int(active_mask.sum())
    print(f"\n  Active dwell positions: {n_active}")

    optimized_times, history = ipsa_optimize(
        dwell_positions, dwell_directions, dwell_times,
        initial_dose_gy, masks, ctv_names, oar_names,
        volume, spacing, origin, L, S_k, Lambda,
        rx_params=rx_params, ipsa_params=ipsa_params,
    )

    opt_active = optimized_times[optimized_times > 0]
    print(f"\n  Optimised dwell time stats:")
    print(f"    Mean  : {opt_active.mean():.4f} s")
    print(f"    Std   : {opt_active.std():.4f} s")
    print(f"    Min   : {opt_active.min():.4f} s")
    print(f"    Max   : {opt_active.max():.4f} s")
    print(f"    Total : {opt_active.sum():.2f} s")

    return optimized_times, history
