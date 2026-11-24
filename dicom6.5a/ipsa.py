import numpy as np
import time
from cost_function import active_objectives, resolve_params
from dose_contribution import dose_contribution, dose_influence_matrix
from simultaneous_loss import evaluate_simultaneous_loss, copy_simultaneous_loss_params

# =====================================================================
# IPSA hyper-parameters (adjust as needed)
# =====================================================================

IPSA_DEFAULTS = {
    # ---- Safety limits (the run normally stops earlier, see below) ----
    # Each iteration proposes a change of ONE dwell.
    "n_iterations": 200000,   # maximum iterations
    "max_time_s": 120.0,      # maximum wall-clock time of the search (s); None = off

    # ---- Cooling: epoch based, independent of the iteration limit ----
    # One epoch = epoch_length proposals (None: one per candidate dwell, i.e.
    # one sweep).  The temperature falls by alpha_epoch per epoch, spread
    # smoothly over its iterations.
    # Initial temperature. None: T_0_fraction x initial loss.  On pt1, 1e-4
    # (first-epoch acceptance ~55%) reached the same plan quality as 1e-2
    # (~87% acceptance, a long random walk) in ~40% fewer iterations.
    "T_0": None,
    "T_0_fraction": 1e-4,
    "epoch_length": None,
    "alpha_epoch": 0.9,
    # Explicit per-iteration cooling rate; overrides alpha_epoch when set.
    "alpha": None,

    # ---- Adaptive termination (checked at the end of every epoch) ----
    # The run stops when ALL of these hold:
    #   cooled   : T <= stop_temperature_ratio * T_0, or the epoch's
    #              acceptance rate < stop_accept_rate;
    #   loss     : best loss improved by < plateau_tol (relative) over the
    #              last patience_epochs epochs;
    #   clinical : (if clinical_plateau) over the same epochs no target
    #              V100 / D90 moved by more than clinical_tol_target
    #              percentage points and no OAR D10 / D1 by more than
    #              clinical_tol_oar.
    "adaptive_stop": True,
    "min_epochs": 5,
    "stop_temperature_ratio": 0.01,
    "stop_accept_rate": 0.02,
    "plateau_tol": 1e-3,
    "patience_epochs": 5,
    "clinical_plateau": True,
    "clinical_tol_target": 0.2,
    "clinical_tol_oar": 0.5,
    # Greedy sweeps (T = 0, improvements only) after the search stops.
    "polish_epochs": 1,

    "perturbation_std": None, # std of dwell-time perturbation in seconds (auto-set)
    "t_min": 0.0,             # minimum dwell time (s)
    "t_max": None,            # maximum dwell time (auto-set to 3× max initial)
    "seed": 42,
    # Probability of proposing a mesh-index move instead of a time change.
    # Position moves require channel_ids. 0 disables them.
    "move_probability_position": 0.5,
    # Std of the mesh-index step for a position move (vertices).
    "position_step_std": 3.0,
    # Precompute the dose per second of every candidate on the objective
    # voxels, so a move costs one vector update instead of a TG-43 grid.
    "use_influence_matrix": True,
    "influence_max_bytes": 3e9,   # fall back to per-move TG-43 above this
}


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


# =====================================================================
# Dose state: how a proposed move changes the dose and the loss
# =====================================================================

def _clinical_metrics(dose_by_name, objectives, D_p):
    """Per-structure DVH metrics (percent) used by the clinical-plateau test.

    Targets: V100 (% volume >= D_p) and D90 (% of D_p).
    OARs   : D10 and D1 (% of D_p) -- a near-max and a small-volume dose.
    Returns {name: (role, (m1, m2))}.
    """
    out = {}
    for obj in objectives:
        d = dose_by_name[obj["name"]]
        if d.size == 0 or D_p <= 0:
            continue
        if obj["role"] == "Target":
            vals = (100.0 * float(np.mean(d >= D_p)),
                    100.0 * float(np.percentile(d, 10)) / D_p)
        else:
            vals = (100.0 * float(np.percentile(d, 90)) / D_p,
                    100.0 * float(np.percentile(d, 99)) / D_p)
        out[obj["name"]] = (obj["role"], vals)
    return out


def _max_metric_change(history_metrics, tol_target, tol_oar):
    """Largest change over the window relative to its tolerance (<=1 = stable).

    ``history_metrics`` is a list of _clinical_metrics results, oldest first.
    Returns (ratio, description of the largest change).
    """
    worst, what = 0.0, ""
    first = history_metrics[0]
    for name, (role, _) in first.items():
        series = np.array([m[name][1] for m in history_metrics if name in m])
        span = series.max(axis=0) - series.min(axis=0)
        tol = tol_target if role == "Target" else tol_oar
        labels = ("V100", "D90") if role == "Target" else ("D10", "D1")
        for k in range(len(span)):
            r = span[k] / max(tol, 1e-12)
            if r > worst:
                worst, what = r, f"{name} {labels[k]} moved {span[k]:.2f} pp"
    return worst, what


class _GridDose:
    """Full TG-43 grid; every move recomputes one dwell's dose (slow)."""

    def __init__(self, dose_gy, positions, directions, volume, spacing, origin,
                 L, S_k, Lambda, masks, ctv_names, oar_names, rx_params, loss_params):
        self.dose = dose_gy.astype(np.float64).copy()
        self.pos, self.dir = positions, directions
        self.geom = (volume, spacing, origin, L, S_k, Lambda)
        self.eval_args = (masks, ctv_names, oar_names)
        self.rx, self.lp = rx_params, loss_params

    def _one(self, i, time_s):
        return _one_dwell_dose_gy(self.pos[i], self.dir[i], time_s, *self.geom)

    def time_change(self, i, dt):
        sign = 1.0 if dt > 0.0 else -1.0
        return self.dose + sign * self._one(i, abs(dt))

    def transfer(self, i, j, ti):
        return self.dose - self._one(i, ti) + self._one(j, ti)

    def loss(self, dose, t):
        L, b = evaluate_simultaneous_loss(dose, t, *self.eval_args,
                                          rx_params=self.rx, loss_params=self.lp)
        return L, b["J_dose"], b["P_l2"]

    def resync(self, t):
        pass

    def dose_for(self, t):
        # A grid recompute per epoch would be costly; at the low temperatures
        # where the clinical test matters, the current plan is the best one.
        return self.dose

    def metrics(self, dose, objectives, D_p):
        masks = self.eval_args[0]
        return _clinical_metrics({o["name"]: dose[masks[o["name"]]] for o in objectives},
                                 objectives, D_p)


class _MatrixDose:
    """Dose on the objective voxels only, from a precomputed influence matrix.

    ``A[:, i]`` is the dose (Gy) per second of candidate i on those voxels,
    so a move updates the dose with one column.  The loss is the same as
    ``evaluate_simultaneous_loss`` (same objectives, same voxels).
    """

    def __init__(self, A, union_mask, masks, objectives, t, w_l2):
        self.A = A
        self.w_l2 = float(w_l2)
        idx, lo, hi, wlo, whi = [], [], [], [], []
        self.where = {}   # objective name -> voxel indices into the dose vector
        for obj in objectives:
            k = np.flatnonzero(masks[obj["name"]][union_mask])
            n = len(k)
            idx.append(k)
            self.where[obj["name"]] = k
            use_lo = obj["lo"] is not None and obj["w_lo"] > 0.0
            use_hi = obj["hi"] is not None and obj["w_hi"] > 0.0
            lo.append(np.full(n, obj["lo"] if use_lo else -np.inf))
            hi.append(np.full(n, obj["hi"] if use_hi else np.inf))
            wlo.append(np.full(n, obj["w_lo"] if use_lo else 0.0))
            whi.append(np.full(n, obj["w_hi"] if use_hi else 0.0))
        self.idx = np.concatenate(idx)
        self.lo, self.hi = np.concatenate(lo), np.concatenate(hi)
        self.wlo, self.whi = np.concatenate(wlo), np.concatenate(whi)
        self.resync(t)

    def resync(self, t):
        """Recompute the dose from scratch (removes float drift)."""
        self.dose = self.A @ np.asarray(t, dtype=np.float32)
        self.dose = self.dose.astype(np.float64)

    def dose_for(self, t):
        return (self.A @ np.asarray(t, dtype=np.float32)).astype(np.float64)

    def metrics(self, dose, objectives, D_p):
        return _clinical_metrics({o["name"]: dose[self.where[o["name"]]] for o in objectives},
                                 objectives, D_p)

    def time_change(self, i, dt):
        return self.dose + dt * self.A[:, i]

    def transfer(self, i, j, ti):
        return self.dose + ti * (self.A[:, j].astype(np.float64) - self.A[:, i])

    def loss(self, dose, t):
        d = dose[self.idx]
        under = self.lo - d
        over = d - self.hi
        np.maximum(under, 0.0, out=under)
        np.maximum(over, 0.0, out=over)
        J = float(self.wlo @ under + self.whi @ over)
        tt = np.asarray(t, dtype=np.float64)
        P = self.w_l2 * float(np.sqrt(np.dot(tt, tt)))
        return J + P, J, P


def _build_matrix_state(positions, directions, t, masks, ctv_names, oar_names,
                        volume, spacing, origin, L, S_k, Lambda,
                        rx_params, loss_params, max_bytes):
    """_MatrixDose for this problem, or None when it would not fit."""
    objectives = active_objectives(masks, ctv_names, oar_names, rx_params)
    if not objectives:
        return None
    union = np.zeros(volume.shape, dtype=bool)
    for obj in objectives:
        union |= masks[obj["name"]]
    n_vox = int(union.sum())
    nbytes = 4.0 * n_vox * len(positions)
    if nbytes > max_bytes:
        print(f"    Influence matrix would need {nbytes / 1e9:.1f} GB "
              f"(> {max_bytes / 1e9:.1f} GB): using per-move TG-43 instead.")
        return None
    print(f"    Building influence matrix: {n_vox} objective voxels x "
          f"{len(positions)} candidates ({nbytes / 1e6:.0f} MB)")
    A = dose_influence_matrix(positions, directions, np.nonzero(union), volume,
                              spacing, origin, L, S_k, Lambda, verbose=False)
    w_l2 = float(copy_simultaneous_loss_params(loss_params)["w_l2"])
    return _MatrixDose(A, union, masks, objectives, t, w_l2)


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
    history : dict with per-iteration loss / temperature / acceptance, the
        per-epoch statistics (``epochs``), ``stop_reason`` ("converged",
        "time_limit" or "max_iterations") and ``stop_detail``.

    The search cools by ``alpha_epoch`` per epoch (one sweep of proposals)
    and, with ``adaptive_stop``, ends at the first epoch where it has cooled,
    the best loss has plateaued and the clinical metrics are stable (see
    IPSA_DEFAULTS).  The iteration and time limits are safety backstops.
    A short greedy polish from the best plan follows.
    """
    ip = dict(IPSA_DEFAULTS)
    if ipsa_params:
        ip.update(ipsa_params)
    # ``L`` is the TG-43 active source length; keep it apart from the loss,
    # which is also called L below.
    L_src = float(L)

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
    n_iter = int(ip["n_iterations"])
    max_time = ip["max_time_s"]
    epoch_len = int(ip["epoch_length"] or n_dwells)
    epoch_len = max(epoch_len, 1)
    alpha = ip["alpha"]
    if alpha is None:
        alpha = float(ip["alpha_epoch"]) ** (1.0 / epoch_len)
    p_pos = float(ip["move_probability_position"])
    if channel_ids is None:
        p_pos = 0.0
    step_std = float(ip["position_step_std"])
    adaptive = bool(ip["adaptive_stop"])
    patience = max(1, int(ip["patience_epochs"]))
    min_epochs = max(int(ip["min_epochs"]), patience)
    plateau_tol = float(ip["plateau_tol"])
    stop_accept = float(ip["stop_accept_rate"])
    use_clinical = bool(ip["clinical_plateau"])
    tol_target = float(ip["clinical_tol_target"])
    tol_oar = float(ip["clinical_tol_oar"])
    rng = np.random.default_rng(ip["seed"])
    objectives = active_objectives(masks, ctv_names, oar_names, rx_params)
    D_p = float(resolve_params(rx_params)["D_p"])
    loss_params = copy_simultaneous_loss_params(loss_params)

    L, breakdown = evaluate_simultaneous_loss(
        dose_gy, t, masks, ctv_names, oar_names,
        rx_params=rx_params, loss_params=loss_params,
    )

    state = None
    t_build = time.time()
    if ip["use_influence_matrix"]:
        state = _build_matrix_state(
            positions, directions, t, masks, ctv_names, oar_names,
            volume, spacing, origin, L_src, S_k, Lambda,
            rx_params, loss_params, float(ip["influence_max_bytes"]))
    if state is None:
        state = _GridDose(dose_gy, positions, directions, volume, spacing, origin,
                          L_src, S_k, Lambda, masks, ctv_names, oar_names,
                          rx_params, loss_params)
        engine = "per-move TG-43 on the full grid"
    else:
        engine = f"influence matrix (built in {time.time() - t_build:.1f}s)"
        L_m, _, _ = state.loss(state.dose, t)
        print(f"    Influence-matrix loss check: {L_m:.4f} vs grid {L:.4f}")

    if ip["T_0"] is None:
        ip["T_0"] = max(L * float(ip["T_0_fraction"]), 1e-6)
    T = ip["T_0"]
    T_0 = T
    cold_T = float(ip["stop_temperature_ratio"]) * T

    best_L = L
    best_t = t.copy()

    history = {
        "cost": [L],
        "j_dose": [breakdown["J_dose"]],
        "p_l2": [breakdown["P_l2"]],
        "temperature": [T],
        "accept_rate": [],
        "move": [],
        "n_active": [int(np.sum(t > 0))],
        "w_l2": breakdown["w_l2"],
        "epoch_length": epoch_len,
        # One entry per completed epoch: iteration, T, best / mean loss,
        # acceptance rate, clinical metrics of the best plan.
        "epochs": [],
        "converged": False,
        "stop_reason": "max_iterations",
        "stop_detail": "",
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
    print(f"    Max time           : {'off' if max_time is None else f'{float(max_time):g} s'}")
    print(f"    Epoch              : {epoch_len} proposals")
    print(f"    Initial T          : {T:.4f}")
    print(f"    Cooling            : x{alpha ** epoch_len:.4f} per epoch "
          f"(alpha {alpha:.6f} per iteration)")
    print(f"    Perturbation std   : {sigma_t:.4f} s")
    print(f"    Dwell time bounds  : [{t_min:.2f}, {t_max:.2f}] s")
    print(f"    Loss               : L = J_dose + w_l2 * ||t||_2")
    print(f"    w_l2               : {breakdown['w_l2']:.4g}")
    print(f"    Initial J_dose     : {breakdown['J_dose']:.2f}")
    print(f"    Initial P_l2       : {breakdown['P_l2']:.2f}")
    if not adaptive:
        print("    Adaptive stop      : off (iteration / time limit only)")
    else:
        print(f"    Adaptive stop      : after >= {min_epochs} epochs, when cooled "
              f"(T <= {ip['stop_temperature_ratio']:g} T0 or accept < {stop_accept:.0%}),")
        print(f"                         best loss gain < {plateau_tol:g} over {patience} epochs"
              + (f", clinical metrics within {tol_target:g} pp (targets) / "
                 f"{tol_oar:g} pp (OARs)" if use_clinical else ""))
    print(f"    Initial loss L     : {L:.2f}")
    print(f"    Dose update        : {engine}")

    J_cur, P_cur = breakdown["J_dose"], breakdown["P_l2"]
    report_every = max(10, epoch_len)
    resync_every = 2000

    def step(T_now):
        """Propose one move and accept it by the Metropolis rule at T_now."""
        nonlocal t, L, J_cur, P_cur, best_L, best_t
        move_name = "none"
        proposed_dose = None
        t_trial = t

        want_position = (p_pos > 0.0) and (rng.random() < p_pos)
        active_indices = np.flatnonzero(t > 0)

        if want_position and len(active_indices) > 0:
            i = int(rng.choice(active_indices))
            ix = groups[_as_channel_key(channel_ids[i])]
            new_local = _propose_mesh_index(
                int(local_index[i]), len(ix), step_std, rng
            )
            j = int(ix[new_local])
            if j != i and (t[j] + t[i]) <= t_max + 1e-12:
                proposed_dose = state.transfer(i, j, t[i])
                t_trial = t.copy()
                t_trial[j] = t[j] + t[i]
                t_trial[i] = 0.0
                move_name = "position"

        if proposed_dose is None:
            i = int(rng.integers(n_dwells))
            t_new = float(np.clip(t[i] + rng.normal(0.0, sigma_t), t_min, t_max))
            actual_dt = t_new - t[i]
            if abs(actual_dt) >= 1e-12:
                proposed_dose = state.time_change(i, actual_dt)
                t_trial = t.copy()
                t_trial[i] = t_new
                move_name = "time"

        accepted = False
        if proposed_dose is not None:
            L_new, J_new, P_new = state.loss(proposed_dose, t_trial)
            dL = L_new - L
            if dL < 0.0 or (T_now > 0.0 and rng.random() < np.exp(-dL / T_now)):
                t = t_trial
                state.dose = proposed_dose
                L, J_cur, P_cur = L_new, J_new, P_new
                accepted = True
                if L < best_L:
                    best_L = L
                    best_t = t.copy()
        return accepted, move_name

    def record(T_now, accepted, move_name):
        history["cost"].append(L)
        history["j_dose"].append(J_cur)
        history["p_l2"].append(P_cur)
        history["temperature"].append(T_now)
        history["accept_rate"].append(1.0 if accepted else 0.0)
        history["move"].append(move_name)
        history["n_active"].append(int(np.sum(t > 0)))

    def end_of_epoch(iteration):
        """Epoch statistics, and whether the adaptive stop criteria all hold."""
        ep_accept = float(np.mean(history["accept_rate"][-epoch_len:]))
        entry = {
            "iteration": iteration + 1,
            "T": T,
            "best": best_L,
            "mean": float(np.mean(history["cost"][-epoch_len:])),
            "accept": ep_accept,
            "metrics": (state.metrics(state.dose_for(best_t), objectives, D_p)
                        if use_clinical else {}),
        }
        epochs = history["epochs"]
        epochs.append(entry)
        n_ep = len(epochs)

        cooled = T <= cold_T or ep_accept < stop_accept
        gain, clin_ratio, clin_what = np.nan, 0.0, ""
        if n_ep > patience:
            ref = epochs[-1 - patience]["best"]
            gain = (ref - best_L) / max(abs(ref), 1e-12)
            if use_clinical:
                clin_ratio, clin_what = _max_metric_change(
                    [e["metrics"] for e in epochs[-1 - patience:]], tol_target, tol_oar)
        loss_flat = np.isfinite(gain) and gain < plateau_tol
        clin_flat = (not use_clinical) or clin_ratio <= 1.0
        entry.update(gain=gain, cooled=cooled, clin_ratio=clin_ratio)

        print(f"    Epoch {n_ep:>3}  iter {iteration + 1:>6}  best={best_L:>12.2f}  "
              f"mean={entry['mean']:>12.2f}  accept={ep_accept:5.1%}  "
              f"T/T0={T / T_0:.2e}{' cooled' if cooled else ''}  "
              f"gain{patience}={'n/a' if not np.isfinite(gain) else f'{gain:.2e}'}"
              + (f"  clinical={clin_ratio:.2f}" if use_clinical and n_ep > patience else ""))

        if adaptive and n_ep >= min_epochs and cooled and loss_flat and clin_flat:
            detail = (f"epoch {n_ep}: best loss improved {gain:.2e} (< {plateau_tol:g}) "
                      f"over {patience} epochs; T = {T / T_0:.2e} T0, "
                      f"acceptance {ep_accept:.1%}")
            if use_clinical:
                detail += (f"; clinical metrics stable (largest change: "
                           f"{clin_what or 'none'})")
            return detail
        return None

    t0 = time.time()
    for iteration in range(n_iter):
        accepted, move_name = step(T)

        if (iteration + 1) % resync_every == 0:
            state.resync(t)
            L, J_cur, P_cur = state.loss(state.dose, t)

        T *= alpha
        record(T, accepted, move_name)

        if (iteration + 1) % report_every == 0 or iteration == 0 or iteration == n_iter - 1:
            recent_accept = np.mean(history["accept_rate"][-min(report_every, iteration + 1):])
            print(f"    Iter {iteration + 1:>4}/{n_iter}  "
                  f"L={L:>12.2f}  best={best_L:>12.2f}  "
                  f"J={J_cur:>10.2f}  "
                  f"P_l2={P_cur:>8.2f}  "
                  f"active={int(np.sum(t > 0)):>4}  "
                  f"T={T:.4f}  accept={recent_accept:.0%}  "
                  f"[{time.time() - t0:.1f}s]")

        if (iteration + 1) % epoch_len == 0:
            detail = end_of_epoch(iteration)
            if detail:
                history["converged"] = True
                history["stop_reason"] = "converged"
                history["stop_detail"] = detail
                break

        if max_time is not None and (iteration + 1) % 100 == 0 \
                and time.time() - t0 > float(max_time):
            history["stop_reason"] = "time_limit"
            history["stop_detail"] = (f"wall-clock limit {float(max_time):g} s reached "
                                      f"after {iteration + 1} iterations")
            break
    else:
        history["stop_detail"] = f"iteration limit {n_iter} reached"

    n_search = len(history["accept_rate"])

    # Greedy polish from the best plan: T = 0, improvements only.
    n_polish = int(ip["polish_epochs"]) * epoch_len
    if n_polish > 0:
        t = best_t.copy()
        state.resync(t)
        L, J_cur, P_cur = state.loss(state.dose, t)
        best_L = L
        L_before = L
        n_better = 0
        for _ in range(n_polish):
            accepted, move_name = step(0.0)
            n_better += int(accepted)
            record(0.0, accepted, move_name)
        state.resync(best_t)
        best_L, _, _ = state.loss(state.dose, best_t)
        print(f"    Polish: {n_polish} greedy proposals, {n_better} improvements, "
              f"best loss {L_before:.2f} -> {best_L:.2f}")

    elapsed = time.time() - t0
    n_run = len(history["accept_rate"])
    history["n_iterations_run"] = n_run
    history["n_search_iterations"] = n_search
    history["n_polish_iterations"] = n_run - n_search
    history["elapsed_s"] = elapsed
    J0 = history["cost"][0]
    moves = np.asarray(history["move"])
    n_time = int(np.sum(moves == "time"))
    n_pos = int(np.sum(moves == "position"))
    print(f"\n  IPSA completed in {elapsed:.1f}s  ({n_search} search + "
          f"{n_run - n_search} polish iterations, {len(history['epochs'])} epochs)")
    print(f"  Stopped ({history['stop_reason']}): {history['stop_detail']}")
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
