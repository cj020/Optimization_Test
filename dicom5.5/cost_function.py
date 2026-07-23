import numpy as np

# =====================================================================
# Prescription & penalty parameters (adjust as needed)
# Doses are stored in Gy (1 Gy = 100 cGy).
# =====================================================================

PRESCRIPTION_PARAMS = {
    # PTV: lower bound 1575 cGy, upper bound 2250 cGy
    "D_p": 15.75,       # Prescription / lower-bound dose (Gy)
    "D_max": 22.50,     # Max tolerable dose in target (Gy)
    "w_L": 100.0,       # Impact factor for target under-dose
    "w_H": 4.0,         # Impact factor for target over-dose
    # Default OAR (used when structure name does not match oar_params)
    "D_tol": 11.25,     # Default OAR tolerance dose (Gy)
    "w_O": 5.0,         # Default OAR over-dose impact factor
    # Per-OAR overrides — matched by case-insensitive substring of structure name
    "oar_params": {
        "urethra": {"D_tol": 17.10, "w_O": 57.0},   # 1710 cGy, impact 57
        "bladder": {"D_tol": 11.25, "w_O": 5.0},    # 1125 cGy, impact 5
        "rectum":  {"D_tol": 11.25, "w_O": 29.3},   # 1125 cGy, impact 29.3
    },
}


def _oar_limits(name, p):
    """Resolve (D_tol, w_O) for an OAR by name, falling back to defaults."""
    name_lower = name.lower()
    for key, vals in p.get("oar_params", {}).items():
        if key in name_lower:
            return vals["D_tol"], vals["w_O"]
    return p["D_tol"], p["w_O"]


def penalty_target(dose_values, D_p, D_max, w_L, w_H):
    """Compute the linear penalty for target (PTV) voxels.

    P_target(D) = w_L * max(D_p - D, 0) + w_H * max(D - D_max, 0)

    Parameters
    ----------
    dose_values : np.ndarray
        Dose in Gy for each target voxel.
    D_p : float
        Prescription dose (Gy).
    D_max : float
        Maximum tolerable dose in target (Gy).
    w_L : float
        Under-dose penalty weight.
    w_H : float
        Over-dose penalty weight.

    Returns
    -------
    penalties : np.ndarray
        Per-voxel penalty values (same shape as *dose_values*).
    """
    underdose = np.maximum(D_p - dose_values, 0.0)
    overdose = np.maximum(dose_values - D_max, 0.0)
    return w_L * underdose + w_H * overdose


def penalty_oar(dose_values, D_tol, w_O):
    """Compute the linear penalty for OAR voxels.

    P_OAR(D) = w_O * max(D - D_tol, 0)

    Parameters
    ----------
    dose_values : np.ndarray
        Dose in Gy for each OAR voxel.
    D_tol : float
        OAR tolerance dose (Gy).
    w_O : float
        OAR over-dose penalty weight.

    Returns
    -------
    penalties : np.ndarray
        Per-voxel penalty values (same shape as *dose_values*).
    """
    return w_O * np.maximum(dose_values - D_tol, 0.0)


def cost_function(dose_gy, masks, ptv_names, oar_names, params=None):
    """Evaluate the total cost function over the dose grid.

    J = sum_target P_target(D_i)  +  sum_OAR P_OAR(D_i)

    Parameters
    ----------
    dose_gy : np.ndarray, shape (nz, ny, nx)
        3-D dose grid in Gy.
    masks : dict[str, np.ndarray(bool)]
        Boolean masks for each structure, keyed by structure name.
    ptv_names : list[str]
        Names of all PTV structures.
    oar_names : list[str]
        Names of all OAR structures.
    params : dict or None
        Override any key in PRESCRIPTION_PARAMS.  If *None*, defaults are
        used.

    Returns
    -------
    J : float
        Total cost.
    breakdown : dict
        Per-structure cost and voxel-level statistics.
    """
    p = dict(PRESCRIPTION_PARAMS)
    if params is not None:
        p.update(params)
        # Deep-merge oar_params if both sides provide it
        if "oar_params" in PRESCRIPTION_PARAMS and "oar_params" in (params or {}):
            merged = dict(PRESCRIPTION_PARAMS["oar_params"])
            merged.update(params["oar_params"])
            p["oar_params"] = merged

    D_p   = p["D_p"]
    D_max = p["D_max"]
    w_L   = p["w_L"]
    w_H   = p["w_H"]

    total_cost = 0.0
    breakdown = {}

    for name in ptv_names:
        if name not in masks:
            continue
        m = masks[name]
        n_vox = int(m.sum())
        if n_vox == 0:
            continue
        d = dose_gy[m]
        pen = penalty_target(d, D_p, D_max, w_L, w_H)
        struct_cost = float(pen.sum())
        total_cost += struct_cost
        n_underdosed = int(np.sum(d < D_p))
        n_overdosed  = int(np.sum(d > D_max))
        breakdown[name] = {
            "type": "PTV",
            "voxels": n_vox,
            "cost": struct_cost,
            "mean_penalty": float(pen.mean()),
            "n_underdosed": n_underdosed,
            "n_overdosed": n_overdosed,
            "Dmin": float(d.min()),
            "Dmean": float(d.mean()),
            "Dmax": float(d.max()),
        }

    for name in oar_names:
        if name not in masks:
            continue
        m = masks[name]
        n_vox = int(m.sum())
        if n_vox == 0:
            continue
        D_tol, w_O = _oar_limits(name, p)
        d = dose_gy[m]
        pen = penalty_oar(d, D_tol, w_O)
        struct_cost = float(pen.sum())
        total_cost += struct_cost
        n_over_tol = int(np.sum(d > D_tol))
        breakdown[name] = {
            "type": "OAR",
            "voxels": n_vox,
            "cost": struct_cost,
            "mean_penalty": float(pen.mean()),
            "n_over_tolerance": n_over_tol,
            "D_tol": D_tol,
            "w_O": w_O,
            "Dmin": float(d.min()),
            "Dmean": float(d.mean()),
            "Dmax": float(d.max()),
        }

    return total_cost, breakdown


def print_cost_report(J, breakdown, params=None):
    """Pretty-print the cost function evaluation."""
    p = dict(PRESCRIPTION_PARAMS)
    if params is not None:
        p.update(params)

    print(f"\n{'='*70}")
    print("COST FUNCTION EVALUATION")
    print(f"{'='*70}")
    print(f"  PTV lower bound    D_p   = {p['D_p']:.2f} Gy  "
          f"({p['D_p']*100:.2f} cGy),  impact w_L = {p['w_L']:.2f}")
    print(f"  PTV upper bound    D_max = {p['D_max']:.2f} Gy  "
          f"({p['D_max']*100:.2f} cGy),  impact w_H = {p['w_H']:.2f}")
    print(f"  OAR limits (per structure):")
    for key, vals in p.get("oar_params", {}).items():
        print(f"    {key.capitalize():<12} D_tol = {vals['D_tol']:.2f} Gy "
              f"({vals['D_tol']*100:.2f} cGy),  impact w_O = {vals['w_O']:.2f}")

    print(f"\n  {'Structure':<30} {'Type':<5} {'Voxels':>8} "
          f"{'Cost':>12} {'Mean pen':>10} {'Dmin':>8} {'Dmean':>8} {'Dmax':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for name, info in breakdown.items():
        print(f"  {name:<30} {info['type']:<5} {info['voxels']:>8} "
              f"{info['cost']:>12.2f} {info['mean_penalty']:>10.4f} "
              f"{info['Dmin']:>8.4f} {info['Dmean']:>8.4f} {info['Dmax']:>8.4f}")

    if any(v["type"] == "PTV" for v in breakdown.values()):
        print(f"\n  PTV detail:")
        for name, info in breakdown.items():
            if info["type"] != "PTV":
                continue
            print(f"    {name}: {info['n_underdosed']} underdosed voxels (D < D_p), "
                  f"{info['n_overdosed']} overdosed voxels (D > D_max)")

    if any(v["type"] == "OAR" for v in breakdown.values()):
        print(f"\n  OAR detail:")
        for name, info in breakdown.items():
            if info["type"] != "OAR":
                continue
            D_tol = info.get("D_tol", p["D_tol"])
            w_O = info.get("w_O", p["w_O"])
            print(f"    {name}: {info['n_over_tolerance']} voxels above "
                  f"tolerance (D > {D_tol:.2f} Gy), impact = {w_O:.2f}")


    print(f"\n  TOTAL COST J = {J:.2f}")
    print()
