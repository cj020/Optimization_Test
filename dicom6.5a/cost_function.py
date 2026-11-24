import numpy as np

# =====================================================================
# Prescription & penalty parameters (adjust as needed)
# Doses are stored in Gy (1 Gy = 100 cGy).
# =====================================================================

PRESCRIPTION_PARAMS = {
    # Treatment type: selects the plan-evaluation criteria
    # (plan_evaluation.PROTOCOLS: "mono" = prostate monotherapy, 1350 cGy;
    # "boost" = prostate boost, 1500 cGy).
    "protocol": "boost",
    "D_p": 15.00,       # Prescription / lower-bound dose (Gy); boost default
    "D_max": 30.00,     # Max tolerable dose in target (Gy)
    "w_L": 15.0,        # Impact factor for target under-dose
    "w_H": 5.0,         # Impact factor for target over-dose
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


def copy_prescription_params(params=None):
    """Return a deep copy of prescription / penalty parameters."""
    src = PRESCRIPTION_PARAMS if params is None else params
    copied = {key: value for key, value in src.items()
              if key not in ("oar_params", "objectives")}
    copied["oar_params"] = {
        name: dict(vals) for name, vals in src.get("oar_params", {}).items()
    }
    if src.get("objectives") is not None:
        copied["objectives"] = [dict(obj) for obj in src["objectives"]]
    return copied


# ---------------------------------------------------------------------
# Per-structure objective goals
#
# ``params["objectives"]`` (optional) is a list of dicts, one per selected
# structure:
#     name        structure name (key into ``masks``)
#     role        "Target" or "OAR"
#     enabled     bool -- unchecked objectives are ignored
#     min_dose    Gy or None -- under-dose penalised below this (targets)
#     min_weight  impact factor of the under-dose term
#     max_dose    Gy or None -- over-dose penalised above this
#     max_weight  impact factor of the over-dose term
# When it is absent, objectives are built from ``ctv_names`` / ``oar_names``
# and the scalar parameters above (the original behaviour).
# ---------------------------------------------------------------------

def target_objective(name, params=None):
    """Default Target objective for *name* (min D_p / w_L, max D_max / w_H)."""
    p = dict(PRESCRIPTION_PARAMS)
    p.update(params or {})
    return {
        "name": name, "role": "Target", "enabled": True,
        "min_dose": float(p["D_p"]), "min_weight": float(p["w_L"]),
        "max_dose": float(p["D_max"]), "max_weight": float(p["w_H"]),
    }


def oar_objective(name, params=None):
    """Default OAR objective for *name* (max D_tol / w_O, matched by name)."""
    p = dict(PRESCRIPTION_PARAMS)
    p.update(params or {})
    D_tol, w_O = _oar_limits(name, p)
    return {
        "name": name, "role": "OAR", "enabled": True,
        "min_dose": None, "min_weight": 0.0,
        "max_dose": float(D_tol), "max_weight": float(w_O),
    }


def default_objectives(ctv_names, oar_names, params=None):
    """Objectives equivalent to the original CTV / OAR cost function."""
    return ([target_objective(n, params) for n in ctv_names]
            + [oar_objective(n, params) for n in oar_names])


def _oar_limits(name, p):
    """Resolve (D_tol, w_O) for an OAR by name, falling back to defaults."""
    name_lower = name.lower()
    for key, vals in p.get("oar_params", {}).items():
        if key in name_lower:
            return vals["D_tol"], vals["w_O"]
    return p["D_tol"], p["w_O"]


def penalty_target(dose_values, D_p, D_max, w_L, w_H):
    """Compute the linear penalty for target (CTV) voxels.

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


def resolve_params(params=None):
    """PRESCRIPTION_PARAMS merged with *params* (oar_params deep-merged)."""
    p = dict(PRESCRIPTION_PARAMS)
    if params is not None:
        p.update(params)
        if "oar_params" in params:
            merged = dict(PRESCRIPTION_PARAMS["oar_params"])
            merged.update(params["oar_params"])
            p["oar_params"] = merged
    return p


def active_objectives(masks, ctv_names, oar_names, params=None):
    """Objectives that contribute to J: enabled, first per name, non-empty mask.

    Each returned dict has name, role, lo, hi, w_lo, w_hi (lo / hi are None
    when that goal is off).  ``cost_function`` and the fast IPSA path both
    use this, so they penalise exactly the same voxels.
    """
    p = resolve_params(params)
    objectives = p.get("objectives")
    if objectives is None:
        objectives = default_objectives(ctv_names, oar_names, p)
    out, seen = [], set()
    for obj in objectives:
        if not obj.get("enabled", True):
            continue
        name = obj["name"]
        if name not in masks or name in seen or not masks[name].any():
            continue
        seen.add(name)
        out.append({
            "name": name,
            "role": obj.get("role", "OAR"),
            "lo": obj.get("min_dose"),
            "hi": obj.get("max_dose"),
            "w_lo": float(obj.get("min_weight") or 0.0),
            "w_hi": float(obj.get("max_weight") or 0.0),
        })
    return out


def cost_function(dose_gy, masks, ctv_names, oar_names, params=None):
    """Evaluate the total cost function over the dose grid.

    J = sum_target P_target(D_i)  +  sum_OAR P_OAR(D_i)

    Parameters
    ----------
    dose_gy : np.ndarray, shape (nz, ny, nx)
        3-D dose grid in Gy.
    masks : dict[str, np.ndarray(bool)]
        Boolean masks for each structure, keyed by structure name.
    ctv_names : list[str]
        Names of all CTV structures.
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
    total_cost = 0.0
    breakdown = {}

    for obj in active_objectives(masks, ctv_names, oar_names, params):
        name = obj["name"]
        m = masks[name]  # Boolean 3-D array, shape (nz, ny, nx)
        n_vox = int(m.sum())
        d = dose_gy[m]   # dose values inside this structure
        lo, hi = obj["lo"], obj["hi"]
        w_lo, w_hi = obj["w_lo"], obj["w_hi"]
        pen = np.zeros_like(d, dtype=np.float64)
        if lo is not None and w_lo > 0.0:
            pen += w_lo * np.maximum(lo - d, 0.0)
        if hi is not None and w_hi > 0.0:
            pen += w_hi * np.maximum(d - hi, 0.0)
        struct_cost = float(pen.sum())
        total_cost += struct_cost
        n_under = int(np.sum(d < lo)) if lo is not None else 0
        n_over = int(np.sum(d > hi)) if hi is not None else 0
        breakdown[name] = {
            "type": obj["role"],
            "voxels": n_vox,
            "cost": struct_cost,
            "mean_penalty": float(pen.mean()),
            "min_dose": lo,
            "min_weight": w_lo,
            "max_dose": hi,
            "max_weight": w_hi,
            "n_underdosed": n_under,
            "n_overdosed": n_over,
            # OAR-style aliases
            "n_over_tolerance": n_over,
            "D_tol": hi,
            "w_O": w_hi,
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
    print(f"  Prescription D_p = {p['D_p']:.2f} Gy ({p['D_p']*100:.2f} cGy)")
    print(f"  Objective goals:")
    for name, info in breakdown.items():
        parts = []
        if info.get("min_dose") is not None:
            parts.append(f"min {info['min_dose']*100:.2f} cGy (imp {info['min_weight']:.3f})")
        if info.get("max_dose") is not None:
            parts.append(f"max {info['max_dose']*100:.2f} cGy (imp {info['max_weight']:.3f})")
        print(f"    {name:<30} {info['type']:<6} " + ",  ".join(parts))

    print(f"\n  {'Structure':<30} {'Type':<6} {'Voxels':>8} "
          f"{'Cost':>12} {'Mean pen':>10} {'Dmin':>8} {'Dmean':>8} {'Dmax':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for name, info in breakdown.items():
        print(f"  {name:<30} {info['type']:<6} {info['voxels']:>8} "
              f"{info['cost']:>12.2f} {info['mean_penalty']:>10.4f} "
              f"{info['Dmin']:>8.4f} {info['Dmean']:>8.4f} {info['Dmax']:>8.4f}")

    print(f"\n  Objective detail:")
    for name, info in breakdown.items():
        parts = []
        if info.get("min_dose") is not None:
            parts.append(f"{info['n_underdosed']} voxels below {info['min_dose']:.2f} Gy")
        if info.get("max_dose") is not None:
            parts.append(f"{info['n_overdosed']} voxels above {info['max_dose']:.2f} Gy")
        print(f"    {name}: " + ", ".join(parts))


    print(f"\n  TOTAL COST J = {J:.2f}")
    print()
