"""Post-optimization clinical plan evaluation against DVH criteria.

Percentages are relative to the prescription dose D_p (Gy).
Absolute volumes use the CT voxel size (mm³ → cc).
"""

import numpy as np
from cost_function import PRESCRIPTION_PARAMS


# =====================================================================
# Clinical acceptance criteria
#   kind: "ge" (>=), "le" (<=), "eq" (== / essentially zero), "range" (inclusive)
#   unit: "pct" (relative volume %) or "cc" (absolute volume) for V metrics;
#         "pct_rx" (dose as % of D_p) for D metrics
#   For soft ranges like "90-95%" on a lower bound: lo=goal, hi=ideal
#   For soft ranges like "90%-100%" on an upper bound: lo=ideal, hi=limit
# =====================================================================

_TARGET_CRITERIA = [
    # V100% >= 90–95%  (pass ≥90%, ideal ≥95%)
    {"metric": "V100", "kind": "ge_soft", "lo": 90.0, "hi": 95.0, "unit": "pct",
     "label": "V100% >= 90-95%"},
    # D90%: 105%–115% of Rx
    {"metric": "D90", "kind": "range", "lo": 105.0, "hi": 115.0, "unit": "pct_rx",
     "label": "D90% in 105%-115%"},
    {"metric": "V150", "kind": "le", "limit": 35.0, "unit": "pct",
     "label": "V150% <= 35%"},
    {"metric": "V200", "kind": "le", "limit": 11.0, "unit": "pct",
     "label": "V200% <= 11%"},
]

CLINICAL_CRITERIA = {
    "ptv": list(_TARGET_CRITERIA),
    "ctv": list(_TARGET_CRITERIA),
    "urethra": [
        {"metric": "V115", "kind": "le", "limit": 5.0, "unit": "pct",
         "label": "V115% <= 5%"},
        {"metric": "V125", "kind": "le", "limit": 1.0, "unit": "cc",
         "label": "V125% <= 1 cc"},
        {"metric": "V150", "kind": "eq", "limit": 0.0, "unit": "cc",
         "label": "V150% = 0 cc"},
        {"metric": "D10", "kind": "le", "limit": 120.0, "unit": "pct_rx",
         "label": "D10% <= 120%"},
        {"metric": "D0.01cc", "kind": "le", "limit": 125.0, "unit": "pct_rx",
         "label": "D0.01cc <= 125%"},
    ],
    "bladder": [
        {"metric": "V75", "kind": "le", "limit": 1.0, "unit": "cc",
         "label": "V75% <= 1 cc"},
        # D0.01cc <= 90%–100%  (pass ≤100%, ideal ≤90%)
        {"metric": "D0.01cc", "kind": "le_soft", "lo": 90.0, "hi": 100.0, "unit": "pct_rx",
         "label": "D0.01cc <= 90%-100%"},
    ],
    "rectum": [
        {"metric": "V75", "kind": "le", "limit": 1.0, "unit": "cc",
         "label": "V75% <= 1 cc"},
        {"metric": "V80", "kind": "le", "limit": 0.5, "unit": "cc",
         "label": "V80% <= 0.5 cc"},
        {"metric": "V100", "kind": "eq", "limit": 0.0, "unit": "cc",
         "label": "V100% = 0 cc"},
        {"metric": "D2cc", "kind": "le", "limit": 70.0, "unit": "pct_rx",
         "label": "D2cc <= 70%"},
    ],
}


def _match_structure(name, key):
    return key in name.lower()


def _voxel_volume_cc(spacing_mm):
    """Voxel volume in cc from spacing (dx, dy, dz) in mm."""
    return float(np.prod(spacing_mm)) / 1000.0


def volume_receiving(dose_values, threshold_gy, voxel_cc):
    """Absolute volume (cc) and relative volume (%) receiving ≥ threshold."""
    n = len(dose_values)
    if n == 0:
        return 0.0, 0.0
    n_above = int(np.sum(dose_values >= threshold_gy))
    vol_cc = n_above * voxel_cc
    vol_pct = 100.0 * n_above / n
    return vol_cc, vol_pct


def dose_covering_fraction(dose_values, fraction_pct):
    """Dose (Gy) covering *fraction_pct*% of the volume (Dx%).

    D90 means 90% of voxels receive at least this dose → 10th percentile.
    """
    if len(dose_values) == 0:
        return float("nan")
    return float(np.percentile(dose_values, 100.0 - fraction_pct))


def dose_to_hottest_volume(dose_values, vol_cc, voxel_cc):
    """Minimum dose (Gy) in the hottest *vol_cc* of the structure (e.g. D0.01cc, D2cc)."""
    if len(dose_values) == 0 or voxel_cc <= 0:
        return float("nan")
    n_vox = max(1, int(np.ceil(vol_cc / voxel_cc)))
    n_vox = min(n_vox, len(dose_values))
    hottest = np.partition(dose_values, -n_vox)[-n_vox:]
    return float(np.min(hottest))


def compute_structure_metrics(dose_values, D_p, voxel_cc):
    """Compute the DVH metrics used by the clinical criteria."""
    metrics = {}
    if len(dose_values) == 0 or D_p <= 0:
        return metrics

    for pct in (75, 80, 100, 115, 125, 150, 200):
        thr = D_p * pct / 100.0
        vol_cc, vol_pct = volume_receiving(dose_values, thr, voxel_cc)
        metrics[f"V{pct}_cc"] = vol_cc
        metrics[f"V{pct}_pct"] = vol_pct

    for frac in (10, 90):
        d_gy = dose_covering_fraction(dose_values, frac)
        metrics[f"D{frac}_gy"] = d_gy
        metrics[f"D{frac}_pct"] = 100.0 * d_gy / D_p

    for label, vol in (("D0.01cc", 0.01), ("D2cc", 2.0)):
        d_gy = dose_to_hottest_volume(dose_values, vol, voxel_cc)
        metrics[f"{label}_gy"] = d_gy
        metrics[f"{label}_pct"] = 100.0 * d_gy / D_p

    metrics["volume_cc"] = len(dose_values) * voxel_cc
    metrics["Dmin_gy"] = float(dose_values.min())
    metrics["Dmean_gy"] = float(dose_values.mean())
    metrics["Dmax_gy"] = float(dose_values.max())
    return metrics


def _metric_value(metrics, metric_name, unit):
    """Look up a criterion metric value in the requested reporting unit."""
    if metric_name.startswith("V"):
        key = f"{metric_name}_{'pct' if unit == 'pct' else 'cc'}"
        return metrics.get(key, float("nan"))
    if metric_name.startswith("D"):
        if unit == "pct_rx":
            return metrics.get(f"{metric_name}_pct", float("nan"))
        return metrics.get(f"{metric_name}_gy", float("nan"))
    return float("nan")


def _check_criterion(value, criterion, tol_cc=1e-6):
    """Return (status, message) where status is PASS / IDEAL / FAIL / N/A."""
    if value != value:  # NaN
        return "N/A", "no data"

    kind = criterion["kind"]
    unit = criterion["unit"]
    suffix = "%" if unit in ("pct", "pct_rx") else " cc"

    if kind == "le":
        ok = value <= criterion["limit"] + (tol_cc if unit == "cc" else 0.0)
        return ("PASS" if ok else "FAIL",
                f"{value:.2f}{suffix}  (limit <= {criterion['limit']:g}{suffix})")

    if kind == "ge":
        ok = value >= criterion["limit"]
        return ("PASS" if ok else "FAIL",
                f"{value:.2f}{suffix}  (limit >= {criterion['limit']:g}{suffix})")

    if kind == "eq":
        ok = abs(value - criterion["limit"]) <= (tol_cc if unit == "cc" else 1e-3)
        return ("PASS" if ok else "FAIL",
                f"{value:.2f}{suffix}  (limit = {criterion['limit']:g}{suffix})")

    if kind == "range":
        lo, hi = criterion["lo"], criterion["hi"]
        ok = lo <= value <= hi
        return ("PASS" if ok else "FAIL",
                f"{value:.2f}{suffix}  (target {lo:g}-{hi:g}{suffix})")

    if kind == "ge_soft":
        # lo = minimum acceptable, hi = ideal
        lo, hi = criterion["lo"], criterion["hi"]
        if value >= hi:
            return "IDEAL", f"{value:.2f}{suffix}  (>= {hi:g}{suffix} ideal, >= {lo:g} ok)"
        if value >= lo:
            return "PASS", f"{value:.2f}{suffix}  (>= {lo:g}{suffix} ok, ideal >= {hi:g})"
        return "FAIL", f"{value:.2f}{suffix}  (need >= {lo:g}{suffix}, ideal >= {hi:g})"

    if kind == "le_soft":
        # lo = ideal upper, hi = maximum acceptable
        lo, hi = criterion["lo"], criterion["hi"]
        if value <= lo:
            return "IDEAL", f"{value:.2f}{suffix}  (<= {lo:g}{suffix} ideal, <= {hi:g} ok)"
        if value <= hi:
            return "PASS", f"{value:.2f}{suffix}  (<= {hi:g}{suffix} ok, ideal <= {lo:g})"
        return "FAIL", f"{value:.2f}{suffix}  (need <= {hi:g}{suffix}, ideal <= {lo:g})"

    return "N/A", "unknown criterion"


def evaluate_plan(dose_gy, masks, ctv_names, ptv_names, oar_names, spacing_mm,
                  D_p=None):
    """Evaluate optimised dose against clinical DVH criteria.

    Parameters
    ----------
    dose_gy : ndarray
        Dose grid in Gy (same shape as masks).
    masks : dict[str, bool ndarray]
    ctv_names, ptv_names, oar_names : list[str]
    spacing_mm : sequence of 3 floats (dx, dy, dz) in mm
    D_p : float or None
        Prescription dose in Gy.  Defaults to PRESCRIPTION_PARAMS["D_p"].

    Returns
    -------
    report : dict
        Keys are structure roles ("ptv", "ctv", "urethra", ...).  Each value
        has structure name, metrics, and a list of criterion check results.
    """
    if D_p is None:
        D_p = PRESCRIPTION_PARAMS["D_p"]
    voxel_cc = _voxel_volume_cc(spacing_mm)

    # Map role → structure name(s)
    role_names = {
        "ptv": list(ptv_names),
        "ctv": list(ctv_names),
    }
    for role in ("urethra", "bladder", "rectum"):
        role_names[role] = [n for n in oar_names if _match_structure(n, role)]

    report = {}
    for role, names in role_names.items():
        if role not in CLINICAL_CRITERIA:
            continue
        if not names:
            report[role] = {
                "structures": [],
                "metrics": {},
                "checks": [],
                "missing": True,
            }
            continue

        # Union mask when multiple structures share a role (e.g. several CTVs)
        combined = None
        for name in names:
            if name not in masks:
                continue
            combined = masks[name].copy() if combined is None else (combined | masks[name])
        if combined is None or int(combined.sum()) == 0:
            report[role] = {
                "structures": names,
                "metrics": {},
                "checks": [],
                "missing": True,
            }
            continue

        d = dose_gy[combined]
        metrics = compute_structure_metrics(d, D_p, voxel_cc)
        checks = []
        for crit in CLINICAL_CRITERIA[role]:
            val = _metric_value(metrics, crit["metric"], crit["unit"])
            status, detail = _check_criterion(val, crit)
            checks.append({
                "label": crit["label"],
                "metric": crit["metric"],
                "value": val,
                "status": status,
                "detail": detail,
            })

        report[role] = {
            "structures": names,
            "metrics": metrics,
            "checks": checks,
            "missing": False,
        }

    return report


def print_evaluation_report(report, D_p=None):
    """Pretty-print the clinical evaluation report."""
    if D_p is None:
        D_p = PRESCRIPTION_PARAMS["D_p"]

    print(f"\n{'='*70}")
    print("CLINICAL PLAN EVALUATION (post-optimization)")
    print(f"{'='*70}")
    print(f"  Prescription dose D_p = {D_p:.2f} Gy ({D_p*100:.2f} cGy)")
    print(f"  Percentages below are relative to D_p.")

    n_pass = n_fail = n_ideal = 0
    role_order = ("ptv", "ctv", "urethra", "bladder", "rectum")

    for role in role_order:
        info = report.get(role)
        if info is None:
            continue
        title = role.upper()
        print(f"\n  --- {title} ---")
        if info.get("missing"):
            print(f"    (structure not found in plan)")
            continue

        names = ", ".join(info["structures"])
        m = info["metrics"]
        print(f"    Structures : {names}")
        print(f"    Volume     : {m.get('volume_cc', float('nan')):.3f} cc")
        print(f"    Dmin/mean/max : {m.get('Dmin_gy', float('nan')):.2f} / "
              f"{m.get('Dmean_gy', float('nan')):.2f} / "
              f"{m.get('Dmax_gy', float('nan')):.2f} Gy")

        print(f"    {'Criterion':<28} {'Status':<7}  Result")
        print(f"    {'-'*28} {'-'*7}  {'-'*40}")
        for c in info["checks"]:
            status = c["status"]
            if status == "PASS":
                n_pass += 1
            elif status == "IDEAL":
                n_ideal += 1
            elif status == "FAIL":
                n_fail += 1
            print(f"    {c['label']:<28} {status:<7}  {c['detail']}")

    total = n_pass + n_ideal + n_fail
    print(f"\n  SUMMARY: {n_pass + n_ideal}/{total} criteria met "
          f"({n_ideal} ideal, {n_pass} acceptable), {n_fail} failed")
    print()
    return n_fail == 0
