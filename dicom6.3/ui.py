import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from conversion import world_to_voxel
from cost_function import cost_function, copy_prescription_params

level = 150
width = 100
vmin = level - width / 2
vmax = level + width / 2

_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}


def _ensure_interactive_backend():
    """Switch off a headless backend so plt.show() can open a window."""
    if matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS:
        return matplotlib.get_backend()
    for name in ("TkAgg", "QtAgg", "Qt5Agg", "WxAgg"):
        try:
            matplotlib.use(name, force=True)
            return matplotlib.get_backend()
        except Exception:
            continue
    return matplotlib.get_backend()


def _can_show_interactive():
    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


def _parse_float(text, fallback):
    try:
        return float(text)
    except (TypeError, ValueError):
        return fallback


def _format_value(value):
    return f"{float(value):.2f}"


_MESH_REGION_COLORS = {
    "CTV": "r",
    "PTV": "tab:orange",
    "OAR": "c",
    "CTV+OAR": "m",
    "PTV+OAR": "m",
    "TARGET+OAR": "m",
    "OUTSIDE": "y",
}


def _mesh_marker_color(label):
    if isinstance(label, dict):
        region = label.get("region", "OUTSIDE")
    else:
        region = label or "OUTSIDE"
    return _MESH_REGION_COLORS.get(region, "y")


def start_ui(volume, spacing, origin, dwell_positions, show=None,
             dose_gy=None, masks=None, ctv_names=None, oar_names=None,
             initial_params=None, mesh_labels=None):
    """
    Image slice viewer with cost-function parameters shown and editable.

    Parameters
    ----------
    show : bool or None
        If None, opens the viewer only when matplotlib has a GUI backend.
        If False, skip the viewer (e.g. batch runs with Agg).
    dose_gy, masks, ctv_names, oar_names
        Optional; if provided, the cost J is recomputed when a value is applied.
    initial_params : dict or None
        Starting cost-function values (defaults to PRESCRIPTION_PARAMS).
    mesh_labels : list[dict] or None
        Per-vertex organ labels from needle meshing. Markers are coloured
        by region (CTV/PTV red/orange, OAR cyan, overlap magenta, outside yellow).

    Returns
    -------
    params : dict
        Cost-function parameters as last set in the GUI, or defaults if
        the viewer is skipped.
    """
    params = copy_prescription_params(initial_params)
    
    if show is None:
        _ensure_interactive_backend()
        show = _can_show_interactive()
    if not show:
        print("Image viewer skipped (non-interactive matplotlib backend: "
              f"{matplotlib.get_backend()}). Using default cost-function parameters.")
        return params

    print(f"\n  Opening GUI window (backend: {matplotlib.get_backend()}) ...")
    print("  Edit cost-function parameters, then click Continue.")


    z_max = volume.shape[0] 
    oar_keys = list(params.get("oar_params", {}).keys())
    can_eval_cost = (
        dose_gy is not None
        and masks is not None
        and ctv_names is not None
        and oar_names is not None
    )

    fig = plt.figure(figsize=(14.8, 8.2))
    ax = fig.add_axes([0.05, 0.16, 0.46, 0.76])
    ax.imshow(volume[0], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Image Viewer")

    # slider axis
    ax_slider = fig.add_axes([0.10, 0.07, 0.38, 0.035])
    slice_slider = Slider(ax_slider, "Slice", 0, z_max - 1, valinit=0, valstep=1)

    voxel_points = [
        world_to_voxel(p, origin, spacing)
        for p in dwell_positions
    ] # the dwell positions converted to voxel coordinates (z, y, x)
    label_list = list(mesh_labels) if mesh_labels is not None else None


    def draw_slice(z):
        ax.clear()
        ax.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)

        # plot mesh points in this slice, coloured by organ label
        seen = set()
        for i, (zz, yy, xx) in enumerate(voxel_points):
            if zz != z:
                continue
            lab = label_list[i] if label_list is not None and i < len(label_list) else None
            color = _mesh_marker_color(lab)
            ax.plot(xx, yy, "o", color=color, markersize=4)
            region = lab.get("region") if isinstance(lab, dict) else None
            if region and region not in seen:
                seen.add(region)
                ax.plot([], [], "o", color=color, markersize=4, label=region)
        if seen:
            ax.legend(loc="upper right", fontsize=7, title="Mesh")
        z_mm = origin[2] + z * spacing[2]
        ax.set_title(f"Slice {z}  |  z = {z_mm:.2f} mm")


    # update function
    def update_slice(val):
        draw_slice(int(val))
        fig.canvas.draw_idle()

    slice_slider.on_changed(update_slice)
    draw_slice(0)

    # ------------------------------------------------------------------
    # Cost-function parameter panel
    # ------------------------------------------------------------------
    fig.text(0.56, 0.93, "Cost-function parameters", fontsize=13, fontweight="bold")
    fig.text(
        0.56, 0.90,
        "Edit a value and press Enter.  Continue uses the current values.",
        fontsize=8.5, color="0.35",
    )

    boxes = {}
    y = 0.85

    def add_section(title):
        nonlocal y
        fig.text(0.56, y, title, fontsize=11, fontweight="bold")
        y -= 0.045

    def add_pair(left_key, left_label, left_val, right_key, right_label, right_val):
        nonlocal y
        fig.text(0.56, y + 0.006, left_label, fontsize=9)
        ax_l = fig.add_axes([0.68, y, 0.08, 0.032])
        boxes[left_key] = TextBox(ax_l, "", initial=_format_value(left_val))

        fig.text(0.78, y + 0.006, right_label, fontsize=9)
        ax_r = fig.add_axes([0.90, y, 0.08, 0.032])
        boxes[right_key] = TextBox(ax_r, "", initial=_format_value(right_val))
        y -= 0.050

    add_section("CTV / Target")
    add_pair("D_p", "D_p (Gy)", params["D_p"],
             "D_max", "D_max (Gy)", params["D_max"])
    add_pair("w_L", "w_L  underdose", params["w_L"],
             "w_H", "w_H  overdose", params["w_H"])

    for oar_key in oar_keys:
        add_section(oar_key.capitalize())
        oar_vals = params["oar_params"][oar_key]
        add_pair(f"{oar_key}.D_tol", "D_tol (Gy)", oar_vals["D_tol"],
                 f"{oar_key}.w_O", "w_O  impact", oar_vals["w_O"])

    add_section("Default OAR (unmatched)")
    add_pair("D_tol", "D_tol (Gy)", params["D_tol"],
             "w_O", "w_O  impact", params["w_O"])

    ax_info = fig.add_axes([0.56, 0.08, 0.28, 0.08])
    ax_info.axis("off")
    info_text = ax_info.text(0.0, 0.5, "", va="center", ha="left", fontsize=9,
                             family="monospace")

    def current_params():
        p = copy_prescription_params(params)
        p["D_p"] = _parse_float(boxes["D_p"].text, p["D_p"])
        p["D_max"] = _parse_float(boxes["D_max"].text, p["D_max"])
        p["w_L"] = _parse_float(boxes["w_L"].text, p["w_L"])
        p["w_H"] = _parse_float(boxes["w_H"].text, p["w_H"])
        p["D_tol"] = _parse_float(boxes["D_tol"].text, p["D_tol"])
        p["w_O"] = _parse_float(boxes["w_O"].text, p["w_O"])
        for oar_key in oar_keys:
            p["oar_params"][oar_key] = {
                "D_tol": _parse_float(
                    boxes[f"{oar_key}.D_tol"].text,
                    p["oar_params"][oar_key]["D_tol"],
                ),
                "w_O": _parse_float(
                    boxes[f"{oar_key}.w_O"].text,
                    p["oar_params"][oar_key]["w_O"],
                ),
            }
        return p

    held = {"params": current_params()}

    def refresh_cost(_text=None):
        held["params"] = current_params()
        if not can_eval_cost:
            info_text.set_text("Continue to use these parameters.")
            fig.canvas.draw_idle()
            return
        J, breakdown = cost_function(
            dose_gy, masks, ctv_names, oar_names, held["params"],
        )
        n_struct = len(breakdown)
        info_text.set_text(f"Cost  J = {J:.2f}    ({n_struct} structures)")
        fig.canvas.draw_idle()

    for box in boxes.values():
        box.on_submit(refresh_cost)

    def set_box(key, value):
        boxes[key].set_val(_format_value(value))

    def on_reset(_event):
        defaults = copy_prescription_params()
        set_box("D_p", defaults["D_p"])
        set_box("D_max", defaults["D_max"])
        set_box("w_L", defaults["w_L"])
        set_box("w_H", defaults["w_H"])
        set_box("D_tol", defaults["D_tol"])
        set_box("w_O", defaults["w_O"])
        for oar_key in oar_keys:
            if oar_key in defaults["oar_params"]:
                set_box(f"{oar_key}.D_tol", defaults["oar_params"][oar_key]["D_tol"])
                set_box(f"{oar_key}.w_O", defaults["oar_params"][oar_key]["w_O"])
        refresh_cost()

    def on_continue(_event):
        held["params"] = current_params()
        plt.close(fig)

    ax_reset = fig.add_axes([0.86, 0.12, 0.12, 0.04])
    ax_cont = fig.add_axes([0.86, 0.07, 0.12, 0.04])
    btn_reset = Button(ax_reset, "Reset")
    btn_continue = Button(ax_cont, "Continue")
    btn_reset.on_clicked(on_reset)
    btn_continue.on_clicked(on_continue)

    def on_close(_evt):
        try:
            held["params"] = current_params()
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", on_close)

    refresh_cost()
    plt.show(block=True)

    # Keep widget references alive until the window closes.
    _ = (slice_slider, boxes, btn_reset, btn_continue)
    return held["params"]