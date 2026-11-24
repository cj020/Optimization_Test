"""Main application window -- File menu import, 2x2 viewer grid, results panel.

Layout follows the HDR Brachy TPS reference UI: the window opens empty,
File > Import DICOM Folder... loads a case in the background, and
Optimization > Optimization Settings... > Run runs IPSA and shows the optimised dwell
positions and dwell times.
"""

import csv
import os
import re
import sys
import threading
import traceback

import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QActionGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.cm as cm

try:
    import vtk
    import vtkmodules.qt
    vtkmodules.qt.PyQtImpl = "PyQt5"
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.util import numpy_support
    _HAS_VTK = True
except Exception:
    _HAS_VTK = False

from cost_function import (
    PRESCRIPTION_PARAMS,
    copy_prescription_params,
    default_objectives,
    oar_objective,
    target_objective,
)
from ipsa import IPSA_DEFAULTS
from plan_evaluation import DEFAULT_PROTOCOL, PROTOCOLS, protocol_info
from needle_mesh import EXTENDED_CTV_NAME, EXTENDED_PTV_NAME
from pipeline import (
    CANDIDATES_ALL,
    CANDIDATES_EXTENDED_CTV,
    CANDIDATES_EXTENDED_PTV,
    EXTENDED_CTV_MARGIN_MM,
    EXTENDED_PTV_MARGIN_MM,
    GRID_1MM,
    GRID_3MM,
    build_candidate_mesh,
    load_case,
    optimize_case,
)


_HEADER_STYLE = (
    "background-color: rgba(0,0,0,180); color: #00ff00; "
    "font-weight: bold; padding: 2px 6px;"
)

_MESH_REGION_COLORS = {
    "CTV": "#ff3030",
    "PTV": "#ff8c00",
    "OAR": "#00e5ee",
    "CTV+OAR": "#ff00ff",
    "PTV+OAR": "#ff00ff",
    "TARGET+OAR": "#ff00ff",
    "OUTSIDE": "#ffff00",
}

_CLASS_COLORS = {"CTV": "#ff3030", "PTV": "#ff8c00"}
_OAR_PALETTE = ["#00bfff", "#7fff00", "#ffd700", "#ff69b4", "#9370db", "#40e0d0"]
_EXTENDED_PTV_COLOR = "#ffcc66"
_EXTENDED_CTV_COLOR = "#66ff99"

# Isodose levels as fractions of the prescription dose D_p.
_ISODOSE_LEVELS = [(0.5, "#00a0ff"), (1.0, "#00ff00"), (1.5, "#ffff00"), (2.0, "#ff0000")]

_DWELL_CMAP = "jet"

# Dose color wash: semi-transparent dose overlay on the CT / US image.
_WASH_CMAP = "jet"
WASH_DEFAULTS = {"min_pct": 50.0, "max_pct": 200.0, "opacity": 0.40}

# numpy volume axis (z, y, x) -> index into world (x, y, z) arrays
_WORLD = {0: 2, 1: 1, 2: 0}
_AXIS_LETTER = {0: "z", 1: "y", 2: "x"}


def _region_color(label):
    region = label.get("region", "OUTSIDE") if isinstance(label, dict) else "OUTSIDE"
    return _MESH_REGION_COLORS.get(region, _MESH_REGION_COLORS["OUTSIDE"])


def _structure_colors(structures, masks):
    colors = {}
    oar_i = 0
    for s in structures:
        name = s["name"]
        cls = s["classification"]
        # Unclassified structures (body, markers, ...) are available as
        # optimization objectives but are not drawn.
        if name not in masks or cls == "OTHER":
            continue
        if cls in _CLASS_COLORS:
            colors[name] = _CLASS_COLORS[cls]
        else:
            colors[name] = _OAR_PALETTE[oar_i % len(_OAR_PALETTE)]
            oar_i += 1
    return colors


# ----------------------------------------------------------------------
# Console capture and background work
# ----------------------------------------------------------------------

class _ConsoleStream:
    """Tee for sys.stdout: keeps the terminal output and buffers it for the log panel."""

    def __init__(self, original):
        self._original = original
        self._lock = threading.Lock()
        self._chunks = []

    def write(self, text):
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        with self._lock:
            self._chunks.append(text)
        return len(text)

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass

    def take(self):
        with self._lock:
            text = "".join(self._chunks)
            self._chunks.clear()
        return text


class _Worker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())
            return
        self.succeeded.emit(result)


# ----------------------------------------------------------------------
# Shared viewer state
# ----------------------------------------------------------------------

class ViewerState:
    """Case data, optimisation result, and display options shared by all views."""

    def __init__(self):
        self.case = None
        self.result = None
        self.indices = {0: 0, 1: 0, 2: 0}   # current slice per numpy axis (z, y, x)
        self.window = (0.0, 1.0)
        self.structure_colors = {}
        self.selected = None                 # row of the highlighted dwell in dwells()
        self.show_structures = True
        self.show_extended_ptv = True
        self.show_extended_ctv = True
        self.show_candidates = False
        self.show_dwells = True
        self.show_isodose = True
        self.show_colorwash = True
        self.wash = dict(WASH_DEFAULTS)      # min_pct / max_pct of Rx, opacity

    def display_dose(self):
        """(dose in Gy on the native image grid, D_p in Gy), or (None, None) before optimizing."""
        r = self.result
        if r is None:
            return None, None
        return r.get("opt_dose_gy_display", r["opt_dose_gy"]), r["rx_params"]["D_p"]

    def coords(self, axis):
        c = self.case
        w = _WORLD[axis]
        n = c["volume"].shape[axis]
        return c["origin"][w] + np.arange(n) * c["spacing"][w]

    def world_to_index(self, axis, value):
        c = self.case
        w = _WORLD[axis]
        idx = int(round((value - c["origin"][w]) / c["spacing"][w]))
        return int(np.clip(idx, 0, c["volume"].shape[axis] - 1))

    def dwells(self):
        """Displayed dwell set: RTPLAN dwells after loading, active IPSA dwells after optimizing.

        Returns (positions (N, 3), times (N,), channels (N,), regions list or None).
        """
        r = self.result
        if r is not None:
            t = r["optimized_dwell_times"]
            ix = np.flatnonzero(t > 0)
            regions = []
            for i in ix:
                lab = r["mesh_labels"][i]
                region = lab.get("region", "")
                if lab.get("in_extended_ptv"):
                    region += f" | {EXTENDED_PTV_NAME}"
                if lab.get("in_extended_ctv"):
                    region += f" | {EXTENDED_CTV_NAME}"
                regions.append(region)
            return r["mesh_positions"][ix], t[ix], r["mesh_channels"][ix], regions
        c = self.case
        return c["plan_dwell_positions"], c["plan_dwell_times"], c["plan_dwell_channels"], None

    def extended_ptv(self):
        """(mask, margin_mm) of the Extended PTV: the optimisation's if run, else the load-time one."""
        if self.result is not None:
            return self.result["extended_ptv_mask"], self.result["ptv_margin_mm"]
        if self.case is not None:
            return self.case["extended_ptv_mask"], self.case["extended_ptv_margin_mm"]
        return None, None

    def extended_ctv(self):
        """(mask, margin_mm) of the Extended CTV: the optimisation's if run, else the load-time one."""
        if self.result is not None:
            return self.result["extended_ctv_mask"], self.result["ctv_margin_mm"]
        if self.case is not None:
            return self.case["extended_ctv_mask"], self.case["extended_ctv_margin_mm"]
        return None, None

    def dwell_source(self):
        return "Optimized (IPSA)" if self.result is not None else "Original RTPLAN"


# ----------------------------------------------------------------------
# 2-D slice views
# ----------------------------------------------------------------------

_ORIENTATIONS = {
    "Axial":    {"axis": 0, "row_axis": 1, "col_axis": 2, "origin": "upper"},
    "Sagittal": {"axis": 2, "row_axis": 0, "col_axis": 1, "origin": "lower"},
    "Coronal":  {"axis": 1, "row_axis": 0, "col_axis": 2, "origin": "lower"},
}


class SliceView(QFrame):
    """Matplotlib slice viewer with a reference-style header label.

    Mouse wheel scrolls slices; left click moves the crosshair (and the
    other two views) to the clicked point.
    """

    def __init__(self, name, state, on_crosshair, parent=None):
        super().__init__(parent)
        self.name = name
        self.state = state
        self._on_crosshair = on_crosshair
        o = _ORIENTATIONS[name]
        self.axis = o["axis"]
        self.row_axis = o["row_axis"]
        self.col_axis = o["col_axis"]
        self.origin = o["origin"]

        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel(name)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(_HEADER_STYLE)
        layout.addWidget(self.label)

        self.figure = Figure(facecolor="black")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background-color: black;")
        layout.addWidget(self.canvas, stretch=1)

        self.ax = self.figure.add_axes([0.0, 0.0, 1.0, 1.0])
        self.cax = None
        self.dose_cax = None
        if name == "Axial":
            self.cax = self.figure.add_axes([0.93, 0.15, 0.018, 0.7])
            self.cax.set_visible(False)
            self.dose_cax = self.figure.add_axes([0.25, 0.085, 0.5, 0.02])
            self.dose_cax.set_visible(False)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.redraw()

    # ------------------------------------------------------------------
    def _extent(self):
        s = self.state
        rows = s.coords(self.row_axis)
        cols = s.coords(self.col_axis)
        hr = 0.5 * s.case["spacing"][_WORLD[self.row_axis]]
        hc = 0.5 * s.case["spacing"][_WORLD[self.col_axis]]
        left, right = cols[0] - hc, cols[-1] + hc
        if self.origin == "upper":
            bottom, top = rows[-1] + hr, rows[0] - hr
        else:
            bottom, top = rows[0] - hr, rows[-1] + hr
        return [left, right, bottom, top]

    def _slice(self, arr):
        return np.take(arr, self.state.indices[self.axis], axis=self.axis)

    def _in_slice(self, positions):
        s = self.state
        w = _WORLD[self.axis]
        z = s.coords(self.axis)[s.indices[self.axis]]
        half = 0.5 * s.case["spacing"][w]
        return np.abs(positions[:, w] - z) <= half + 1e-6

    def _plot_xy(self, positions):
        return positions[:, _WORLD[self.col_axis]], positions[:, _WORLD[self.row_axis]]

    # ------------------------------------------------------------------
    def redraw(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("black")
        ax.set_axis_off()
        s = self.state

        if self.dose_cax is not None:
            self.dose_cax.set_visible(False)
        if s.case is None:
            if self.cax is not None:
                self.cax.set_visible(False)
            ax.text(0.5, 0.5, "No image loaded\nFile > Import DICOM Folder...",
                    color="0.6", ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            self.label.setText(self.name)
            self.canvas.draw_idle()
            return

        case = s.case
        extent = self._extent()
        idx = s.indices[self.axis]
        n = case["volume"].shape[self.axis]
        ax.imshow(self._slice(case["volume"]), cmap="gray",
                  vmin=s.window[0], vmax=s.window[1],
                  origin=self.origin, extent=extent, interpolation="nearest")

        legend = []
        dose, D_p = s.display_dose()
        dose2 = self._slice(dose) if dose is not None else None

        if s.show_colorwash and dose2 is not None:
            w = s.wash
            lo_cgy = w["min_pct"] / 100.0 * D_p * 100.0
            hi_cgy = max(w["max_pct"] / 100.0 * D_p * 100.0, lo_cgy + 1e-6)
            dose_cgy = dose2 * 100.0
            # Below the threshold is transparent; above the max shows the top color.
            ax.imshow(np.ma.masked_less(dose_cgy, lo_cgy), cmap=_WASH_CMAP,
                      vmin=lo_cgy, vmax=hi_cgy, alpha=w["opacity"],
                      origin=self.origin, extent=extent, interpolation="bilinear")
            if self.dose_cax is not None:
                self.dose_cax.set_visible(True)
                self.dose_cax.clear()
                cb = self.figure.colorbar(
                    cm.ScalarMappable(norm=Normalize(lo_cgy, hi_cgy), cmap=_WASH_CMAP),
                    cax=self.dose_cax, orientation="horizontal",
                )
                cb.set_label(f"Dose (cGy)   [{w['min_pct']:g}%-{w['max_pct']:g}% of Rx "
                             f"{D_p * 100:g} cGy]", color="white", fontsize=7)
                cb.ax.tick_params(colors="white", labelsize=7)

        if s.show_structures:
            for name, color in s.structure_colors.items():
                m2 = self._slice(case["masks"][name])
                if m2.any() and not m2.all():
                    ax.contour(m2.astype(np.uint8), levels=[0.5], colors=[color],
                               linewidths=1.2, origin=self.origin, extent=extent)
                legend.append(Line2D([], [], color=color, lw=1.5, label=name))

        ext_mask, ext_margin = s.extended_ptv()
        if s.show_extended_ptv and ext_mask is not None:
            m2 = self._slice(ext_mask)
            if m2.any() and not m2.all():
                ax.contour(m2.astype(np.uint8), levels=[0.5], colors=[_EXTENDED_PTV_COLOR],
                           linewidths=1.2, linestyles=":", origin=self.origin, extent=extent)
            legend.append(Line2D([], [], color=_EXTENDED_PTV_COLOR, lw=1.5, ls=":",
                                 label=f"{EXTENDED_PTV_NAME} (+{ext_margin:g} mm)"))

        ext_ctv_mask, ext_ctv_margin = s.extended_ctv()
        if s.show_extended_ctv and ext_ctv_mask is not None:
            m2 = self._slice(ext_ctv_mask)
            if m2.any() and not m2.all():
                ax.contour(m2.astype(np.uint8), levels=[0.5], colors=[_EXTENDED_CTV_COLOR],
                           linewidths=1.2, linestyles=":", origin=self.origin, extent=extent)
            legend.append(Line2D([], [], color=_EXTENDED_CTV_COLOR, lw=1.5, ls=":",
                                 label=f"{EXTENDED_CTV_NAME} (+{ext_ctv_margin:g} mm)"))

        if s.show_isodose and dose2 is not None:
            for frac, color in _ISODOSE_LEVELS:
                level = frac * D_p
                if dose2.max() > level > dose2.min():
                    ax.contour(dose2, levels=[level], colors=[color], linewidths=1.0,
                               linestyles="--", origin=self.origin, extent=extent)
                    legend.append(Line2D([], [], color=color, lw=1.0, ls="--",
                                         label=f"{frac*100:.0f}% Rx"))

        if s.show_candidates and s.result is not None:
            mesh_pos = s.result["mesh_positions"]
            sel = np.flatnonzero(self._in_slice(mesh_pos))
            if len(sel):
                colors = [_region_color(s.result["mesh_labels"][i]) for i in sel]
                x, y = self._plot_xy(mesh_pos[sel])
                ax.scatter(x, y, s=6, c=colors, alpha=0.5, linewidths=0)

        pos, times, _channels, _regions = s.dwells()
        if self.cax is not None:
            self.cax.set_visible(False)
        if s.show_dwells and len(pos):
            t_max = float(times.max()) if times.max() > 0 else 1.0
            sel = np.flatnonzero(self._in_slice(pos))
            if len(sel):
                x, y = self._plot_xy(pos[sel])
                ax.scatter(x, y, c=times[sel], cmap=_DWELL_CMAP, vmin=0.0, vmax=t_max,
                           s=18 + 60 * times[sel] / t_max,
                           edgecolors="white", linewidths=0.6, zorder=5)
            if self.cax is not None:
                self.cax.set_visible(True)
                self.cax.clear()
                cb = self.figure.colorbar(
                    cm.ScalarMappable(norm=Normalize(0.0, t_max), cmap=_DWELL_CMAP),
                    cax=self.cax,
                )
                cb.set_label("Dwell time (s)", color="white", fontsize=8)
                cb.ax.tick_params(colors="white", labelsize=7)

        if s.selected is not None and self._in_slice(pos[[s.selected]])[0]:
            x, y = self._plot_xy(pos[[s.selected]])
            ax.scatter(x, y, s=160, facecolors="none", edgecolors="white",
                       linewidths=1.8, zorder=6)

        # crosshair at the other two views' slices
        xc = s.coords(self.col_axis)[s.indices[self.col_axis]]
        yc = s.coords(self.row_axis)[s.indices[self.row_axis]]
        ax.axvline(xc, color="#00ff00", lw=0.5, alpha=0.5)
        ax.axhline(yc, color="#00ff00", lw=0.5, alpha=0.5)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

        if legend and self.name == "Axial":
            leg = ax.legend(handles=legend, loc="upper left", fontsize=7,
                            facecolor="black", edgecolor="0.4", labelcolor="white",
                            framealpha=0.6)
            leg.set_zorder(10)

        z_mm = s.coords(self.axis)[idx]
        self.label.setText(
            f"{self.name}  [{idx}/{n - 1}]   {_AXIS_LETTER[self.axis]} = {z_mm:.1f} mm"
        )
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        s = self.state
        if s.case is None:
            return
        step = 1 if event.button == "up" else -1
        n = s.case["volume"].shape[self.axis]
        s.indices[self.axis] = int(np.clip(s.indices[self.axis] + step, 0, n - 1))
        self._on_crosshair(self)

    def _on_click(self, event):
        s = self.state
        if s.case is None or event.inaxes is not self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        s.indices[self.col_axis] = s.world_to_index(self.col_axis, event.xdata)
        s.indices[self.row_axis] = s.world_to_index(self.row_axis, event.ydata)
        self._on_crosshair(self)


# ----------------------------------------------------------------------
# 3-D view: structure surfaces, needles, and dwells
# ----------------------------------------------------------------------

def _mask_surface_actor(mask, spacing, origin, color, opacity, wireframe=False):
    """Smoothed iso-surface actor of a binary mask (z, y, x) in world mm."""
    nz, ny, nx = mask.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*[float(v) for v in spacing])
    img.SetOrigin(*[float(v) for v in origin])
    arr = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(mask, dtype=np.uint8).ravel(), deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    img.GetPointData().SetScalars(arr)

    surf = vtk.vtkFlyingEdges3D()
    surf.SetInputData(img)
    surf.SetValue(0, 0.5)
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(surf.GetOutputPort())
    smooth.SetNumberOfIterations(15)
    smooth.SetPassBand(0.1)
    smooth.NormalizeCoordinatesOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(smooth.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    c = QColor(color)
    actor.GetProperty().SetColor(c.redF(), c.greenF(), c.blueF())
    actor.GetProperty().SetOpacity(opacity)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
    return actor


class Scene3DView(QFrame):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel("3D View")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(_HEADER_STYLE)
        layout.addWidget(self.label)

        self._structure_actors = []
        self._ext_ptv_actors = []
        self._ext_ctv_actors = []
        self._candidate_actors = []
        self._dwell_actors = []

        if not _HAS_VTK:
            msg = QLabel("VTK is not available -- 3D view disabled.")
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("background-color: black; color: #888;")
            layout.addWidget(msg, stretch=1)
            self.vtk_widget = None
            return

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, stretch=1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

    def initialize(self):
        """Must be called once after the window is shown."""
        if self.vtk_widget is None:
            return
        self.vtk_widget.Initialize()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.vtk_widget.GetRenderWindow().Render()

    def _render(self):
        if self.vtk_widget is not None:
            self.vtk_widget.GetRenderWindow().Render()

    def _remove(self, actors):
        for a in actors:
            self.renderer.RemoveViewProp(a)
        actors.clear()

    @staticmethod
    def _rgb(hex_color):
        c = QColor(hex_color)
        return c.redF(), c.greenF(), c.blueF()

    def _surface_actor(self, mask, color, opacity, wireframe=False):
        case = self.state.case
        return _mask_surface_actor(mask, case["spacing"], case["origin"],
                                   color, opacity, wireframe)

    def _set_extended_ptv(self):
        self._remove(self._ext_ptv_actors)
        mask, _margin = self.state.extended_ptv()
        if mask is None or not mask.any():
            return
        actor = self._surface_actor(mask, _EXTENDED_PTV_COLOR, 0.15, wireframe=True)
        self.renderer.AddActor(actor)
        self._ext_ptv_actors.append(actor)

    def _set_extended_ctv(self):
        self._remove(self._ext_ctv_actors)
        mask, _margin = self.state.extended_ctv()
        if mask is None or not mask.any():
            return
        actor = self._surface_actor(mask, _EXTENDED_CTV_COLOR, 0.15, wireframe=True)
        self.renderer.AddActor(actor)
        self._ext_ctv_actors.append(actor)

    # ------------------------------------------------------------------
    def set_case(self):
        """Rebuild structures and candidate needles for the current case."""
        if self.vtk_widget is None:
            return
        self._remove(self._structure_actors)
        self._remove(self._candidate_actors)
        self._remove(self._dwell_actors)
        case = self.state.case
        if case is None:
            self._render()
            return

        for name, color in self.state.structure_colors.items():
            mask = case["masks"][name]
            if not mask.any():
                continue
            actor = self._surface_actor(mask, color, 0.25)
            self.renderer.AddActor(actor)
            self._structure_actors.append(actor)

        # Catheter lines through each channel's RTPLAN dwell positions.
        pos = case["plan_dwell_positions"]
        channels = case["plan_dwell_channels"]
        points = vtk.vtkPoints()
        for p in pos:
            points.InsertNextPoint(*p)
        lines = vtk.vtkCellArray()
        for ch in np.unique(channels):
            ix = np.flatnonzero(channels == ch)
            if len(ix) < 2:
                continue
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(ix))
            for k, i in enumerate(ix):
                line.GetPointIds().SetId(k, int(i))
            lines.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        needle_actor = vtk.vtkActor()
        needle_actor.SetMapper(mapper)
        needle_actor.GetProperty().SetColor(0.75, 0.75, 0.75)
        needle_actor.GetProperty().SetLineWidth(1.5)
        self.renderer.AddActor(needle_actor)
        self._structure_actors.append(needle_actor)

        self.set_dwells()
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Elevation(-60)
        cam.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self._render()

    def _set_candidates(self):
        self._remove(self._candidate_actors)
        r = self.state.result
        if r is None:
            return
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for p, lab in zip(r["mesh_positions"], r["mesh_labels"]):
            points.InsertNextPoint(*p)
            cr, cg, cb = self._rgb(_region_color(lab))
            colors.InsertNextTuple3(int(cr * 255), int(cg * 255), int(cb * 255))
        cpoly = vtk.vtkPolyData()
        cpoly.SetPoints(points)
        cpoly.GetPointData().SetScalars(colors)
        verts = vtk.vtkVertexGlyphFilter()
        verts.SetInputData(cpoly)
        cmapper = vtk.vtkPolyDataMapper()
        cmapper.SetInputConnection(verts.GetOutputPort())
        cand_actor = vtk.vtkActor()
        cand_actor.SetMapper(cmapper)
        cand_actor.GetProperty().SetPointSize(3)
        self.renderer.AddActor(cand_actor)
        self._candidate_actors.append(cand_actor)

    def set_dwells(self):
        """Show the displayed dwell set as spheres coloured by dwell time."""
        if self.vtk_widget is None:
            return
        self._remove(self._dwell_actors)
        self._set_candidates()
        self._set_extended_ptv()
        self._set_extended_ctv()
        s = self.state
        if s.case is None:
            self.update_visibility()
            return
        positions, times, _channels, _regions = s.dwells()
        t_max = float(times.max()) if len(times) and times.max() > 0 else 1.0

        points = vtk.vtkPoints()
        scalars = vtk.vtkDoubleArray()
        scalars.SetName("dwell_time")
        for p, t in zip(positions, times):
            points.InsertNextPoint(*p)
            scalars.InsertNextValue(float(t))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.GetPointData().SetScalars(scalars)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.2)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(12)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetColorModeToColorByScalar()

        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.SetTableRange(0.0, t_max)
        lut.Build()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0.0, t_max)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)

        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle("Dwell time (s)")
        bar.SetNumberOfLabels(5)
        bar.SetWidth(0.08)
        bar.SetHeight(0.6)
        bar.SetPosition(0.9, 0.2)
        bar.GetTitleTextProperty().SetFontSize(10)
        bar.GetLabelTextProperty().SetFontSize(9)
        self.renderer.AddViewProp(bar)

        self._dwell_actors.extend([actor, bar])
        self.update_visibility()

    def update_visibility(self):
        if self.vtk_widget is None:
            return
        s = self.state
        for a in self._structure_actors:
            a.SetVisibility(s.show_structures)
        for a in self._ext_ptv_actors:
            a.SetVisibility(s.show_extended_ptv)
        for a in self._ext_ctv_actors:
            a.SetVisibility(s.show_extended_ctv)
        for a in self._candidate_actors:
            a.SetVisibility(s.show_candidates)
        for a in self._dwell_actors:
            a.SetVisibility(s.show_dwells)
        self._render()


# ----------------------------------------------------------------------
# Debug window: candidate (blue) and activated (red) dwell positions in 3-D
# ----------------------------------------------------------------------

_CANDIDATE_COLOR = "#2f6fff"   # blue
_ACTIVE_COLOR = "#ff1a1a"      # red
_STRUCT_POINTS_MAX = 2500      # surface points drawn per structure (subsampled)


class CandidateDwellWindow(QWidget):
    """Separate 3-D window (matplotlib, no VTK needed) for debugging the optimizer.

    Every candidate dwell position is a blue dot; after an optimization the
    activated positions (dwell time > 0) are drawn as larger red dots.
    Structure surfaces (sparse points) and needles are shown faintly for
    orientation. Drag with the left mouse button to rotate, right button to zoom.
    """

    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Candidate Dwell Positions (3D)")
        self.resize(900, 780)
        self._case = None
        self._structure_points = []     # [(name, color, (M, 3) world points)]
        self._needles = []              # [((3,), (3,))]
        self._positions = np.zeros((0, 3))
        self._active = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self._show_cand = QCheckBox("Candidates (blue)")
        self._show_active = QCheckBox("Activated (red)")
        self._show_struct = QCheckBox("Structures / needles")
        for cb in (self._show_cand, self._show_active, self._show_struct):
            cb.setChecked(True)
            cb.toggled.connect(self._update_visibility)
            bar.addWidget(cb)
        bar.addStretch(1)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Show the last optimization's dwells, or compute the "
                           "candidates from the current settings if not optimized yet.")
        refresh.clicked.connect(self.refresh_requested.emit)
        bar.addWidget(refresh)
        reset = QPushButton("Reset View")
        reset.clicked.connect(self._reset_camera)
        bar.addWidget(reset)
        layout.addLayout(bar)

        self._info = QLabel("No candidates.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        self.figure = Figure(facecolor="#14141e")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self._view = None              # (elev, azim) kept across redraws
        self._redraw()

    # -- data ---------------------------------------------------------
    @staticmethod
    def _surface_points(mask, spacing, origin, max_points, rng):
        """Sparse world-coordinate points on the boundary of a (z, y, x) mask."""
        from scipy.ndimage import binary_erosion
        shell = mask & ~binary_erosion(mask)
        idx = np.argwhere(shell)                      # (M, 3) as (z, y, x)
        if len(idx) > max_points:
            idx = idx[rng.choice(len(idx), max_points, replace=False)]
        xyz = idx[:, ::-1].astype(np.float64)         # -> (x, y, z) voxel index
        return np.asarray(origin, float) + xyz * np.asarray(spacing, float)

    # -- public -------------------------------------------------------
    def set_case(self, case, structure_colors):
        """New case: prepare structures and needles; drop any old candidates."""
        self._case = case
        self._positions = np.zeros((0, 3))
        self._active = None
        self._view = None
        rng = np.random.default_rng(0)
        self._structure_points = []
        for name, color in structure_colors.items():
            mask = case["masks"][name]
            if mask.any():
                pts = self._surface_points(mask, case["spacing"], case["origin"],
                                           _STRUCT_POINTS_MAX, rng)
                self._structure_points.append((name, color, pts))
        self._needles = []
        pos, channels = case["plan_dwell_positions"], case["plan_dwell_channels"]
        for ch in np.unique(channels):
            ix = np.flatnonzero(channels == ch)
            if len(ix) >= 2:
                self._needles.append((pos[ix[0]], pos[ix[-1]]))
        self._info.setText("No candidates computed yet -- click Refresh.")
        self._redraw()

    def set_dwells(self, positions, active=None, source=""):
        """positions (N, 3) candidates; active (N,) bool or None (not optimized)."""
        self._positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self._active = None if active is None else np.asarray(active, dtype=bool)
        n = len(self._positions)
        if self._active is None:
            text = (f"<b>{n}</b> candidate dwell positions (blue). "
                    f"Not optimized yet -- run the optimization to see activated positions.")
        else:
            k = int(self._active.sum())
            text = (f"<b>{n}</b> candidate dwell positions (blue), "
                    f"<b><span style='color:#e00'>{k}</span></b> activated after optimization "
                    f"(red, dwell time &gt; 0), {n - k} unused.")
        if source:
            text += f"<br>{source}"
        self._info.setText(text)
        self._redraw()

    # -- drawing ------------------------------------------------------
    def _update_visibility(self, *_):
        self._redraw()

    def _reset_camera(self):
        self._view = None
        self._redraw(keep_view=False)

    def _redraw(self, keep_view=True):
        ax = self.ax
        if keep_view and self._view is not None:
            elev, azim = ax.elev, ax.azim
        else:
            elev, azim = 20.0, -60.0
        ax.clear()
        ax.set_facecolor("#14141e")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0.1, 0.1, 0.15, 1.0))
            pane.label.set_color("white")
        ax.tick_params(colors="0.75", labelsize=7)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

        extent_pts = []
        handles = []
        if self._show_struct.isChecked():
            for name, color, pts in self._structure_points:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=color,
                           alpha=0.25, depthshade=False, linewidths=0)
                handles.append(Line2D([], [], ls="", marker="o", ms=4, color=color,
                                      alpha=0.6, label=name))
                extent_pts.append(pts)
            for p0, p1 in self._needles:
                ax.plot(*zip(p0, p1), color="0.6", lw=0.8, alpha=0.6)
                extent_pts.append(np.vstack([p0, p1]))

        pos = self._positions
        if len(pos):
            extent_pts.append(pos)
        if len(pos) and self._show_cand.isChecked():
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=6, c=_CANDIDATE_COLOR,
                       depthshade=False, linewidths=0)
            handles.append(Line2D([], [], ls="", marker="o", ms=5, color=_CANDIDATE_COLOR,
                                  label=f"Candidates ({len(pos)})"))
        if (self._active is not None and self._active.any()
                and self._show_active.isChecked()):
            act = pos[self._active]
            ax.scatter(act[:, 0], act[:, 1], act[:, 2], s=26, c=_ACTIVE_COLOR,
                       edgecolors="white", linewidths=0.3, depthshade=False)
            handles.append(Line2D([], [], ls="", marker="o", ms=7, color=_ACTIVE_COLOR,
                                  label=f"Activated ({len(act)})"))

        if extent_pts:
            allp = np.vstack(extent_pts)
            lo, hi = allp.min(axis=0), allp.max(axis=0)
            ctr, half = (lo + hi) / 2.0, max(float((hi - lo).max()) / 2.0, 1.0) * 1.05
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_zlim(ctr[2] - half, ctr[2] + half)
            ax.set_box_aspect((1, 1, 1))        # equal mm scale on all axes
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=8, facecolor="black",
                      edgecolor="0.4", labelcolor="white", framealpha=0.6)
        ax.view_init(elev=elev, azim=azim)
        self._view = (elev, azim)
        self.canvas.draw_idle()


# ----------------------------------------------------------------------
# Optimization settings dialog
# ----------------------------------------------------------------------

_ROLE_TARGET = "Target"
_ROLE_OAR = "OAR"


class ObjectivesEditor(QGroupBox):
    """VOI table: pick targets / OARs from the structure set and set their goals.

    One row per selected structure. Targets have a minimum-dose (under-dose)
    and a maximum-dose (over-dose) goal; OARs have a maximum-dose goal. Each
    goal has an absolute dose limit in cGy and an impact factor (penalty
    weight). Dose limits are independent of the prescription dose: changing
    the prescription never changes them. Unchecked rows are ignored.
    """

    COLUMNS = ("Name", "Type", "Class",
               "Min dose [cGy]", "Min imp. factor",
               "Max dose [cGy]", "Max imp. factor")
    (C_NAME, C_TYPE, C_CLASS, C_MIN_CGY, C_MIN_W, C_MAX_CGY, C_MAX_W) = range(7)

    def __init__(self, structures, objectives, rx_cgy, params=None, parent=None):
        super().__init__("Objective Goals -- VOI Settings", parent)
        self._structures = list(structures)          # [(name, classification)]
        self._classes = dict(self._structures)
        self._params = params
        self._rx_cgy = float(rx_cgy)
        self._rows = []                              # widgets per table row

        v = QVBoxLayout(self)
        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.C_NAME, QHeaderView.ResizeToContents)
        hdr.setMinimumSectionSize(90)
        self._table.setMinimumHeight(220)
        self._table.setStyleSheet(
            "QTableWidget::item:selected { background-color: #0a3d91; color: white; }"
        )
        v.addWidget(self._table, stretch=1)

        h = QHBoxLayout()
        h.addWidget(QLabel("Available structures:"))
        self._add_combo = QComboBox()
        self._add_combo.setMinimumWidth(220)
        self._add_combo.currentIndexChanged.connect(self._suggest_role)
        h.addWidget(self._add_combo)
        h.addWidget(QLabel("as"))
        self._add_role = QComboBox()
        self._add_role.addItems([_ROLE_TARGET, _ROLE_OAR])
        h.addWidget(self._add_role)
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._on_add)
        h.addWidget(self._add_btn)
        h.addStretch(1)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._on_remove)
        h.addWidget(remove)
        v.addLayout(h)

        hint = QLabel(
            "Check a row to include it in the optimization. Dose limits are absolute "
            "(cGy) and independent of the prescription dose; the minimum-dose goal "
            "applies to targets only. An impact factor of 0 switches that goal off."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        v.addWidget(hint)

        self.set_objectives(objectives)

    # -- rows ---------------------------------------------------------
    @staticmethod
    def _spin(maximum, decimals, step):
        s = QDoubleSpinBox()
        s.setRange(0.0, maximum)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setFrame(False)
        s.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return s

    def _append_row(self, obj):
        name = obj["name"]
        r = self._table.rowCount()
        self._table.insertRow(r)

        name_item = QTableWidgetItem(name)
        name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        name_item.setCheckState(Qt.Checked if obj.get("enabled", True) else Qt.Unchecked)
        self._table.setItem(r, self.C_NAME, name_item)

        cls_item = QTableWidgetItem(self._classes.get(name, ""))
        cls_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        cls_item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(r, self.C_CLASS, cls_item)

        role = QComboBox()
        role.addItems([_ROLE_TARGET, _ROLE_OAR])
        role.setCurrentText(obj.get("role", _ROLE_OAR))

        w = {
            "name": name, "item": name_item, "role": role,
            "min_cgy": self._spin(1e5, 2, 10.0),
            "min_w": self._spin(1e4, 3, 1.0),
            "max_cgy": self._spin(1e5, 2, 10.0),
            "max_w": self._spin(1e4, 3, 1.0),
        }

        min_dose = obj.get("min_dose")
        if min_dose is None:
            min_dose = self._rx_cgy / 100.0
        w["min_cgy"].setValue(float(min_dose) * 100.0)
        w["min_w"].setValue(float(obj.get("min_weight") or 0.0))
        max_dose = obj.get("max_dose")
        w["max_cgy"].setValue(float(max_dose if max_dose is not None else 0.0) * 100.0)
        w["max_w"].setValue(float(obj.get("max_weight") or 0.0))

        self._table.setCellWidget(r, self.C_TYPE, role)
        for col, key in ((self.C_MIN_CGY, "min_cgy"), (self.C_MIN_W, "min_w"),
                         (self.C_MAX_CGY, "max_cgy"), (self.C_MAX_W, "max_w")):
            self._table.setCellWidget(r, col, w[key])

        role.currentTextChanged.connect(lambda _text, w=w: self._apply_role(w, changed=True))
        self._rows.append(w)
        self._apply_role(w)

    def _apply_role(self, w, changed=False):
        is_target = w["role"].currentText() == _ROLE_TARGET
        for key in ("min_cgy", "min_w"):
            w[key].setEnabled(is_target)
        if changed and is_target and w["min_w"].value() == 0.0:
            d = target_objective(w["name"], self._params)
            w["min_cgy"].setValue(self._rx_cgy)
            w["min_w"].setValue(d["min_weight"])

    def _refresh_add_combo(self):
        used = {w["name"] for w in self._rows}
        self._add_combo.blockSignals(True)
        self._add_combo.clear()
        for name, cls in self._structures:
            if name not in used:
                self._add_combo.addItem(f"{name}  ({cls})", name)
        available = self._add_combo.count() > 0
        if not available:
            self._add_combo.addItem("(all structures already in the table)", None)
        self._add_combo.blockSignals(False)
        self._add_combo.setEnabled(available)
        self._add_btn.setEnabled(available)
        self._suggest_role()

    def _suggest_role(self, *_):
        name = self._add_combo.currentData()
        if name is None:
            return
        cls = self._classes.get(name, "")
        self._add_role.setCurrentText(_ROLE_TARGET if cls in ("CTV", "PTV") else _ROLE_OAR)

    def _on_add(self):
        name = self._add_combo.currentData()
        if name is None:
            return
        if self._add_role.currentText() == _ROLE_TARGET:
            obj = target_objective(name, self._params)
            obj["min_dose"] = self._rx_cgy / 100.0
        else:
            obj = oar_objective(name, self._params)
        self._append_row(obj)
        self._refresh_add_combo()
        self._table.selectRow(self._table.rowCount() - 1)

    def _on_remove(self):
        r = self._table.currentRow()
        if r < 0:
            return
        self._table.removeRow(r)
        del self._rows[r]
        self._refresh_add_combo()

    # -- public -------------------------------------------------------
    def set_objectives(self, objectives):
        self._table.setRowCount(0)
        self._rows = []
        for obj in objectives or []:
            if obj["name"] in self._classes:
                self._append_row(obj)
        self._refresh_add_combo()

    def set_default_min_dose(self, rx_cgy):
        """Suggested min dose for targets added later; existing rows are untouched."""
        self._rx_cgy = float(rx_cgy)

    def objectives(self):
        out = []
        for w in self._rows:
            is_target = w["role"].currentText() == _ROLE_TARGET
            out.append({
                "name": w["name"],
                "role": w["role"].currentText(),
                "enabled": w["item"].checkState() == Qt.Checked,
                "min_dose": w["min_cgy"].value() / 100.0 if is_target else None,
                "min_weight": w["min_w"].value() if is_target else 0.0,
                "max_dose": w["max_cgy"].value() / 100.0,
                "max_weight": w["max_w"].value(),
            })
        return out

    def validate(self):
        objs = [o for o in self.objectives() if o["enabled"]]
        if not objs:
            return "Select (check) at least one target or OAR objective."
        for o in objs:
            if (o["min_dose"] is not None and o["min_weight"] > 0 and o["max_weight"] > 0
                    and o["min_dose"] > o["max_dose"]):
                return (f"{o['name']}: the minimum dose ({o['min_dose'] * 100:.2f} cGy) "
                        f"is above the maximum dose ({o['max_dose'] * 100:.2f} cGy).")
        return None


class OptimizationSettingsDialog(QDialog):
    """Objective goals (per structure), candidate dwells, and IPSA settings."""

    def __init__(self, params, ipsa_params, mesh_params, structures=(),
                 ctv_names=(), oar_names=(), has_ptv=True, has_ctv=True,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Settings")
        self._base_params = copy_prescription_params(params)
        self._ctv_names = list(ctv_names)
        self._oar_names = list(oar_names)

        layout = QVBoxLayout(self)

        rx_row = QHBoxLayout()
        rx_row.addWidget(QLabel("Treatment type:"))
        self._protocol = QComboBox()
        for key, info in PROTOCOLS.items():
            self._protocol.addItem(f"{info['label']} ({info['D_p'] * 100:.0f} cGy)", key)
        self._protocol.setToolTip(
            "Selects the plan-evaluation criteria and sets the default prescription "
            "dose of that treatment type. The prescription can still be edited.")
        rx_row.addWidget(self._protocol)
        rx_row.addSpacing(16)
        rx_row.addWidget(QLabel("Prescription dose:"))
        self._rx = QDoubleSpinBox()
        self._rx.setRange(1.0, 1e5)
        self._rx.setDecimals(2)
        self._rx.setSingleStep(50.0)
        self._rx.setSuffix(" cGy")
        self._rx.setValue(float(params["D_p"]) * 100.0)
        rx_row.addWidget(self._rx)
        rx_row.addStretch(1)
        layout.addLayout(rx_row)
        self._set_protocol(params.get("protocol", DEFAULT_PROTOCOL))
        # Connected after the initial value, so a saved prescription is kept.
        self._protocol.currentIndexChanged.connect(self._on_protocol_changed)

        objectives = params.get("objectives")
        if objectives is None:
            objectives = default_objectives(self._ctv_names, self._oar_names, params)
        self._objectives = ObjectivesEditor(
            structures, objectives, self._rx.value(), params=self._base_params,
        )
        self._rx.valueChanged.connect(self._objectives.set_default_min_dose)
        layout.addWidget(self._objectives, stretch=1)

        bottom = QHBoxLayout()
        layout.addLayout(bottom)

        box = QGroupBox("Candidate dwell positions")
        v = QVBoxLayout(box)
        self._all_radio = QRadioButton("All mesh points in the search area")
        self._ext_radio = QRadioButton(f"Only mesh points inside the {EXTENDED_PTV_NAME}")
        self._ext_ctv_radio = QRadioButton(f"Only mesh points inside the {EXTENDED_CTV_NAME}")
        v.addWidget(self._all_radio)
        v.addWidget(self._ext_radio)
        v.addWidget(self._ext_ctv_radio)
        form = QFormLayout()
        self._margin = QDoubleSpinBox()
        self._margin.setRange(0.0, 50.0)
        self._margin.setDecimals(1)
        self._margin.setSingleStep(0.5)
        self._margin.setSuffix(" mm")
        self._margin.setValue(float(mesh_params.get("ptv_margin_mm", EXTENDED_PTV_MARGIN_MM)))
        form.addRow(f"{EXTENDED_PTV_NAME} margin beyond PTV", self._margin)
        self._ctv_margin = QDoubleSpinBox()
        self._ctv_margin.setRange(0.0, 50.0)
        self._ctv_margin.setDecimals(1)
        self._ctv_margin.setSingleStep(0.5)
        self._ctv_margin.setSuffix(" mm")
        self._ctv_margin.setValue(float(mesh_params.get("ctv_margin_mm", EXTENDED_CTV_MARGIN_MM)))
        form.addRow(f"{EXTENDED_CTV_NAME} margin beyond CTV", self._ctv_margin)
        v.addLayout(form)
        region = mesh_params.get("candidate_region", CANDIDATES_ALL)
        want_ptv = region == CANDIDATES_EXTENDED_PTV and has_ptv
        want_ctv = region == CANDIDATES_EXTENDED_CTV and has_ctv
        self._ext_radio.setChecked(want_ptv)
        self._ext_ctv_radio.setChecked(want_ctv)
        self._all_radio.setChecked(not want_ptv and not want_ctv)
        if not has_ptv:
            self._ext_radio.setEnabled(False)
            self._ext_radio.setToolTip("No PTV structure in this case.")
        if not has_ctv:
            self._ext_ctv_radio.setEnabled(False)
            self._ext_ctv_radio.setToolTip("No CTV structure in this case.")
        self._ext_radio.toggled.connect(self._update_margin_enabled)
        self._ext_ctv_radio.toggled.connect(self._update_margin_enabled)
        self._update_margin_enabled()
        bottom.addWidget(box, stretch=2)

        box = QGroupBox("IPSA")
        form = QFormLayout(box)

        def ip(key):
            return ipsa_params.get(key, IPSA_DEFAULTS[key])

        self._iters = QSpinBox()
        self._iters.setRange(1, 10_000_000)
        self._iters.setSingleStep(10000)
        self._iters.setToolTip("Safety limit. With automatic termination the run "
                               "normally stops earlier.")
        form.addRow("Max iterations", self._iters)
        self._max_time = QDoubleSpinBox()
        self._max_time.setRange(0.0, 36000.0)
        self._max_time.setDecimals(0)
        self._max_time.setSingleStep(30.0)
        self._max_time.setSuffix(" s")
        self._max_time.setSpecialValueText("off")
        self._max_time.setToolTip("Safety limit on the search time (0 = off).")
        form.addRow("Max time", self._max_time)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2**31 - 1)
        form.addRow("Random seed", self._seed)

        self._adaptive = QCheckBox("Automatic termination")
        self._adaptive.setToolTip(
            "Stop when the search has cooled, the best loss has stopped improving "
            "and the clinical metrics are stable. Checked once per epoch "
            "(one sweep = one proposal per candidate dwell).")
        form.addRow(self._adaptive)
        self._plateau = QDoubleSpinBox()
        self._plateau.setRange(1e-6, 0.5)
        self._plateau.setDecimals(3)
        self._plateau.setSingleStep(0.05)
        self._plateau.setSuffix(" %")
        self._plateau.setToolTip("Stop when the best loss improved by less than this "
                                 "over the last 'Patience' epochs.")
        form.addRow("Loss plateau tol.", self._plateau)
        self._patience = QSpinBox()
        self._patience.setRange(1, 100)
        self._patience.setSuffix(" epochs")
        form.addRow("Patience", self._patience)
        self._clinical = QCheckBox("Require stable clinical metrics")
        self._clinical.setToolTip(
            "Also require target V100 / D90 (within 0.2 pp) and OAR D10 / D1 "
            "(within 0.5 pp) to be stable over the patience window.")
        form.addRow(self._clinical)
        self._adaptive.toggled.connect(self._update_stop_enabled)
        self._set_ipsa_values(ip)
        bottom.addWidget(box, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        run = QPushButton("Run")
        run.setDefault(True)
        run.setToolTip("Save these settings and run the optimization.")
        buttons.addButton(run, QDialogButtonBox.AcceptRole)
        reset = QPushButton("Reset Defaults")
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        reset.clicked.connect(self._on_reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(1050, 650)

    def accept(self):
        error = self._objectives.validate()
        if error:
            QMessageBox.warning(self, "Objective goals", error)
            return
        if not self._confirm_prescription():
            return
        super().accept()

    def _set_protocol(self, key):
        i = self._protocol.findData(key)
        self._protocol.setCurrentIndex(i if i >= 0 else self._protocol.findData(DEFAULT_PROTOCOL))

    def _on_protocol_changed(self, *_):
        """New treatment type: load its default prescription (still editable)."""
        info = protocol_info(self._protocol.currentData())
        self._rx.setValue(info["D_p"] * 100.0)

    def _confirm_prescription(self):
        """Ask before running when the prescription is far from the target goals.

        The clinical criteria (V100, D90, ...) are relative to the prescription,
        so a typo such as 15000 instead of 1500 cGy makes every target fail
        even though the objectives (absolute doses) optimise a normal plan.
        """
        rx_cgy = self._rx.value()
        mins = [o["min_dose"] * 100.0 for o in self._objectives.objectives()
                if o["enabled"] and o["role"] == _ROLE_TARGET
                and o["min_dose"] is not None and o["min_weight"] > 0]
        if not mins:
            return True
        lo, hi = min(mins), max(mins)
        if 0.5 * lo <= rx_cgy <= 2.0 * hi:
            return True
        answer = QMessageBox.question(
            self, "Check prescription dose",
            f"The prescription dose is {rx_cgy:.2f} cGy, but the target minimum-dose "
            f"goals are {lo:.2f}"
            + (f"-{hi:.2f}" if hi != lo else "")
            + " cGy.\n\nThe plan evaluation (V100, D90, ...) is relative to the "
            "prescription, so the results would not be meaningful.\n\n"
            "Run anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def params(self):
        p = copy_prescription_params(self._base_params)
        p["D_p"] = self._rx.value() / 100.0
        p["protocol"] = self._protocol.currentData()
        p["objectives"] = self._objectives.objectives()
        return p

    def ipsa_params(self):
        max_time = self._max_time.value()
        return {
            "n_iterations": self._iters.value(),
            "max_time_s": max_time if max_time > 0 else None,
            "seed": self._seed.value(),
            "adaptive_stop": self._adaptive.isChecked(),
            "plateau_tol": self._plateau.value() / 100.0,
            "patience_epochs": self._patience.value(),
            "clinical_plateau": self._clinical.isChecked(),
        }

    def _set_ipsa_values(self, ip):
        self._iters.setValue(int(ip("n_iterations")))
        self._max_time.setValue(float(ip("max_time_s") or 0.0))
        self._seed.setValue(int(ip("seed")))
        self._adaptive.setChecked(bool(ip("adaptive_stop")))
        self._plateau.setValue(100.0 * float(ip("plateau_tol")))
        self._patience.setValue(int(ip("patience_epochs")))
        self._clinical.setChecked(bool(ip("clinical_plateau")))
        self._update_stop_enabled()

    def _update_stop_enabled(self, *_):
        on = self._adaptive.isChecked()
        for w in (self._plateau, self._patience, self._clinical):
            w.setEnabled(on)

    def _update_margin_enabled(self, *_):
        self._margin.setEnabled(self._ext_radio.isChecked())
        self._ctv_margin.setEnabled(self._ext_ctv_radio.isChecked())

    def mesh_params(self):
        if self._ext_radio.isChecked():
            region = CANDIDATES_EXTENDED_PTV
        elif self._ext_ctv_radio.isChecked():
            region = CANDIDATES_EXTENDED_CTV
        else:
            region = CANDIDATES_ALL
        return {
            "candidate_region": region,
            "ptv_margin_mm": self._margin.value(),
            "ctv_margin_mm": self._ctv_margin.value(),
        }

    def _on_reset(self):
        self._base_params = copy_prescription_params()
        self._protocol.blockSignals(True)
        self._set_protocol(PRESCRIPTION_PARAMS.get("protocol", DEFAULT_PROTOCOL))
        self._protocol.blockSignals(False)
        self._rx.setValue(PRESCRIPTION_PARAMS["D_p"] * 100.0)
        self._objectives.set_objectives(
            default_objectives(self._ctv_names, self._oar_names, self._base_params)
        )
        self._set_ipsa_values(IPSA_DEFAULTS.get)
        self._all_radio.setChecked(True)
        self._margin.setValue(EXTENDED_PTV_MARGIN_MM)
        self._ctv_margin.setValue(EXTENDED_CTV_MARGIN_MM)
        self._update_margin_enabled()


# ----------------------------------------------------------------------
# Dose color wash settings
# ----------------------------------------------------------------------

class ColorWashDialog(QDialog):
    """Dose range (% of prescription) and opacity of the dose color wash.

    Emits ``changed`` on every edit so the slice views preview it live.
    """

    changed = pyqtSignal(dict)

    def __init__(self, wash, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dose Color Wash")
        form = QFormLayout(self)

        self._min = QDoubleSpinBox()
        self._min.setRange(0.0, 1000.0)
        self._min.setDecimals(1)
        self._min.setSingleStep(5.0)
        self._min.setSuffix(" % of Rx")
        self._min.setValue(wash["min_pct"])
        self._min.setToolTip("Dose below this level is not colored (transparent).")
        form.addRow("Lower threshold", self._min)

        self._max = QDoubleSpinBox()
        self._max.setRange(1.0, 2000.0)
        self._max.setDecimals(1)
        self._max.setSingleStep(10.0)
        self._max.setSuffix(" % of Rx")
        self._max.setValue(wash["max_pct"])
        self._max.setToolTip("Dose at or above this level gets the top color.")
        form.addRow("Upper limit", self._max)

        self._opacity = QSpinBox()
        self._opacity.setRange(5, 100)
        self._opacity.setSingleStep(5)
        self._opacity.setSuffix(" %")
        self._opacity.setValue(int(round(wash["opacity"] * 100)))
        form.addRow("Opacity", self._opacity)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = QPushButton("Reset Defaults")
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        reset.clicked.connect(self._on_reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        for spin in (self._min, self._max, self._opacity):
            spin.valueChanged.connect(self._emit)

    def wash(self):
        lo, hi = self._min.value(), self._max.value()
        if hi <= lo:
            hi = lo + 1.0
        return {"min_pct": lo, "max_pct": hi, "opacity": self._opacity.value() / 100.0}

    def _emit(self, *_):
        self.changed.emit(self.wash())

    def _on_reset(self):
        self._min.setValue(WASH_DEFAULTS["min_pct"])
        self._max.setValue(WASH_DEFAULTS["max_pct"])
        self._opacity.setValue(int(round(WASH_DEFAULTS["opacity"] * 100)))


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, console=None):
        super().__init__()
        self.setWindowTitle("HDR Brachy TPS -- Dwell Optimization")
        self.resize(1500, 950)

        self.state = ViewerState()
        self._console = console
        self._worker = None
        self._rx_params = copy_prescription_params()
        self._ipsa_params = {}
        self._mesh_params = {
            "candidate_region": CANDIDATES_ALL,
            "ptv_margin_mm": EXTENDED_PTV_MARGIN_MM,
            "ctv_margin_mm": EXTENDED_CTV_MARGIN_MM,
        }
        self._grid_mm = GRID_1MM
        self._cand_window = None

        self._build_views()
        self._build_docks()
        self._build_menu()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(260)
        self._progress.setVisible(False)
        self.status_bar.addPermanentWidget(self._progress)
        self.status_bar.showMessage("Ready -- use File > Import DICOM Folder... to load a study.")

        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._flush_log)
        self._log_timer.start(100)

        self._update_actions()
        QTimer.singleShot(0, self.view_3d.initialize)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        self._import_action = QAction("&Import DICOM Folder...", self)
        self._import_action.setShortcut("Ctrl+O")
        self._import_action.triggered.connect(self._on_import)
        file_menu.addAction(self._import_action)

        self._export_action = QAction("&Export Dwell Times (CSV)...", self)
        self._export_action.setShortcut("Ctrl+S")
        self._export_action.triggered.connect(self._on_export)
        file_menu.addAction(self._export_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        grid_menu = menu_bar.addMenu("Dose / Calculation &Grid")
        grid_group = QActionGroup(self)
        grid_group.setExclusive(True)
        self._grid_1_action = QAction("1 mm × 1 mm × 1 mm", self)
        self._grid_1_action.setCheckable(True)
        self._grid_1_action.setChecked(True)
        self._grid_1_action.setActionGroup(grid_group)
        self._grid_1_action.triggered.connect(lambda: self._set_grid_mm(GRID_1MM))
        grid_menu.addAction(self._grid_1_action)
        self._grid_3_action = QAction("3 mm × 3 mm × 3 mm", self)
        self._grid_3_action.setCheckable(True)
        self._grid_3_action.setActionGroup(grid_group)
        self._grid_3_action.triggered.connect(lambda: self._set_grid_mm(GRID_3MM))
        grid_menu.addAction(self._grid_3_action)

        optim_menu = menu_bar.addMenu("&Optimization")
        self._settings_action = QAction("Optimization &Settings...", self)
        self._settings_action.setShortcut("Ctrl+P")
        self._settings_action.triggered.connect(self._on_settings)
        optim_menu.addAction(self._settings_action)

        self._candidates_action = QAction("Candidate &Dwell Positions (3D)...", self)
        self._candidates_action.setShortcut("Ctrl+D")
        self._candidates_action.setToolTip(
            "Debug view: candidate dwell positions (blue) and, after optimizing, "
            "the activated positions (red)."
        )
        self._candidates_action.triggered.connect(self._on_show_candidates)
        optim_menu.addAction(self._candidates_action)

        view_menu = menu_bar.addMenu("&View")
        self._toggle_actions = {}
        for attr, text in (
            ("show_structures", "Show &Structures"),
            ("show_extended_ptv", f"Show &{EXTENDED_PTV_NAME}"),
            ("show_extended_ctv", f"Show Extended &CTV"),
            ("show_candidates", "Show &Candidate Mesh"),
            ("show_dwells", "Show &Dwells"),
            ("show_isodose", "Show &Isodose Lines"),
            ("show_colorwash", "Show Dose Color &Wash"),
        ):
            act = QAction(text, self)
            act.setCheckable(True)
            act.setChecked(getattr(self.state, attr))
            act.toggled.connect(lambda on, a=attr: self._on_toggle(a, on))
            view_menu.addAction(act)
            self._toggle_actions[attr] = act
        wash_action = QAction("Dose Color Wash Settings...", self)
        wash_action.triggered.connect(self._on_wash_settings)
        view_menu.addAction(wash_action)
        view_menu.addSeparator()
        view_menu.addAction(self.results_dock.toggleViewAction())
        view_menu.addAction(self.log_dock.toggleViewAction())

    # ------------------------------------------------------------------
    # Views (2x2 grid) and docks
    # ------------------------------------------------------------------
    def _build_views(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        self.axial_view = SliceView("Axial", self.state, self._on_crosshair)
        self.sagittal_view = SliceView("Sagittal", self.state, self._on_crosshair)
        self.coronal_view = SliceView("Coronal", self.state, self._on_crosshair)
        self.view_3d = Scene3DView(self.state)
        self.slice_views = [self.axial_view, self.sagittal_view, self.coronal_view]

        grid.addWidget(self.axial_view, 0, 0)
        grid.addWidget(self.sagittal_view, 0, 1)
        grid.addWidget(self.coronal_view, 1, 0)
        grid.addWidget(self.view_3d, 1, 1)
        for i in range(2):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)

    def _build_docks(self):
        # Results: dwell table + clinical evaluation
        self.results_dock = QDockWidget("Dwell Positions / Times", self)
        self.results_dock.setObjectName("results_dock")
        tabs = QTabWidget()

        dwell_tab = QWidget()
        v = QVBoxLayout(dwell_tab)
        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self._summary_label)
        self._dwell_table = QTableWidget(0, 7)
        self._dwell_table.setHorizontalHeaderLabels(
            ["Channel", "Dwell #", "X (mm)", "Y (mm)", "Z (mm)", "Time (s)", "Region"]
        )
        self._dwell_table.verticalHeader().setVisible(False)
        self._dwell_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._dwell_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._dwell_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._dwell_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._dwell_table.itemSelectionChanged.connect(self._on_dwell_selected)
        v.addWidget(self._dwell_table, stretch=1)
        tabs.addTab(dwell_tab, "Dwell Positions / Times")

        self._channel_table = QTableWidget(0, 3)
        self._channel_table.setHorizontalHeaderLabels(["Channel", "Active dwells", "Total time (s)"])
        self._channel_table.verticalHeader().setVisible(False)
        self._channel_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self._channel_table, "Per Channel")

        self._eval_table = QTableWidget(0, 4)
        self._eval_table.setHorizontalHeaderLabels(["Structure", "Criterion", "Status", "Result"])
        self._eval_table.verticalHeader().setVisible(False)
        self._eval_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._eval_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._eval_table.horizontalHeader().setStretchLastSection(True)
        eval_page = QWidget()
        eval_layout = QVBoxLayout(eval_page)
        eval_layout.setContentsMargins(0, 0, 0, 0)
        self._eval_header = QLabel()
        self._eval_header.setContentsMargins(4, 2, 4, 2)
        eval_layout.addWidget(self._eval_header)
        eval_layout.addWidget(self._eval_table)
        tabs.addTab(eval_page, "Plan Evaluation")

        self.results_dock.setWidget(tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self.results_dock)
        self.results_dock.setMinimumWidth(430)
        self.results_dock.setVisible(False)

        # Console log
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("log_dock")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(20000)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(8)
        self._log.setFont(font)
        self.log_dock.setWidget(self._log)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [160], Qt.Vertical)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _busy(self):
        return self._worker is not None

    def _update_actions(self):
        has_case = self.state.case is not None
        busy = self._busy()
        self._import_action.setEnabled(not busy)
        self._settings_action.setEnabled(has_case and not busy)
        self._candidates_action.setEnabled(has_case and not busy)
        self._export_action.setEnabled(has_case and not busy)
        self._grid_1_action.setEnabled(not busy)
        self._grid_3_action.setEnabled(not busy)

    def _start_worker(self, fn, on_success, message, *args, **kwargs):
        self._worker = _Worker(fn, *args, **kwargs)
        self._worker.succeeded.connect(on_success)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._progress.setRange(0, 0)
        self._progress.setVisible(True)
        self.status_bar.showMessage(message)
        self.setCursor(Qt.BusyCursor)
        self._update_actions()
        self._worker.start()

    def _on_worker_finished(self):
        self._flush_log()
        self._worker = None
        self._progress.setVisible(False)
        self.unsetCursor()
        self._update_actions()

    def _on_worker_failed(self, message, tb):
        self.status_bar.showMessage(f"Failed: {message}")
        QMessageBox.critical(self, "Error", f"{message}\n\n{tb}")

    def _flush_log(self):
        if self._console is None:
            return
        text = self._console.take()
        if not text:
            return
        self._log.moveCursor(self._log.textCursor().End)
        self._log.insertPlainText(text)
        self._log.ensureCursorVisible()
        matches = re.findall(r"Iter\s+(\d+)/(\d+)", text)
        if matches and self._busy():
            k, n = (int(v) for v in matches[-1])
            self._progress.setRange(0, n)
            self._progress.setValue(k)

    def _redraw_all(self):
        for view in self.slice_views:
            view.redraw()

    def _on_crosshair(self, _source):
        self._redraw_all()

    def _on_toggle(self, attr, on):
        setattr(self.state, attr, on)
        self._redraw_all()
        self.view_3d.update_visibility()

    def _on_wash_settings(self):
        dlg = ColorWashDialog(self.state.wash, parent=self)
        dlg.changed.connect(self._apply_wash)
        before = dict(self.state.wash)
        if dlg.exec_() != QDialog.Accepted:
            self._apply_wash(before)          # Cancel undoes the live preview

    def _apply_wash(self, wash):
        self.state.wash = dict(wash)
        self._redraw_all()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def _on_import(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder", "", QFileDialog.ShowDirsOnly
        )
        if not folder:
            return
        self._start_worker(load_case, self._on_case_loaded,
                           f"Loading {folder} ...", folder)

    def _set_grid_mm(self, grid_mm):
        self._grid_mm = grid_mm
        self.status_bar.showMessage(
            f"Dose / calculation grid set to {grid_mm:g} mm "
            f"(image display stays native; applies at the next optimization)."
        )

    def _on_case_loaded(self, case):
        s = self.state
        s.case = case
        s.result = None
        s.selected = None
        # Objective goals are per structure set: rebuild defaults for the new case.
        self._rx_params.pop("objectives", None)
        s.structure_colors = _structure_colors(case["structures"], case["masks"])
        sample = case["volume"][::2, ::4, ::4]
        lo, hi = np.percentile(sample, [1.0, 99.5])
        s.window = (float(lo), float(hi) if hi > lo else float(lo) + 1.0)

        # Centre the slices on the RTPLAN dwells (the implant region).
        center = case["plan_dwell_positions"].mean(axis=0)
        for axis in (0, 1, 2):
            s.indices[axis] = s.world_to_index(axis, center[_WORLD[axis]])

        self._fill_results()
        self._redraw_all()
        self.view_3d.set_case()
        if self._cand_window is not None:
            self._cand_window.set_case(case, s.structure_colors)

        basename = os.path.basename(case["folder"].rstrip("/\\"))
        self.setWindowTitle(f"HDR Brachy TPS - {basename}")
        nz, ny, nx = case["volume"].shape
        times = case["plan_dwell_times"]
        self.status_bar.showMessage(
            f"Loaded {basename}  ({case['modality']} {nx}x{ny}x{nz} voxels, "
            f"spacing {tuple(float(v) for v in case['spacing'])} mm, "
            f"RTPLAN: {len(times)} dwells on "
            f"{len(np.unique(case['plan_dwell_channels']))} channels, "
            f"total time {times.sum():.1f} s) "
            f"-- Optimization > Optimization Settings... > Run to optimise dwell times"
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def _on_settings(self):
        case = self.state.case
        structures = [] if case is None else [
            (st["name"], st["classification"]) for st in case["structures"]
            if st["name"] in case["masks"] and case["masks"][st["name"]].any()
        ]
        dlg = OptimizationSettingsDialog(
            self._rx_params, self._ipsa_params, self._mesh_params,
            structures=structures,
            ctv_names=[] if case is None else case["ctv_names"],
            oar_names=[] if case is None else case["oar_names"],
            has_ptv=case is not None and case["extended_ptv_mask"] is not None,
            has_ctv=case is not None and case["extended_ctv_mask"] is not None,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._rx_params = dlg.params()
            self._ipsa_params = dlg.ipsa_params()
            self._mesh_params = dlg.mesh_params()
            self._on_run_optimization()

    # -- candidate dwell debug window ------------------------------------
    @staticmethod
    def _region_text(region, ptv_margin_mm, ctv_margin_mm):
        if region == CANDIDATES_EXTENDED_PTV:
            return f"{EXTENDED_PTV_NAME} only (PTV + {ptv_margin_mm:g} mm)"
        if region == CANDIDATES_EXTENDED_CTV:
            return f"{EXTENDED_CTV_NAME} only (CTV + {ctv_margin_mm:g} mm)"
        return "all mesh points in the search area"

    def _on_show_candidates(self):
        case = self.state.case
        if case is None:
            return
        if self._cand_window is None:
            self._cand_window = CandidateDwellWindow(self)
            self._cand_window.refresh_requested.connect(self._on_show_candidates)
            self._cand_window.set_case(case, self.state.structure_colors)
        self._cand_window.show()
        self._cand_window.raise_()
        self._cand_window.activateWindow()
        if self.state.result is not None:
            self._show_result_candidates()
        elif not self._busy():
            # Not optimized yet: mesh the needles with the current settings (no dose).
            self._start_worker(build_candidate_mesh, self._on_candidates_built,
                               "Computing candidate dwell positions ...",
                               case, **self._mesh_params)

    def _on_candidates_built(self, cand):
        if self._cand_window is None:
            return
        region = self._region_text(cand["candidate_region"], cand["ptv_margin_mm"],
                                   cand["ctv_margin_mm"])
        self._cand_window.set_dwells(
            cand["positions"], None,
            source=f"Candidate region (current settings): {region}",
        )
        self.status_bar.showMessage(f"{len(cand['positions'])} candidate dwell positions.")

    def _show_result_candidates(self):
        r = self.state.result
        region = self._region_text(r["candidate_region"], r["ptv_margin_mm"], r["ctv_margin_mm"])
        self._cand_window.set_dwells(
            r["mesh_positions"], r["optimized_dwell_times"] > 0,
            source=f"From the last optimization -- candidate region: {region}",
        )

    def _on_run_optimization(self):
        if self.state.case is None:
            return
        self._start_worker(
            optimize_case, self._on_optimized, "Running IPSA optimization ...",
            self.state.case, rx_params=self._rx_params, ipsa_params=self._ipsa_params,
            mesh_params={**self._mesh_params, "grid_mm": self._grid_mm},
        )

    def _on_optimized(self, result):
        s = self.state
        s.result = result
        s.selected = None
        self._fill_results()
        self._redraw_all()
        self.view_3d.set_dwells()
        if self._cand_window is not None and self._cand_window.isVisible():
            self._show_result_candidates()

        n_active = int(np.sum(result["optimized_dwell_times"] > 0))
        total = float(result["optimized_dwell_times"].sum())
        self.status_bar.showMessage(
            f"Optimization done: {n_active} active dwells, total time {total:.1f} s, "
            f"loss reduction {result['loss_reduction_pct']:.1f}%, "
            f"clinical criteria {result['n_checks'] - result['n_fail']}/{result['n_checks']} met"
        )

    # ------------------------------------------------------------------
    # Results panel
    # ------------------------------------------------------------------
    @staticmethod
    def _item(value, fmt=None, align=Qt.AlignRight | Qt.AlignVCenter):
        item = QTableWidgetItem(fmt.format(value) if fmt else str(value))
        item.setTextAlignment(align)
        return item

    @staticmethod
    def _dwell_numbers(channels):
        """1-based dwell number of each dwell along its channel."""
        counts = {}
        numbers = []
        for ch in channels:
            counts[ch] = counts.get(ch, 0) + 1
            numbers.append(counts[ch])
        return numbers

    def _fill_results(self):
        """Fill the results panel with the displayed dwell set (RTPLAN or optimized)."""
        s = self.state
        result = s.result
        pos, times, channels, regions = s.dwells()
        numbers = self._dwell_numbers(channels)
        self.results_dock.setWindowTitle(f"Dwell Positions / Times -- {s.dwell_source()}")

        self._dwell_table.clearSelection()
        self._dwell_table.setRowCount(len(pos))
        for row in range(len(pos)):
            cells = [
                self._item(int(channels[row])),
                self._item(numbers[row]),
                self._item(pos[row, 0], "{:.2f}"),
                self._item(pos[row, 1], "{:.2f}"),
                self._item(pos[row, 2], "{:.2f}"),
                self._item(times[row], "{:.3f}"),
                self._item(regions[row] if regions else "",
                           align=Qt.AlignLeft | Qt.AlignVCenter),
            ]
            for col, cell in enumerate(cells):
                self._dwell_table.setItem(row, col, cell)
        self._dwell_table.setColumnHidden(6, not regions)

        # Per-channel totals over all positions of the plan / candidate mesh.
        if result is not None:
            all_channels, all_times = result["mesh_channels"], result["optimized_dwell_times"]
        else:
            all_channels, all_times = channels, times
        uniq = sorted(np.unique(all_channels).tolist())
        self._channel_table.setRowCount(len(uniq))
        for row, ch in enumerate(uniq):
            m = all_channels == ch
            self._channel_table.setItem(row, 0, self._item(int(ch)))
            self._channel_table.setItem(row, 1, self._item(f"{int(np.sum(all_times[m] > 0))} / {int(m.sum())}"))
            self._channel_table.setItem(row, 2, self._item(float(all_times[m].sum()), "{:.2f}"))

        active = times[times > 0]
        time_text = (
            f"<b>Dwell time:</b> min {active.min():.3f} s, mean {active.mean():.3f} s, "
            f"max {active.max():.3f} s" if len(active) else "<b>No active dwells.</b>"
        )
        if result is None:
            self._eval_table.setRowCount(0)
            self._eval_header.setText("")
            self._summary_label.setText(
                f"<b>{s.dwell_source()}:</b> {len(times)} dwell positions "
                f"({len(active)} with time &gt; 0) &nbsp; "
                f"<b>Total time:</b> {times.sum():.2f} s<br>{time_text}"
            )
            self.results_dock.setVisible(True)
            return

        # Clinical evaluation.
        rx = result["rx_params"]
        self._eval_header.setText(
            f"<b>{protocol_info(rx.get('protocol'))['label']}</b> criteria &nbsp; "
            f"Rx {rx['D_p'] * 100:.0f} cGy (percentages relative to Rx)")
        rows = []
        for role in ("ptv", "ctv", "urethra", "bladder", "rectum"):
            info = result["eval_report"].get(role)
            if info is None:
                continue
            names = ", ".join(info["structures"]) or role.upper()
            if info.get("missing"):
                rows.append((names, "-", "N/A", "structure not found"))
                continue
            for c in info["checks"]:
                rows.append((names, c["label"], c["status"], c["detail"]))
        status_colors = {"IDEAL": "#c8f7c5", "PASS": "#e6f7d9", "FAIL": "#f7c5c5"}
        self._eval_table.setRowCount(len(rows))
        for r, (name, label, status, detail) in enumerate(rows):
            for col, text in enumerate((name, label, status, detail)):
                item = QTableWidgetItem(text)
                if status in status_colors:
                    item.setBackground(QColor(status_colors[status]))
                self._eval_table.setItem(r, col, item)

        bd0, bd1 = result["breakdown_initial"], result["breakdown_opt"]
        hist = result["ipsa_history"]
        if result["candidate_region"] == CANDIDATES_EXTENDED_PTV:
            region_text = f"{EXTENDED_PTV_NAME} only (PTV + {result['ptv_margin_mm']:g} mm)"
        elif result["candidate_region"] == CANDIDATES_EXTENDED_CTV:
            region_text = f"{EXTENDED_CTV_NAME} only (CTV + {result['ctv_margin_mm']:g} mm)"
        else:
            region_text = "all mesh points in the search area"
        self._summary_label.setText(
            f"<b>Calculation grid:</b> {result.get('grid_mm', 1):g} mm "
            f"({tuple(int(n) for n in result['opt_dose_gy'].shape)} voxels) "
            f"-- display is native "
            f"{tuple(int(n) for n in s.case['volume'].shape)}<br>"
            f"<b>Candidate region:</b> {region_text}<br>"
            f"<b>{s.dwell_source()}:</b> {len(active)} active / "
            f"{len(all_times)} candidates &nbsp; "
            f"<b>Total time:</b> {times.sum():.2f} s "
            f"(RTPLAN {s.case['plan_dwell_times'].sum():.2f} s)<br>{time_text}<br>"
            f"<b>Loss L:</b> {result['L_initial']:.2f} &rarr; {result['L_opt']:.2f} "
            f"({result['loss_reduction_pct']:.1f}% reduction); "
            f"J_dose {bd0['J_dose']:.2f} &rarr; {bd1['J_dose']:.2f}<br>"
            f"<b>IPSA:</b> {hist.get('n_search_iterations', hist.get('n_iterations_run', 0))} "
            f"iterations, {len(hist.get('epochs', []))} epochs, "
            f"{hist.get('elapsed_s', 0.0):.1f} s -- stopped: {hist.get('stop_reason', '')}"
            f"{' (' + hist['stop_detail'] + ')' if hist.get('stop_detail') else ''}<br>"
            f"<b>Clinical criteria</b> ({protocol_info(result['rx_params'].get('protocol'))['label']}, "
            f"Rx {result['rx_params']['D_p'] * 100:.0f} cGy): "
            f"{result['n_checks'] - result['n_fail']}/{result['n_checks']} met"
        )
        self.results_dock.setVisible(True)

    def _on_dwell_selected(self):
        rows = self._dwell_table.selectionModel().selectedRows()
        if not rows or self.state.case is None:
            return
        s = self.state
        s.selected = rows[0].row()
        p = s.dwells()[0][s.selected]
        for axis in (0, 1, 2):
            s.indices[axis] = s.world_to_index(axis, p[_WORLD[axis]])
        self._redraw_all()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _on_export(self):
        s = self.state
        if s.case is None:
            return
        default = "optimized_dwells.csv" if s.result is not None else "rtplan_dwells.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Dwell Times", default, "CSV files (*.csv)"
        )
        if not path:
            return
        pos, times, channels, regions = s.dwells()
        numbers = self._dwell_numbers(channels)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["channel", "dwell_number",
                             "x_mm", "y_mm", "z_mm", "dwell_time_s", "region"])
            for row in range(len(pos)):
                writer.writerow([
                    int(channels[row]), numbers[row],
                    f"{pos[row, 0]:.3f}", f"{pos[row, 1]:.3f}", f"{pos[row, 2]:.3f}",
                    f"{times[row]:.4f}", regions[row] if regions else "",
                ])
        self.status_bar.showMessage(f"Exported dwell times to {path}")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self._busy():
            answer = QMessageBox.question(
                self, "Quit",
                "A calculation is still running. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            sys.stdout.flush()
            os._exit(0)
        event.accept()


def run_app(argv=None):
    """Open the main window (no case loaded) and run the Qt event loop."""
    if _HAS_VTK:
        vtk.vtkObject.GlobalWarningDisplayOff()

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("HDR Brachy TPS")

    console = _ConsoleStream(sys.stdout)
    original_stdout = sys.stdout
    sys.stdout = console
    try:
        window = MainWindow(console=console)
        window.show()
        return app.exec_()
    finally:
        sys.stdout = original_stdout
