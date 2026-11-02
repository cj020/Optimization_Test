"""Main application window -- File menu import, 2x2 viewer grid, results panel.

Layout follows the HDR Brachy TPS reference UI: the window opens empty,
File > Import DICOM Folder... loads a case in the background, and
Optimization > Run Optimization runs IPSA and shows the optimised dwell
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
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
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

from cost_function import copy_prescription_params
from ipsa import IPSA_DEFAULTS
from pipeline import load_case, optimize_case


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

# Isodose levels as fractions of the prescription dose D_p.
_ISODOSE_LEVELS = [(0.5, "#00a0ff"), (1.0, "#00ff00"), (1.5, "#ffff00"), (2.0, "#ff0000")]

_DWELL_CMAP = "jet"

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
        if name not in masks:
            continue
        cls = s["classification"]
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
        self.show_candidates = False
        self.show_dwells = True
        self.show_isodose = True

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
            regions = [r["mesh_labels"][i].get("region", "") for i in ix]
            return r["mesh_positions"][ix], t[ix], r["mesh_channels"][ix], regions
        c = self.case
        return c["plan_dwell_positions"], c["plan_dwell_times"], c["plan_dwell_channels"], None

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
        if name == "Axial":
            self.cax = self.figure.add_axes([0.93, 0.15, 0.018, 0.7])
            self.cax.set_visible(False)

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

        if s.show_structures:
            for name, color in s.structure_colors.items():
                m2 = self._slice(case["masks"][name])
                if m2.any() and not m2.all():
                    ax.contour(m2.astype(np.uint8), levels=[0.5], colors=[color],
                               linewidths=1.2, origin=self.origin, extent=extent)
                legend.append(Line2D([], [], color=color, lw=1.5, label=name))

        if s.show_isodose and s.result is not None:
            dose2 = self._slice(s.result["opt_dose_gy"])
            D_p = s.result["rx_params"]["D_p"]
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

        nz, ny, nx = case["volume"].shape
        for name, color in self.state.structure_colors.items():
            mask = case["masks"][name]
            if not mask.any():
                continue
            img = vtk.vtkImageData()
            img.SetDimensions(nx, ny, nz)
            img.SetSpacing(*[float(v) for v in case["spacing"]])
            img.SetOrigin(*[float(v) for v in case["origin"]])
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
            actor.GetProperty().SetColor(*self._rgb(color))
            actor.GetProperty().SetOpacity(0.25)
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
        for a in self._candidate_actors:
            a.SetVisibility(s.show_candidates)
        for a in self._dwell_actors:
            a.SetVisibility(s.show_dwells)
        self._render()


# ----------------------------------------------------------------------
# Optimization settings dialog
# ----------------------------------------------------------------------

class OptimizationSettingsDialog(QDialog):
    """Cost-function parameters and IPSA iteration settings."""

    def __init__(self, params, ipsa_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Settings")
        self._oar_keys = list(params.get("oar_params", {}).keys())
        self._spins = {}

        layout = QVBoxLayout(self)

        target = QGroupBox("CTV / Target")
        form = QFormLayout(target)
        self._add(form, "D_p", "D_p  prescription (Gy)", params["D_p"])
        self._add(form, "D_max", "D_max  max target dose (Gy)", params["D_max"])
        self._add(form, "w_L", "w_L  underdose weight", params["w_L"])
        self._add(form, "w_H", "w_H  overdose weight", params["w_H"])
        layout.addWidget(target)

        for key in self._oar_keys:
            box = QGroupBox(key.capitalize())
            form = QFormLayout(box)
            vals = params["oar_params"][key]
            self._add(form, f"{key}.D_tol", "D_tol  tolerance (Gy)", vals["D_tol"])
            self._add(form, f"{key}.w_O", "w_O  overdose weight", vals["w_O"])
            layout.addWidget(box)

        box = QGroupBox("Default OAR (unmatched names)")
        form = QFormLayout(box)
        self._add(form, "D_tol", "D_tol  tolerance (Gy)", params["D_tol"])
        self._add(form, "w_O", "w_O  overdose weight", params["w_O"])
        layout.addWidget(box)

        box = QGroupBox("IPSA")
        form = QFormLayout(box)
        self._iters = QSpinBox()
        self._iters.setRange(1, 1_000_000)
        self._iters.setValue(int(ipsa_params.get("n_iterations", IPSA_DEFAULTS["n_iterations"])))
        form.addRow("Max iterations", self._iters)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2**31 - 1)
        self._seed.setValue(int(ipsa_params.get("seed", IPSA_DEFAULTS["seed"])))
        form.addRow("Random seed", self._seed)
        layout.addWidget(box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = QPushButton("Reset Defaults")
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        reset.clicked.connect(self._on_reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add(self, form, key, label, value):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1e6)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setValue(float(value))
        form.addRow(label, spin)
        self._spins[key] = spin

    def params(self):
        p = copy_prescription_params()
        for key in ("D_p", "D_max", "w_L", "w_H", "D_tol", "w_O"):
            p[key] = self._spins[key].value()
        for key in self._oar_keys:
            p["oar_params"][key] = {
                "D_tol": self._spins[f"{key}.D_tol"].value(),
                "w_O": self._spins[f"{key}.w_O"].value(),
            }
        return p

    def ipsa_params(self):
        return {"n_iterations": self._iters.value(), "seed": self._seed.value()}

    def _on_reset(self):
        d = copy_prescription_params()
        for key in ("D_p", "D_max", "w_L", "w_H", "D_tol", "w_O"):
            self._spins[key].setValue(d[key])
        for key in self._oar_keys:
            if key in d["oar_params"]:
                self._spins[f"{key}.D_tol"].setValue(d["oar_params"][key]["D_tol"])
                self._spins[f"{key}.w_O"].setValue(d["oar_params"][key]["w_O"])
        self._iters.setValue(IPSA_DEFAULTS["n_iterations"])
        self._seed.setValue(IPSA_DEFAULTS["seed"])


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

        optim_menu = menu_bar.addMenu("&Optimization")
        self._settings_action = QAction("Optimization &Settings...", self)
        self._settings_action.setShortcut("Ctrl+P")
        self._settings_action.triggered.connect(self._on_settings)
        optim_menu.addAction(self._settings_action)

        self._run_action = QAction("&Run Optimization", self)
        self._run_action.setShortcut("Ctrl+R")
        self._run_action.triggered.connect(self._on_run_optimization)
        optim_menu.addAction(self._run_action)

        view_menu = menu_bar.addMenu("&View")
        self._toggle_actions = {}
        for attr, text in (
            ("show_structures", "Show &Structures"),
            ("show_candidates", "Show &Candidate Mesh"),
            ("show_dwells", "Show &Dwells"),
            ("show_isodose", "Show &Isodose Lines"),
        ):
            act = QAction(text, self)
            act.setCheckable(True)
            act.setChecked(getattr(self.state, attr))
            act.toggled.connect(lambda on, a=attr: self._on_toggle(a, on))
            view_menu.addAction(act)
            self._toggle_actions[attr] = act
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
        tabs.addTab(self._eval_table, "Plan Evaluation")

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
        self._run_action.setEnabled(has_case and not busy)
        self._export_action.setEnabled(has_case and not busy)

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

    def _on_case_loaded(self, case):
        s = self.state
        s.case = case
        s.result = None
        s.selected = None
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

        basename = os.path.basename(case["folder"].rstrip("/\\"))
        self.setWindowTitle(f"HDR Brachy TPS - {basename}")
        nz, ny, nx = case["volume"].shape
        times = case["plan_dwell_times"]
        self.status_bar.showMessage(
            f"Loaded {basename}  ({case['modality']} {nx}x{ny}x{nz} voxels, "
            f"RTPLAN: {len(times)} dwells on "
            f"{len(np.unique(case['plan_dwell_channels']))} channels, "
            f"total time {times.sum():.1f} s) "
            f"-- Optimization > Run Optimization to optimise dwell times"
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def _on_settings(self):
        dlg = OptimizationSettingsDialog(
            self._rx_params, self._ipsa_params, parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._rx_params = dlg.params()
            self._ipsa_params = dlg.ipsa_params()
            self.status_bar.showMessage("Optimization settings updated.")

    def _on_run_optimization(self):
        if self.state.case is None:
            return
        self._start_worker(
            optimize_case, self._on_optimized, "Running IPSA optimization ...",
            self.state.case, rx_params=self._rx_params, ipsa_params=self._ipsa_params,
        )

    def _on_optimized(self, result):
        s = self.state
        s.result = result
        s.selected = None
        self._fill_results()
        self._redraw_all()
        self.view_3d.set_dwells()

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
            self._summary_label.setText(
                f"<b>{s.dwell_source()}:</b> {len(times)} dwell positions "
                f"({len(active)} with time &gt; 0) &nbsp; "
                f"<b>Total time:</b> {times.sum():.2f} s<br>{time_text}"
            )
            self.results_dock.setVisible(True)
            return

        # Clinical evaluation.
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
        self._summary_label.setText(
            f"<b>{s.dwell_source()}:</b> {len(active)} active / "
            f"{len(all_times)} candidates &nbsp; "
            f"<b>Total time:</b> {times.sum():.2f} s "
            f"(RTPLAN {s.case['plan_dwell_times'].sum():.2f} s)<br>{time_text}<br>"
            f"<b>Loss L:</b> {result['L_initial']:.2f} &rarr; {result['L_opt']:.2f} "
            f"({result['loss_reduction_pct']:.1f}% reduction); "
            f"J_dose {bd0['J_dose']:.2f} &rarr; {bd1['J_dose']:.2f}<br>"
            f"<b>IPSA:</b> {hist.get('n_iterations_run', 0)} iterations "
            f"({hist.get('stop_reason', '')}) &nbsp; "
            f"<b>Clinical criteria:</b> {result['n_checks'] - result['n_fail']}"
            f"/{result['n_checks']} met"
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
