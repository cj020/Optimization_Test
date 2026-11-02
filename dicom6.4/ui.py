"""HDR brachy planning window: import a study, edit cost-function parameters, run IPSA.

Layout follows the reference TPS UI: menu bar, 2x2 viewer grid (axial,
sagittal, coronal, 3D volume) and a status bar. The window opens empty;
File > Import DICOM Folder loads a study and Optimization > Run Optimization
shows the optimised dwell positions on all four views.
"""

import os
import sys

import numpy as np

from cost_function import cost_function, copy_prescription_params

level = 150
width = 100

_MESH_REGION_RGB = {
    "CTV": (255, 40, 40),
    "PTV": (255, 150, 0),
    "OAR": (0, 220, 230),
    "CTV+OAR": (230, 40, 230),
    "PTV+OAR": (230, 40, 230),
    "TARGET+OAR": (230, 40, 230),
    "OUTSIDE": (240, 220, 40),
}

_LEGEND_ORDER = ("CTV", "PTV", "OAR", "CTV+OAR", "OUTSIDE")

# Dwell-time colour ramp, short -> long.
_DWELL_RAMP = (
    (0.00, (40, 110, 255)),
    (0.33, (0, 230, 230)),
    (0.66, (255, 235, 0)),
    (1.00, (255, 30, 30)),
)

_HEADER_STYLE = (
    "background-color: rgba(0,0,0,180); color: #00ff00; "
    "font-weight: bold; padding: 2px 6px;"
)


def _mesh_region(label):
    if isinstance(label, dict):
        return label.get("region", "OUTSIDE")
    return label or "OUTSIDE"


def _mesh_colors(mesh_labels, n_points):
    colors = np.empty((n_points, 3), dtype=np.uint8)
    colors[:] = _MESH_REGION_RGB["OUTSIDE"]
    if mesh_labels is None:
        return colors
    for i, lab in enumerate(list(mesh_labels)[:n_points]):
        colors[i] = _MESH_REGION_RGB.get(_mesh_region(lab), _MESH_REGION_RGB["OUTSIDE"])
    return colors


def _dwell_time_colors(times):
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = float(times.min()), float(times.max())
    t = np.zeros_like(times) if hi - lo < 1e-12 else (times - lo) / (hi - lo)
    stops = np.array([s for s, _ in _DWELL_RAMP])
    rgb = np.array([c for _, c in _DWELL_RAMP], dtype=np.float64)
    colors = np.column_stack([np.interp(t, stops, rgb[:, k]) for k in range(3)])
    return colors.round().astype(np.uint8)


def _load_gui_modules():
    """Import PyQt5 and VTK lazily so batch runs do not need them."""
    try:
        import PyQt5
        plugin_root = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
        if os.path.isdir(plugin_root):
            os.environ.setdefault(
                "QT_QPA_PLATFORM_PLUGIN_PATH", os.path.join(plugin_root, "platforms"),
            )
        import vtk
        from PyQt5 import QtCore, QtGui, QtWidgets
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError as exc:
        return None, exc
    return (vtk, QtCore, QtGui, QtWidgets, QVTKRenderWindowInteractor), None


def start_ui(load_folder=None, run_optimization=None, initial_params=None,
             start_dir="", show=None):
    """
    Open the planning window and block until it is closed.

    Parameters
    ----------
    load_folder : callable(folder) -> dict or None
        Called on a background thread after File > Import DICOM Folder.
        Returns keyword data for the views: volume, spacing, origin,
        dwell_positions (mesh vertices, world mm), dose_gy, masks,
        ctv_names, oar_names, mesh_labels and optionally channel_ids.
        Returns None if the study cannot be planned.
    run_optimization : callable(params) -> dict
        Called on a background thread by Optimization > Run Optimization with
        the current cost-function parameters. Returns dwell_times (one per
        mesh vertex), dose_gy (optimised dose) and summary (list of str).
    initial_params : dict or None
        Starting cost-function values (defaults to PRESCRIPTION_PARAMS).
    start_dir : str
        Folder the import dialog opens in.
    show : bool or None
        False skips the window (e.g. batch runs). None opens it when PyQt5
        and VTK are installed.

    Returns
    -------
    params : dict
        Cost-function parameters as last set in the window.
    """
    params = copy_prescription_params(initial_params)
    if show is False:
        print("Planning window skipped.")
        return params

    modules, error = _load_gui_modules()
    if modules is None:
        print(f"Planning window skipped ({error}). Install PyQt5 and vtk to use it.")
        return params

    print("\n  Opening planning window ... use File > Import DICOM Folder to load a study.")

    window_cls = _build_window_class(*modules)
    QtWidgets = modules[3]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HDR Brachy TPS")

    window = window_cls(params, load_folder, run_optimization, start_dir)
    window.show()
    app.exec_()
    return window.result_params()


def _build_window_class(vtk, QtCore, QtGui, QtWidgets, QVTKRenderWindowInteractor):
    from vtk.util.numpy_support import numpy_to_vtk

    Qt = QtCore.Qt

    def numpy_volume_to_vtk(array, spacing, origin):
        data = np.ascontiguousarray(array, dtype=np.float32)
        nz, ny, nx = data.shape
        scalars = numpy_to_vtk(data.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT)
        image = vtk.vtkImageData()
        image.SetDimensions(int(nx), int(ny), int(nz))
        image.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
        image.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
        image.GetPointData().SetScalars(scalars)
        return image

    def points_polydata(coords, colors):
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for i, p in enumerate(coords):
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        rgb = numpy_to_vtk(
            np.ascontiguousarray(colors, dtype=np.uint8), deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        rgb.SetName("Colors")
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.GetPointData().SetScalars(rgb)
        return poly

    def point_actor(size):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vtk.vtkPolyData())
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUsePointData()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetPointSize(size)
        prop.SetRenderPointsAsSpheres(True)
        prop.SetLighting(False)
        return actor

    class PointSet:
        """World coordinates, voxel indices (z, y, x) and colours of overlay points."""

        def __init__(self, world=None, spacing=None, origin=None, colors=None):
            self.world = np.zeros((0, 3)) if world is None else np.asarray(world, dtype=np.float64)
            if len(self.world) and spacing is not None:
                # Same truncation as conversion.world_to_voxel, ordered (z, y, x).
                rel = (self.world - np.asarray(origin, dtype=np.float64)) / np.asarray(spacing, dtype=np.float64)
                self.voxel = rel.astype(int)[:, ::-1]
            else:
                self.voxel = np.zeros((0, 3), dtype=int)
            self.colors = (np.zeros((0, 3), dtype=np.uint8) if colors is None
                           else np.asarray(colors, dtype=np.uint8))

        def __len__(self):
            return len(self.world)

    def _framed(parent, title):
        frame_layout = QtWidgets.QVBoxLayout(parent)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        label = QtWidgets.QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(_HEADER_STYLE)
        frame_layout.addWidget(label)
        return frame_layout, label

    class PlaceholderView(QtWidgets.QFrame):
        """Empty viewer shown before a study is imported."""

        def __init__(self, title, parent=None):
            super().__init__(parent)
            self.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Sunken)
            layout, _label = _framed(self, title)
            body = QtWidgets.QLabel("No study loaded\nFile > Import DICOM Folder...")
            body.setAlignment(Qt.AlignCenter)
            body.setStyleSheet("background-color: black; color: #777;")
            layout.addWidget(body, stretch=1)

        def initialize(self):
            pass

        def finalize(self):
            pass

        def set_layers(self, candidates=None, dwells=None):
            pass

        def set_visibility(self, show_candidates, show_dwells):
            pass

    class SliceView(QtWidgets.QFrame):
        """vtkResliceImageViewer with mesh-candidate and optimised-dwell overlays."""

        _NAMES = {2: "Axial", 0: "Sagittal", 1: "Coronal"}

        def __init__(self, axis, image, parent=None):
            super().__init__(parent)
            self.axis = axis
            self._image = image
            self._candidates = PointSet()
            self._dwells = None
            self._show_candidates = True
            self._show_dwells = True

            self.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Sunken)
            layout, self.label = _framed(self, self._NAMES[axis])
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget, stretch=1)

            render_window = self.vtk_widget.GetRenderWindow()
            self.viewer = vtk.vtkResliceImageViewer()
            self.viewer.SetRenderWindow(render_window)
            self.viewer.SetupInteractor(self.vtk_widget)
            self.viewer.SetInputData(image)
            self.viewer.SetSliceOrientation(axis)
            self.viewer.SetResliceModeToAxisAligned()
            self.viewer.GetRenderer().SetBackground(0.0, 0.0, 0.0)
            self.viewer.SetColorWindow(width)
            self.viewer.SetColorLevel(level)

            # Overlay layer keeps points in front of the image slice.
            render_window.SetNumberOfLayers(2)
            self.overlay = vtk.vtkRenderer()
            self.overlay.SetLayer(1)
            self.overlay.InteractiveOff()
            self.overlay.SetActiveCamera(self.viewer.GetRenderer().GetActiveCamera())
            render_window.AddRenderer(self.overlay)
            self.candidate_actor = point_actor(6)
            self.dwell_actor = point_actor(11)
            self.overlay.AddActor(self.candidate_actor)
            self.overlay.AddActor(self.dwell_actor)

            self.viewer.SetSlice(image.GetDimensions()[axis] // 2)

        def initialize(self):
            self.vtk_widget.Initialize()
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            for event in ("MouseWheelForwardEvent", "MouseWheelBackwardEvent", "KeyPressEvent"):
                interactor.AddObserver(event, self._on_slice_event)
            self.viewer.Render()
            self.refresh()

        def finalize(self):
            self.vtk_widget.Finalize()

        def _on_slice_event(self, _obj, _event):
            QtCore.QTimer.singleShot(0, self.refresh)

        def set_layers(self, candidates=None, dwells=None):
            if candidates is not None:
                self._candidates = candidates
            self._dwells = dwells
            self.refresh()

        def set_visibility(self, show_candidates, show_dwells):
            self._show_candidates = show_candidates
            self._show_dwells = show_dwells
            self.refresh()

        def _draw(self, actor, points, index, plane, visible):
            if points is None or not len(points):
                actor.SetVisibility(False)
                return 0
            chosen = points.voxel[:, 2 - self.axis] == index
            count = int(np.count_nonzero(chosen))
            if visible and count:
                coords = points.world[chosen].copy()
                coords[:, self.axis] = plane
                actor.GetMapper().SetInputData(points_polydata(coords, points.colors[chosen]))
                actor.SetVisibility(True)
            else:
                actor.SetVisibility(False)
            return count

        def refresh(self):
            index = int(self.viewer.GetSlice())
            plane = self._image.GetOrigin()[self.axis] + index * self._image.GetSpacing()[self.axis]
            n_cand = self._draw(self.candidate_actor, self._candidates, index, plane,
                                self._show_candidates)
            n_dwell = self._draw(self.dwell_actor, self._dwells, index, plane, self._show_dwells)

            unit = {2: "z", 0: "x", 1: "y"}[self.axis]
            text = (f"{self._NAMES[self.axis]}  [{index}/{self.viewer.GetSliceMax()}]   "
                    f"{unit} = {plane:.2f} mm   |   {n_cand} mesh pts")
            if self._dwells is not None:
                text += f"   |   {n_dwell} optimized dwells"
            self.label.setText(text)
            self.vtk_widget.GetRenderWindow().Render()

    class VolumeView(QtWidgets.QFrame):
        """3-D ray-cast volume with mesh candidates and optimised dwells."""

        def __init__(self, image, parent=None):
            super().__init__(parent)
            self.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Sunken)
            layout, self.label = _framed(self, "3D Volume")
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget, stretch=1)

            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.15)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            self.renderer.AddVolume(self._build_volume(image))

            self.candidate_actor = point_actor(5)
            self.dwell_actor = point_actor(10)
            self.renderer.AddActor(self.candidate_actor)
            self.renderer.AddActor(self.dwell_actor)
            self._n_cand = 0
            self._n_dwell = None
            self.renderer.ResetCamera()

        @staticmethod
        def _build_volume(image):
            lo, hi = image.GetScalarRange()
            span = max(hi - lo, 1.0)
            color_tf = vtk.vtkColorTransferFunction()
            opacity_tf = vtk.vtkPiecewiseFunction()
            stops = (
                (0.00, (0.0, 0.0, 0.0), 0.00),
                (0.25, (0.18, 0.12, 0.08), 0.00),
                (0.45, (0.45, 0.28, 0.18), 0.04),
                (0.65, (0.75, 0.58, 0.42), 0.15),
                (0.85, (0.92, 0.84, 0.74), 0.40),
                (1.00, (1.0, 1.0, 1.0), 0.85),
            )
            for t, rgb, alpha in stops:
                color_tf.AddRGBPoint(lo + t * span, *rgb)
                opacity_tf.AddPoint(lo + t * span, alpha)

            prop = vtk.vtkVolumeProperty()
            prop.SetColor(color_tf)
            prop.SetScalarOpacity(opacity_tf)
            prop.ShadeOn()
            prop.SetInterpolationTypeToLinear()
            prop.SetAmbient(0.4)
            prop.SetDiffuse(0.6)
            prop.SetSpecular(0.2)

            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(image)
            actor = vtk.vtkVolume()
            actor.SetMapper(mapper)
            actor.SetProperty(prop)
            return actor

        def initialize(self):
            self.vtk_widget.Initialize()
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
            self.vtk_widget.GetRenderWindow().Render()

        def finalize(self):
            self.vtk_widget.Finalize()

        @staticmethod
        def _fill(actor, points):
            if points is None or not len(points):
                actor.GetMapper().SetInputData(vtk.vtkPolyData())
                return 0
            actor.GetMapper().SetInputData(points_polydata(points.world, points.colors))
            return len(points)

        def set_layers(self, candidates=None, dwells=None):
            if candidates is not None:
                self._n_cand = self._fill(self.candidate_actor, candidates)
            n_dwell = self._fill(self.dwell_actor, dwells)
            self._n_dwell = None if dwells is None else n_dwell
            text = f"3D Volume   |   {self._n_cand} mesh pts"
            if self._n_dwell is not None:
                text += f"   |   {self._n_dwell} optimized dwells"
            self.label.setText(text)
            self.vtk_widget.GetRenderWindow().Render()

        def set_visibility(self, show_candidates, show_dwells):
            self.candidate_actor.SetVisibility(show_candidates)
            self.dwell_actor.SetVisibility(show_dwells)
            self.vtk_widget.GetRenderWindow().Render()

    class SettingsDialog(QtWidgets.QDialog):
        """Non-modal cost-function parameter dialog (Optimization menu)."""

        changed = QtCore.pyqtSignal()
        runRequested = QtCore.pyqtSignal()

        def __init__(self, params, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Optimization Settings")
            self.setMinimumWidth(380)
            self.setWindowFlags(self.windowFlags() | Qt.Tool)

            self._oar_keys = list(params.get("oar_params", {}).keys())
            self._spins = {}

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(10, 10, 10, 10)

            group, grid = self._group("CTV / Target")
            self._row(grid, 0, "D_p", "D_p (Gy)", params["D_p"], "D_max", "D_max (Gy)", params["D_max"])
            self._row(grid, 1, "w_L", "w_L  underdose", params["w_L"], "w_H", "w_H  overdose", params["w_H"])
            layout.addWidget(group)

            for key in self._oar_keys:
                vals = params["oar_params"][key]
                group, grid = self._group(key.capitalize())
                self._row(grid, 0, f"{key}.D_tol", "D_tol (Gy)", vals["D_tol"],
                          f"{key}.w_O", "w_O  impact", vals["w_O"])
                layout.addWidget(group)

            group, grid = self._group("Default OAR (unmatched)")
            self._row(grid, 0, "D_tol", "D_tol (Gy)", params["D_tol"], "w_O", "w_O  impact", params["w_O"])
            layout.addWidget(group)

            cost_group = QtWidgets.QGroupBox("Cost")
            cost_layout = QtWidgets.QVBoxLayout(cost_group)
            self.cost_label = QtWidgets.QLabel("--")
            self.cost_label.setStyleSheet("font-weight: bold; font-family: monospace;")
            cost_layout.addWidget(self.cost_label)
            layout.addWidget(cost_group)

            actions = QtWidgets.QGroupBox("Actions")
            actions_layout = QtWidgets.QVBoxLayout(actions)
            self.run_button = QtWidgets.QPushButton("Run Optimization")
            self.run_button.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; "
                "font-weight: bold; padding: 6px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:disabled { background-color: #9e9e9e; }"
            )
            self.run_button.clicked.connect(self.runRequested.emit)
            actions_layout.addWidget(self.run_button)
            reset_btn = QtWidgets.QPushButton("Reset to Defaults")
            reset_btn.setStyleSheet("QPushButton { padding: 4px; }")
            reset_btn.clicked.connect(self.reset)
            actions_layout.addWidget(reset_btn)
            layout.addWidget(actions)
            layout.addStretch()

            for spin in self._spins.values():
                spin.valueChanged.connect(self.changed.emit)

        @staticmethod
        def _group(title):
            group = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout(group)
            return group, grid

        def _row(self, grid, row, left_key, left_label, left_val, right_key, right_label, right_val):
            for col, (key, text, value) in enumerate(
                ((left_key, left_label, left_val), (right_key, right_label, right_val))
            ):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0.0, 1000.0)
                spin.setDecimals(2)
                spin.setSingleStep(0.5)
                spin.setValue(float(value))
                spin.setKeyboardTracking(False)
                grid.addWidget(QtWidgets.QLabel(text), row, 2 * col)
                grid.addWidget(spin, row, 2 * col + 1)
                self._spins[key] = spin

        def apply_to(self, params):
            p = copy_prescription_params(params)
            for key in ("D_p", "D_max", "w_L", "w_H", "D_tol", "w_O"):
                p[key] = self._spins[key].value()
            for key in self._oar_keys:
                p["oar_params"][key] = {
                    "D_tol": self._spins[f"{key}.D_tol"].value(),
                    "w_O": self._spins[f"{key}.w_O"].value(),
                }
            return p

        def reset(self):
            defaults = copy_prescription_params()
            for key in ("D_p", "D_max", "w_L", "w_H", "D_tol", "w_O"):
                self._spins[key].setValue(defaults[key])
            for key in self._oar_keys:
                if key in defaults["oar_params"]:
                    self._spins[f"{key}.D_tol"].setValue(defaults["oar_params"][key]["D_tol"])
                    self._spins[f"{key}.w_O"].setValue(defaults["oar_params"][key]["w_O"])

    class DwellTableDialog(QtWidgets.QDialog):
        """Optimised dwell positions and times, grouped by channel."""

        def __init__(self, summary, rows, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Optimized Dwell Positions")
            self.resize(640, 560)
            self.setWindowFlags(self.windowFlags() | Qt.Tool)
            layout = QtWidgets.QVBoxLayout(self)

            text = QtWidgets.QLabel("\n".join(summary))
            text.setStyleSheet("font-family: monospace; font-weight: bold; padding: 4px;")
            layout.addWidget(text)

            headers = ["Channel", "Dwell #", "X (mm)", "Y (mm)", "Z (mm)", "Time (s)", "Region"]
            table = QtWidgets.QTableWidget(len(rows), len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
            header = table.horizontalHeader()
            for col in range(len(headers)):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(len(headers) - 1, QtWidgets.QHeaderView.Stretch)
            for r, values in enumerate(rows):
                for c, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if 0 < c < 6:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(r, c, item)
            layout.addWidget(table)

            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            buttons.rejected.connect(self.close)
            layout.addWidget(buttons)

    class Worker(QtCore.QThread):
        succeeded = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, fn, arg):
            super().__init__()
            self._fn = fn
            self._arg = arg

        def run(self):
            try:
                self.succeeded.emit(self._fn(self._arg))
            except Exception:
                import traceback
                self.failed.emit(traceback.format_exc())

    class PlanningWindow(QtWidgets.QMainWindow):
        def __init__(self, params, load_folder, run_optimization, start_dir):
            super().__init__()
            self.setWindowTitle("HDR Brachy TPS -- Pre-Insertion Planning")
            self.resize(1200, 800)

            self._params = params
            self._load_folder = load_folder
            self._run_optimization = run_optimization
            self._last_dir = start_dir or ""
            self._worker = None
            self._progress = None
            self._data = None
            self._result = None
            self._dose_gy = None
            self._dwell_dialog = None

            self.settings = SettingsDialog(params, self)
            self.settings.changed.connect(self._refresh_cost)
            self.settings.runRequested.connect(self._on_run_optimization)

            self._build_menu()
            self._build_views()
            self._build_status_bar()
            self._update_actions()
            self._refresh_cost()
            self.statusBar().showMessage("Ready -- use File > Import DICOM Folder to load a study.")

        # --------------------------------------------------------------
        # Menu
        # --------------------------------------------------------------
        def _action(self, menu, text, slot, shortcut=None, checkable=False):
            action = QtWidgets.QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            if checkable:
                action.setCheckable(True)
                action.setChecked(True)
            action.triggered.connect(slot)
            menu.addAction(action)
            return action

        def _build_menu(self):
            bar = self.menuBar()

            file_menu = bar.addMenu("&File")
            self._import_action = self._action(
                file_menu, "&Import DICOM Folder...", self._on_import, "Ctrl+O")
            file_menu.addSeparator()
            self._action(file_menu, "&Quit", self.close, "Ctrl+Q")

            plan_menu = bar.addMenu("&Planning")
            self._candidates_action = self._action(
                plan_menu, "Show &Mesh Candidates", self._apply_visibility, checkable=True)
            self._dwells_action = self._action(
                plan_menu, "Show &Optimized Dwells", self._apply_visibility, checkable=True)

            calc_menu = bar.addMenu("&Calculation")
            self._cost_action = self._action(
                calc_menu, "Recompute &Cost J", self._refresh_cost, "Ctrl+J")

            optim_menu = bar.addMenu("&Optimization")
            self._action(optim_menu, "Optimization &Settings...", self._open_settings, "Ctrl+P")
            self._action(optim_menu, "&Reset Parameters to Defaults", self.settings.reset)
            optim_menu.addSeparator()
            self._run_action = self._action(
                optim_menu, "&Run Optimization", self._on_run_optimization, "Ctrl+R")

            analysis_menu = bar.addMenu("&Analysis")
            self._dwell_table_action = self._action(
                analysis_menu, "Optimized &Dwell Positions...", self._show_dwell_table, "Ctrl+T")

        def _update_actions(self):
            busy = self._worker is not None and self._worker.isRunning()
            loaded = self._data is not None
            optimized = self._result is not None
            self._import_action.setEnabled(self._load_folder is not None and not busy)
            can_run = loaded and self._run_optimization is not None and not busy
            self._run_action.setEnabled(can_run)
            self.settings.run_button.setEnabled(can_run)
            self._cost_action.setEnabled(loaded)
            self._candidates_action.setEnabled(loaded)
            self._dwells_action.setEnabled(optimized)
            self._dwell_table_action.setEnabled(optimized)

        # --------------------------------------------------------------
        # Views
        # --------------------------------------------------------------
        def _build_views(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            grid = QtWidgets.QGridLayout(central)
            grid.setContentsMargins(4, 4, 4, 4)
            grid.setSpacing(4)
            for i in range(2):
                grid.setRowStretch(i, 1)
                grid.setColumnStretch(i, 1)
            self._grid = grid
            self._views = [
                PlaceholderView("Axial"), PlaceholderView("Sagittal"),
                PlaceholderView("Coronal"), PlaceholderView("3D Volume"),
            ]
            self._place_views()

        def _place_views(self):
            for view, (row, col) in zip(self._views, ((0, 0), (0, 1), (1, 0), (1, 1))):
                self._grid.addWidget(view, row, col)

        def _replace_views(self, image):
            for view in self._views:
                self._grid.removeWidget(view)
                view.finalize()
                view.deleteLater()
            self._views = [SliceView(2, image), SliceView(0, image), SliceView(1, image),
                           VolumeView(image)]
            self._place_views()
            for view in self._views:
                view.initialize()

        def _build_status_bar(self):
            status = QtWidgets.QStatusBar()
            self.setStatusBar(status)
            self._legend = QtWidgets.QLabel("")
            self._legend.setTextFormat(Qt.RichText)
            status.addPermanentWidget(self._legend)
            self._cost_status = QtWidgets.QLabel("")
            self._cost_status.setStyleSheet("font-weight: bold; padding: 0 8px;")
            status.addPermanentWidget(self._cost_status)

        def _update_legend(self):
            parts = []
            if self._data is not None and self._candidates_action.isChecked():
                parts.append("Mesh: " + "  ".join(
                    f"<span style='color: rgb{_MESH_REGION_RGB[name]};'>&#9679;</span> {name}"
                    for name in _LEGEND_ORDER
                ))
            if self._result is not None and self._dwells_action.isChecked():
                times = self._result["active_times"]
                ramp = "".join(
                    f"<span style='color: rgb{rgb};'>&#9679;</span>" for _, rgb in _DWELL_RAMP
                )
                if len(times):
                    parts.append(f"Dwell time: {times.min():.2f} s {ramp} {times.max():.2f} s")
            self._legend.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts))

        def _apply_visibility(self):
            show_cand = self._candidates_action.isChecked()
            show_dwell = self._dwells_action.isChecked()
            for view in self._views:
                view.set_visibility(show_cand, show_dwell)
            self._update_legend()

        # --------------------------------------------------------------
        # Background jobs
        # --------------------------------------------------------------
        def _busy(self):
            if self._worker is not None and self._worker.isRunning():
                QtWidgets.QMessageBox.information(
                    self, "Busy", "Another operation is still running. Please wait.",
                )
                return True
            return False

        def _start_job(self, fn, arg, title, message, on_success):
            self._progress = QtWidgets.QProgressDialog(message, None, 0, 0, self)
            self._progress.setWindowTitle(title)
            self._progress.setWindowModality(Qt.WindowModal)
            self._progress.setMinimumDuration(0)
            self._progress.show()
            self.setCursor(Qt.WaitCursor)

            self._worker = Worker(fn, arg)
            self._worker.succeeded.connect(on_success)
            self._worker.failed.connect(lambda details: self._on_job_failed(title, details))
            self._worker.finished.connect(self._update_actions)
            self._worker.start()
            self._update_actions()

        def _finish_job(self):
            if self._progress is not None:
                self._progress.close()
                self._progress = None
            self.unsetCursor()

        def _on_job_failed(self, title, details):
            self._finish_job()
            self.statusBar().showMessage(f"{title} failed.")
            QtWidgets.QMessageBox.critical(self, f"{title} Error", details)

        # --------------------------------------------------------------
        # Import
        # --------------------------------------------------------------
        def _on_import(self):
            if self._load_folder is None or self._busy():
                return
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select DICOM Folder", self._last_dir, QtWidgets.QFileDialog.ShowDirsOnly,
            )
            if not folder:
                return
            self._last_dir = os.path.dirname(folder)
            self.statusBar().showMessage(f"Loading {folder} ...")
            self._start_job(
                self._load_folder, folder, "Import DICOM Folder",
                f"Loading {folder}\n\nReading DICOM, meshing needles, and calculating "
                "the initial TG-43 dose. Progress is printed in the console.",
                lambda data: self._on_loaded(folder, data),
            )

        def _on_loaded(self, folder, data):
            self._finish_job()
            if data is None:
                self.statusBar().showMessage("Import failed.")
                QtWidgets.QMessageBox.warning(
                    self, "Import DICOM Folder",
                    f"{folder}\n\nThis study cannot be planned (missing RTSTRUCT, "
                    "RTPLAN, or mesh points). See the console for details.",
                )
                return

            self._data = data
            self._result = None
            if self._dwell_dialog is not None:
                self._dwell_dialog.close()
                self._dwell_dialog = None
            self._dose_gy = data.get("dose_gy")

            self._candidates = PointSet(
                data["dwell_positions"], data["spacing"], data["origin"],
                _mesh_colors(data.get("mesh_labels"), len(data["dwell_positions"])),
            )
            self._replace_views(numpy_volume_to_vtk(data["volume"], data["spacing"], data["origin"]))
            for view in self._views:
                view.set_layers(self._candidates, None)
            self._candidates_action.setChecked(True)
            self._dwells_action.setChecked(True)
            self._apply_visibility()
            self._update_actions()
            self._refresh_cost()

            name = os.path.basename(os.path.normpath(folder))
            self.setWindowTitle(f"HDR Brachy TPS - {name}")
            shape = data["volume"].shape
            self.statusBar().showMessage(
                f"Loaded {name}  ({shape[2]}x{shape[1]}x{shape[0]} voxels, "
                f"{len(self._candidates)} mesh dwells) -- set parameters, then Run Optimization"
            )
            self._open_settings()

        # --------------------------------------------------------------
        # Optimization
        # --------------------------------------------------------------
        def _on_run_optimization(self):
            if self._data is None or self._run_optimization is None or self._busy():
                return
            self._params = self.settings.apply_to(self._params)
            self.statusBar().showMessage("Running IPSA optimization ...")
            self._start_job(
                self._run_optimization, copy_prescription_params(self._params),
                "Run Optimization",
                "Running IPSA (dwell time + mesh position) and the optimized TG-43 dose.\n"
                "This can take several minutes. Progress is printed in the console.",
                self._on_optimized,
            )

        def _on_optimized(self, result):
            self._finish_job()
            data = self._data
            times = np.asarray(result["dwell_times"], dtype=np.float64)
            active = times > 0
            positions = np.asarray(data["dwell_positions"], dtype=np.float64)
            active_times = times[active]

            self._result = dict(result)
            self._result["active_index"] = np.flatnonzero(active)
            self._result["active_times"] = active_times
            self._dose_gy = result.get("dose_gy", self._dose_gy)

            dwells = PointSet(positions[active], data["spacing"], data["origin"],
                              _dwell_time_colors(active_times))
            for view in self._views:
                view.set_layers(None, dwells)
            # Optimised dwells are the focus now; candidates stay available in Planning.
            self._candidates_action.setChecked(False)
            self._dwells_action.setChecked(True)
            self._apply_visibility()
            self._update_actions()
            self._refresh_cost()

            summary = list(result.get("summary", []))
            self.statusBar().showMessage(
                summary[0] if summary else f"Optimization done: {int(active.sum())} active dwells"
            )
            self._show_dwell_table()

        def _dwell_rows(self):
            data = self._data
            positions = np.asarray(data["dwell_positions"], dtype=np.float64)
            labels = data.get("mesh_labels")
            channels = data.get("channel_ids")
            times = np.asarray(self._result["dwell_times"], dtype=np.float64)
            rows = []
            counters = {}
            for i in self._result["active_index"]:
                ch = channels[i] if channels is not None else "-"
                key = str(ch)
                counters[key] = counters.get(key, 0) + 1
                region = _mesh_region(labels[i]) if labels is not None else ""
                x, y, z = positions[i]
                rows.append([str(ch), str(counters[key]), f"{x:.2f}", f"{y:.2f}",
                             f"{z:.2f}", f"{times[i]:.3f}", region])
            return rows

        def _show_dwell_table(self):
            if self._result is None:
                return
            if self._dwell_dialog is not None:
                self._dwell_dialog.close()
            self._dwell_dialog = DwellTableDialog(
                self._result.get("summary", []), self._dwell_rows(), self,
            )
            self._dwell_dialog.show()

        # --------------------------------------------------------------
        # Parameters / cost
        # --------------------------------------------------------------
        def _open_settings(self):
            self.settings.show()
            self.settings.raise_()
            self.settings.activateWindow()

        def _refresh_cost(self):
            self._params = self.settings.apply_to(self._params)
            data = self._data
            if data is None or self._dose_gy is None or data.get("masks") is None:
                text = "Import a study to evaluate the cost J."
            else:
                J, breakdown = cost_function(
                    self._dose_gy, data["masks"], data["ctv_names"], data["oar_names"],
                    self._params,
                )
                plan = "optimized" if self._result is not None else "uniform start"
                text = f"Cost  J = {J:.2f}    ({len(breakdown)} structures, {plan})"
            self.settings.cost_label.setText(text)
            self._cost_status.setText(text)

        def result_params(self):
            return self._params

        def closeEvent(self, event):
            if self._busy():
                event.ignore()
                return
            self._params = self.settings.apply_to(self._params)
            self.settings.close()
            if self._dwell_dialog is not None:
                self._dwell_dialog.close()
            for view in self._views:
                view.finalize()
            super().closeEvent(event)

    return PlanningWindow
