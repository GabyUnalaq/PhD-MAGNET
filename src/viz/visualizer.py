import os

from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QGraphicsView, QPlainTextEdit, QTextBrowser,
    QPushButton, QSlider, QLabel, QSpinBox, QComboBox, QRadioButton,
    QAction, QFileDialog, QMessageBox,
)

from .sim_view import SimView

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SIM_DT           = 1.0 / 60.0
TICK_INTERVAL_MS = 16            # ~60 fps

_POLICY_MAP = {
    "Empty":          "NoPolicy",
    "Direct to goal": "DirectToGoalPolicy",
    "A*":             "AStarPolicy",
}


class VisualizerWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(os.path.join(CURRENT_DIR, "vizualiser.ui"), self)

        # ── Replace placeholder QGraphicsView with SimView ────────────────────
        placeholder = self.findChild(QGraphicsView, "viewMain")
        self.simView = SimView(self)
        placeholder.parentWidget().layout().replaceWidget(placeholder, self.simView)
        placeholder.deleteLater()

        # ── Actions ───────────────────────────────────────────────────────────
        self._actionLoadRun = self.findChild(QAction, "actionLoadRun")
        self._actionSaveRun = self.findChild(QAction, "actionSaveRun")
        self._actionFitView = self.findChild(QAction, "actionFitView")

        # ── Timeline controls ─────────────────────────────────────────────────
        self._buttPlay      = self.findChild(QPushButton, "buttPlay")
        self._buttPause     = self.findChild(QPushButton, "buttPause")
        self._buttReset     = self.findChild(QPushButton, "buttReset")
        self._buttFwd       = self.findChild(QPushButton, "buttStepForward")
        self._buttBack      = self.findChild(QPushButton, "buttStepBackwards")
        self._slider        = self.findChild(QSlider,     "timelineSlider")
        self._timeLabel     = self.findChild(QLabel,      "timeLabel")
        self._stepLabel     = self.findChild(QLabel,      "stepLabel")

        # ── Load Sim tab ──────────────────────────────────────────────────────
        self._browseBut     = self.findChild(QPushButton,  "pushButton")
        self._runSimBut     = self.findChild(QPushButton,  "buttRunSim")
        self._pathLabel     = self.findChild(QLabel,       "label_3")
        self._spinAgents    = self.findChild(QSpinBox,     "spinBox")
        self._comboPolicy   = self.findChild(QComboBox,    "comboBox")
        self._radioFull     = self.findChild(QRadioButton, "radioKnowledgeFull")
        self._radioLocal    = self.findChild(QRadioButton, "radioKnowledgeLocal")

        # ── Info / debug ──────────────────────────────────────────────────────
        self._textBrowser   = self.findChild(QTextBrowser,    "textBrowser")
        self._debugBox      = self.findChild(QPlainTextEdit,   "boxDebug")

        # ── Internal state ────────────────────────────────────────────────────
        self._sim          = None   # Simulator  (live mode)
        self._run          = None   # SimulationRun (replay mode)
        self._grid_map     = None   # GridMap last loaded via Browse
        self._map_path     = None   # path to that map
        self._sim_dt       = SIM_DT
        self._replay_frame = 0

        # ── Timer (live step + replay auto-play) ──────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # ── Connect signals ───────────────────────────────────────────────────
        self._actionLoadRun.triggered.connect(self._on_load_run)
        self._actionSaveRun.triggered.connect(self._on_save_run)
        self._actionFitView.triggered.connect(self._on_fit_view)

        self._browseBut.clicked.connect(self._on_browse_map)
        self._runSimBut.clicked.connect(self._on_run_sim)

        self._buttPlay.clicked.connect(self._on_play)
        self._buttPause.clicked.connect(self._on_pause)
        self._buttReset.clicked.connect(self._on_reset)
        self._buttFwd.clicked.connect(self._on_step_forward)
        self._buttBack.clicked.connect(self._on_step_backward)
        self._slider.valueChanged.connect(self._on_slider_changed)

        # ── Initial state ─────────────────────────────────────────────────────
        self._radioFull.setChecked(True)
        self._update_labels(0, 0.0)

    # ── Public API ───────────────────────────────────────────────────────────────

    def set_simulator(self, sim, dt: float = SIM_DT):
        """Attach a pre-built Simulator. Does not start the loop."""
        self._timer.stop()
        self._run    = None
        self._sim    = sim
        self._sim_dt = dt
        self.simView.set_simulator(sim)

    def start(self):
        """Start (or resume) the simulation / replay loop."""
        self._timer.start(TICK_INTERVAL_MS)

    def stop(self):
        """Pause the loop."""
        self._timer.stop()

    def log(self, message: str):
        self._debugBox.appendPlainText(message)

    # ── Timer callback ────────────────────────────────────────────────────────

    def _on_tick(self):
        if self._sim is not None:
            self._sim.step(self._sim_dt)
            step = self._sim._step_count
            self._update_labels(step, step * self._sim_dt)
            self.simView.draw()
            if self._sim.is_done:
                self._timer.stop()
                self.log(f"Simulation stopped at step {step} — condition met.")

        elif self._run is not None:
            next_frame = self._replay_frame + 1
            if next_frame > self.simView.max_frame:
                self._timer.stop()
                return
            self._seek(next_frame)

    # ── Timeline slots ────────────────────────────────────────────────────────

    def _on_play(self):
        self._timer.start(TICK_INTERVAL_MS)

    def _on_pause(self):
        self._timer.stop()

    def _on_reset(self):
        self._timer.stop()
        if self._run is not None:
            self._seek(0)
            return
        if self._sim is not None:
            cfg = self._sim._config
            gm  = self._sim._grid_map
            if cfg and gm:
                self._sim.setup(cfg, grid_map=gm)
                self._update_labels(0, 0.0)
                self.simView.draw()

    def _on_step_forward(self):
        if self._run is not None:
            self._seek(self._replay_frame + 1)
        elif self._sim is not None:
            self._sim.step(self._sim_dt)
            step = self._sim._step_count
            self._update_labels(step, step * self._sim_dt)
            self.simView.draw()

    def _on_step_backward(self):
        if self._run is not None:
            self._seek(self._replay_frame - 1)

    def _on_slider_changed(self, value: int):
        if self._run is None:
            return
        max_frame = self.simView.max_frame
        frame = int(value / 1000 * max_frame)
        self._replay_frame = frame
        self.simView.set_frame(frame)
        dt = self._run.config.dt
        self._update_labels(frame, frame * dt)

    # ── Load Sim tab slots ────────────────────────────────────────────────────

    def _on_run_sim(self):
        if self._grid_map is None:
            self.log("Browse a map first.")
            return
        if self._sim is not None or self._run is not None:
            answer = QMessageBox.question(
                self,
                "Replace simulation?",
                "A simulation is already loaded. Replace it with the new one?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._timer.stop()
        self._run = None
        self._build_and_start()

    def _on_browse_map(self):
        from ..map._grid_map import GridMap
        from ..map import MAPS_FILES_DIR, EXT

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Map", MAPS_FILES_DIR, f"Map Files (*.{EXT})"
        )
        if not path:
            return
        try:
            grid_map = GridMap.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load map:\n{e}")
            return

        self._map_path  = path
        self._grid_map  = grid_map
        n_pairs = len(grid_map.linked_points)
        self._pathLabel.setText(os.path.basename(path))
        self._spinAgents.setRange(1, max(1, n_pairs))
        self._spinAgents.setValue(n_pairs)
        self.log(f"Map loaded: {grid_map.name}  ({grid_map.size[0]}×{grid_map.size[1]})  |  {n_pairs} pair(s)")

    # ── Toolbar action slots ──────────────────────────────────────────────────

    def _on_load_run(self):
        from ..simulator import RUN_FILES_DIR, RUN_EXT
        from ..simulation.run import SimulationRun

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Run", RUN_FILES_DIR, f"Run Files (*.{RUN_EXT})"
        )
        if not path:
            return
        try:
            run = SimulationRun.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load run:\n{e}")
            return

        self._timer.stop()
        self._sim = None
        self._run = run
        self.simView.set_run(run)
        gm = run.grid_map
        self.simView.fit_view(gm.size[0], gm.size[1])
        self._seek(0)
        self.log(f"Run loaded: {os.path.basename(path)}  |  {len(run.agent_records)} agent(s)  |  {self.simView.max_frame} frames")
        self._textBrowser.setPlainText(run.summary())

    def _on_save_run(self):
        from ..simulator import RUN_FILES_DIR, RUN_EXT

        if self._sim is None:
            self.log("No live simulation to save.")
            return
        try:
            run = self._sim.finalize()
        except RuntimeError as e:
            QMessageBox.warning(self, "Save Error", str(e))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Run", RUN_FILES_DIR, f"Run Files (*.{RUN_EXT})"
        )
        if not path:
            return
        if not path.endswith(f".{RUN_EXT}"):
            path += f".{RUN_EXT}"
        try:
            run.save(path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save run:\n{e}")
            return

        self.log(f"Run saved → {os.path.basename(path)}")
        self._textBrowser.setPlainText(run.summary())

    def _on_fit_view(self):
        gm = (self._grid_map
              or (self._sim._grid_map if self._sim else None)
              or (self._run.grid_map  if self._run else None))
        if gm:
            self.simView.fit_view(gm.size[0], gm.size[1])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_and_start(self):
        from ..simulator import Simulator
        from ..simulation.run import AgentConfig, AlgorithmConfig, RunConfig

        grid_map    = self._grid_map
        n_use       = self._spinAgents.value()
        policy_name = _POLICY_MAP.get(self._comboPolicy.currentText(), "DirectToGoalPolicy")
        full_map    = self._radioFull.isChecked()

        agents = [
            AgentConfig(
                agent_type="SimpleAgent",
                start_pos=(float(lp[1]), float(lp[0])),
                goal_pos =(float(lp[3]), float(lp[2])),
                algorithm=AlgorithmConfig(policy_name),
                full_map_access=full_map,
            )
            for lp in grid_map.linked_points[:n_use]
        ]
        config = RunConfig.from_map(self._map_path, agents, dt=self._sim_dt)
        sim = Simulator(headless=True)
        sim.setup(config, grid_map=grid_map)

        w, h = grid_map.size
        self.set_simulator(sim)
        self.simView.fit_view(w, h)
        self._timer.start(TICK_INTERVAL_MS)
        self.log(f"Started  |  {n_use} agent(s)  |  policy: {policy_name}  |  full_map: {full_map}")

    def _seek(self, frame: int):
        """Set replay frame and sync slider + labels."""
        max_frame = self.simView.max_frame
        frame = max(0, min(frame, max_frame))
        self._replay_frame = frame
        self.simView.set_frame(frame)
        dt = self._run.config.dt if self._run else self._sim_dt
        self._update_labels(frame, frame * dt)
        if max_frame > 0:
            slider_val = int(frame / max_frame * 1000)
            self._slider.blockSignals(True)
            self._slider.setValue(slider_val)
            self._slider.blockSignals(False)

    def _update_labels(self, step: int, time_s: float):
        self._timeLabel.setText(f"Time: {time_s:.2f}s")
        self._stepLabel.setText(f"Step: {step}")
