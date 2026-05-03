"""
Interactive preview for generate_structured_grid_map.
Adjust parameters and click Generate (or press Enter) to see the result.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QCheckBox, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

from src.map.generator.grid_map_generator import generate_structured_grid_map


OBSTACLE_COLOR = QColor(40, 40, 40)
FREE_COLOR = QColor(220, 220, 220)


def grid_to_pixmap(grid: np.ndarray, cell_size: int = 10) -> QPixmap:
    h, w = grid.shape
    img = QImage(w * cell_size, h * cell_size, QImage.Format_RGB32)
    painter = QPainter(img)
    for row in range(h):
        for col in range(w):
            color = OBSTACLE_COLOR if grid[row, col] == 1 else FREE_COLOR
            painter.fillRect(col * cell_size, row * cell_size, cell_size, cell_size, color)
    painter.end()
    return QPixmap.fromImage(img)


class ParamSpinBox(QDoubleSpinBox):
    def __init__(self, min_val, max_val, step, value, decimals=2):
        super().__init__()
        self.setDecimals(decimals)
        self.setRange(min_val, max_val)
        self.setSingleStep(step)
        self.setValue(value)
        self.setFixedWidth(90)


class IntSpinBox(QSpinBox):
    def __init__(self, min_val, max_val, value):
        super().__init__()
        self.setRange(min_val, max_val)
        self.setValue(value)
        self.setFixedWidth(90)


class MapGeneratorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Generator Preview")
        self._build_ui()
        self._generate()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(16)

        # --- Left panel: controls ---
        controls = QWidget()
        controls.setFixedWidth(220)
        left = QVBoxLayout(controls)
        left.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)

        self.sp_width   = IntSpinBox(5, 500, 50)
        self.sp_height  = IntSpinBox(5, 500, 50)
        self.sp_scale   = ParamSpinBox(0.5, 100.0, 0.5, 5.0)
        self.sp_octaves = IntSpinBox(1, 8, 3)
        self.sp_thresh  = ParamSpinBox(-1.0, 1.0, 0.01, 0.05)
        self.sp_birth   = IntSpinBox(1, 8, 4)
        self.sp_death   = IntSpinBox(1, 8, 3)
        self.cb_rand    = QCheckBox("Random seed")
        self.cb_rand.setChecked(True)
        self.sp_seed    = IntSpinBox(0, 99999, 0)
        self.sp_seed.setEnabled(False)

        self.cb_rand.stateChanged.connect(lambda s: self.sp_seed.setEnabled(not s))

        form.addRow("Width:", self.sp_width)
        form.addRow("Height:", self.sp_height)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken)
        form.addRow(sep)

        form.addRow("Scale:", self.sp_scale)
        form.addRow("Octaves:", self.sp_octaves)
        form.addRow("Threshold:", self.sp_thresh)
        form.addRow("Birth limit:", self.sp_birth)
        form.addRow("Death limit:", self.sp_death)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setFrameShadow(QFrame.Sunken)
        form.addRow(sep2)

        form.addRow(self.cb_rand)
        form.addRow("Seed:", self.sp_seed)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color: gray; font-size: 11px;")

        btn_generate = QPushButton("Generate  [Enter]")
        btn_generate.setDefault(True)
        btn_generate.clicked.connect(self._generate)

        left.addLayout(form)
        left.addSpacing(4)
        left.addWidget(self.lbl_info)
        left.addStretch()
        left.addWidget(btn_generate)

        # --- Right panel: map display ---
        self.map_label = QLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setStyleSheet("background: #111; border: 1px solid #444;")
        self.map_label.setMinimumSize(400, 400)

        root.addWidget(controls)
        root.addWidget(self.map_label, stretch=1)

    def _generate(self):
        if self.cb_rand.isChecked():
            seed = int(np.random.randint(0, 99999))
            self.sp_seed.blockSignals(True)
            self.sp_seed.setValue(seed)
            self.sp_seed.blockSignals(False)
        else:
            seed = self.sp_seed.value()

        grid_map = generate_structured_grid_map(
            width=self.sp_width.value(),
            height=self.sp_height.value(),
            scale=self.sp_scale.value(),
            octaves=self.sp_octaves.value(),
            threshold=self.sp_thresh.value(),
            birth_limit=self.sp_birth.value(),
            death_limit=self.sp_death.value(),
            seed=seed,
        )
        grid = grid_map.grid

        density = grid.sum() / grid.size * 100
        auto_iter = max(2, round(max(self.sp_width.value(), self.sp_height.value()) / 12))
        self.lbl_info.setText(f"Obstacles: {density:.1f}%  |  iterations: {auto_iter}")

        available = self.map_label.size()
        cell_size = max(1, min(
            available.width() // grid.shape[1],
            available.height() // grid.shape[0],
        ))
        pixmap = grid_to_pixmap(grid, cell_size=cell_size)
        self.map_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._generate()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-render at new cell size when window is resized — re-generate to update
        self._generate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapGeneratorWindow()
    window.resize(800, 560)
    window.show()
    sys.exit(app.exec_())
