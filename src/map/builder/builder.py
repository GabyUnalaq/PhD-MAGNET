import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from . import resources_rc

from .. import MAPS_FILES_DIR, EXT
from .map_view import MapGraphicsView
from .utils import *
from .._base_map import *
from .._grid_map import GridMap

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class MapBuilderWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MapBuilderWindow, self).__init__(parent)
        uic.loadUi(os.path.join(CURRENT_DIR, "map_builder.ui"), self)

        self.setWindowTitle("MAGNET - Map Builder")
        self.setGeometry(100, 100, 800, 600)

        # Variables
        self.current_map: BaseMap = None
        self.object_selected: ObjectMarker = ObjectMarker.OBSTACLE

        # Actions
        self.qtLoadMapAction = self.findChild(QAction, "actionLoadMap")
        self.qtLoadTemplateAction = self.findChild(QAction, "actionLoadTemplate")
        self.qtSaveAction = self.findChild(QAction, "actionSave")

        # Widgets
        self.qtDebugBox = self.findChild(QPlainTextEdit, "boxDebug")
        self.qtNameBox = self.findChild(QLineEdit, "boxName")
        self.qtNatureCombo = self.findChild(QComboBox, "boxNature")  # Grid / Continuous
        self.qtTypeCombo = self.findChild(QComboBox, "boxType")  # Map / Template
        self.qtSizeW = self.findChild(QSpinBox, "boxWidth")
        self.qtSizeH = self.findChild(QSpinBox, "boxHeight")
        self.qtGenerateButton = self.findChild(QPushButton, "buttGenerateMap")
        self.qtUpdateButton = self.findChild(QPushButton, "buttUpdateMap")
        self.qtCheckButton = self.findChild(QPushButton, "buttCheckMap")

        self.qtObstacleRadio = self.findChild(QRadioButton, "rbuttObstacle")
        self.qtStartRadio = self.findChild(QRadioButton, "rbuttStart")
        self.qtFinishRadio = self.findChild(QRadioButton, "rbuttFinish")
        self.qtInterestRadio = self.findChild(QRadioButton, "rbuttInterest")
        self.qtEntranceRadio = self.findChild(QRadioButton, "rbuttEntrance")

        self.qtLinkButton = self.findChild(QPushButton, "buttLinkStartFinish")
        self.qtInterestCounter = self.findChild(QLineEdit, "boxInterestCounter")
        self.qtStartCounter = self.findChild(QLineEdit, "boxStartCounter")
        self.qtFinishCounter = self.findChild(QLineEdit, "boxFinishCounter")
        self.qtLinkedCounter = self.findChild(QLineEdit, "boxLinkedCounter")
        self.qtPtsWarnLabel = self.findChild(QLabel, "labelPtsWarn")
        self.qtLinkedWarnLabel = self.findChild(QLabel, "labelLinkedWarn")

        # Replace the placeholder QGraphicsView the custom MapGraphicsView
        placeholder_view = self.findChild(QGraphicsView, "viewMain")
        self.mapView = MapGraphicsView(self)
        layout = placeholder_view.parentWidget().layout()
        layout.replaceWidget(placeholder_view, self.mapView)
        placeholder_view.deleteLater()

        # Configuration Signals
        self.qtLoadMapAction.triggered.connect(self.on_load_map_clicked)
        self.qtLoadTemplateAction.triggered.connect(self.on_load_template_clicked)
        self.qtSaveAction.triggered.connect(self.on_save_map_clicked)

        self.qtGenerateButton.clicked.connect(self.on_generate_map_clicked)
        self.qtUpdateButton.clicked.connect(self.on_update_map_clicked)
        self.qtCheckButton.clicked.connect(self.on_check_map_clicked)
        self.qtLinkButton.clicked.connect(self.on_link_button_clicked)

        self.qtObstacleRadio.toggled.connect(lambda checked: self.on_marker_selected(ObjectMarker.OBSTACLE, checked))
        self.qtStartRadio.toggled.connect(lambda checked: self.on_marker_selected(ObjectMarker.START, checked))
        self.qtFinishRadio.toggled.connect(lambda checked: self.on_marker_selected(ObjectMarker.FINISH, checked))
        self.qtInterestRadio.toggled.connect(lambda checked: self.on_marker_selected(ObjectMarker.INTEREST, checked))
        self.qtEntranceRadio.toggled.connect(lambda checked: self.on_marker_selected(ObjectMarker.ENTRANCE, checked))

        self.mapView.markers_changed.connect(self._update_counters)

        # Initialize
        self._init_templates()
        self._refresh_ui()

    def log_debug(self, message: str):
        self.qtDebugBox.appendPlainText(message)
    
    def _init_templates(self):
        pass

    def _refresh_ui(self):
        if self.current_map is None:
            self._update_counters()
            return

        is_map: bool = self.current_map.map_type == MapType.MAP

        self.qtStartRadio.setEnabled(is_map)
        self.qtFinishRadio.setEnabled(is_map)
        self.qtLinkButton.setEnabled(is_map)
        self.qtEntranceRadio.setEnabled(not is_map)

        self.qtObstacleRadio.click()
        if self.qtLinkButton.isChecked():
            self.qtLinkButton.setChecked(False)
        
        self._update_counters()

    # Events --------------------------------------------------------------------
    def closeEvent(self, event):
        result = QMessageBox.question(self,
                                      "Confirm Exit...",
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes | QMessageBox.No)
        event.ignore()
        if result == QMessageBox.Yes:
            event.accept()

    # Slots ---------------------------------------------------------------------
    def on_load_map_clicked(self):
        if not self.check_overwrite_map():
            return
        self._load_file(MAPS_FILES_DIR, "Map")

    def on_load_template_clicked(self):
        QMessageBox.warning(self,
                            "Not implemented yet",
                            "Load Template feature not implemented yet.",
                            QMessageBox.OK)
        return

    def on_save_map_clicked(self):
        if not self.check_map_loaded():
            return
        if not self.current_map.check_validity(print_issues=False):
            QMessageBox.warning(self,
                                "Invalid Map",
                                "The current map is invalid and cannot be saved. Please check the debug log for details.",
                                QMessageBox.Ok)
            self.log_debug("Save operation aborted: Map is invalid.")
            return
        
        if self.current_map.map_type == MapType.TEMPLATE:
            QMessageBox.warning(self,
                                "Not implemented yet",
                                "Save Template feature not implemented yet.",
                                QMessageBox.Ok)
            self.log_debug("Save operation aborted: Map is invalid.")
            return

        if os.path.exists(os.path.join(MAPS_FILES_DIR, f"{self.current_map.name}.{EXT}")):
            result = QMessageBox.question(self,
                                          "Overwrite Map?",
                                          f"A map named '{self.current_map.name}' already exists. Do you want to overwrite it?",
                                          QMessageBox.Yes | QMessageBox.No)
            if result != QMessageBox.Yes:
                self.log_debug("Save operation cancelled by user.")
                return

        self.current_map.save()
        self.log_debug(f"Map '{self.current_map.name}' saved successfully.")

    def on_generate_map_clicked(self):
        if not self.check_overwrite_map():
            return
        
        name = self.qtNameBox.text().strip()
        if not self.check_map_name(name):
            return
        
        map_kind = MapKind.from_string(self.qtNatureCombo.currentText())
        map_type = MapType.from_string(self.qtTypeCombo.currentText())
        width = self.qtSizeW.value()
        height = self.qtSizeH.value()

        if map_type == MapType.TEMPLATE:
            print("For templates, the size is fixed to 5x5 for now.")
            width = 5
            height = 5
            self.qtSizeW.setValue(5)
            self.qtSizeW.setDisabled(True)
            self.qtSizeH.setValue(5)
            self.qtSizeH.setDisabled(True)
        else:
            self.qtSizeW.setDisabled(False)
            self.qtSizeH.setDisabled(False)

        if map_kind == MapKind.GRID:
            self.current_map = GridMap(map_type, width, height, name)
            self.log_debug(f"Generated new Grid {map_type.name.lower()}: {name} ({width}x{height})")
        else:
            self.log_debug("Map kind not implemented yet.")
            return

        self.mapView.set_map(self.current_map)
        self._refresh_ui()
    
    def on_update_map_clicked(self):
        if not self.check_map_loaded():
            return

        new_width = self.qtSizeW.value()
        new_height = self.qtSizeH.value()
        new_map_type = self.qtTypeCombo.currentText()
        new_name = self.qtNameBox.text().strip()
        if not self.check_map_name(new_name):
            return

        if self.current_map.map_type != MapType.from_string(new_map_type):
            QMessageBox.warning(self,
                                "Map Type Change Not Allowed",
                                "Changing the map type of an existing map is not supported.",
                                QMessageBox.Ok)
            return

        if new_name != self.current_map.name:
            self.current_map.name = new_name
            self.log_debug(f"Updated map name to: {new_name}")

        if (new_width, new_height) != self.current_map.size:
            self.current_map.update_size(new_width, new_height)
            self.mapView.draw_map()
            self.log_debug(f"Updated map size to: {new_width}x{new_height}")

    def on_check_map_clicked(self):
        if not self.check_map_loaded():
            return

        print("# Checking map validity...")
        valid = self.current_map.check_validity()
        if valid:
            QMessageBox.information(self,
                                    "Map Validity Check",
                                    "The current map is valid.",
                                    QMessageBox.Ok)
            self.log_debug("Map validity check passed.")
        else:
            QMessageBox.warning(self,
                                "Map Validity Check",
                                "The current map is invalid. Please check the debug log for details.",
                                QMessageBox.Ok)
            self.log_debug("Map validity check failed.")

    def on_link_button_clicked(self):
        checked = self.qtLinkButton.isChecked()
        self.mapView.set_link_mode(checked)
        self.log_debug(f"Link mode {'enabled' if checked else 'disabled'}.")

    def on_marker_selected(self, marker: ObjectMarker, checked: bool):
        if checked:
            self.object_selected = marker
            self.mapView.set_object_mode(marker)
            self.log_debug(f"Selected: {marker.name}")

    def _update_counters(self):
        if self.current_map is None:
            self.qtPtsWarnLabel.setVisible(False)
            self.qtLinkedWarnLabel.setVisible(False)
            return

        start_pts = len(self.current_map.start_points)
        finish_pts = len(self.current_map.finish_points)
        interest_pts = len(self.current_map.interest_points)
        linked_pts = len(self.current_map.linked_points)

        self.qtStartCounter.setText(str(start_pts))
        self.qtFinishCounter.setText(str(finish_pts))
        self.qtInterestCounter.setText(str(interest_pts))
        self.qtLinkedCounter.setText(str(linked_pts))

        if start_pts == finish_pts:
            self.qtPtsWarnLabel.setVisible(False)
        else:
            self.qtPtsWarnLabel.setVisible(True)
        
        if linked_pts == 0:
            self.qtLinkedWarnLabel.setToolTip("There should be at least a pair of linked start/finish points.")
            self.qtLinkedWarnLabel.setVisible(True)
        elif start_pts != 0 or finish_pts != 0:
            self.qtLinkedWarnLabel.setToolTip("All start/finish points should be linked.")
            self.qtLinkedWarnLabel.setVisible(True)
        else:
            self.qtLinkedWarnLabel.setVisible(False)

    # Utils ---------------------------------------------------------------------
    def _load_file(self, directory: str, label: str):
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Load {label}", directory, f"{label} Files (*.{EXT})")
        if not file_path:
            return
        try:
            loaded_map = GridMap.load(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}", QMessageBox.Ok)
            return

        self.current_map = loaded_map
        self.qtNameBox.setText(loaded_map.name)
        self.qtSizeW.setValue(loaded_map.size[0])
        self.qtSizeH.setValue(loaded_map.size[1])
        self.mapView.set_map(self.current_map)
        self._refresh_ui()
        self.log_debug(f"Loaded {label}: {loaded_map.name} ({loaded_map.size[0]}x{loaded_map.size[1]})")

    def check_overwrite_map(self) -> bool:
        if self.current_map is not None:
            result = QMessageBox.question(self,
                                          "Are you sure?",
                                          "A map is already loaded, if not saved it will be lost. Continue?",
                                          QMessageBox.Yes | QMessageBox.No)
            return result == QMessageBox.Yes
        return True

    def check_map_name(self, name) -> bool:
        if name == "":
            QMessageBox.warning(self,
                                "Invalid Name",
                                "Please provide a valid name for the map.",
                                QMessageBox.Ok)
            return False
        return True

    def check_map_loaded(self) -> bool:
        if self.current_map is None:
            QMessageBox.warning(self,
                                "No Map Loaded",
                                "Please generate or load a map first.",
                                QMessageBox.Ok)
            return False
        return True

