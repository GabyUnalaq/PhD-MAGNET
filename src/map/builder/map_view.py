import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QBrush, QPen, QFont
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem

from .utils import *
from .._base_map import BaseMap


class MapGraphicsView(QGraphicsView):
    markers_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QBrush(Qt.white))

        # Map reference
        self._map: BaseMap = None

        # Visual item storage
        self._cell_items: list[list[QGraphicsRectItem]] = []
        self._marker_items: dict[tuple, QGraphicsTextItem] = {}  # (marker_type, x, y) -> item
        self._link_lines: list[QGraphicsLineItem] = []

        # Interaction state
        self._object_mode: ObjectMarker = ObjectMarker.OBSTACLE
        self._link_mode: bool = False
        self._link_pending_start: tuple = None  # (gx, gy) of selected start
        self._link_highlight: QGraphicsRectItem = None

        # Pan state
        self._panning = False
        self._pan_start = QPointF()

        # Zoom limits
        self._zoom_factor = 1.15
        self._zoom_min = 0.1
        self._zoom_max = 10.0

    def set_object_mode(self, mode: ObjectMarker):
        self._object_mode = mode

    def set_link_mode(self, enabled: bool):
        self._link_mode = enabled
        self._link_pending_start = None
        self._clear_link_highlight()
        for line in self._link_lines:
            line.setVisible(enabled)

    def set_map(self, grid_map: BaseMap):
        self._map = grid_map
        self.draw_map()

    # Drawing -------------------------------------------------------------------
    def draw_map(self):
        if self._map is None:
            return

        self._scene.clear()
        self._cell_items = []
        self._marker_items = {}
        self._link_lines = []

        w, h = self._map.size  # w=width(cols), h=height(rows)
        pen = QPen(COLOR_GRID_LINE, 1)

        for row in range(h):
            row_items = []
            for col in range(w):
                x = col * CELL_SIZE
                y = row * CELL_SIZE
                val = self._map.grid[row, col] if hasattr(self._map, 'grid') else 0
                color = COLOR_OBSTACLE if val == 1 else COLOR_EMPTY
                rect = self._scene.addRect(x, y, CELL_SIZE, CELL_SIZE, pen, QBrush(color))
                row_items.append(rect)
            self._cell_items.append(row_items)

        # Draw existing markers
        self._draw_all_markers()

        self._scene.setSceneRect(0, 0, w * CELL_SIZE, h * CELL_SIZE)
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _update_cell(self, row: int, col: int):
        if not self._cell_items or row >= len(self._cell_items) or col >= len(self._cell_items[0]):
            return
        val = self._map.grid[row, col]
        color = COLOR_OBSTACLE if val == 1 else COLOR_EMPTY
        self._cell_items[row][col].setBrush(QBrush(color))

    def _draw_all_markers(self):
        # Unlinked start/finish with "?" suffix
        for pt in self._map.start_points:
            self._draw_marker(pt[0], pt[1], ObjectMarker.START, "S?")
        for pt in self._map.finish_points:
            self._draw_marker(pt[0], pt[1], ObjectMarker.FINISH, "F?")

        # Linked pairs with numbered labels and dashed lines
        for i, lp in enumerate(self._map.linked_points):
            n = i + 1
            self._draw_marker(lp[0], lp[1], ObjectMarker.START, f"S{n}")
            self._draw_marker(lp[2], lp[3], ObjectMarker.FINISH, f"F{n}")
            self._draw_link_line(lp[0], lp[1], lp[2], lp[3])

        # Interest points unchanged
        for pt in self._map.interest_points:
            self._draw_marker(pt[0], pt[1], ObjectMarker.INTEREST)

        # Link lines only visible in link mode
        for line in self._link_lines:
            line.setVisible(self._link_mode)

    def _draw_marker(self, gx: float, gy: float, marker_type: ObjectMarker, label: str = None):
        key = (marker_type, gx, gy)
        if key in self._marker_items:
            return  # already drawn

        if label is None:
            label, color = MARKER_DESIGN.get(marker_type, ("?", COLOR_INTEREST))
        else:
            _, color = MARKER_DESIGN.get(marker_type, ("?", COLOR_INTEREST))

        text_item = QGraphicsTextItem(label)
        font = QFont("Arial", 12, QFont.Bold)
        text_item.setFont(font)
        text_item.setDefaultTextColor(color)

        # Position: gx/gy are in grid coords, convert to pixel and center the text
        px = gy * CELL_SIZE
        py = gx * CELL_SIZE
        br = text_item.boundingRect()
        text_item.setPos(px - br.width() / 2, py - br.height() / 2)
        text_item.setZValue(10)  # above grid cells

        self._scene.addItem(text_item)
        self._marker_items[key] = text_item

    def _draw_link_line(self, gx1: float, gy1: float, gx2: float, gy2: float):
        px1 = gy1 * CELL_SIZE
        py1 = gx1 * CELL_SIZE
        px2 = gy2 * CELL_SIZE
        py2 = gx2 * CELL_SIZE

        pen = QPen(COLOR_LINKED, 2, Qt.DashLine)
        line = self._scene.addLine(px1, py1, px2, py2, pen)
        line.setZValue(5)  # above cells, below markers
        self._link_lines.append(line)

    def _remove_marker(self, gx: float, gy: float, marker_type: ObjectMarker):
        key = (marker_type, gx, gy)
        item = self._marker_items.pop(key, None)
        if item is not None:
            self._scene.removeItem(item)

    def _redraw_markers(self):
        """Clear and redraw all markers and link lines (grid cells untouched)."""
        for item in self._marker_items.values():
            self._scene.removeItem(item)
        self._marker_items = {}
        for line in self._link_lines:
            self._scene.removeItem(line)
        self._link_lines = []
        self._clear_link_highlight()
        self._draw_all_markers()

    def _clear_link_highlight(self):
        if self._link_highlight is not None:
            self._scene.removeItem(self._link_highlight)
            self._link_highlight = None

    def _set_link_pending(self, gx: float, gy: float):
        self._link_pending_start = (gx, gy)
        self._clear_link_highlight()
        px = gy * CELL_SIZE
        py = gx * CELL_SIZE
        size = CELL_SIZE * 0.8
        self._link_highlight = self._scene.addRect(
            px - size / 2, py - size / 2, size, size,
            QPen(COLOR_LINKED, 2), QBrush(Qt.transparent)
        )
        self._link_highlight.setZValue(8)

    def _find_in_array(self, points: np.ndarray, gx: float, gy: float):
        """Find point index in array by position, or None."""
        for i, pt in enumerate(points):
            if abs(pt[0] - gx) < 0.01 and abs(pt[1] - gy) < 0.01:
                return i
        return None

    def _find_linked(self, gx: float, gy: float, as_start: bool = True):
        """Find index in linked_points where (gx,gy) matches start (cols 0,1) or finish (cols 2,3)."""
        offset = 0 if as_start else 2
        for i, lp in enumerate(self._map.linked_points):
            if abs(lp[offset] - gx) < 0.01 and abs(lp[offset + 1] - gy) < 0.01:
                return i
        return None

    # Interaction ---------------------------------------------------------------
    def mousePressEvent(self, event):
        # Middle button: start panning
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        # Left button: interact with map
        if event.button() == Qt.LeftButton and self._map is not None:
            scene_pos = self.mapToScene(event.pos())
            self._handle_click(scene_pos)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle == 0:
            return

        factor = self._zoom_factor if angle > 0 else 1.0 / self._zoom_factor

        # Check zoom bounds
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        if new_scale < self._zoom_min or new_scale > self._zoom_max:
            return

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        event.accept()

    def _handle_click(self, scene_pos: QPointF):
        if self._link_mode:
            self._handle_link_click(scene_pos)
        elif self._object_mode == ObjectMarker.OBSTACLE:
            self._handle_obstacle_click(scene_pos)
        else:
            self._handle_marker_click(scene_pos)

    def _handle_link_click(self, scene_pos: QPointF):
        """ Handling clicks when the `Link` option is checked. """
        gx = round(scene_pos.y() / CELL_SIZE * 2) / 2
        gy = round(scene_pos.x() / CELL_SIZE * 2) / 2

        # Check what was clicked
        unlinked_start = self._find_in_array(self._map.start_points, gx, gy)
        linked_as_start = self._find_linked(gx, gy, as_start=True)
        unlinked_finish = self._find_in_array(self._map.finish_points, gx, gy)
        linked_as_finish = self._find_linked(gx, gy, as_start=False)

        is_start = unlinked_start is not None or linked_as_start is not None
        is_finish = unlinked_finish is not None or linked_as_finish is not None

        if is_start:
            # Break existing link if this start was linked
            if linked_as_start is not None:
                self._break_link(linked_as_start)
            self._set_link_pending(gx, gy)
            return

        if is_finish and self._link_pending_start is not None:
            # Break existing link if this finish was linked
            if linked_as_finish is not None:
                self._break_link(linked_as_finish)
            self._create_link(self._link_pending_start, (gx, gy))
            self._link_pending_start = None
            self._clear_link_highlight()
            return

    def _break_link(self, linked_idx: int):
        """ Break a link, moving both points back to unlinked arrays. """
        lp = self._map.linked_points[linked_idx]
        start_pt = np.array([[lp[0], lp[1]]], dtype=np.float32)
        finish_pt = np.array([[lp[2], lp[3]]], dtype=np.float32)

        self._map.linked_points = np.delete(self._map.linked_points, linked_idx, axis=0)

        sp = self._map.start_points
        self._map.start_points = np.vstack([sp, start_pt]) if len(sp) > 0 else start_pt
        fp = self._map.finish_points
        self._map.finish_points = np.vstack([fp, finish_pt]) if len(fp) > 0 else finish_pt

        self._redraw_markers()
        self.markers_changed.emit()

    def _create_link(self, start_pos: tuple, finish_pos: tuple):
        """ Link a start and finish point, removing them from unlinked arrays. """
        sgx, sgy = start_pos
        fgx, fgy = finish_pos

        # Remove from unlinked
        start_idx = self._find_in_array(self._map.start_points, sgx, sgy)
        if start_idx is not None:
            self._map.start_points = np.delete(self._map.start_points, start_idx, axis=0)
        finish_idx = self._find_in_array(self._map.finish_points, fgx, fgy)
        if finish_idx is not None:
            self._map.finish_points = np.delete(self._map.finish_points, finish_idx, axis=0)

        # Add to linked
        new_link = np.array([[sgx, sgy, fgx, fgy]], dtype=np.float32)
        lp = self._map.linked_points
        self._map.linked_points = np.vstack([lp, new_link]) if len(lp) > 0 else new_link

        self._redraw_markers()
        self.markers_changed.emit()

    def _handle_obstacle_click(self, scene_pos: QPointF):
        """ Handling clicks when the `Obstacle` marker is selected. """
        col = int(scene_pos.x() / CELL_SIZE)
        row = int(scene_pos.y() / CELL_SIZE)

        w, h = self._map.size  # w=width(cols), h=height(rows)
        if row < 0 or row >= h or col < 0 or col >= w:
            return

        # Toggle obstacle
        current = self._map.grid[row, col]
        self._map.grid[row, col] = 0 if current == 1 else 1
        self._update_cell(row, col)

    def _handle_marker_click(self, scene_pos: QPointF):
        """ Handling clicks when a interest (S/F/I) marker is selected. """
        # Snap to 0.5 grid
        gx = round(scene_pos.y() / CELL_SIZE * 2) / 2
        gy = round(scene_pos.x() / CELL_SIZE * 2) / 2

        if self._object_mode == ObjectMarker.ENTRANCE:
            print("Not yet implemented")
            return

        w, h = self._map.size  # w=width(cols), h=height(rows)
        if gx < 0 or gx > h or gy < 0 or gy > w:
            return

        marker_type = self._object_mode

        # Check if clicking on a linked marker — break link, remove this marker, orphan partner
        if marker_type in (ObjectMarker.START, ObjectMarker.FINISH):
            as_start = marker_type == ObjectMarker.START
            linked_idx = self._find_linked(gx, gy, as_start=as_start)
            if linked_idx is not None:
                lp = self._map.linked_points[linked_idx]
                self._map.linked_points = np.delete(self._map.linked_points, linked_idx, axis=0)
                # Orphan the partner into its unlinked array
                if as_start:
                    partner = np.array([[lp[2], lp[3]]], dtype=np.float32)
                    fp = self._map.finish_points
                    self._map.finish_points = np.vstack([fp, partner]) if len(fp) > 0 else partner
                else:
                    partner = np.array([[lp[0], lp[1]]], dtype=np.float32)
                    sp = self._map.start_points
                    self._map.start_points = np.vstack([sp, partner]) if len(sp) > 0 else partner
                self._redraw_markers()
                self.markers_changed.emit()
                return

        points_attr = {
            ObjectMarker.START: "start_points",
            ObjectMarker.FINISH: "finish_points",
            ObjectMarker.INTEREST: "interest_points"
        }[marker_type]

        points = getattr(self._map, points_attr)

        # Check if unlinked marker already exists at this position
        existing_idx = self._find_in_array(points, gx, gy)

        if existing_idx is not None:
            # Remove existing unlinked marker
            setattr(self._map, points_attr, np.delete(points, existing_idx, axis=0))
            self._remove_marker(gx, gy, marker_type)
        else:
            # Add new unlinked marker
            new_point = np.array([[gx, gy]], dtype=np.float32)
            setattr(self._map, points_attr, np.vstack([points, new_point]) if len(points) > 0 else new_point)
            if marker_type == ObjectMarker.START:
                self._draw_marker(gx, gy, marker_type, "S?")
            elif marker_type == ObjectMarker.FINISH:
                self._draw_marker(gx, gy, marker_type, "F?")
            else:
                self._draw_marker(gx, gy, marker_type)

        self.markers_changed.emit()
