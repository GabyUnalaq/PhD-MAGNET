import math
from typing import Optional

import pymunk
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QPolygonF, QFont
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene

from ..theme import (
    CELL_SIZE as SCALE,
    COLOR_OBSTACLE, COLOR_SIM_BG, COLOR_SIM_WALL, COLOR_FINISH,
    COLOR_AGENT, COLOR_AGENT_OUT, COLOR_HEADING, COLOR_TRAJECTORY,
)



class SimView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(COLOR_SIM_BG))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)

        self._sim = None

        # Replay mode state
        self._run: Optional[object] = None   # SimulationRun when in replay mode
        self._frame: int = 0

        # Pan state
        self._panning = False
        self._pan_start = QPointF()

        # Zoom limits
        self._zoom_factor = 1.15
        self._zoom_min = 0.05
        self._zoom_max = 20.0

    # Public API ----------------------------------------------------------------

    def set_simulator(self, sim):
        self._run = None
        self._sim = sim

    def set_run(self, run):
        """Switch to replay mode with a loaded SimulationRun."""
        self._sim = None
        self._run = run
        self._frame = 0

    def set_frame(self, frame: int):
        """Seek the replay to a specific frame (0-based). Triggers a repaint."""
        if self._run is None:
            return
        max_frame = max(len(r.trajectory) for r in self._run.agent_records) - 1
        self._frame = max(0, min(frame, max_frame))
        self.viewport().update()

    @property
    def max_frame(self) -> int:
        """Total number of replay frames available (0 when not in replay mode)."""
        if self._run is None:
            return 0
        return max(len(r.trajectory) for r in self._run.agent_records) - 1

    def fit_view(self, world_width: float, world_height: float):
        w_px = world_width * SCALE
        h_px = world_height * SCALE
        self._scene.setSceneRect(-SCALE, -SCALE, w_px + 2 * SCALE, h_px + 2 * SCALE)
        self.fitInView(QRectF(0, 0, w_px, h_px), Qt.KeepAspectRatio)

    def draw(self):
        """Refresh the rendered frame. Call once per simulation tick."""
        self.viewport().update()

    # Rendering -----------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._sim is not None:
            self._paint_shapes(painter)
            self._paint_agents(painter)
        elif self._run is not None:
            self._paint_map(painter, self._run.grid_map)
            self._paint_replay(painter)

    def _paint_shapes(self, painter: QPainter):
        # Skip shapes that belong to agent bodies — those are drawn by _paint_agents
        agent_bodies = {a.body for a in self._sim.agents if a.body is not None}
        for shape in self._sim.space.shapes:
            if shape.body in agent_bodies:
                continue
            if isinstance(shape, pymunk.Segment):
                self._draw_segment(painter, shape)
            elif isinstance(shape, pymunk.Poly):
                self._draw_poly(painter, shape)
            elif isinstance(shape, pymunk.Circle):
                self._draw_circle(painter, shape)

    def _draw_segment(self, painter: QPainter, seg: pymunk.Segment):
        width = max(1.0, seg.radius * 2 * SCALE)
        pen = QPen(COLOR_SIM_WALL, width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        a = seg.body.local_to_world(seg.a)
        b = seg.body.local_to_world(seg.b)
        painter.drawLine(QPointF(a.x * SCALE, a.y * SCALE),
                         QPointF(b.x * SCALE, b.y * SCALE))

    def _draw_poly(self, painter: QPainter, poly: pymunk.Poly):
        verts = [poly.body.local_to_world(v) for v in poly.get_vertices()]
        polygon = QPolygonF([QPointF(v.x * SCALE, v.y * SCALE) for v in verts])
        painter.setPen(QPen(COLOR_SIM_WALL, 1))
        painter.setBrush(QBrush(COLOR_OBSTACLE))
        painter.drawPolygon(polygon)

    def _draw_circle(self, painter: QPainter, circle: pymunk.Circle):
        pos = circle.body.local_to_world(circle.offset)
        r = circle.radius * SCALE
        painter.setPen(QPen(COLOR_SIM_WALL, 1))
        painter.setBrush(QBrush(COLOR_OBSTACLE))
        painter.drawEllipse(QPointF(pos.x * SCALE, pos.y * SCALE), r, r)

    def _paint_agents(self, painter: QPainter):
        for agent in self._sim.agents:
            if agent.body is None:
                continue
            self._draw_trajectory(painter, agent)
            self._draw_goal(painter, agent)
            self._draw_agent(painter, agent)

    def _draw_trajectory(self, painter: QPainter, agent):
        traj = agent.trajectory
        if len(traj) < 2:
            return
        pen = QPen(COLOR_TRAJECTORY, 1, Qt.DotLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        pts = list(traj)  # trajectory is stored in world coordinates
        for i in range(1, len(pts)):
            painter.drawLine(QPointF(pts[i - 1][0] * SCALE, pts[i - 1][1] * SCALE),
                             QPointF(pts[i][0] * SCALE, pts[i][1] * SCALE))

    def _draw_goal(self, painter: QPainter, agent):
        gx = agent.goal_pos[0] * SCALE
        gy = agent.goal_pos[1] * SCALE
        r = 0.2 * SCALE
        painter.setPen(QPen(COLOR_FINISH.darker(120), 1))
        painter.setBrush(QBrush(COLOR_FINISH))
        painter.drawRect(QRectF(gx - r, gy - r, 2 * r, 2 * r))
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", max(6, int(r * 0.9)), QFont.Bold))
        painter.drawText(QRectF(gx - r, gy - r, 2 * r, 2 * r),
                         Qt.AlignCenter, str(agent.id + 1))

    def _draw_agent(self, painter: QPainter, agent):
        pos = agent.body.position
        angle = agent.body.angle
        r = 0.3 * SCALE
        cx, cy = pos.x * SCALE, pos.y * SCALE

        painter.setPen(QPen(COLOR_AGENT_OUT, 1))
        painter.setBrush(QBrush(COLOR_AGENT))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        if agent.kinematic.heading is not None:
            painter.setPen(QPen(COLOR_HEADING, 2))
            painter.drawLine(QPointF(cx, cy),
                             QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))

        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", max(6, int(r * 0.8)), QFont.Bold))
        painter.drawText(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                         Qt.AlignCenter, str(agent.id + 1))

    # Replay rendering ----------------------------------------------------------

    def _paint_map(self, painter: QPainter, grid_map):
        """Draw obstacle cells from a GridMap (used in replay mode)."""
        w, h = grid_map.size
        for row in range(h):
            for col in range(w):
                if grid_map.grid[row, col] == 1:
                    painter.setPen(QPen(COLOR_SIM_WALL, 1))
                    painter.setBrush(QBrush(COLOR_OBSTACLE))
                    painter.drawRect(QRectF(col * SCALE, row * SCALE, SCALE, SCALE))

    def _paint_replay(self, painter: QPainter):
        """Draw all agents at the current replay frame."""
        frame = self._frame
        for record in self._run.agent_records:
            traj = record.trajectory          # (N, 2) world coords
            goal = record.config.goal_pos

            # Trajectory up to current frame
            if frame > 0:
                pts = traj[:frame + 1]
                pen = QPen(COLOR_TRAJECTORY, 1, Qt.DotLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                for i in range(1, len(pts)):
                    painter.drawLine(
                        QPointF(pts[i - 1][0] * SCALE, pts[i - 1][1] * SCALE),
                        QPointF(pts[i][0] * SCALE, pts[i][1] * SCALE),
                    )

            # Goal marker
            agent_id = self._run.agent_records.index(record)
            gx, gy = goal[0] * SCALE, goal[1] * SCALE
            r = 0.2 * SCALE
            painter.setPen(QPen(COLOR_FINISH.darker(120), 1))
            painter.setBrush(QBrush(COLOR_FINISH))
            painter.drawRect(QRectF(gx - r, gy - r, 2 * r, 2 * r))
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont("Arial", max(6, int(r * 0.9)), QFont.Bold))
            painter.drawText(QRectF(gx - r, gy - r, 2 * r, 2 * r),
                             Qt.AlignCenter, str(agent_id + 1))

            # Agent circle at current frame position
            idx = min(frame, len(traj) - 1)
            wx, wy = traj[idx][0] * SCALE, traj[idx][1] * SCALE
            ar = 0.3 * SCALE
            painter.setPen(QPen(COLOR_AGENT_OUT, 1))
            painter.setBrush(QBrush(COLOR_AGENT))
            painter.drawEllipse(QPointF(wx, wy), ar, ar)
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont("Arial", max(6, int(ar * 0.8)), QFont.Bold))
            painter.drawText(QRectF(wx - ar, wy - ar, 2 * ar, 2 * ar),
                             Qt.AlignCenter, str(agent_id + 1))

    # Input events --------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
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
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        if new_scale < self._zoom_min or new_scale > self._zoom_max:
            return
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        event.accept()
