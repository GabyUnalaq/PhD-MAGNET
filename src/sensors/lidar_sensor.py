import numpy as np
import pymunk
from .base_sensor import BaseSensor


class LidarSensor(BaseSensor):
    """
    Raycasting lidar sensor using Pymunk's built-in raycasting.

    Casts rays in a circular pattern around the agent and returns
    distances to the nearest obstacles.
    """

    def __init__(self, num_rays: int = 16, max_range: float = 5.0, angle_offset: float = 0.0):
        """
        Initialize the lidar sensor.

        Args:
            num_rays: Number of rays to cast (evenly distributed in 360°)
            max_range: Maximum sensing distance in world units
            angle_offset: Angular offset in radians (for non-centered sensors)
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.angle_offset = angle_offset

    def observe(self, space: pymunk.Space, agent_body: pymunk.Body) -> np.ndarray:
        """
        Cast rays and return distances to nearest obstacles.

        Args:
            space: Pymunk space containing all collision objects
            agent_body: Agent's body (provides position and heading)

        Returns:
            Array of shape (num_rays,) with distances [0, max_range]
            Returns max_range if no hit detected
        """
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)
        pos = agent_body.position
        heading = agent_body.angle

        for i in range(self.num_rays):
            # Calculate ray angle (evenly distributed around agent)
            angle = heading + self.angle_offset + (2 * np.pi * i / self.num_rays)

            # Ray endpoint
            end_x = pos.x + self.max_range * np.cos(angle)
            end_y = pos.y + self.max_range * np.sin(angle)
            end_pos = pymunk.Vec2d(end_x, end_y)

            # Raycast (returns first hit info)
            query_info = space.segment_query_first(pos, end_pos, 0, pymunk.ShapeFilter())

            if query_info is not None:
                # Calculate distance to hit point
                hit_point = query_info.point
                distance = pos.get_distance(hit_point)
                distances[i] = distance

        return distances


if __name__ == "__main__":
    # Simple example with PyQt visualization
    import sys
    from PyQt5.QtWidgets import QApplication, QWidget
    from PyQt5.QtCore import QTimer, QPointF
    from PyQt5.QtGui import QPainter, QColor, QPen
    import math

    class LidarDemo(QWidget):
        def __init__(self):
            super().__init__()
            self.setGeometry(100, 100, 800, 600)
            self.setMouseTracking(True)
            self.space = pymunk.Space()
            # Add walls
            b = self.space.static_body
            self.space.add(pymunk.Segment(b, (50,50), (750,50), 5),
                          pymunk.Segment(b, (750,50), (750,550), 5),
                          pymunk.Segment(b, (750,550), (50,550), 5),
                          pymunk.Segment(b, (50,550), (50,50), 5))
            # Add boxes
            for x, y in [(200,200), (500,150), (300,400)]:
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
                body.position = x, y
                self.space.add(body, pymunk.Poly(body, [(-40,-40), (40,-40), (40,40), (-40,40)]))

            self.lidar = LidarSensor(num_rays=36, max_range=300)
            self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            self.body.position = (400, 300)
            QTimer(self).timeout.connect(self.update)
            QTimer.singleShot(0, lambda: QTimer(self).start(16))

        def mouseMoveEvent(self, e):
            self.body.position = (e.pos().x(), e.pos().y())
            self.update()

        def paintEvent(self, e):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(240,240,245))
            # Draw shapes
            for s in self.space.shapes:
                if isinstance(s, pymunk.Segment):
                    p.setPen(QPen(QColor(100,100,100), 8))
                    p.drawLine(int(s.a.x), int(s.a.y), int(s.b.x), int(s.b.y))
                elif isinstance(s, pymunk.Poly):
                    p.setBrush(QColor(150,100,200,180))
                    p.drawPolygon(*[QPointF(v.x+s.body.position.x, v.y+s.body.position.y) for v in s.get_vertices()])
            # Draw lidar
            distances = self.lidar.observe(self.space, self.body)
            pos = self.body.position
            for i, d in enumerate(distances):
                angle = 2*math.pi*i/self.lidar.num_rays
                ex, ey = pos.x + d*math.cos(angle), pos.y + d*math.sin(angle)
                intensity = int(255*d/self.lidar.max_range)
                p.setPen(QPen(QColor(255-intensity, 50, intensity, 100), 1))
                p.drawLine(int(pos.x), int(pos.y), int(ex), int(ey))
                if d < self.lidar.max_range:
                    p.setPen(QPen(QColor(0,255,0), 4))
                    p.drawPoint(int(ex), int(ey))
            p.setPen(QPen(QColor(255,0,0), 8))
            p.drawPoint(int(pos.x), int(pos.y))

    app = QApplication(sys.argv)
    w = LidarDemo()
    w.show()
    sys.exit(app.exec_())