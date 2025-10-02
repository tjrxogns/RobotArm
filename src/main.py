#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Arm Viewer (v2.9, Matplotlib + PySide6)
- 각 관절(J1~J5) 슬라이더에 실제 로봇 서보 각도 표시
- 시뮬레이터 각도 ↔ 로봇 서보 각도 변환 테이블 포함
- 3D 뷰에 스크린 위치/사이즈를 UI로 지정하여 연한 하늘색 사각형으로 표시
"""

import sys
import math
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QSlider, QDoubleSpinBox, QLabel, QPushButton, QGroupBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ------------------------------ import from robot_arm ------------------------------
from robot_arm import RobotArm, ArmConfig, sim2robot

# ------------------------------ UI widgets ------------------------------

class SliderWithBox(QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, lo: float, hi: float, step: float, init: float, unit: str = "", parent=None):
        super().__init__(parent)
        self._scale = 100.0
        self.sld = QSlider(Qt.Horizontal)
        self.sld.setFixedWidth(180)
        self.sld.setMinimum(int(lo * self._scale))
        self.sld.setMaximum(int(hi * self._scale))
        self.sld.setSingleStep(int(step * self._scale))
        self.sld.setValue(int(init * self._scale))

        self.box = QDoubleSpinBox()
        self.box.setDecimals(2)
        self.box.setRange(lo, hi)
        self.box.setSingleStep(step)
        self.box.setValue(init)

        self.lbl = QLabel(unit)
        self.lbl_robot = QLabel("Robot: 000.0°")
        self.lbl_robot.setMinimumWidth(80)
        self.lbl_robot.setAlignment(Qt.AlignRight)

        lay = QHBoxLayout(self)
        lay.addWidget(self.sld)
        lay.addWidget(self.box)
        lay.addWidget(self.lbl)
        lay.addWidget(self.lbl_robot)
        lay.setContentsMargins(0,0,0,0)

        self.sld.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)

    def _from_slider(self, v: int):
        x = v / self._scale
        if abs(self.box.value() - x) > 1e-9:
            self.box.blockSignals(True)
            self.box.setValue(x)
            self.box.blockSignals(False)
        self.valueChanged.emit(x)

    def _from_box(self, x: float):
        v = int(round(x * self._scale))
        if self.sld.value() != v:
            self.sld.blockSignals(True)
            self.sld.setValue(v)
            self.sld.blockSignals(False)
        self.valueChanged.emit(float(x))

    def setValue(self, x: float):
        self._from_box(x)

    def value(self) -> float:
        return float(self.box.value())

# ------------------------------ Touch Screen widget ------------------------------

class TouchScreen(QGraphicsView):
    clicked = QtCore.Signal(float, float)

    def __init__(self, width=155, height=90, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setFixedSize(width+2, height+2)
        rect = QGraphicsRectItem(0, 0, width, height)
        rect.setBrush(Qt.lightGray)
        self.scene.addItem(rect)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)

# ------------------------------ Main Window ------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Arm Viewer (Matplotlib)")
        self.resize(1280, 800)

        self.cfg = ArmConfig()
        self.arm = RobotArm(self.cfg)

        # 초기 스크린 값
        self.screen_x0 = 60
        self.screen_y0 = -45
        self.screen_z0 = 0
        self.screen_w  = 90
        self.screen_h  = 155

        # Matplotlib Figure
        self.fig = Figure(figsize=(7.5, 6.25))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.toolbar = NavigationToolbar(self.canvas, self.canvas)

        self.joint_group = self._make_controls()

        self.ee_label = QLabel("")
        self.btn_home = QPushButton("Home pose")
        self.btn_home.clicked.connect(self._go_home)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setFixedSize(300, 50)
        self.btn_exit.clicked.connect(QApplication.quit)

        # XYZ 입력칸
        self.target_x = QDoubleSpinBox(); self.target_x.setRange(0, 200); self.target_x.setSuffix(" mm")
        self.target_y = QDoubleSpinBox(); self.target_y.setRange(-100, 100); self.target_y.setSuffix(" mm")
        self.target_z = QDoubleSpinBox(); self.target_z.setRange(0, 200); self.target_z.setSuffix(" mm")
        self.btn_move = QPushButton("Move to XYZ")
        self.btn_move.clicked.connect(self._move_to_xyz)

        xyz_layout = QHBoxLayout()
        xyz_layout.addWidget(QLabel("X:")); xyz_layout.addWidget(self.target_x)
        xyz_layout.addWidget(QLabel("Y:")); xyz_layout.addWidget(self.target_y)
        xyz_layout.addWidget(QLabel("Z:")); xyz_layout.addWidget(self.target_z)
        xyz_layout.addWidget(self.btn_move)

        # 📌 스크린 설정 입력칸
        self.screen_x_in = QDoubleSpinBox(); self.screen_x_in.setRange(-200,200); self.screen_x_in.setValue(self.screen_x0)
        self.screen_y_in = QDoubleSpinBox(); self.screen_y_in.setRange(-200,200); self.screen_y_in.setValue(self.screen_y0)
        self.screen_z_in = QDoubleSpinBox(); self.screen_z_in.setRange(0,200);   self.screen_z_in.setValue(self.screen_z0)
        self.screen_w_in = QDoubleSpinBox(); self.screen_w_in.setRange(10,400);  self.screen_w_in.setValue(self.screen_w)
        self.screen_h_in = QDoubleSpinBox(); self.screen_h_in.setRange(10,400);  self.screen_h_in.setValue(self.screen_h)
        self.btn_screen = QPushButton("Update Screen")
        self.btn_screen.clicked.connect(self._update_screen)

        # 좌표 (X,Y,Z) 한 줄
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("screenX0:")); coord_layout.addWidget(self.screen_x_in)
        coord_layout.addWidget(QLabel("screenY0:")); coord_layout.addWidget(self.screen_y_in)
        coord_layout.addWidget(QLabel("screenZ0:")); coord_layout.addWidget(self.screen_z_in)

        # 크기 (screenW, screenH) 한 줄
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("screenW:")); size_layout.addWidget(self.screen_w_in)
        size_layout.addWidget(QLabel("screenH:")); size_layout.addWidget(self.screen_h_in)

        # 전체 스크린 입력 레이아웃
        screen_layout = QVBoxLayout()
        screen_layout.addLayout(coord_layout)
        screen_layout.addLayout(size_layout)
        screen_layout.addWidget(self.btn_screen)


        # Touch screen
        self.touch_screen = TouchScreen(int(self.screen_w*2), int(self.screen_h*2))
        self.touch_screen.clicked.connect(self._on_screen_click)

        right = QVBoxLayout()
        right.addWidget(self.joint_group)
        right.addWidget(self.btn_home)
        right.addWidget(self.ee_label)
        right.addLayout(xyz_layout)
        right.addLayout(screen_layout)   # ✅ 새로 만든 레이아웃
        right.addWidget(self.touch_screen)
        right.addStretch(1)
        right.addWidget(self.btn_exit, alignment=Qt.AlignRight | Qt.AlignBottom)


        center = QWidget()
        main = QHBoxLayout(center)
        fig_col = QVBoxLayout()
        fig_col.addWidget(self.toolbar)
        fig_col.addWidget(self.canvas, 1)
        main.addLayout(fig_col, 2)
        main.addLayout(right, 1)
        self.setCentralWidget(center)

        self._elev = self.ax.elev
        self._azim = self.ax.azim

        self._redraw()
        self.canvas.mpl_connect("button_release_event", self._update_view)
        self.canvas.mpl_connect("motion_notify_event", self._update_view)

    # --- controls (sliders + buttons)
    def _make_controls(self) -> QGroupBox:
        gb = QGroupBox("Controls")
        fl = QFormLayout(gb)
        q = self.arm.q
        self.j1 = SliderWithBox(self.cfg.j1.soft_min, self.cfg.j1.soft_max, self.cfg.j1.resolution, q[0], unit="J1")
        self.j2 = SliderWithBox(self.cfg.j2.soft_min, self.cfg.j2.soft_max, self.cfg.j2.resolution, q[1], unit="J2")
        self.j3 = SliderWithBox(self.cfg.j3.soft_min, self.cfg.j3.soft_max, self.cfg.j3.resolution, q[2], unit="J3")
        self.j4 = SliderWithBox(self.cfg.j4.soft_min, self.cfg.j4.soft_max, self.cfg.j4.resolution, q[3], unit="J4")
        self.j5 = SliderWithBox(self.cfg.j5.soft_min, self.cfg.j5.soft_max, self.cfg.j5.resolution, q[4], unit="J5")
        fl.addRow("J1 (Base yaw)", self.j1)
        fl.addRow("J2 (Shoulder pitch)", self.j2)
        fl.addRow("J3 (Mid joint)", self.j3)
        fl.addRow("J4 (Elbow pitch)", self.j4)
        fl.addRow("J5 (Wrist roll, L3 axis)", self.j5)

        self.j1.valueChanged.connect(lambda v: self._on_joint_change(0, v))
        self.j2.valueChanged.connect(lambda v: self._on_joint_change(1, v))
        self.j3.valueChanged.connect(lambda v: self._on_joint_change(2, v))
        self.j4.valueChanged.connect(lambda v: self._on_joint_change(3, v))
        self.j5.valueChanged.connect(lambda v: self._on_joint_change(4, v))

        btn_x = QPushButton("View X-axis")
        btn_x.clicked.connect(self._set_view_x)
        btn_y = QPushButton("View Y-axis")
        btn_y.clicked.connect(self._set_view_y)
        btn_z = QPushButton("View Z-axis")
        btn_z.clicked.connect(self._set_view_z)
        fl.addRow(btn_x)
        fl.addRow(btn_y)
        fl.addRow(btn_z)
        return gb

    # --- joint handlers
    def _go_home(self):
        self.j1.setValue(self.cfg.j1.home); self._on_joint_change(0, self.cfg.j1.home)
        self.j2.setValue(self.cfg.j2.home); self._on_joint_change(1, self.cfg.j2.home)
        self.j3.setValue(self.cfg.j3.home); self._on_joint_change(2, self.cfg.j3.home)
        self.j4.setValue(self.cfg.j4.home); self._on_joint_change(3, self.cfg.j4.home)
        self.j5.setValue(self.cfg.j5.home); self._on_joint_change(4, self.cfg.j5.home)

    def _on_joint_change(self, idx: int, val: float):
        self.arm.q[idx] = float(val)
        self._update_robot_angle(idx, val)
        self._redraw()

    def _update_robot_angle(self, idx: int, sim_val: float):
        name = f"J{idx+1}"
        if name in self.cfg.servo_map:
            sim_min, sim_max, rob_min, rob_max = self.cfg.servo_map[name]
            rob_angle = sim2robot(sim_val, sim_min, sim_max, rob_min, rob_max)
            [self.j1, self.j2, self.j3, self.j4, self.j5][idx].lbl_robot.setText(f"Robot: {rob_angle:5.1f}°")

    def _move_to_xyz(self):
        # 간단한 IK (현재 L1,L2,L3만 사용)
        x, y, z = self.target_x.value(), self.target_y.value(), self.target_z.value()
        L1, L2, L3 = self.cfg.L1, self.cfg.L2, self.cfg.L3

        q1 = math.atan2(y, x)
        r_xy = math.hypot(x, y)
        x2 = r_xy - L3
        z2 = z - self.cfg.BaseHeight
        d = math.sqrt(x2**2 + z2**2)
        if d > (L1 + L2) or d < abs(L1 - L2):
            self.ee_label.setText("Out of reach")
            return

        cos_q3 = (d**2 - L1**2 - L2**2) / (2*L1*L2)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        q3 = math.acos(cos_q3)
        q2 = math.atan2(z2, x2) - math.atan2(L2*math.sin(q3), L1 + L2*math.cos(q3))

        q1_deg, q2_deg, q3_deg = map(math.degrees, (q1, q2, q3))
        self.j1.setValue(q1_deg); self._on_joint_change(0, q1_deg)
        self.j2.setValue(q2_deg); self._on_joint_change(1, q2_deg)
        self.j3.setValue(q3_deg); self._on_joint_change(2, q3_deg)

    # --- screen
    def _update_screen(self):
    # 입력값 반영
        self.screen_x0 = self.screen_x_in.value()
        self.screen_y0 = self.screen_y_in.value()
        self.screen_z0 = self.screen_z_in.value()
        self.screen_w = self.screen_w_in.value()
        self.screen_h = self.screen_h_in.value()

        # 기존 touch_screen 제거 후 새로 생성
        new_touch = TouchScreen(int(self.screen_w*2), int(self.screen_h*2))
        new_touch.clicked.connect(self._on_screen_click)

        # 레이아웃에서 기존 위젯 교체
        parent_layout = self.touch_screen.parent().layout()
        parent_layout.replaceWidget(self.touch_screen, new_touch)
        self.touch_screen.deleteLater()
        self.touch_screen = new_touch

        # 3D View 갱신
        self._redraw()


    def _draw_screen(self):
        X, Y = np.meshgrid(
            [self.screen_x0, self.screen_x0 + self.screen_w],
            [self.screen_y0, self.screen_y0 + self.screen_h]
        )
        Z = np.ones_like(X) * self.screen_z0
        self.ax.plot_surface(X, Y, Z, color='skyblue', alpha=0.3)

    def _on_screen_click(self, sx, sy):
        u = sx / self.touch_screen.width()
        v = sy / self.touch_screen.height()
        x = self.screen_x0 + u * self.screen_w
        y = self.screen_y0 + v * self.screen_h
        z = self.screen_z0
        self.target_x.setValue(x)
        self.target_y.setValue(y)
        self.target_z.setValue(z)
        self._move_to_xyz()

    # --- view
    def _update_view(self, event=None):
        self._elev, self._azim = self.ax.elev, self.ax.azim

    def _set_view_x(self):
        self.ax.view_init(elev=0, azim=0); self._update_view(); self.canvas.draw_idle()
    def _set_view_y(self):
        self.ax.view_init(elev=0, azim=90); self._update_view(); self.canvas.draw_idle()
    def _set_view_z(self):
        self.ax.view_init(elev=90, azim=-90); self._update_view(); self.canvas.draw_idle()

    # --- redraw
    def _redraw(self):
        elev, azim = self._elev, self._azim
        pts = self.arm.fk_points()
        P = np.array(pts)
        self.ax.cla()

        span = (self.cfg.L1 + self.cfg.L2 + self.cfg.L3 + 80) * 0.75
        self.ax.set_xlim(0, span)
        self.ax.set_ylim(-span/2, span/2)
        self.ax.set_zlim(0, span)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_box_aspect([1,1,1])

        self.ax.plot(P[:,0], P[:,1], P[:,2], marker='o')
        self.ax.scatter(P[-1,0], P[-1,1], P[-1,2], s=40)

        ee = P[-1]
        self.ee_label.setText(f"EE: X={ee[0]:.1f} mm, Y={ee[1]:.1f} mm, Z={ee[2]:.1f} mm")

        # 스크린 그리기
        self._draw_screen()
        self.canvas.draw_idle()

# ------------------------------ main ------------------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == '__main__':
    raise SystemExit(main())
