#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Arm Viewer (v2.0, Matplotlib + PySide6)
- J1 범위: -90° ~ +90°, 홈포즈 0°
- J2 범위: -90° ~ 0°, 홈포즈 -70°
- J3 범위: -20° ~ 140°, 홈포즈 120°
- J4, J5: -90° ~ +90° (기존값 유지)
- 3D 뷰 회전 상태 유지
- X축 음수 부분 제거
- 로봇팔 표시 크기 1.5배 확대
"""

from __future__ import annotations
import math, sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QSlider, QDoubleSpinBox, QLabel, QPushButton, QGroupBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

@dataclass
class RevoluteLimit:
    soft_min: float
    soft_max: float
    hard_min: float
    hard_max: float
    resolution: float = 0.1
    home: float = 0.0

@dataclass
class ArmConfig:
    BaseHeight: float = 20.0
    L1: float = 120.0
    L2: float = 110.0
    L3: float = 60.0

    # 각 조인트 범위 및 홈포즈
    j1: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90,90,-90,90,0.1,0))
    j2: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90,0,-90,0,0.1,-70))
    j3: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-20,140,-20,140,0.1,120))
    j4: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90,90,-90,90,0.1,10))
    j5: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90,90,-90,90,0.1,0))

class RobotArm:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.q = np.array([cfg.j1.home, cfg.j2.home, cfg.j3.home, cfg.j4.home, cfg.j5.home], dtype=float)

    def joint_frames(self, q: np.ndarray | None = None) -> list[np.ndarray]:
        if q is None:
            q = self.q
        q1_deg, q2_deg, q3_deg, q4_deg, q5_deg = q
        q1, q2, q3, q4, q5 = map(deg2rad, (q1_deg, q2_deg, q3_deg, q4_deg, q5_deg))
        BH, L1, L2, L3 = self.cfg.BaseHeight, self.cfg.L1, self.cfg.L2, self.cfg.L3

        Ts = []
        T = np.eye(4)
        Ts.append(T.copy())
        T = T @ np.array([[math.cos(q1), -math.sin(q1), 0, 0],
                          [math.sin(q1),  math.cos(q1), 0, 0],
                          [0,             0,            1, 0],
                          [0,             0, 0, 1]], dtype=float)
        T = T @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,BH],[0,0,0,1]], dtype=float)
        Ts.append(T.copy())
        T = T @ np.array([[ math.cos(q2), 0, math.sin(q2), 0],
                          [ 0,            1, 0,            0],
                          [-math.sin(q2), 0, math.cos(q2), 0],
                          [ 0,            0, 0, 1]], dtype=float)
        Ts.append(T.copy())
        T = T @ np.array([[1,0,0,L1],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)
        Ts.append(T.copy())
        T = T @ np.array([[ math.cos(q3), 0, math.sin(q3), 0],
                          [ 0,            1, 0,            0],
                          [-math.sin(q3), 0, math.cos(q3), 0],
                          [ 0,            0, 0, 1]], dtype=float)
        Ts.append(T.copy())
        T = T @ np.array([[ math.cos(q4), 0, math.sin(q4), L2],
                          [ 0,            1, 0,             0],
                          [-math.sin(q4), 0, math.cos(q4),  0],
                          [ 0,            0, 0,             1]], dtype=float)
        Ts.append(T.copy())
        T = T @ np.array([[ math.cos(q5), -math.sin(q5), 0, L3],
                          [ math.sin(q5),  math.cos(q5), 0, 0 ],
                          [ 0,             0,            1, 0 ],
                          [ 0,             0, 0, 1 ]], dtype=float)
        Ts.append(T.copy())
        return Ts

    def fk_points(self, q: np.ndarray | None = None) -> List[np.ndarray]:
        Ts = self.joint_frames(q)
        return [T[:3,3].copy() for T in Ts]

    def ee(self) -> np.ndarray:
        return self.fk_points()[-1]

class SliderWithBox(QWidget):
    valueChanged = QtCore.Signal(float)
    def __init__(self, lo: float, hi: float, step: float, init: float, unit: str = "", parent=None):
        super().__init__(parent)
        self._scale = 100.0
        self.sld = QSlider(Qt.Horizontal)
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

        lay = QHBoxLayout(self)
        lay.addWidget(self.sld)
        lay.addWidget(self.box)
        lay.addWidget(self.lbl)
        lay.setContentsMargins(0, 0, 0, 0)

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Arm Viewer (Matplotlib)")
        self.resize(1280, 800)

        self.cfg = ArmConfig()
        self.arm = RobotArm(self.cfg)

        self.fig = Figure(figsize=(7.5, 6.25))  # 기존보다 1.5배 확대
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.joint_group = self._make_joint_panel()

        self.ee_label = QLabel("")
        self.btn_home = QPushButton("Home pose")
        self.btn_home.clicked.connect(self._go_home)

        right = QVBoxLayout()
        right.addWidget(self.joint_group)
        right.addWidget(self.btn_home)
        right.addWidget(self.ee_label)
        right.addStretch(1)

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
        self.canvas.mpl_connect("motion_notify_event", self._update_view)

    def _update_view(self, event):
        self._elev, self._azim = self.ax.elev, self.ax.azim

    def _make_joint_panel(self) -> QGroupBox:
        gb = QGroupBox("Joints (deg)")
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
        fl.addRow("J5 (Wrist yaw ⊥ L3)", self.j5)
        self.j1.valueChanged.connect(lambda v: self._on_joint_change(0, v))
        self.j2.valueChanged.connect(lambda v: self._on_joint_change(1, v))
        self.j3.valueChanged.connect(lambda v: self._on_joint_change(2, v))
        self.j4.valueChanged.connect(lambda v: self._on_joint_change(3, v))
        self.j5.valueChanged.connect(lambda v: self._on_joint_change(4, v))
        return gb

    def _on_joint_change(self, idx: int, val: float):
        self.arm.q[idx] = val
        self._redraw()

    def _go_home(self):
        self.arm.q[:] = [
            self.cfg.j1.home,
            self.cfg.j2.home,
            self.cfg.j3.home,
            self.cfg.j4.home,
            self.cfg.j5.home,
        ]
        self.j1.setValue(self.arm.q[0])
        self.j2.setValue(self.arm.q[1])
        self.j3.setValue(self.arm.q[2])
        self.j4.setValue(self.arm.q[3])
        self.j5.setValue(self.arm.q[4])
        self._redraw()

    def _redraw(self):
        elev, azim = self._elev, self._azim
        pts = self.arm.fk_points()
        P = np.array(pts)
        self.ax.cla()

        # 좌표 범위를 절반으로 줄임 (화면 확대 효과)
        reach = (self.cfg.L1 + self.cfg.L2 + self.cfg.L3 + 80) * 0.75
        self.ax.set_xlim(0, reach)   # X축 음수 부분 제거
        self.ax.set_ylim(-reach, reach)
        self.ax.set_zlim(0, reach)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.view_init(elev=elev, azim=azim)

        self.ax.plot(P[:,0], P[:,1], P[:,2], marker='o')
        self.ax.scatter(P[-1,0], P[-1,1], P[-1,2], s=40)

        Ts = self.arm.joint_frames()
        for idx in [2,3,4,5,6]:
            self._draw_frame(Ts[idx], axis_len=25)

        ee = P[-1]
        self.ee_label.setText(f"EE: X={ee[0]:.1f} mm, Y={ee[1]:.1f} mm, Z={ee[2]:.1f} mm")

        self.ax.plot([0, reach], [0, 0], [0, 0], alpha=0.2)
        self.ax.plot([0, 0], [-reach, reach], [0, 0], alpha=0.2)

        self.canvas.draw_idle()

    def _draw_frame(self, T: np.ndarray, axis_len: float = 20.0):
        o = T[:3, 3]
        R = T[:3, :3]
        axes = [R[:,0], R[:,1], R[:,2]]
        colors = ['r', 'g', 'b']
        for a, c in zip(axes, colors):
            u, v, w = (a * axis_len)
            self.ax.quiver(o[0], o[1], o[2], u, v, w, color=c, length=1, normalize=False)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == '__main__':
    raise SystemExit(main())