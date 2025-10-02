# 프로젝트 파일 구조를 두 개로 분리합니다.
# 1. robot_arm.py → 로봇팔 계산 로직 전용 (Forward/Inverse Kinematics)
# 2. main.py → UI 및 그림 전용 (PySide6 + Matplotlib)

# ----------------------------- robot_arm.py -----------------------------

import math
import numpy as np
from dataclasses import dataclass, field


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def sim2robot(sim_angle: float, sim_min: float, sim_max: float, rob_min: float, rob_max: float) -> float:
    return float(np.interp(sim_angle, [sim_min, sim_max], [rob_min, rob_max]))


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

    j1: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90, 90, -90, 90, 0.1, 0))
    j2: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90, 0, -90, 0, 0.1, -70))
    j3: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-20, 140, -20, 140, 0.1, 120))
    j4: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90, 90, -90, 90, 0.1, 10))
    j5: RevoluteLimit = field(default_factory=lambda: RevoluteLimit(-90, 90, -90, 90, 0.1, 0))

    servo_map: dict = field(default_factory=lambda: {
        "J1": (-90, 90, 0, 180),
        "J2": (-90, 10, 120, 20),
        "J3": (0, 140, 30, 170),
        "J4": (-90, 90, 0, 180),
        "J5": (-90, 90, 0, 180),
    })


class RobotArm:
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg
        self.q = np.array([cfg.j1.home, cfg.j2.home, cfg.j3.home, cfg.j4.home, cfg.j5.home], dtype=float)

    def joint_frames(self, q=None):
        if q is None:
            q = self.q
        q1_deg, q2_deg, q3_deg, q4_deg, q5_deg = q
        q1, q2, q3, q4, q5 = map(deg2rad, (q1_deg, q2_deg, q3_deg, q4_deg, q5_deg))
        BH, L1, L2, L3 = self.cfg.BaseHeight, self.cfg.L1, self.cfg.L2, self.cfg.L3

        Ts = []
        T = np.eye(4)
        Ts.append(T.copy())

        # J1
        T = T @ np.array([[ math.cos(q1), -math.sin(q1), 0, 0],
                          [ math.sin(q1),  math.cos(q1), 0, 0],
                          [ 0,             0,            1, 0],
                          [ 0,             0,            0, 1]])
        T = T @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,BH],[0,0,0,1]])
        Ts.append(T.copy())

        # J2
        T = T @ np.array([[ math.cos(q2), 0, math.sin(q2), 0],
                          [ 0,            1, 0,            0],
                          [-math.sin(q2), 0, math.cos(q2), 0],
                          [ 0,            0, 0, 1]])
        Ts.append(T.copy())

        # L1
        T = T @ np.array([[1,0,0,L1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        Ts.append(T.copy())

        # J3
        T = T @ np.array([[ math.cos(q3), 0, math.sin(q3), 0],
                          [ 0,            1, 0,            0],
                          [-math.sin(q3), 0, math.cos(q3), 0],
                          [ 0,            0, 0, 1]])
        Ts.append(T.copy())

        # J4 + L2
        T = T @ np.array([[ math.cos(q4), 0, math.sin(q4), L2],
                          [ 0,            1, 0,             0],
                          [-math.sin(q4), 0, math.cos(q4),  0],
                          [ 0,            0, 0,             1]])
        Ts.append(T.copy())

        # J5 + L3
        T = T @ np.array([[ 1, 0,            0,           L3],
                          [ 0, math.cos(q5),-math.sin(q5),0 ],
                          [ 0, math.sin(q5), math.cos(q5),0 ],
                          [ 0, 0,            0,           1 ]])
        Ts.append(T.copy())

        return Ts

    def fk_points(self, q=None):
        return [T[:3,3].copy() for T in self.joint_frames(q)]

    def ee(self):
        return self.fk_points()[-1]

