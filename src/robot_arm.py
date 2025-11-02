# robot_arm_251014.py — IK/기구해석 전담 모듈 (CCD + 시리얼 연동 완전판)
# * 좌표계: Z up, XY 수평. J0=Yaw(베이스), J1=어깨 Pitch, J2=팔꿈치, J3=손목 Pitch, J4=Roll
# * CCD(재귀/반복) IK로 J1~J3 동시 해결, 시리얼 수신 대응 set_servo_angles() 복구 포함

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def normalize_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def _angle_of(vx: float, vz: float) -> float:
    return float(np.degrees(np.arctan2(vx, vz)))  # Z축 기준 각도(우리 FK 기준)

# ──────────────────────────────────────────────────────────────
# 본체
# ──────────────────────────────────────────────────────────────
class RobotArm:
    def __init__(self):
        # 기하
        self.base = np.array([0.0, 0.0, 20.0])  # 베이스 원점
        self.L1, self.L2, self.L3 = 80.0, 80.0, 70.0

        # 관절각(시뮬레이터 각도) [J0..J4]
        self.joints: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_effector = np.array([0.0, 0.0, 0.0])

        # 서보 맵 (sim_min, sim_max, real_min, real_max)
        self.servo_map: Dict[int, Tuple[float, float, float, float]] = {
            0: (0, 180, 0, 180),
            1: (0, 90, 120, 30),
            2: (-10, 120, 20, 150),
            3: (-90, 90, 180, 0),
            4: (-90, 90, 0, 180),
        }
        self.sim_limits = {i: (v[0], v[1]) for i, v in self.servo_map.items()}

        # IK 한계(물리 범위 의미)
        self.ik_limits: Dict[str, Tuple[float, float]] = {
            'J1': (-180.0, 180.0),  # J0(Yaw)
            'J2': (0.0, 90.0),      # J1(Shoulder)
            'J3': (0.0, 135.0),     # J2(Elbow)
            'J4': (-90.0, 90.0),    # J3(Wrist Pitch)
            'J5': (-90.0, 90.0),    # J4(Roll)
        }

        # 스크린 캘리브레이션(기본: 동일 좌표계)
        self.screen_origin = np.array([0.0, 0.0, 0.0])
        self.screen_u = np.array([1.0, 0.0, 0.0])
        self.screen_v = np.array([0.0, 1.0, 0.0])
        self.su, self.sv = 1.0, 1.0
        self.lock_z_to_plane = True
        self._rebuild_screen_basis()

        self._ik_trace: List[dict] | None = None
        self.update_end_effector()

    # ───────── 스크린 캘리브레이션 ─────────
    def set_screen_calibration(self, origin, u_axis, v_axis, su, sv, lock_z_to_plane=True):
        self.screen_origin = np.array(origin, dtype=float)
        self.screen_u = np.array(u_axis, dtype=float)
        self.screen_v = np.array(v_axis, dtype=float)
        self.su = float(su); self.sv = float(sv)
        self.lock_z_to_plane = bool(lock_z_to_plane)
        self._rebuild_screen_basis()

    def _rebuild_screen_basis(self):
        u = np.array(self.screen_u, dtype=float)
        v = np.array(self.screen_v, dtype=float)
        un = np.linalg.norm(u); u = u / (un if un > 1e-9 else 1.0)
        v = v - np.dot(v, u) * u
        vn = np.linalg.norm(v); v = v / (vn if vn > 1e-9 else 1.0)
        n = np.cross(u, v)
        nn = np.linalg.norm(n); n = n / (nn if nn > 1e-9 else 1.0)
        self.screen_u, self.screen_v, self.screen_normal = u, v, n
        self.R_screen_to_world = np.column_stack((u, v, n))

    def map_touch_to_world(self, u, v, z=0.0):
        if not hasattr(self, "R_screen_to_world"):
            self._rebuild_screen_basis()
        local = np.array([
            self.su * float(u),
            self.sv * float(v),
            0.0 if self.lock_z_to_plane else float(z)
        ], dtype=float)
        P = self.screen_origin + self.R_screen_to_world @ local
        return P[0], P[1], P[2]

    # ───────── 유틸 ─────────
    @staticmethod
    def _yaw_from_points(base_xyz, tgt_xyz) -> float:
        dx = float(tgt_xyz[0] - base_xyz[0])
        dy = float(tgt_xyz[1] - base_xyz[1])
        return normalize_deg(np.degrees(np.arctan2(dy, dx)))

    # ───────── 정기구학(3D) ─────────
    def forward_kinematics(self, joints: List[float] | None = None):
        if joints is None:
            joints = self.joints
        j0, j1, j2, j3, j4 = np.radians(joints)
        x0, y0, z0 = self.base
        # 0°: +Z(수직), 90°: 수평(+X′)
        x1 = x0 + self.L1 * np.cos(j0) * np.sin(j1)
        y1 = y0 + self.L1 * np.sin(j0) * np.sin(j1)
        z1 = z0 + self.L1 * np.cos(j1)
        s2 = j1 + j2
        x2 = x1 + self.L2 * np.cos(j0) * np.sin(s2)
        y2 = y1 + self.L2 * np.sin(j0) * np.sin(s2)
        z2 = z1 + self.L2 * np.cos(s2)
        s3 = s2 + j3
        x3 = x2 + self.L3 * np.cos(j0) * np.sin(s3)
        y3 = y2 + self.L3 * np.sin(j0) * np.sin(s3)
        z3 = z2 + self.L3 * np.cos(s3)
        self.end_effector = np.array([x3, y3, z3])
        return np.array([self.base, [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], dtype=float)

    def joint_positions(self):
        pts = self.forward_kinematics()
        j2_pos = self.base
        j3_pos = np.array(pts[1])
        j4_pos = np.array(pts[2])
        ee_pos = np.array(pts[3])
        return j2_pos, j3_pos, j4_pos, ee_pos

    # ───────── 평면 FK (J1~J3, X′Z) ─────────
    def _fk_planar(self, j1: float, j2: float, j3: float) -> Tuple[float, float]:
        r1 = self.L1 * np.sin(np.radians(j1)); h1 = self.L1 * np.cos(np.radians(j1))
        s2 = j1 + j2
        r2 = r1 + self.L2 * np.sin(np.radians(s2)); h2 = h1 + self.L2 * np.cos(np.radians(s2))
        s3 = s2 + j3
        r3 = r2 + self.L3 * np.sin(np.radians(s3)); h3 = h2 + self.L3 * np.cos(np.radians(s3))
        return float(r3), float(h3)

    def _fk_planar_joints(self, j1: float, j2: float, j3: float):
        r0, h0 = 0.0, 0.0
        r1 = self.L1 * np.sin(np.radians(j1)); h1 = self.L1 * np.cos(np.radians(j1))
        s2 = j1 + j2
        r2 = r1 + self.L2 * np.sin(np.radians(s2)); h2 = h1 + self.L2 * np.cos(np.radians(s2))
        s3 = s2 + j3
        r3 = r2 + self.L3 * np.sin(np.radians(s3)); h3 = h2 + self.L3 * np.cos(np.radians(s3))
        return (r0, h0), (r1, h1), (r2, h2), (r3, h3)

    # ───────── 역기구학 (CCD 반복법) ─────────
    def inverse_kinematics(self, x: float, y: float, z: float):
        tx, ty, tz = float(x), float(y), float(z)
        dx, dy, dz = tx - self.base[0], ty - self.base[1], tz - self.base[2]

        j0 = clamp(self._yaw_from_points(self.base, [tx, ty, tz]), *self.ik_limits['J1'])
        r_tgt = float(np.hypot(dx, dy)); h_tgt = float(dz)

        j1, j2, j3 = self.joints[1], self.joints[2], self.joints[3]
        j1 = clamp(j1, *self.ik_limits['J2'])
        j2 = clamp(j2, *self.ik_limits['J3'])
        j3 = clamp(j3, *self.ik_limits['J4'])

        max_iter = 60; tol = 0.8; damping = 1.0

        for _ in range(max_iter):
            (r0,h0),(r1,h1),(r2,h2),(re,he) = self._fk_planar_joints(j1,j2,j3)
            er, eh = r_tgt - re, h_tgt - he
            if (er*er + eh*eh) ** 0.5 <= tol:
                break

            # J3 보정
            v_cur = np.array([re - r2, he - h2]); v_tgt = np.array([r_tgt - r2, h_tgt - h2])
            if np.linalg.norm(v_cur) > 1e-9 and np.linalg.norm(v_tgt) > 1e-9:
                ang_cur = _angle_of(v_cur[0], v_cur[1]); ang_tgt = _angle_of(v_tgt[0], v_tgt[1])
                delta = normalize_deg(ang_tgt - ang_cur) * damping
                j3 += delta; j3 = clamp(j3, *self.ik_limits['J4'])

            # J2 보정
            (r0,h0),(r1,h1),(r2,h2),(re,he) = self._fk_planar_joints(j1,j2,j3)
            v_cur = np.array([re - r1, he - h1]); v_tgt = np.array([r_tgt - r1, h_tgt - h1])
            if np.linalg.norm(v_cur) > 1e-9 and np.linalg.norm(v_tgt) > 1e-9:
                ang_cur = _angle_of(v_cur[0], v_cur[1]); ang_tgt = _angle_of(v_tgt[0], v_tgt[1])
                delta = normalize_deg(ang_tgt - ang_cur) * damping
                j2 += delta; j2 = clamp(j2, *self.ik_limits['J3'])

            # J1 보정
            (r0,h0),(r1,h1),(r2,h2),(re,he) = self._fk_planar_joints(j1,j2,j3)
            v_cur = np.array([re - r0, he - h0]); v_tgt = np.array([r_tgt - r0, h_tgt - h0])
            if np.linalg.norm(v_cur) > 1e-9 and np.linalg.norm(v_tgt) > 1e-9:
                ang_cur = _angle_of(v_cur[0], v_cur[1]); ang_tgt = _angle_of(v_tgt[0], v_tgt[1])
                delta = normalize_deg(ang_tgt - ang_cur) * damping
                j1 += delta; j1 = clamp(j1, *self.ik_limits['J2'])

        self.joints = [float(j0), float(j1), float(j2), float(j3), 0.0]
        self.update_end_effector()
        return self.joints

    # ───────── 기타 ─────────
    def set_servo_angles(self, angles: List[float]):
        """실기 서보 각도(0~180)를 시뮬레이터 joint로 역변환."""
        if len(angles) != 5:
            return
        for i, ang in enumerate(angles):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            if (real_max - real_min) * (sim_max - sim_min) > 0:
                sim_angle = np.interp(ang, [real_min, real_max], [sim_min, sim_max])
            else:
                sim_angle = np.interp(ang, [real_max, real_min], [sim_min, sim_max])
            self.joints[i] = float(sim_angle)
        self.update_end_effector()

    def set_ik_limits(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.ik_limits and isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                if lo > hi:
                    lo, hi = hi, lo
                self.ik_limits[k] = (lo, hi)

    def update_end_effector(self):
        _ = self.forward_kinematics()

    def get_servo_angles(self) -> List[int]:
        out: List[int] = []
        for i, v in enumerate(self.joints):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            vv = clamp(float(v), float(sim_min), float(sim_max))
            out.append(int(np.interp(vv, [sim_min, sim_max], [real_min, real_max])))
        return out

    def get_ik_trace(self) -> List[dict] | None:
        return self._ik_trace
