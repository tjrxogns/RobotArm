# robot_arm_251021_full.py (Full version with corrected J1 range and servo mapping)
# ---------------------------------------------------------------------------
# 로봇팔 3D 시뮬레이터 — 전체 역기구학 및 베이스 회전 오프셋 보정 포함 완전판
# 수정사항:
#   - self.ik_limits['J1'] 범위를 (0.0, 180.0)으로 변경
#   - set_servo_angles() 내 J1 -90 보정 제거
# ---------------------------------------------------------------------------
import numpy as np

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def normalize_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

class RobotArm:
    def __init__(self):
        self.base = np.array([0.0, 0.0, 20.0])
        self.L1, self.L2, self.L3 = 80.0, 80.0, 70.0
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_effector = np.array([0.0, 0.0, 0.0])

        self.servo_map = {
            0: (0, 180, 0, 180),
            1: (0, 90, 120, 30),
            2: (-10, 120, 20, 150),
            3: (-90, 90, 180, 0),
            4: (-90, 90, 0, 180),
        }
        self.sim_limits = {i: (v[0], v[1]) for i, v in self.servo_map.items()}
        self.ik_limits = {
            'J1': (0.0, 180.0),  # ✅ 변경됨
            'J2': (-10.0, 120.0),
            'J3': (-10.0, 120.0),
            'J4': (-10.0, 120.0),
            'J5': (-90.0, 90.0),
        }
        self.screen_origin = np.array([0.0, 0.0, 0.0])
        self.screen_u = np.array([1.0, 0.0, 0.0])
        self.screen_v = np.array([0.0, 1.0, 0.0])
        self.su, self.sv = 1.0, 1.0
        self.lock_z_to_plane = True
        self._rebuild_screen_basis()

        self.min_ee_z = float(self.base[2])
        self.update_end_effector()

    def _rebuild_screen_basis(self):
        u = np.array(self.screen_u, dtype=float)
        v = np.array(self.screen_v, dtype=float)
        u /= np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else 1.0
        v = v - np.dot(v, u) * u
        v /= np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else 1.0
        n = np.cross(u, v)
        n /= np.linalg.norm(n) if np.linalg.norm(n) > 1e-9 else 1.0
        self.screen_u, self.screen_v, self.screen_normal = u, v, n
        self.R_screen_to_world = np.column_stack((u, v, n))

    def map_touch_to_world(self, u, v, z=0.0):
        if not hasattr(self, "R_screen_to_world"):
            self._rebuild_screen_basis()
        local = np.array([self.su*u, self.sv*v, 0.0 if self.lock_z_to_plane else z], dtype=float)
        P = self.screen_origin + self.R_screen_to_world @ local
        return P[0], P[1], P[2]

    def inverse_kinematics(self, x, y, z):
        EPS = 1e-9
        RAD = np.radians
        DEG = np.degrees

        tx, ty, tz = float(x), float(y), float(z)
        dx, dy, dz = tx - self.base[0], ty - self.base[1], tz - self.base[2]
        prev = list(self.joints)

        j1_full = (DEG(np.arctan2(dy, dx)) + 360.0) % 360.0
        cur_j1 = float(self.joints[0]) if hasattr(self, 'joints') else 0.0
        cands = [j1_full - 360.0, j1_full, j1_full + 360.0]
        j1_sel = min(cands, key=lambda a: abs(((a - cur_j1 + 180) % 360) - 180))
        j1 = normalize_deg(j1_sel)
        j1 = self._clamp_ik('J1', j1)

        er = np.array([np.cos(np.radians(j1)), np.sin(np.radians(j1)), 0.0])
        ez = np.array([0.0, 0.0, 1.0])

        r = np.hypot(dx, dy)
        h = dz
        L1, L2, L3 = self.L1, self.L2, self.L3

        dmag = max(np.hypot(r, h), EPS)
        ur, uh = r/dmag, h/dmag
        rw, hw = r - L3*ur, h - L3*uh

        beta = np.arctan2(rw, hw)
        c2 = np.clip((L1**2 + rw**2 + hw**2 - L2**2) / (2*L1*np.hypot(rw, hw)), -1, 1)
        c3 = np.clip((L1**2 + L2**2 - rw**2 - hw**2) / (2*L1*L2), -1, 1)
        a2 = np.arccos(c2)
        a3 = np.arccos(c3)

        j2 = np.degrees(beta - a2)
        j3 = np.degrees(np.pi - a3)
        s234 = np.arctan2(h - (L1*np.cos(RAD(j2)) + L2*np.cos(RAD(j2+j3))), r - (L1*np.sin(RAD(j2)) + L2*np.sin(RAD(j2+j3))))
        j4 = np.degrees(s234) - (j2 + j3)

        self.joints = [j1, j2, j3, j4, 0.0]
        self.update_end_effector()
        return self.joints

    def _clamp_ik(self, name, val):
        lo, hi = self.ik_limits[name]
        return clamp(val, lo, hi)

    def set_servo_angles(self, angles):
        if len(angles) != len(self.joints):
            return
        for i, ang in enumerate(angles):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            if (real_max - real_min) * (sim_max - sim_min) > 0:
                sim_angle = np.interp(ang, [real_min, real_max], [sim_min, sim_max])
            else:
                sim_angle = np.interp(ang, [real_max, real_min], [sim_min, sim_max])
            # ✅ J1의 -90 보정 제거
            if i == 2:
                sim_angle -= 45
            self.joints[i] = sim_angle
        self.update_end_effector()

    def update_end_effector(self):
        j1, j2, j3, j4, _ = np.radians(self.joints)
        x0, y0, z0 = self.base
        x1 = x0 + self.L1 * np.cos(j1) * np.sin(j2)
        y1 = y0 + self.L1 * np.sin(j1) * np.sin(j2)
        z1 = z0 + self.L1 * np.cos(j2)
        s2 = j2 + j3
        x2 = x1 + self.L2 * np.cos(j1) * np.sin(s2)
        y2 = y1 + self.L2 * np.sin(j1) * np.sin(s2)
        z2 = z1 + self.L2 * np.cos(s2)
        s3 = s2 + j4
        x3 = x2 + self.L3 * np.cos(j1) * np.sin(s3)
        y3 = y2 + self.L3 * np.sin(j1) * np.sin(s3)
        z3 = z2 + self.L3 * np.cos(s3)
        self.end_effector = np.array([x3, y3, z3])
