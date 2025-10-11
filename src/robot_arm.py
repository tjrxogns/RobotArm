# robot_arm.py (251011 복원 + 튜닝 통합 — FINAL)
import numpy as np

class RobotArm:
    def __init__(self):
        self.base = np.array([0.0, 0.0, 20.0])
        self.L1, self.L2, self.L3 = 60.0, 80.0, 70.0
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_effector = np.array([0.0, 0.0, 0.0])

        # servo map 최신화 (sim → robot)
        self.servo_map = {
            0: (-90, 90, 0, 180),   # J1
            1: (-90, 10, 120, 20),  # J2
            2: (0, 140, 30, 170),   # J3
            3: (-90, 90, 0, 180),   # J4
            4: (-90, 90, 0, 180),   # J5
        }

    # =========================
    # 순기구학 (Forward Kinematics)
    # =========================
    def forward_kinematics(self):
        j1, j2, j3, j4, j5 = np.radians(self.joints)
        x0, y0, z0 = self.base

        x1 = x0
        y1 = y0
        z1 = z0 + self.L1 * np.cos(j2)

        x2 = x1 + self.L2 * np.cos(j2 + j3) * np.cos(j1)
        y2 = y1 + self.L2 * np.cos(j2 + j3) * np.sin(j1)
        z2 = z1 + self.L2 * np.sin(j2 + j3)

        x3 = x2 + self.L3 * np.cos(j2 + j3 + j4) * np.cos(j1)
        y3 = y2 + self.L3 * np.cos(j2 + j3 + j4) * np.sin(j1)
        z3 = z2 + self.L3 * np.sin(j2 + j3 + j4)

        self.end_effector = np.array([x3, y3, z3])
        return [self.base, [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]

    # =========================
    # 역기구학 (Inverse Kinematics)
    # =========================
    def inverse_kinematics(self, x, y, z):
        dx, dy, dz = x - self.base[0], y - self.base[1], z - self.base[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        max_reach = self.L1 + self.L2 + self.L3 - 5

        if dist > max_reach:
            print("[WARN] Target out of reach.")
            return self.joints

        j1 = np.degrees(np.arctan2(dy, dx))

        r = np.sqrt(dx**2 + dy**2)
        h = dz
        d = np.sqrt(r**2 + h**2)
        d = np.clip(d, 1e-6, max_reach)

        # 삼각형 계산 (NaN 방지용 클램프)
        val2 = (self.L1**2 + d**2 - self.L2**2) / (2 * self.L1 * d)
        val3 = (self.L1**2 + self.L2**2 - d**2) / (2 * self.L1 * self.L2)
        val2 = np.clip(val2, -1.0, 1.0)
        val3 = np.clip(val3, -1.0, 1.0)

        a1 = np.arctan2(h, r)
        a2 = np.arccos(val2)
        a3 = np.arccos(val3)

        j2 = np.degrees(a1 + a2)
        j3 = np.degrees(np.pi - a3)

        j4 = -(j2 + j3 - 90)
        j5 = 0.0

        self.joints = [j1, j2 - 90, -(j3 - 90), j4, j5]
        self.update_end_effector()

        # ===== Z 오차 보정 =====
        dz_error = self.end_effector[2] - z
        if abs(dz_error) > 0.1:
            correction = np.degrees(np.arctan2(dz_error, self.L2 + self.L3))
            self.joints[1] -= correction
            self.update_end_effector()

        # NaN 방지 처리
        self.joints = [0.0 if np.isnan(j) else j for j in self.joints]
        return self.joints

    # =========================
    # EE 갱신
    # =========================
    def update_end_effector(self):
        pts = self.forward_kinematics()
        self.end_effector = np.array(pts[-1])

    # =========================
    # 서보 각도 변환
    # =========================
    def get_servo_angles(self):
        servo_angles = []
        for i, val in enumerate(self.joints):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            val = np.clip(val, sim_min, sim_max)
            mapped = np.interp(val, [sim_min, sim_max], [real_min, real_max])
            servo_angles.append(int(mapped))
        return servo_angles
