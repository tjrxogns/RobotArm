# robot_arm.py (251011 — IK 재설계: 목표점 직달 방식 / 3링크 위치합치)
import numpy as np


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def normalize_deg(a: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


class RobotArm:
    def __init__(self):
        self.base = np.array([0.0, 0.0, 20.0])  # J2 회전축 위치
        self.L1, self.L2, self.L3 = 80.0, 80.0, 70.0
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0]  # [J1,J2,J3,J4,J5] (deg, 물리각)
        self.end_effector = np.array([0.0, 0.0, 0.0])

        # (sim_min, sim_max, real_min, real_max) — 실기 전송용(그대로 유지)
        self.servo_map = {
            0: (-180, 180, 0, 360),
            1: (-90, 10, 120, 20),
            2: (0, 140, 30, 170),
            3: (-90, 90, 0, 180),
            4: (-90, 90, 0, 180),
        }
        self.sim_limits = {i: (v[0], v[1]) for i, v in self.servo_map.items()}

        # IK 전용 각도 제한(물리각 기준). 필요 시 set_ik_limits()로 변경 가능
        self.ik_limits = {
            'J1': (-180.0, 180.0),
            'J2': (0.0, 90.0),
            'J3': (-10, 120.0),
            'J4': (-10, 120.0),
            'J5': (-90.0, 90.0),
        }

        # === 안전 최소 높이(EE가 절대 내려가지 않을 Z) ===
        # 기본값은 베이스 상단(=20). main.py에서 화면 적용 시 screenZ0로 설정 권장.
        self.min_ee_z = float(self.base[2])

    def set_ik_limits(self, **kwargs):
        """예: set_ik_limits(J2=(5,80), J4=(-45,45)) — 물리각 기준.
        주어진 키만 업데이트."""
        for k, v in kwargs.items():
            if k in self.ik_limits and isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                if lo > hi:
                    lo, hi = hi, lo
                self.ik_limits[k] = (lo, hi)

    def _clamp_ik(self, name, val):
        lo, hi = self.ik_limits[name]
        return clamp(val, lo, hi)

    # 안전 최소 높이 설정 API
    def set_min_ee_z(self, z: float):
        """EE의 최소 높이(Z)를 설정. 화면 평면 Z(screenZ0)로 맞춰 호출 권장."""
        self.min_ee_z = max(0.0, float(z))
        return self.min_ee_z

    def forward_kinematics(self):
        j1, j2, j3, j4, j5 = np.radians(self.joints)
        x0, y0, z0 = self.base

        # L1 (at J2)
        x1 = x0 + self.L1 * np.cos(j1) * np.sin(j2)
        y1 = y0 + self.L1 * np.sin(j1) * np.sin(j2)
        z1 = z0 + self.L1 * np.cos(j2)

        # L2 (at J3)
        s2 = j2 + j3
        x2 = x1 + self.L2 * np.cos(j1) * np.sin(s2)
        y2 = y1 + self.L2 * np.sin(j1) * np.sin(s2)
        z2 = z1 + self.L2 * np.cos(s2)

        # L3 (at J4)
        s3 = s2 + j4
        x3 = x2 + self.L3 * np.cos(j1) * np.sin(s3)
        y3 = y2 + self.L3 * np.sin(j1) * np.sin(s3)
        z3 = z2 + self.L3 * np.cos(s3)

        self.end_effector = np.array([x3, y3, z3])
        return [self.base, [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]

    # 시각화/디버깅용 관절 좌표
    def joint_positions(self):
        pts = self.forward_kinematics()
        j2_pos = self.base
        j3_pos = np.array(pts[1])
        j4_pos = np.array(pts[2])
        ee_pos = np.array(pts[3])
        return j2_pos, j3_pos, j4_pos, ee_pos

    # =========================
    # 역기구학 (IK) — 목표점 직달(orientation 자유)
    #   1) J1 = atan2(y,x)
    #   2) 수직평면(r,h)에서, 타겟 T까지의 단위벡터 u를 따라 L3만큼 뒤쪽에 Wrist W를 놓음
    #   3) L1,L2로 W를 정확히 맞춤
    #   4) 남은 L3로 T를 맞추는 s3 계산 → J4 = s3 - (J2+J3)
    # =========================
    def inverse_kinematics(self, x, y, z):
        """IK (물리각 기준, 서보맵 무시) + 각 관절 제한 적용 + 안전높이 보장."""
        # 0) 타겟 Z를 안전높이 이상으로 끌어올림
        z_req = max(float(z), getattr(self, 'min_ee_z', 0.0), 0.0)
        dx, dy, dz = float(x) - self.base[0], float(y) - self.base[1], z_req - self.base[2]
        prev = list(self.joints)  # 롤백 대비

        # --- J1: 360° 연속성 보장 + 제한 ---
        j1_full = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        cur = float(self.joints[0]) if hasattr(self, 'joints') else 0.0
        cands = [j1_full - 360.0, j1_full, j1_full + 360.0]
        j1_sel = min(cands, key=lambda a: abs(((a - cur + 180) % 360) - 180))
        j1 = normalize_deg(j1_sel)
        j1 = self._clamp_ik('J1', j1)

        # 수직평면 좌표
        r = np.hypot(dx, dy)
        h = dz

        # --- 손목점/2링크/손목 각도: 일관 수렴 루프 ---
        r = float(r); h = float(h)
        h_target = max(dz, (self.min_ee_z - self.base[2]))

        def solve_2link(rw, hw):
            d_w = np.hypot(rw, hw)
            max12 = self.L1 + self.L2 - 1e-9
            if d_w > max12:
                scale = max12 / d_w
                rw *= scale; hw *= scale; d_w = max12
            c2 = np.clip((self.L1**2 + d_w**2 - self.L2**2) / (2*self.L1*d_w), -1.0, 1.0)
            c3 = np.clip((self.L1**2 + self.L2**2 - d_w**2) / (2*self.L1*self.L2), -1.0, 1.0)
            beta  = np.arctan2(rw, hw)
            alpha = np.arccos(c2)
            theta2 = beta - alpha
            j2p = np.degrees(theta2)
            j2p = self._clamp_ik('J2', j2p)
            gamma = np.arccos(c3)
            s2r = np.radians(j2p) + (np.pi - gamma)
            j3p = np.degrees(s2r) - j2p
            j3p = self._clamp_ik('J3', j3p)
            s2r = np.radians(j2p + j3p)
            return j2p, j3p, s2r

        # 초기 Wrist는 타겟 방향으로 L3만큼 뒤로
        d = max(np.hypot(r, h_target), 1e-9)
        ur, uh = r/d, h_target/d
        rw, hw = r - self.L3*ur, h_target - self.L3*uh

        j2_phys = j3_phys = 0.0
        s2 = 0.0
        j4_phys = 0.0
        for _ in range(5):  # 소규모 2회 수렴이면 충분
            # (1) 2링크로 Wrist 맞춤
            j2_phys, j3_phys, s2 = solve_2link(rw, hw)
            # (2) Z를 정확히 맞추는 s3 계산
            r2 = self.L1*np.sin(np.radians(j2_phys)) + self.L2*np.sin(s2)
            h2 = self.L1*np.cos(np.radians(j2_phys)) + self.L2*np.cos(s2)
            cos_s3 = np.clip((h_target - h2) / max(self.L3, 1e-9), -1.0, 1.0)
            s3_abs = np.arccos(cos_s3)
            # r 오차가 작은 부호 선택
            def r_err(s3):
                return abs((r2 + self.L3*np.sin(s3)) - r)
            s3 = s3_abs if r_err(s3_abs) <= r_err(-s3_abs) else -s3_abs
            j4_phys = np.degrees(s3) - (j2_phys + j3_phys)
            j4_phys = self._clamp_ik('J4', j4_phys)
            # (3) s3로 재정의된 Wrist로 업데이트하여 다음 반복 시 2링크를 일관되게 맞춤
            rw = r - self.L3*np.sin(np.radians(j2_phys + j3_phys + j4_phys))
            hw = h_target - self.L3*np.cos(np.radians(j2_phys + j3_phys + j4_phys))

        # J5는 회전만: 기본 0, 제한 적용
        j5_phys = self._clamp_ik('J5', 0.0)

        self.joints = [j1, j2_phys, j3_phys, j4_phys, j5_phys]
        self.update_end_effector()
        # 최종 안전 확인: EE가 안전 높이 아래로 내려가면 이전 자세로 롤백
        if self.end_effector[2] < self.min_ee_z - 1e-6:
            self.joints = prev
            self.update_end_effector()
        return self.joints
        # end IK
        

    # =========================
    # EE 갱신
    # =========================
    def update_end_effector(self):
        pts = self.forward_kinematics()
        self.end_effector = np.array(pts[-1])

    # =========================
    # 서보 각도 변환 (sim → real)
    # =========================
    def get_servo_angles(self):
        out = []
        for i, v in enumerate(self.joints):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            v = clamp(v, sim_min, sim_max)
            out.append(int(np.interp(v, [sim_min, sim_max], [real_min, real_max])))
        return out
