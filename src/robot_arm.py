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
        self.L1, self.L2, self.L3 = 60.0, 80.0, 70.0
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0]  # [J1,J2,J3,J4,J5] (deg, sim)
        self.end_effector = np.array([0.0, 0.0, 0.0])

        # (sim_min, sim_max, real_min, real_max)
        self.servo_map = {
            0: (-180, 180, 0, 360),
            1: (-90, 10, 120, 20),
            2: (0, 140, 30, 170),
            3: (-90, 90, 0, 180),
            4: (-90, 90, 0, 180),
        }
        self.sim_limits = {i: (v[0], v[1]) for i, v in self.servo_map.items()}

    # =========================
    # 순기구학 (FK) — 누적 피치: s2=J2+J3, s3=s2+J4
    # =========================
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
        """
        IK (서보 맵/보정 무시):
        - 각도 정의를 FK와 1:1 일치(모두 '물리각')
          J2 = 어깨 절대 피치(0°=+Z 정위치, 90°=수평)
          J3 = 팔꿈치 절대 추가 피치(누적: s2 = J2+J3)
          J4 = 손목 절대 추가 피치(누적: s3 = J2+J3+J4)
        - 목표점 T(x,y,z)을 정확히 위치 일치
        """
        # 1) 제약: EE는 바닥 아래로 가지 않음
        z = max(0.0, float(z))

        # 2) Yaw (J1) — 360° 전범위 + 연속성 보장(현재 각도에 가장 가까운 해 선택)
        dx, dy, dz = float(x) - self.base[0], float(y) - self.base[1], z - self.base[2]
        j1_full = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0  # [0,360)
        # 후보 3개: j1_full-360, j1_full, j1_full+360 → 현재 각과 가장 가까운 해 선택
        cur = float(self.joints[0]) if hasattr(self, 'joints') else 0.0
        cands = [j1_full - 360.0, j1_full, j1_full + 360.0]
        j1_sel = min(cands, key=lambda a: abs(((a - cur + 180) % 360) - 180))
        j1 = normalize_deg(j1_sel)  # 내부 저장은 [-180,180) 유지

        # 수직평면 좌표 (r: 반경, h: 높이)
        r = np.hypot(dx, dy)
        h = dz

        # 3) 손목점 W = T - L3 * u (u = (r,h)/||T||)
        d = max(np.hypot(r, h), 1e-9)
        ur, uh = r/d, h/d
        rw, hw = r - self.L3*ur, h - self.L3*uh

        # 4) 2링크 IK (L1, L2)로 W 맞춤 — 모든 각도는 '물리각' 기준
        d_w = np.hypot(rw, hw)
        max12 = self.L1 + self.L2 - 1e-9
        if d_w > max12:
            scale = max12 / d_w
            rw *= scale; hw *= scale; d_w = max12

        # (a) 어깨: β=atan2(rw, hw) (from +Z), α=acos(c2), θ2=β−α (elbow-down)
        c2 = np.clip((self.L1**2 + d_w**2 - self.L2**2) / (2*self.L1*d_w), -1.0, 1.0)
        c3 = np.clip((self.L1**2 + self.L2**2 - d_w**2) / (2*self.L1*self.L2), -1.0, 1.0)
        beta = np.arctan2(rw, hw)
        alpha = np.arccos(c2)
        theta2 = beta - alpha                      # rad
        j2_phys = np.degrees(theta2)               # deg (0=up, 90=horizontal)
        j2_phys = clamp(j2_phys, 0.0, 90.0)

        # (b) 팔꿈치: γ=acos(c3) (내각), L2 절대 각 = θ2 + (π − γ)
        gamma = np.arccos(c3)
        s2 = theta2 + (np.pi - gamma)
        j3_phys = np.degrees(s2) - j2_phys        # 추가 피치 (물리각 정의)

        # 5) 손목: Z를 정확히 맞추고 r도 정합 — s3 = atan2(dr, dh)
        r2 = self.L1*np.sin(np.radians(j2_phys)) + self.L2*np.sin(s2)
        h2 = self.L1*np.cos(np.radians(j2_phys)) + self.L2*np.cos(s2)
        dr, dh = r - r2, h - h2
        s3 = np.arctan2(dr, dh)
        j4_phys = np.degrees(s3) - (j2_phys + j3_phys)

        # 6) 결과 반영 (모두 '물리각') — 서보맵/보정 무시
        j5_phys = 0.0
        self.joints = [j1, j2_phys, j3_phys, j4_phys, j5_phys]
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
