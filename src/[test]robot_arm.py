# robot_arm_251021_full.py
# ---------------------------------------------------------------------------
# 로봇팔 3D 시뮬레이터 — 최종 통합판 (251021)
#  - 화면(터치) → 월드 좌표 캘리브레이션 체계 고정 (U,V 정규직교화)
#  - J1(베이스) 연속각(unwrapped) 추적으로 터치 지점마다 달라지는 회전 각도 문제 완화
#  - J1은 atan2(dy, dx)에서 계산하되 wrap/unwrap으로 급격한 점프 억제
#  - DLS/자코비안, 한계각 처리, 서보 맵핑 등 기존 API/동작 유지
#  - 필요시 디버그 로그 on/off 가능 (self.ik_debug)
# ---------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


# --------------------------- 유틸리티 ---------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def normalize_deg(a: float) -> float:
    """[-180, 180) 범위로 wrap."""
    return (a + 180.0) % 360.0 - 180.0


# =========================== 로봇 모델 ===========================
class RobotArm:
    """
    kinematic convention
      - joints = [J1, J2, J3, J4, J5] (deg, 물리각 기준)
      - base = J2 회전축의 월드 좌표
      - L1,L2,L3: 각 구간 길이 (J2→J3, J3→J4, J4→EE)

    화면 캘리브레이션
      - screen_origin (O), screen_u (U), screen_v (V), screen_normal (N)
      - su, sv: 스크린 u,v 축 스케일(픽셀→mm 등)
      - lock_z_to_plane: True면 터치 z는 평면에 고정
    """

    # ------------------------- 생성자 -------------------------
    def __init__(self):
        # 기하
        self.base = np.array([0.0, 0.0, 20.0])  # J2 회전축 위치
        self.L1, self.L2, self.L3 = 80.0, 80.0, 70.0

        # 관절 상태 (deg)
        self.joints: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_effector = np.array([0.0, 0.0, 0.0])

        # (sim_min, sim_max, real_min, real_max)
        #  시뮬레이터 각도 ↔ 실기 서보각 변환 테이블
        self.servo_map: Dict[int, Tuple[float, float, float, float]] = {
            0: (0, 180, 0, 180),        # J1 베이스 회전
            1: (0, 90, 120, 30),        # J2
            2: (-10, 120, 20, 150),     # J3
            3: (-90, 90, 180, 0),       # J4
            4: (-90, 90, 0, 180),       # J5
        }
        self.sim_limits = {i: (v[0], v[1]) for i, v in self.servo_map.items()}

        # IK 전용 각도 제한(물리각 기준)
        self.ik_limits: Dict[str, Tuple[float, float]] = {
            'J1': (-180.0, 180.0),
            'J2': (-10.0, 120.0),
            'J3': (-10.0, 120.0),
            'J4': (-10.0, 120.0),
            'J5': (-90.0, 90.0),
        }

        # 화면-로봇 캘리브레이션 (정규직교화 포함)
        self.screen_origin = np.array([0.0, 0.0, 0.0])
        self.screen_u = np.array([1.0, 0.0, 0.0])
        self.screen_v = np.array([0.0, 1.0, 0.0])
        self.su, self.sv = 1.0, 1.0
        self.lock_z_to_plane = True
        self._rebuild_screen_basis()

        # 안전 최소 Z (디폴트는 베이스 상단)
        self.min_ee_z = float(self.base[2])

        # 디버그 트레이스
        self.ik_debug = False
        self._ik_trace: List[dict] | None = None

        # J1 연속각(unwrapped) 상태
        self._j1_unwrapped: float | None = None

        # EE 초기화
        self.update_end_effector()

    # -------------------- 화면 좌표계 캘리브레이션 --------------------
    def _rebuild_screen_basis(self):
        u = np.array(self.screen_u, dtype=float)
        v = np.array(self.screen_v, dtype=float)
        un = np.linalg.norm(u)
        vn = np.linalg.norm(v)
        u = u / (un if un > 1e-9 else 1.0)
        # v를 u에 직교화 후 정규화
        v = v - np.dot(v, u) * u
        vn = np.linalg.norm(v)
        v = v / (vn if vn > 1e-9 else 1.0)
        n = np.cross(u, v)
        nn = np.linalg.norm(n)
        n = n / (nn if nn > 1e-9 else 1.0)
        self.screen_u, self.screen_v, self.screen_normal = u, v, n
        self.R_screen_to_world = np.column_stack((u, v, n))

    def set_screen_calibration(self, origin, u_axis, v_axis, su, sv, lock_z_to_plane=True):
        self.screen_origin = np.array(origin, dtype=float)
        self.screen_u = np.array(u_axis, dtype=float)
        self.screen_v = np.array(v_axis, dtype=float)
        self.su = float(su)
        self.sv = float(sv)
        self.lock_z_to_plane = bool(lock_z_to_plane)
        self._rebuild_screen_basis()
        if self.ik_debug:
            print("[apply_screen]")
            print(f"  Origin: {self.screen_origin}")
            print(f"  U: {self.screen_u}, V: {self.screen_v}, N: {self.screen_normal}")
            print(f"  su,sv: {self.su:.4f}, {self.sv:.4f}")
            print("  R =")
            print(self.R_screen_to_world)

    def map_touch_to_world(self, u, v, z=0.0):
        if not hasattr(self, "R_screen_to_world"):
            self._rebuild_screen_basis()
        local = np.array([
            self.su * float(u),
            self.sv * float(v),
            0.0 if self.lock_z_to_plane else float(z)
        ], dtype=float)
        P = self.screen_origin + self.R_screen_to_world @ local
        if self.ik_debug:
            print(f"[map_touch_to_world] (u,v,z)=({u:.2f},{v:.2f},{z:.2f}) → P={P}")
        return P[0], P[1], P[2]

    # ---------------------------- IO ----------------------------
    def set_servo_angles(self, angles: List[float]):
        """실기에서 받은 각도를 시뮬레이터 각도로 변환하여 joints 반영.
        (방향 자동 판별 + 축별 오프셋 보정 유지)
        """
        if len(angles) != len(self.joints):
            return
        for i, ang in enumerate(angles):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            # 방향 자동 판별
            if (real_max - real_min) * (sim_max - sim_min) > 0:
                sim_angle = np.interp(ang, [real_min, real_max], [sim_min, sim_max])
            else:
                sim_angle = np.interp(ang, [real_max, real_min], [sim_min, sim_max])
            # 특정 축 보정 (필요시 유지)
            if i == 0:
                sim_angle -= 90  # 하드웨어 0점 차이를 시뮬 상 중앙에 맞춤
            elif i == 2:
                sim_angle -= 45  # 링크 편차 실측 보정
            self.joints[i] = float(sim_angle)
        self.update_end_effector()

    def get_servo_angles(self) -> List[int]:
        out: List[int] = []
        for i, v in enumerate(self.joints):
            sim_min, sim_max, real_min, real_max = self.servo_map[i]
            vv = clamp(float(v), float(sim_min), float(sim_max))
            out.append(int(np.interp(vv, [sim_min, sim_max], [real_min, real_max])))
        return out

    def set_ik_limits(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.ik_limits and isinstance(v, (list, tuple)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                if lo > hi:
                    lo, hi = hi, lo
                self.ik_limits[k] = (lo, hi)

    def _clamp_ik(self, name: str, val: float) -> float:
        lo, hi = self.ik_limits[name]
        return clamp(val, lo, hi)

    def set_min_ee_z(self, z: float):
        self.min_ee_z = max(0.0, float(z))
        return self.min_ee_z

    # ------------------- 정기구학 / 디버그 포인트 -------------------
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

    def joint_positions(self):
        pts = self.forward_kinematics()
        j2_pos = self.base
        j3_pos = np.array(pts[1])
        j4_pos = np.array(pts[2])
        ee_pos = np.array(pts[3])
        return j2_pos, j3_pos, j4_pos, ee_pos

    # --------------------------- 역기구학 ---------------------------
    def inverse_kinematics(self, x: float, y: float, z: float):
        """L3-직달(orientation 자유) + 화면평면 터치 구속 + DLS 미세조정.
        - 터치에서 넘어온 z는 그대로 추종 (min_ee_z는 제한으로만 사용하지 않음).
        - J1은 atan2(dy,dx)를 unwrapped 방식으로 연속 추적 (wrap 점프 억제).
        """
        EPS = 1e-9
        RAD = np.radians
        DEG = np.degrees

        # ------------------ 헬퍼 ------------------
        def unit(v):
            v = np.array(v, dtype=float)
            n = np.linalg.norm(v)
            return v / n if n > EPS else v

        def wrap_pi(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        def joint_limits(name):
            lo = self._clamp_ik(name, -1e9)
            hi = self._clamp_ik(name, +1e9)
            if lo > hi:
                lo, hi = hi, lo
            return float(lo), float(hi)

        def joint_slack(name, ang):
            lo, hi = joint_limits(name)
            rng = max(1e-6, hi - lo)
            dmin = max(0.0, float(ang) - lo)
            dmax = max(0.0, hi - float(ang))
            slack = max(1e-6, min(dmin, dmax))
            return slack, lo, hi, rng

        # 입력
        tx, ty, tz = float(x), float(y), float(z)
        dx, dy, dz = tx - self.base[0], ty - self.base[1], tz - self.base[2]
        prev = list(self.joints)

        # ------------------ J1 연속각 계산 ------------------
        raw = DEG(np.arctan2(dy, dx)) - 30 # [-180,180] 범위가 아님
        if self._j1_unwrapped is None:
            self._j1_unwrapped = raw
        else:
            # unwrap: raw와 이전 unwrapped의 차가 180을 넘지 않도록 조정
            while raw - self._j1_unwrapped > 180.0:
                raw -= 360.0
            while raw - self._j1_unwrapped < -180.0:
                raw += 360.0
            self._j1_unwrapped = raw
        j1 = clamp(self._j1_unwrapped, *self.ik_limits['J1'])

        # J1-정렬 평면 기저
        er = np.array([np.cos(RAD(j1)), np.sin(RAD(j1)), 0.0])
        ez = np.array([0.0, 0.0, 1.0])

        # 스크린 노멀/원점 (없으면 합리적 기본치)
        n_world = unit(getattr(self, 'screen_normal', er))
        P0 = np.array(getattr(self, 'screen_origin', self.base) + er * max(np.hypot(dx, dy), 1.0), dtype=float)

        # 노멀을 (er, ez) 평면에 사영하여 목표 s_target 얻기
        n_plane = (n_world @ er) * er + (n_world @ ez) * ez
        if np.linalg.norm(n_plane) < 1e-6:
            n_plane = er.copy()
        n_plane = unit(n_plane)
        s_target = np.arctan2(n_plane @ er, n_plane @ ez)

        # 원통화 좌표로 축소
        r = float(np.hypot(dx, dy))
        h = float(dz)
        h_target = h
        L1, L2, L3 = float(self.L1), float(self.L2), float(self.L3)

        def clamp01(v):
            return np.clip(v, -1.0, 1.0)

        def within_limits(j2, j3, j4):
            j2c = self._clamp_ik('J2', j2)
            j3c = self._clamp_ik('J3', j3)
            j4c = self._clamp_ik('J4', j4)
            return (abs(j2 - j2c) < 1e-4) and (abs(j3 - j3c) < 1e-4) and (abs(j4 - j4c) < 1e-4)

        def fk_partial(j2, j3, j4):
            s2 = RAD(j2)
            s23 = RAD(j2 + j3)
            s234 = RAD(j2 + j3 + j4)
            r2 = L1*np.sin(s2) + L2*np.sin(s23) + L3*np.sin(s234)
            h2 = L1*np.cos(s2) + L2*np.cos(s23) + L3*np.cos(s234)
            return r2, h2, s234

        # 초기 팔목점 추정 (타겟에서 L3만큼 뒤)
        dmag = max(np.hypot(r, h_target), EPS)
        ur, uh = r/dmag, h_target/dmag
        rw, hw = r - L3*ur, h_target - L3*uh

        # 2링크 열거
        def solve_2link_all(rw, hw):
            d_w = max(np.hypot(rw, hw), EPS)
            max12 = L1 + L2 - 1e-9
            if d_w > max12:
                scale = max12 / d_w
                rw *= scale
                hw *= scale
                d_w = max12
            beta = np.arctan2(rw, hw)
            c2 = clamp01((L1**2 + d_w**2 - L2**2) / (2*L1*d_w))
            c3 = clamp01((L1**2 + L2**2 - d_w**2) / (2*L1*L2))
            alpha = np.arccos(c2)
            gamma = np.arccos(c3)
            j2_up = DEG(beta - alpha)
            j2_dn = DEG(beta + alpha)
            j3_mag = DEG(np.pi - gamma)
            combos = [(j2_up, +j3_mag), (j2_up, -j3_mag), (j2_dn, +j3_mag), (j2_dn, -j3_mag)]
            seen, out = set(), []
            for j2c, j3c in combos:
                key = (round(j2c, 4), round(j3c, 4))
                if key not in seen:
                    seen.add(key)
                    out.append((j2c, j3c))
            return out

        # wrist from r-error & orientation (s_target)
        def solve_wrist(j2, j3, w_orient=0.45):
            r2 = L1*np.sin(RAD(j2)) + L2*np.sin(RAD(j2 + j3))
            h2 = L1*np.cos(RAD(j2)) + L2*np.cos(RAD(j2 + j3))
            cos_s3 = clamp01((h_target - h2) / max(L3, EPS))
            s3_abs = np.arccos(cos_s3)

            def cost(s3):
                r_err = abs((r2 + L3*np.sin(s3)) - r)
                return r_err + w_orient * abs(wrap_pi(s3 - s_target))

            s3_pos, s3_neg = s3_abs, -s3_abs
            cand = []
            for s3 in (s3_pos, s3_neg):
                j4 = DEG(s3) - (j2 + j3)
                j4c = self._clamp_ik('J4', j4)
                penal = 10.0 if abs(j4c - j4) > 1e-6 else 0.0
                cand.append((cost(s3) + penal, j4c))
            cand.sort(key=lambda t: t[0])
            return cand[0][1]

        # 후보 생성
        raw_candidates = solve_2link_all(rw, hw)
        candidates = []
        for j2c, j3c in raw_candidates:
            j4c = solve_wrist(j2c, j3c, w_orient=0.45)
            if within_limits(j2c, j3c, j4c):
                candidates.append((j2c, j3c, j4c))
        if not candidates:
            for j2c, j3c in raw_candidates:
                j4c = solve_wrist(j2c, j3c, w_orient=0.45)
                candidates.append((self._clamp_ik('J2', j2c), self._clamp_ik('J3', j3c), self._clamp_ik('J4', j4c)))

        def score(j2, j3, j4):
            r2, h2, s234 = fk_partial(j2, j3, j4)
            pos_err = np.hypot(r2 - r, h2 - h_target)
            orient_pen = abs(wrap_pi(s234 - s_target)) * 0.8
            return pos_err * 6.0 + orient_pen, abs(j4)

        best = min(candidates, key=lambda t: score(t[0], t[1], t[2]))
        j2_phys, j3_phys, j4_phys = best

        # 소정렬 루프
        for _ in range(3):
            _, _, s234 = fk_partial(j2_phys, j3_phys, j4_phys)
            rw = r - L3*np.sin(s234)
            hw = h_target - L3*np.cos(s234)
            raw_candidates = solve_2link_all(rw, hw)
            tmp = []
            for j2c, j3c in raw_candidates:
                j4c = solve_wrist(j2c, j3c, w_orient=0.45)
                if within_limits(j2c, j3c, j4c):
                    tmp.append((j2c, j3c, j4c))
            if tmp:
                best = min(tmp, key=lambda t: score(t[0], t[1], t[2]))
                j2_phys, j3_phys, j4_phys = best
            else:
                break

        j5_phys = self._clamp_ik('J5', 0.0)
        self.joints = [j1,
                       self._clamp_ik('J2', j2_phys),
                       self._clamp_ik('J3', j3_phys),
                       self._clamp_ik('J4', j4_phys),
                       j5_phys]
        self.update_end_effector()

        # --------------------- DLS 미세조정 ---------------------
        def numeric_jacobian(theta):
            save = list(self.joints)

            def ee_of(ang):
                self.joints = [save[0], float(ang[0]), float(ang[1]), float(ang[2]), save[4]]
                self.update_end_effector()
                ee = np.array(self.end_effector[:3])
                self.joints = list(save)
                self.update_end_effector()
                return ee

            J = np.zeros((3, 3))
            ee0 = ee_of(theta)
            hdeg = 0.5  # (0.2 ~ 1.0)로 조정 가능
            for i in range(3):
                dth = np.array(theta, dtype=float)
                dth[i] += hdeg
                ee1 = ee_of(dth)
                J[:, i] = (ee1 - ee0) / (hdeg * np.pi / 180.0)
            return J

        def signed_step(jname, cur, d):
            new_try = self._clamp_ik(jname, float(cur + d))
            if abs(new_try - cur) < 1e-9 and abs(d) > 1e-9:
                return 0.0
            return new_try - cur

        target = np.array([tx, ty, tz])
        lam = 0.14
        w_xyz = np.array([1.0, 1.0, 1.0])
        w_plane = 2.5
        w_orient = 0.4

        # DLS 반복
        for it in range(8):
            cur_ee = np.array(self.end_effector[:3])
            err_xyz = (target - cur_ee) * w_xyz

            th = np.array([float(self.joints[1]), float(self.joints[2]), float(self.joints[3])], dtype=float)
            J_xyz = numeric_jacobian(th)
            if J_xyz.shape != (3, 3) or np.any(~np.isfinite(J_xyz)):
                break
            Jw = (w_xyz.reshape(3, 1) * J_xyz)

            plane_res = float(np.dot(self.screen_normal, (cur_ee - self.screen_origin)))
            J_plane = np.array([np.dot(self.screen_normal, J_xyz[:, 0]),
                                np.dot(self.screen_normal, J_xyz[:, 1]),
                                np.dot(self.screen_normal, J_xyz[:, 2])], dtype=float) * w_plane
            b_plane = -w_plane * plane_res

            s234 = RAD(self.joints[1] + self.joints[2] + self.joints[3])
            s2, lo2, hi2, rng2 = joint_slack('J2', th[0])
            s3, lo3, hi3, rng3 = joint_slack('J3', th[1])
            s4, lo4, hi4, rng4 = joint_slack('J4', th[2])
            g = np.array([s2/rng2, s3/rng3, s4/rng4], dtype=float)
            J_orient = g * w_orient
            b_orient = -w_orient * np.dot(g, np.array([1.0, 1.0, 1.0])) * wrap_pi(s234 - np.arctan2(self.screen_normal @ er, self.screen_normal @ ez))

            JTJ = Jw.T @ Jw + np.outer(J_plane, J_plane) + np.outer(J_orient, J_orient) + (lam**2) * np.eye(3)
            JTb = Jw.T @ err_xyz + J_plane * b_plane + J_orient * b_orient
            try:
                dtheta_rad = np.linalg.solve(JTJ, JTb)
            except np.linalg.LinAlgError:
                break
            if np.any(~np.isfinite(dtheta_rad)):
                break
            dtheta_deg = np.asarray(dtheta_rad) * 180.0 / np.pi

            d2 = signed_step('J2', th[0], dtheta_deg[0])
            d3 = signed_step('J3', th[1], dtheta_deg[1])
            d4 = signed_step('J4', th[2], dtheta_deg[2])
            th_new = th + np.array([d2, d3, d4])

            self.joints = [j1,
                           self._clamp_ik('J2', float(th_new[0])),
                           self._clamp_ik('J3', float(th_new[1])),
                           self._clamp_ik('J4', float(th_new[2])),
                           j5_phys]
            self.update_end_effector()

            if self.ik_debug:
                if self._ik_trace is None:
                    self._ik_trace = []
                self._ik_trace.append({
                    'it': int(it),
                    'EE': list(map(float, self.end_effector[:3])),
                    'th': [float(self.joints[1]), float(self.joints[2]), float(self.joints[3])],
                    'ddeg': [float(d2), float(d3), float(d4)],
                    'plane_res': float(plane_res),
                    'j4_limits': [lo4, hi4]
                })

            if (np.linalg.norm(target - np.array(self.end_effector[:3])) < 0.4 and
                abs(np.dot(self.screen_normal, (np.array(self.end_effector[:3]) - self.screen_origin))) < 0.2):
                break

        # 안전 Z 보정: 최종 EE가 하한 미만이면 되돌림
        if self.end_effector[2] < self.min_ee_z - 1e-6:
            self.joints = prev
            self.update_end_effector()

        return self.joints

    # ------------------------- 기타 편의 -------------------------
    def update_end_effector(self):
        _ = self.forward_kinematics()
        # self.end_effector는 forward_kinematics에서 갱신됨

    # (선택) 디버그 트레이스 가져오기
    def get_ik_trace(self) -> List[dict] | None:
        return self._ik_trace
