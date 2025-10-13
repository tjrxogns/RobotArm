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
        
        self.servo_map = {  #물리 각도 / 서보  범위
            0: (0, 180, 0, 180),        # 1번 모터 (베이스 회전)
            1: (0, 90, 120, 30),        # 2번
            2: (-10, 120, 20, 150),     # 3번
            3: (-90, 90, 180, 0),       # 4번
            4: (-90, 90, 0, 180),
        }
        self.sim_limits = {i: (v[0], v[1]) for i, v in self.servo_map.items()}

        # IK 전용 각도 제한(물리각 기준). 필요 시 set_ik_limits()로 변경 가능
        self.ik_limits = {
            'J1': (-180.0, 180.0),
            'J2': (-10, 120.0),
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
        """IK with screen-plane touch constraint & L3 normal alignment.
        Adds joint‑limit awareness so J4가 하한/상한에 붙어도 J2/J3가 더 많이 분담.
        """
        import numpy as np

        EPS = 1e-9
        RAD = np.radians
        DEG = np.degrees

        # ==== helpers ====
        def unit(v):
            v = np.array(v, dtype=float)
            n = np.linalg.norm(v)
            return v / n if n > EPS else v

        def wrap_pi(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        def joint_limits(name):
            # Probe _clamp_ik to infer limits
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
            slack = max(1e-6, min(dmin, dmax))  # distance to nearest bound
            return slack, lo, hi, rng

        # ==== inputs ====
        z_req = max(float(z), getattr(self, 'min_ee_z', 0.0), 0.0)
        tx, ty, tz = float(x), float(y), z_req
        dx, dy, dz = tx - self.base[0], ty - self.base[1], tz - self.base[2]

        prev = list(self.joints)

        # ==== J1 continuity & clamp ====
        j1_full = (DEG(np.arctan2(dy, dx)) + 360.0) % 360.0
        cur_j1 = float(self.joints[0]) if hasattr(self, 'joints') else 0.0
        cands = [j1_full - 360.0, j1_full, j1_full + 360.0]
        j1_sel = min(cands, key=lambda a: abs(((a - cur_j1 + 180) % 360) - 180))
        j1 = normalize_deg(j1_sel)
        j1 = self._clamp_ik('J1', j1)

        # world basis and plane basis
        er = np.array([np.cos(np.radians(j1)), np.sin(np.radians(j1)), 0.0])
        ez = np.array([0.0, 0.0, 1.0])

        # ==== screen normal & origin ====
        if hasattr(self, 'screen_normal') and self.screen_normal is not None:
            n_world = unit(self.screen_normal)
        elif hasattr(self, 'screen_u') and hasattr(self, 'screen_v') and self.screen_u is not None and self.screen_v is not None:
            n_world = unit(np.cross(self.screen_u, self.screen_v))
        else:
            n_world = er.copy()

        if hasattr(self, 'screen_origin') and self.screen_origin is not None:
            P0 = np.array(self.screen_origin, dtype=float)
        elif hasattr(self, 'screen_center') and self.screen_center is not None:
            P0 = np.array(self.screen_center, dtype=float)
        else:
            P0 = np.array(self.base) + er * max(np.hypot(dx, dy), 1.0)

        # project normal into J1 plane to get s_target
        n_plane = (n_world @ er) * er + (n_world @ ez) * ez
        if np.linalg.norm(n_plane) < 1e-6:
            n_plane = er.copy()
        n_plane = unit(n_plane)
        s_target = np.arctan2(n_plane @ er, n_plane @ ez)

        # ==== cylindrical reduction ====
        r = float(np.hypot(dx, dy))
        h = float(dz)
        h_target = max(h, (getattr(self, 'min_ee_z', 0.0) - self.base[2]))

        L1, L2, L3 = float(self.L1), float(self.L2), float(self.L3)

        def clamp01(v):
            return np.clip(v, -1.0, 1.0)

        def within_limits(j2, j3, j4):
            j2c = self._clamp_ik('J2', j2)
            j3c = self._clamp_ik('J3', j3)
            j4c = self._clamp_ik('J4', j4)
            return (abs(j2 - j2c) < 1e-4) and (abs(j3 - j3c) < 1e-4) and (abs(j4 - j4c) < 1e-4)

        def fk_partial(j2, j3, j4):
            s2 = RAD(j2); s23 = RAD(j2 + j3); s234 = RAD(j2 + j3 + j4)
            r2 = L1*np.sin(s2) + L2*np.sin(s23) + L3*np.sin(s234)
            h2 = L1*np.cos(s2) + L2*np.cos(s23) + L3*np.cos(s234)
            return r2, h2, s234

        # initial wrist guess
        dmag = max(np.hypot(r, h_target), EPS)
        ur, uh = r/dmag, h_target/dmag
        rw, hw = r - L3*ur, h_target - L3*uh

        # ==== 2-link enumeration ====
        def solve_2link_all(rw, hw):
            d_w = max(np.hypot(rw, hw), EPS)
            max12 = L1 + L2 - 1e-9
            if d_w > max12:
                scale = max12 / d_w
                rw *= scale; hw *= scale; d_w = max12
            beta = np.arctan2(rw, hw)
            c2 = clamp01((L1**2 + d_w**2 - L2**2) / (2*L1*d_w))
            c3 = clamp01((L1**2 + L2**2 - d_w**2) / (2*L1*L2))
            alpha = np.arccos(c2); gamma = np.arccos(c3)
            j2_up = DEG(beta - alpha); j2_dn = DEG(beta + alpha)
            j3_mag = DEG(np.pi - gamma)
            combos = [(j2_up, +j3_mag),(j2_up,-j3_mag),(j2_dn,+j3_mag),(j2_dn,-j3_mag)]
            seen=set(); out=[]
            for j2c,j3c in combos:
                key=(round(j2c,4),round(j3c,4))
                if key not in seen:
                    seen.add(key); out.append((j2c,j3c))
            return out

        # wrist from r-error & orientation (s_target)
        def solve_wrist(j2, j3, w_orient=0.5):
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
                # if clamped, add penalty so we prefer feasible sign
                penal = 10.0 if abs(j4c - j4) > 1e-6 else 0.0
                cand.append((cost(s3) + penal, j4c))
            cand.sort(key=lambda t: t[0])
            return cand[0][1]

        # candidate scoring
        def score_candidate(j2, j3, j4, cur):
            r2, h2, s234 = fk_partial(j2, j3, j4)
            pos_err = np.hypot(r2 - r, h2 - h_target)
            delta = np.array([j2 - cur[1], j3 - cur[2], j4 - cur[3]])
            move_cost = np.linalg.norm(((delta + 180) % 360) - 180)
            elbow_pref = 0.0
            if h_target > 0 and j3 < 0: elbow_pref = -abs(j3) * 0.4
            elif h_target < 0 and j3 > 0: elbow_pref = -abs(j3) * 0.4
            orient_pen = abs(wrap_pi(s234 - s_target)) * 0.8
            # near‑limit penalty for J4 to avoid sticking at bounds
            s4, lo4, hi4, rng4 = joint_slack('J4', j4)
            near_pen = max(0.0, (5.0 - s4)) * 0.3  # degrees from nearest bound <~5deg penalize
            return pos_err * 6.0 + move_cost * 0.04 + elbow_pref + orient_pen + near_pen

        cur = list(self.joints)

        # near-range DLS pre-pass
        if r < L1 * 0.9:
            j2_init, j3_init = 10.0, 20.0
            j4_init = DEG(s_target) - (j2_init + j3_init)
            self.joints = [j1, self._clamp_ik('J2', j2_init), self._clamp_ik('J3', j3_init), self._clamp_ik('J4', j4_init), 0.0]
            self.update_end_effector()
            target = np.array([tx, ty, tz])
            lam_pre = 0.12
            for _ in range(8):
                cur_ee = np.array(self.end_effector[:3]); err = target - cur_ee
                if np.linalg.norm(err) < 0.5: break
                th = np.array([self.joints[1], self.joints[2], self.joints[3]], dtype=float)
                J = np.zeros((3,3)); ee0 = cur_ee; hstep = 1e-2
                for i in range(3):
                    dth = th.copy(); dth[i] += hstep
                    self.joints[1:4] = dth; self.update_end_effector(); ee1 = np.array(self.end_effector[:3])
                    J[:, i] = (ee1 - ee0) / (hstep * np.pi/180.0)
                self.joints[1:4] = th; self.update_end_effector()
                JT = J.T; H = JT @ J + (lam_pre**2)*np.eye(3)
                dtheta = np.linalg.solve(H, JT @ err) * 180.0/np.pi
                th = th + dtheta
                self.joints[1:4] = [self._clamp_ik('J2', th[0]), self._clamp_ik('J3', th[1]), self._clamp_ik('J4', th[2])]
                self.update_end_effector()

        # candidate set
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

        best = min(candidates, key=lambda t: (score_candidate(t[0], t[1], t[2], cur), abs(t[2])))  # prefer smaller |J4|
        j2_phys, j3_phys, j4_phys = best

        # small realignment loop
        for _ in range(3):
            _, _, s234 = fk_partial(j2_phys, j3_phys, j4_phys)
            rw = r - L3*np.sin(s234); hw = h_target - L3*np.cos(s234)
            raw_candidates = solve_2link_all(rw, hw)
            tmp = []
            for j2c, j3c in raw_candidates:
                j4c = solve_wrist(j2c, j3c, w_orient=0.45)
                if within_limits(j2c, j3c, j4c): tmp.append((j2c, j3c, j4c))
            if tmp:
                best = min(tmp, key=lambda t: (score_candidate(t[0], t[1], t[2], cur), abs(t[2])))
                j2_phys, j3_phys, j4_phys = best
            else:
                break

        j5_phys = self._clamp_ik('J5', 0.0)
        self.joints = [j1, self._clamp_ik('J2', j2_phys), self._clamp_ik('J3', j3_phys), self._clamp_ik('J4', j4_phys), j5_phys]
        self.update_end_effector()

        # ==== Final DLS (XYZ + plane + orientation with slack weighting) ====
        def numeric_jacobian(theta):
            save = list(self.joints)
            def ee_of(ang):
                self.joints = [save[0], float(ang[0]), float(ang[1]), float(ang[2]), save[4]]
                self.update_end_effector(); ee = np.array(self.end_effector[:3])
                self.joints = list(save); self.update_end_effector(); return ee
            J = np.zeros((3,3)); ee0 = ee_of(theta); hstep = 1e-2
            for i in range(3):
                dth = np.array(theta, dtype=float); dth[i] += hstep
                ee1 = ee_of(dth)
                J[:, i] = (ee1 - ee0) / (hstep * np.pi/180.0)
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
        w_orient = 0.4  # 낮춤 (자세는 J4 한계 시 과도하지 않게)

        for it in range(8):
            cur_ee = np.array(self.end_effector[:3])
            err_xyz = (target - cur_ee) * w_xyz

            th = np.array([float(self.joints[1]), float(self.joints[2]), float(self.joints[3])], dtype=float)
            J_xyz = numeric_jacobian(th)
            if J_xyz.shape != (3,3) or np.any(~np.isfinite(J_xyz)):
                break
            Jw = (w_xyz.reshape(3,1) * J_xyz)

            plane_res = float(np.dot(n_world, (cur_ee - P0)))
            J_plane = np.array([np.dot(n_world, J_xyz[:,0]), np.dot(n_world, J_xyz[:,1]), np.dot(n_world, J_xyz[:,2])], dtype=float) * w_plane
            b_plane = -w_plane * plane_res

            s234 = RAD(self.joints[1] + self.joints[2] + self.joints[3])
            # slack‑weighted orientation: joints near limits receive less orientation push
            s2, lo2, hi2, rng2 = joint_slack('J2', th[0])
            s3, lo3, hi3, rng3 = joint_slack('J3', th[1])
            s4, lo4, hi4, rng4 = joint_slack('J4', th[2])
            g = np.array([s2/rng2, s3/rng3, s4/rng4], dtype=float)
            J_orient = g * w_orient
            b_orient = -w_orient * np.dot(g, np.array([1.0,1.0,1.0])) * wrap_pi(s234 - s_target)

            JTJ = Jw.T @ Jw + np.outer(J_plane, J_plane) + np.outer(J_orient, J_orient) + (lam**2) * np.eye(3)
            JTb = Jw.T @ err_xyz + J_plane * b_plane + J_orient * b_orient
            try:
                dtheta_rad = np.linalg.solve(JTJ, JTb)
            except np.linalg.LinAlgError:
                break
            if np.any(~np.isfinite(dtheta_rad)):
                break
            dtheta_deg = np.asarray(dtheta_rad) * 180.0/np.pi

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

            if getattr(self, 'ik_debug', False):
                if not hasattr(self, '_ik_trace'): self._ik_trace = []
                self._ik_trace.append({'it': int(it), 'EE': list(map(float, self.end_effector[:3])), 'th': [float(self.joints[1]), float(self.joints[2]), float(self.joints[3])], 'ddeg': [float(d2), float(d3), float(d4)], 'plane_res': float(plane_res), 'j4_limits': [lo4, hi4]})

            if (np.linalg.norm(target - np.array(self.end_effector[:3])) < 0.4 and abs(np.dot(n_world, (np.array(self.end_effector[:3]) - P0))) < 0.2):
                break

        if self.end_effector[2] < getattr(self, 'min_ee_z', 0.0) - 1e-6:
            self.joints = prev
            self.update_end_effector()

        return self.joints



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
