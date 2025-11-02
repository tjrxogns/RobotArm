













































































































































































































































def on_touchscreen_click(self, x, y, w, h):
    self.apply_screen()
    rel_x = x / max(w, 1)
    rel_y = 1 - (y / max(h, 1))
    tx = self.screenX0 + self.screenW * rel_x
    ty = self.screenY0 + self.screenH * rel_y
    tz = self.screenZ0

    u, v, z = tx, ty, self.screenZ0
    wx, wy, wz = self.robot.map_touch_to_world(u, v, z)
    self.clicked_xy = (u, v)
    self.lbl_click.setText(
        f"터치 목표: ({u:.1f}, {v:.1f}, {z:.1f})"
        f"    → EE 실제: ({wx:.1f}, {wy:.1f}, {wz:.1f})"
    )

    # ✅ inverse_kinematics() 결과를 joints에 반영
    angles = self.robot.inverse_kinematics(wx, wy, wz)
    if angles:
        self.robot.joints = angles
        self.robot.update_end_effector()

    self.update_all()
























def apply_screen(self):
    self.screenX0 = float(self.input_screenX0.text())
    self.screenY0 = float(self.input_screenY0.text())
    self.screenZ0 = float(self.input_screenZ0.text())
    self.screenW  = float(self.input_screenW.text())
    self.screenH  = float(self.input_screenH.text())

    w = max(self.touch_label.width(), 1)
    h = max(self.touch_label.height(), 1)
    self.robot.set_screen_calibration(
        origin=[self.screenX0, self.screenY0, self.screenZ0],
        u_axis=[1, 0, 0],
        v_axis=[0, 1, 0],
        su=self.screenW / w,
        sv=self.screenH / h,
        lock_z_to_plane=True
    )

    # ✅ 화면 반영 (시각 갱신)
    self.update_view()
