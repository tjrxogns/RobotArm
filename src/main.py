from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QLineEdit, QFormLayout, QFrame, QComboBox
)
from PySide6.QtGui import QDoubleValidator, QMouseEvent, QPainter, QColor
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import sys

# ─────────────────────────────────────────────────────────────
# 외부 모듈
#   - robot_arm_251014.py 의 RobotArm 클래스를 사용합니다.
#   - serial_comm.py 는 기존 프로젝트 파일을 그대로 사용합니다.
# ─────────────────────────────────────────────────────────────
from robot_arm import RobotArm
try:
    from serial_comm import SerialComm
except Exception:
    class SerialComm:
        def __init__(self): pass
        def connect(self, *_): return False
        def send_command(self, *_): pass
        def close(self): pass
        @staticmethod
        def list_ports(): return []
        class _Sig: 
            def connect(self, *_): pass
        status_received = _Sig()

# ─────────────────────────────────────────────────────────────
# 터치스크린 위젯 (2D)
# ─────────────────────────────────────────────────────────────
class TouchScreenSim(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: #bfe7ff; border: 2px solid #888;")
        self.click_pos = None
        self.parent_viewer = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.click_pos = event.position().toPoint()
            self.update()
            if self.parent_viewer:
                self.parent_viewer.on_touchscreen_click(
                    self.click_pos.x(), self.click_pos.y(), self.width(), self.height()
                )

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.click_pos:
            painter = QPainter(self)
            painter.setBrush(QColor(255, 64, 64))
            painter.drawEllipse(self.click_pos, 6, 6)

# ─────────────────────────────────────────────────────────────
# 메인 뷰어 (UI 담당)
# ─────────────────────────────────────────────────────────────
class RobotArmViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로봇팔 3D 뷰어 — UI 전체판 (251024)")
        self.setGeometry(50, 50, 1000, 680)

        self.robot = RobotArm()
        self.serial = SerialComm()
        try:
            self.serial.status_received.connect(self.on_status_received)
        except Exception:
            pass

        self.screenX0, self.screenY0, self.screenZ0 = -50.0, 60.0, 10.0
        self.screenW, self.screenH = 155.0, 90.0
        self.clicked_xy = None
        self.view_mode = '3d'

        self._init_ui()
        self._enable_text_selection(self)
        self.apply_screen()
        self.update_all()

    def _enable_text_selection(self, widget):
        for child in widget.findChildren(QWidget):
            if hasattr(child, "setTextInteractionFlags"):
                try:
                    child.setTextInteractionFlags(Qt.TextSelectableByMouse)
                except Exception:
                    pass
            self._enable_text_selection(child)

    def _init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left = QFrame(); left_layout = QVBoxLayout(left)
        self.fig = Figure(); self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(left, stretch=2)

        right = QFrame(); right.setFixedWidth(430)
        right_layout = QVBoxLayout(right)

        port_bar = QHBoxLayout()
        self.port_combo = QComboBox()
        ports = SerialComm.list_ports()
        self.port_combo.addItems(ports if ports else ["포트 없음"])
        self.btn_refresh = QPushButton("새로고침"); self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect = QPushButton("연결"); self.btn_connect.clicked.connect(self.connect_serial)
        self.lbl_status = QLabel("상태: 연결 안됨")
        for w in (self.port_combo, self.btn_refresh, self.btn_connect, self.lbl_status):
            port_bar.addWidget(w)
        right_layout.addLayout(port_bar)

        vbar = QHBoxLayout()
        for text, mode in [("3D", '3d'), ("XY", 'xy'), ("YZ", 'yz'), ("ZX", 'zx')]:
            b = QPushButton(text)
            b.clicked.connect(lambda _, m=mode: self.set_view(m))
            vbar.addWidget(b)
        right_layout.addLayout(vbar)

        right_layout.addWidget(QLabel("<b>터치스크린 위치/크기</b>"))
        form = QFormLayout()
        self.ed_screenX0 = QLineEdit(str(self.screenX0))
        self.ed_screenY0 = QLineEdit(str(self.screenY0))
        self.ed_screenZ0 = QLineEdit(str(self.screenZ0))
        self.ed_screenW  = QLineEdit(str(self.screenW))
        self.ed_screenH  = QLineEdit(str(self.screenH))
        for ed in [self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0, self.ed_screenW, self.ed_screenH]:
            ed.setValidator(QDoubleValidator(-10000.0, 10000.0, 3))
            ed.setMaximumWidth(80)
        form.addRow("X0", self.ed_screenX0)
        form.addRow("Y0", self.ed_screenY0)
        form.addRow("Z0", self.ed_screenZ0)
        form.addRow("W", self.ed_screenW)
        form.addRow("H", self.ed_screenH)
        right_layout.addLayout(form)
        btn_apply = QPushButton("터치스크린 적용"); btn_apply.clicked.connect(self.apply_screen)
        right_layout.addWidget(btn_apply)

        right_layout.addWidget(QLabel("<b>가상 터치스크린</b>"))
        self.touch_sim = TouchScreenSim(); self.touch_sim.parent_viewer = self
        right_layout.addWidget(self.touch_sim)
        self.lbl_click = QLabel("터치: (---, ---) @Z=---    EE: (---, ---, ---)")
        right_layout.addWidget(self.lbl_click)

        right_layout.addWidget(QLabel("<b>Joint 제어</b>"))
        self.joint_sliders, self.joint_labels, self.joint_phys_labels = {}, {}, {}
        for i in range(5):
            hl = QHBoxLayout()
            lbl = QLabel(f"J{i}:")
            sld = QSlider(Qt.Horizontal)
            sim_min, sim_max, _, _ = self.robot.servo_map[i]
            sld.setRange(int(sim_min), int(sim_max))
            sld.setValue(int(self.robot.joints[i]))
            sld.valueChanged.connect(self.update_from_joints)
            lbl_ang = QLabel("0°"); lbl_ang.setFixedWidth(50)
            lbl_phys = QLabel("(0°)"); lbl_phys.setFixedWidth(50)
            hl.addWidget(lbl); hl.addWidget(sld); hl.addWidget(lbl_ang); hl.addWidget(lbl_phys)
            right_layout.addLayout(hl)
            self.joint_sliders[i] = sld
            self.joint_labels[i] = lbl_ang
            self.joint_phys_labels[i] = lbl_phys

        comm = QHBoxLayout()
        btn_all = QPushButton("전체 각도 전송 (ALL)"); btn_all.clicked.connect(self.send_all_servos)
        btn_off = QPushButton("모든 서보 토크 해제"); btn_off.clicked.connect(self.disable_all_servos)
        comm.addWidget(btn_all); comm.addWidget(btn_off)
        right_layout.addLayout(comm)
        self.lbl_status_recv = QLabel("상태회신: ---"); right_layout.addWidget(self.lbl_status_recv)

        right_layout.addStretch(1)
        main_layout.addWidget(right, stretch=1)

    def refresh_ports(self):
        ports = SerialComm.list_ports()
        self.port_combo.clear(); self.port_combo.addItems(ports if ports else ["포트 없음"])

    def connect_serial(self):
        port = self.port_combo.currentText()
        if self.serial.connect(port):
            self.lbl_status.setText(f"상태: {port} 연결됨")
        else:
            self.lbl_status.setText("상태: 연결 실패")

    def send_all_servos(self):
        angs = [int(a) for a in self.robot.get_servo_angles()]
        self.serial.send_command("ALL " + " ".join(map(str, angs)))

    def disable_all_servos(self):
        self.serial.send_command("Disable All")

    def on_status_received(self, angles):
        if len(angles) == 5:
            #self.robot.set_servo_angles(angles)
            self.lbl_status_recv.setText(f"상태회신: {', '.join(map(str, angles))}")
            self.update_all()

    def on_touchscreen_click(self, x, y, w, h):
        self.apply_screen()
        rel_x = x / max(w, 1)
        rel_y = 1 - (y / max(h, 1))
        tx = self.screenX0 + self.screenW * rel_x
        ty = self.screenY0 + self.screenH * rel_y
        tz = self.screenZ0
        base = self.robot.base
        j1 = np.degrees(np.arctan2(ty - base[1], tx - base[0]))
        self.robot.joints[0] = j1
        joints = self.robot.inverse_kinematics(tx, ty, tz)
        ee = self.robot.end_effector
        self.clicked_xy = (tx, ty)
        self.lbl_click.setText(
            f"터치: ({tx:.1f}, {ty:.1f}) @Z={tz:.1f} | J1={j1:.1f}° EE: ({ee[0]:.1f}, {ee[1]:.1f}, {ee[2]:.1f})"
        )
        self.update_all()

    def apply_screen(self):
        try:
            self.screenX0 = float(self.ed_screenX0.text())
            self.screenY0 = float(self.ed_screenY0.text())
            self.screenZ0 = float(self.ed_screenZ0.text())
            self.screenW  = float(self.ed_screenW.text())
            self.screenH  = float(self.ed_screenH.text())
        except Exception:
            pass
        wpx = max(self.touch_sim.width(), 1)
        hpx = max(self.touch_sim.height(), 1)
        su = self.screenW / wpx
        sv = self.screenH / hpx
        self.robot.set_screen_calibration(
            origin=[self.screenX0, self.screenY0, self.screenZ0],
            u_axis=[1, 0, 0], v_axis=[0, 1, 0], su=su, sv=sv,
            lock_z_to_plane=True
        )

    def set_view(self, mode):
        self.view_mode = mode
        self.update_view()

    def update_view(self):
        self.ax.clear()
        origin = np.array([0.0, 0.0, 0.0])
        axes = np.eye(3) * 50
        colors = ['r', 'g', 'b']
        for i in range(3):
            self.ax.quiver(origin[0], origin[1], origin[2], axes[0, i], axes[1, i], axes[2, i], color=colors[i], linewidth=2)
        pts = self.robot.forward_kinematics()
        xs, ys, zs = zip(*pts)
        self.ax.plot(xs, ys, zs, marker='o', linewidth=3, color='blue')
        j2_pos, j3_pos, j4_pos, ee_pos = self.robot.joint_positions()
        self.ax.scatter([j2_pos[0]], [j2_pos[1]], [j2_pos[2]], color='orange', s=50)
        self.ax.scatter([j3_pos[0]], [j3_pos[1]], [j3_pos[2]], color='purple', s=50)
        self.ax.scatter([j4_pos[0]], [j4_pos[1]], [j4_pos[2]], color='cyan', s=40)
        self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], color='red', s=50)
        jdeg = self.robot.joints
        servo = self.robot.get_servo_angles()
        bx, by, bz = self.robot.base
        self.ax.text(bx, by, bz + 10, f"0 {jdeg[0]:.1f}° | S0 {servo[0]}°", fontsize=8)
        self.ax.text(j3_pos[0], j3_pos[1], j3_pos[2] + 8, f"2 {jdeg[2]:.1f}° | S2 {servo[2]}°", fontsize=8)
        self.ax.text(j4_pos[0], j4_pos[1], j4_pos[2] + 8, f"3 {jdeg[3]:.1f}° | S3 {servo[3]}°", fontsize=8)
        self.ax.text(ee_pos[0], ee_pos[1], ee_pos[2] + 10, f"4 {jdeg[4]:.1f}° | S4 {servo[4]}°", fontsize=8)
        x0, y0, z0, w, h = self.screenX0, self.screenY0, self.screenZ0, self.screenW, self.screenH
        verts = [[(x0, y0, z0), (x0 + w, y0, z0), (x0 + w, y0 + h, z0), (x0, y0 + h, z0)]]
        poly = Poly3DCollection(verts, alpha=0.2, facecolor='cyan', edgecolor='black')
        self.ax.add_collection3d(poly)
        if self.clicked_xy is not None:
            cx, cy = self.clicked_xy
            self.ax.scatter([cx], [cy], [self.screenZ0], color='magenta', s=60)
        lim = 220
        self.ax.set_xlim(-lim/2, lim/2)
        self.ax.set_ylim(-10, lim)
        self.ax.set_zlim(-10, lim)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        views = {'xy': (90, -90), 'yz': (0, 0), 'zx': (0, 90), '3d': (30, 45)}
        elev, azim = views.get(self.view_mode, (30, 45))
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def update_from_joints(self):
        for i, s in self.joint_sliders.items():
            self.robot.joints[i] = float(s.value())
        self.robot.update_end_effector()
        self.update_all()

    def update_all(self):
        self._block_signals(True)
        self._sync_joint_sliders()
        self._block_signals(False)
        self.update_view()

    def _block_signals(self, state):
        for s in list(self.joint_sliders.values()):
            s.blockSignals(state)

    def _sync_joint_sliders(self):
        for i in range(5):
            v = float(self.robot.joints[i])
            sim_min, sim_max, _, _ = self.robot.servo_map[i]
            v = min(max(v, sim_min), sim_max)
            self.joint_sliders[i].setValue(int(v))
            self.joint_labels[i].setText(f"{int(v)}°")
            self.joint_phys_labels[i].setText(f"({v:.1f}°)")

    def closeEvent(self, event):
        try:
            if hasattr(self, 'serial') and self.serial:
                self.serial.close()
        finally:
            event.accept()

def main():
    app = QApplication(sys.argv)
    w = RobotArmViewer(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
