from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QLineEdit, QFormLayout, QFrame, QComboBox
)
from PySide6.QtGui import QDoubleValidator, QMouseEvent, QPainter, QColor
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from robot_arm import RobotArm
from serial_comm import SerialComm
import sys, numpy as np

class EMA:
    def __init__(self, alpha=0.25, init=None):
        self.a = alpha; self.y = init
    def update(self, x):
        self.y = x if self.y is None else (self.a*x + (1-self.a)*self.y)
        return self.y

class TouchScreenSim(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: lightblue; border: 2px solid gray;")
        self.click_pos = None; self.parent_viewer = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.click_pos = event.position().toPoint()
            self.update()
            if self.parent_viewer:
                self.parent_viewer.on_touchscreen_click(self.click_pos.x(), self.click_pos.y(), self.width(), self.height())

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.click_pos:
            painter = QPainter(self)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(self.click_pos, 6, 6)

class RobotArmViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로봇팔 3D 뷰어 (251011)")
        self.setGeometry(0, 0, 800, 500)

        self.robot = RobotArm()
        self.serial = SerialComm()

        self.view_mode = '3d'
        self.screenX0, self.screenY0, self.screenZ0 = -50.0, 60.0, 10.0
        self.screenW, self.screenH = 155.0, 90.0
        self.clicked_xy = None
        self.xyz_ema = {k: EMA(0.3) for k in ['X','Y','Z']}
        self.view_zoom = 2.0

        self.init_ui()
        self.update_view()

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_frame = QFrame(); left_layout = QVBoxLayout(left_frame)
        self.fig = Figure(); self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        left_layout.addWidget(self.canvas); main_layout.addWidget(left_frame, stretch=2)

        right_frame = QFrame(); right_frame.setFixedWidth(300)
        right_layout = QVBoxLayout(right_frame)

        # === 시리얼 포트 선택 ===
        port_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        ports = SerialComm.list_ports()
        self.port_combo.addItems(ports if ports else ["포트 없음"])
        self.btn_refresh = QPushButton("새로고침")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect = QPushButton("연결")
        self.btn_connect.clicked.connect(self.connect_serial)
        self.lbl_status = QLabel("상태: 연결 안됨")
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(self.btn_refresh)
        port_layout.addWidget(self.btn_connect)
        port_layout.addWidget(self.lbl_status)
        right_layout.addLayout(port_layout)

        view_bar = QHBoxLayout()
        for text, mode in [("3D", '3d'), ("XY", 'xy'), ("YZ", 'yz'), ("ZX", 'zx')]:
            b = QPushButton(text)
            b.clicked.connect(lambda _, m=mode: self.set_view(m))
            view_bar.addWidget(b)
        right_layout.addLayout(view_bar)

        xyz_label = QLabel("<b>엔드이펙터 XYZ 제어</b>"); right_layout.addWidget(xyz_label)
        self.sliders = {}
        xyz_row_sliders = QHBoxLayout()
        for axis in ["X","Y","Z"]:
            sub = QHBoxLayout()
            lbl = QLabel(f"{axis}:")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(-200, 200)
            sld.setValue(0)
            sld.setTracking(True)
            sld.valueChanged.connect(self.update_from_xyz)
            sld.sliderReleased.connect(self.update_from_xyz)
            sub.addWidget(lbl)
            sub.addWidget(sld)
            wrap = QWidget(); wrap.setLayout(sub)
            xyz_row_sliders.addWidget(wrap)
            self.sliders[axis] = sld
            right_layout.addLayout(xyz_row_sliders)

        joint_label = QLabel("<b>Joint 제어 (서보 각도 / 물리각)</b>"); right_layout.addWidget(joint_label)
        self.joint_sliders={}; self.joint_labels={}
        self.joint_physical_labels = {}   # ← 이 줄 추가
        for i in range(5):
            hl = QHBoxLayout()
            lbl = QLabel(f"J{i+1}:")
            sld = QSlider(Qt.Horizontal)

            # ✅ servo_map의 시뮬레이터 범위를 사용 (물리각 범위)
            sim_min, sim_max, _, _ = self.robot.servo_map[i]
            sld.setRange(int(sim_min), int(sim_max))

            sld.setValue(0)
            sld.setTracking(True)
            sld.valueChanged.connect(self.update_from_joints)
            sld.sliderReleased.connect(self.update_from_joints)

            lbl_ang  = QLabel("0°");  lbl_ang.setFixedWidth(60)
            lbl_phys = QLabel("(0°)"); lbl_phys.setFixedWidth(60)
            btn_send_one = QPushButton("▶")
            btn_send_one.setFixedWidth(30)
            btn_send_one.clicked.connect(lambda _, idx=i: self.send_single_joint(idx))

            hl.addWidget(lbl); hl.addWidget(sld); hl.addWidget(lbl_ang); hl.addWidget(lbl_phys); hl.addWidget(btn_send_one)
            right_layout.addLayout(hl)

            self.joint_sliders[i] = sld
            self.joint_labels[i] = lbl_ang
            self.joint_physical_labels[i] = lbl_phys

        form = QFormLayout()
        self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0 = QLineEdit(str(self.screenX0)), QLineEdit(str(self.screenY0)), QLineEdit(str(self.screenZ0))
        self.ed_screenW, self.ed_screenH = QLineEdit(str(self.screenW)), QLineEdit(str(self.screenH))
        for ed in [self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0, self.ed_screenW, self.ed_screenH]:
            ed.setValidator(QDoubleValidator(-10000.0, 10000.0, 2)); ed.setMaximumWidth(120)
        xyz_row = QWidget()
        xyz_lay = QHBoxLayout(xyz_row); xyz_lay.setContentsMargins(0,0,0,0); xyz_lay.setSpacing(6)
        xyz_lay.addWidget(QLabel("X0")); xyz_lay.addWidget(self.ed_screenX0)
        xyz_lay.addWidget(QLabel("Y0")); xyz_lay.addWidget(self.ed_screenY0)
        xyz_lay.addWidget(QLabel("Z0")); xyz_lay.addWidget(self.ed_screenZ0)
        form.addRow(QLabel("Screen XYZ"), xyz_row)
        size_row = QWidget(); size_lay = QHBoxLayout(size_row)
        size_lay.setContentsMargins(0,0,0,0); size_lay.setSpacing(6)
        size_lay.addWidget(QLabel("W")); size_lay.addWidget(self.ed_screenW)
        size_lay.addWidget(QLabel("H")); size_lay.addWidget(self.ed_screenH)
        form.addRow(QLabel("Screen Size"), size_row)
        btn_apply = QPushButton("터치스크린 적용"); btn_apply.clicked.connect(self.apply_screen)
        right_layout.addLayout(form); right_layout.addWidget(btn_apply)

        right_layout.addWidget(QLabel("<b>가상 터치스크린</b>"))
        self.touch_sim = TouchScreenSim(); self.touch_sim.parent_viewer=self
        right_layout.addWidget(self.touch_sim)

        self.lbl_click=QLabel("터치스크린 클릭 위치: (---,---) @Z=---"); right_layout.addWidget(self.lbl_click)
        btn_send=QPushButton("현재 각도 전송"); btn_send.clicked.connect(self.send_to_robot)
        right_layout.addWidget(btn_send)

        self.lbl_ee = QLabel("EE: (0.0, 0.0, 0.0)")
        right_layout.addWidget(self.lbl_ee)
        right_layout.addStretch(1)
        main_layout.addWidget(right_frame, stretch=1)

    def refresh_ports(self):
        ports = SerialComm.list_ports()
        self.port_combo.clear()
        self.port_combo.addItems(ports if ports else ["포트 없음"])

    def connect_serial(self):
        port = self.port_combo.currentText()
        if self.serial.connect(port):
            self.lbl_status.setText(f"상태: {port} 연결됨")
        else:
            self.lbl_status.setText("상태: 연결 실패")
            
    def send_single_joint(self, idx):
        angle = int(self.robot.get_servo_angles()[idx])
        self.serial.send_angle(idx + 1, angle)

    def on_touchscreen_click(self, x, y, w, h):
        self.apply_screen()
        rel_x = x / max(w, 1)
        rel_y = 1 - (y / max(h, 1))
        tx = self.screenX0 + self.screenW * rel_x
        ty = self.screenY0 + self.screenH * rel_y
        tz = self.screenZ0
        self.clicked_xy = (tx, ty)
        self.lbl_click.setText(f"터치: ({tx:.1f}, {ty:.1f}) @Z={tz:.1f}")
        self.robot.inverse_kinematics(tx, ty, tz)
        self.update_all()

    def update_from_xyz(self):
        x=self.xyz_ema['X'].update(self.sliders['X'].value())
        y=self.xyz_ema['Y'].update(self.sliders['Y'].value())
        z=self.xyz_ema['Z'].update(self.sliders['Z'].value())
        self.robot.inverse_kinematics(x,y,z)
        self.update_all()

    def update_from_joints(self):
        for i,s in self.joint_sliders.items():
            self.robot.joints[i]=s.value()
        self.robot.update_end_effector(); self.update_all()

    def set_view(self, mode):
        self.view_mode = mode; self.update_view()

    def apply_screen(self):
        self.screenX0, self.screenY0, self.screenZ0 = float(self.ed_screenX0.text()), float(self.ed_screenY0.text()), float(self.ed_screenZ0.text())
        self.screenW, self.screenH = float(self.ed_screenW.text()), float(self.ed_screenH.text())
        self.robot.set_min_ee_z(self.screenZ0)
        self.update_view()

    def update_all(self):
        self.block_signals(True)
        self.sync_joint_sliders(); self.sync_xyz_sliders(); self.update_servo_labels()
        self.block_signals(False); self.update_view()

    def block_signals(self,state):
        for s in list(self.sliders.values())+list(self.joint_sliders.values()): s.blockSignals(state)

    def sync_joint_sliders(self):
        for i in range(5):
            v = float(self.robot.joints[i])
            sim_min, sim_max, _, _ = self.robot.servo_map[i]
            v = min(max(v, sim_min), sim_max)  # clamp
            self.joint_sliders[i].setValue(int(v))

    def sync_xyz_sliders(self):
        x, y, z = self.robot.end_effector
        self.sliders['X'].setValue(int(x))
        self.sliders['Y'].setValue(int(y))
        self.sliders['Z'].setValue(int(z))

    def update_servo_labels(self):
        """서보각 라벨과 물리각(계산용) 라벨을 동기화한다."""
        angs = self.robot.get_servo_angles() if hasattr(self.robot, 'get_servo_angles') else []
        # 로봇 모델의 joints가 항상 최신이 아닐 수 있어, 안전하게 슬라이더 값으로도 보조한다.
        for i in range(5):
            servo_ang = angs[i] if i < len(angs) else int(self.joint_sliders[i].value())
            # 물리각: 우선 robot.joints, 없거나 0으로만 유지되면 슬라이더 값으로 대체
            phys_src = getattr(self.robot, 'joints', None)
            phys_ang = float(phys_src[i]) if (phys_src is not None) else float(self.joint_sliders[i].value())
            self.joint_labels[i].setText(f"{servo_ang}°")
            self.joint_physical_labels[i].setText(f"({phys_ang:.1f}°)")

    def send_to_robot(self): self.serial.send_angles(self.robot.get_servo_angles())

    def closeEvent(self, event):
        if hasattr(self, 'serial') and self.serial:
            self.serial.close()
        event.accept()

    def draw_touch_screen(self, half=None):
        x0,y0,z0,w,h=self.screenX0,self.screenY0,self.screenZ0,self.screenW,self.screenH
        if half is None:
            verts=[[(x0,y0,z0),(x0+w,y0,z0),(x0+w,y0+h,z0),(x0,y0+h,z0)]]
        else:
            verts=[[(x0-half,y0-half,z0),(x0+half,y0-half,z0),(x0+half,y0+half,z0),(x0-half,y0+half,z0)]]
        poly=Poly3DCollection(verts,alpha=0.25,facecolor='cyan',edgecolor='black')
        self.ax.add_collection3d(poly)

    def update_view(self):
        self.ax.clear()
        pts=self.robot.forward_kinematics(); xs,ys,zs=zip(*pts)
        self.ax.plot(xs,ys,zs,marker='o',linewidth=3,color='blue')
        j2_pos, j3_pos, j4_pos, ee_pos = self.robot.joint_positions()
        self.ax.scatter([j2_pos[0]],[j2_pos[1]],[j2_pos[2]], color='orange', s=50)
        self.ax.scatter([j3_pos[0]],[j3_pos[1]],[j3_pos[2]], color='purple', s=50)
        self.ax.scatter([j4_pos[0]],[j4_pos[1]],[j4_pos[2]], color='cyan', s=40)
        self.ax.scatter([ee_pos[0]],[ee_pos[1]],[ee_pos[2]], color='red', s=50)
        jdeg = self.robot.joints
        bx, by, bz = self.robot.base
        phys = self.robot.joints                      # 물리각(계산용)
        servo = self.robot.get_servo_angles()         # 서보각(전송용)
        self.ax.text(bx, by, bz + 10,               f"J1 {phys[0]:.1f}° | S1 {servo[0]}°", fontsize=8)
        self.ax.text(j2_pos[0], j2_pos[1], j2_pos[2] + 8,  f"J2 {phys[1]:.1f}° | S2 {servo[1]}°", fontsize=8)
        self.ax.text(j3_pos[0], j3_pos[1], j3_pos[2] + 8,  f"J3 {phys[2]:.1f}° | S3 {servo[2]}°", fontsize=8)
        self.ax.text(j4_pos[0], j4_pos[1], j4_pos[2] + 8,  f"J4 {phys[3]:.1f}° | S4 {servo[3]}°", fontsize=8)
        self.ax.text(ee_pos[0], ee_pos[1], ee_pos[2] + 12, f"J5 {phys[4]:.1f}° | S5 {servo[4]}°", fontsize=8)
        self.ax.text(ee_pos[0], ee_pos[1], ee_pos[2] + 5, f"EE ({ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f})", fontsize=8)
        if hasattr(self, 'lbl_ee'):
            self.lbl_ee.setText(f"EE: ({ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f})")
        self.draw_touch_screen()

        if hasattr(self, 'joint_labels') and hasattr(self, 'joint_physical_labels'):
            self.update_servo_labels()

        if self.clicked_xy is not None:
            cx, cy = self.clicked_xy; self.ax.scatter([cx],[cy],[self.screenZ0],color='magenta',s=60)
        lim=220
        self.ax.set_xlim(-lim,lim)
        self.ax.set_ylim(-lim/2,lim)
        self.ax.set_zlim(-lim/2,lim)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        views={'xy':(90,-90),'yz':(0,0),'zx':(0,90),'3d':(30,45)}
        elev,azim=views[self.view_mode]; self.ax.view_init(elev=elev,azim=azim)
        zx = getattr(self, "view_zoom", 2.0)
        if zx and zx != 1.0:
            for getter, setter in [
                (self.ax.get_xlim3d, self.ax.set_xlim3d),
                (self.ax.get_ylim3d, self.ax.set_ylim3d),
                (self.ax.get_zlim3d, self.ax.set_zlim3d),
            ]:
                lo, hi = getter(); c = (lo + hi) / 2.0; half = (hi - lo) / (2.0 * zx); setter(c - half, c + half)
        span = max(
            self.ax.get_xlim3d()[1]-self.ax.get_xlim3d()[0],
            self.ax.get_ylim3d()[1]-self.ax.get_ylim3d()[0],
            self.ax.get_zlim3d()[1]-self.ax.get_zlim3d()[0],
        )
        ymin, ymax = self.ax.get_ylim3d(); zmin, zmax = self.ax.get_zlim3d()
        if ymin < -10: self.ax.set_ylim(-10, max(ymax, -9))
        if zmin < -10: self.ax.set_zlim(-10, max(zmax, -9))
        axis_len = 0.2 * span; s = 0.03 * span
        bx, by, bz = self.robot.base
        self.ax.plot([bx, bx], [by, by], [0, bz], linewidth=4, color='k')
        xs = [bx-s, bx+s, bx+s, bx-s, bx-s]; ys = [by-s, by-s, by+s, by+s, by-s]; zs = [bz, bz, bz, bz, bz]
        self.ax.plot(xs, ys, zs, color='k')
        self.ax.quiver(bx,by,bz, axis_len,0,0, color='r', arrow_length_ratio=0.2)
        self.ax.quiver(bx,by,bz, 0,axis_len,0, color='g', arrow_length_ratio=0.2)
        self.ax.quiver(bx,by,bz, 0,0,axis_len, color='b', arrow_length_ratio=0.2)
        self.canvas.draw_idle()

def main():
    app=QApplication(sys.argv)
    w=RobotArmViewer(); w.show()
    sys.exit(app.exec())

if __name__=='__main__':
    main()
