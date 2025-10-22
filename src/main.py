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
import time

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
                self.parent_viewer.on_touchscreen_click(
                    self.click_pos.x(), self.click_pos.y(), self.width(), self.height()
                )

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.click_pos:
            painter = QPainter(self)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(self.click_pos, 6, 6)

class RobotArmViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로봇팔 3D 뷰어 (통신 + 상태회신 + 터치스크린 완전판)")
        self.setGeometry(50, 50, 950, 650)

        self.robot = RobotArm()
        self.serial = SerialComm()
        self.serial.status_received.connect(self.on_status_received)

        self.view_mode = '3d'
        self.screenX0, self.screenY0, self.screenZ0 = -50.0, 60.0, 10.0
        self.screenW, self.screenH = 155.0, 90.0
        self.clicked_xy = None
        self.xyz_ema = {k: EMA(0.3) for k in ['X','Y','Z']}
        self.view_zoom = 2.0

        self.last_update_time = 0

        self.init_ui()
        self.enable_text_selection(self)

    def enable_text_selection(self, widget):
        """모든 하위 위젯의 텍스트를 복사 가능하게 설정 (화면 모양 변화 없음)"""
        from PySide6.QtWidgets import QWidget
        for child in widget.findChildren(QWidget):
            if hasattr(child, "setTextInteractionFlags"):
                try:
                    child.setTextInteractionFlags(Qt.TextSelectableByMouse)
                except Exception:
                    pass
            self.enable_text_selection(child)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(left_frame, stretch=2)

        right_frame = QFrame()
        right_frame.setFixedWidth(420)
        right_layout = QVBoxLayout(right_frame)

        # === 시리얼 포트 ===
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

        # === 뷰 전환 ===
        view_bar = QHBoxLayout()
        for text, mode in [("3D", '3d'), ("XY", 'xy'), ("YZ", 'yz'), ("ZX", 'zx')]:
            b = QPushButton(text)
            b.clicked.connect(lambda _, m=mode: self.set_view(m))
            view_bar.addWidget(b)
        right_layout.addLayout(view_bar)

        # === XYZ 슬라이더 ===
        # 🔧 수정된 부분 시작 (엔드이펙터 XYZ 제어 UI)

        xyz_label = QLabel("<b>엔드이펙터 XYZ 제어</b>")
        right_layout.addWidget(xyz_label)

        self.sliders = {}
        xyz_row = QHBoxLayout()

        for axis in ["X", "Y", "Z"]:
            lbl = QLabel(f"{axis}:")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(-200, 200)
            sld.valueChanged.connect(self.update_from_xyz)

            # 👇 라벨과 슬라이더를 같은 줄에 붙임
            axis_row = QHBoxLayout()
            axis_row.addWidget(lbl)
            axis_row.addWidget(sld)

            wrap = QWidget()
            wrap.setLayout(axis_row)
            xyz_row.addWidget(wrap)

            self.sliders[axis] = sld

        # ✅ EE 좌표 표시를 같은 줄 끝에 추가
        self.lbl_ee = QLabel("EE: (0.0, 0.0, 0.0)")
        xyz_row.addWidget(self.lbl_ee)

        right_layout.addLayout(xyz_row)

        # === Joint 슬라이더 ===
        joint_label = QLabel("<b>Joint 제어 (서보 각도 / 물리각)</b>"); right_layout.addWidget(joint_label)
        self.joint_sliders={}; self.joint_labels={}; self.joint_physical_labels={}
        for i in range(5):
            hl = QHBoxLayout()
            lbl = QLabel(f"J{i}:");
            sld = QSlider(Qt.Horizontal)
            sim_min, sim_max, _, _ = self.robot.servo_map[i]
            sld.setRange(int(sim_min), int(sim_max)); sld.setValue(0)
            sld.valueChanged.connect(self.update_from_joints)
            lbl_ang = QLabel("0°"); lbl_ang.setFixedWidth(50)
            lbl_phys = QLabel("(0°)"); lbl_phys.setFixedWidth(50)
            btn_send_one = QPushButton("▶"); btn_send_one.setFixedWidth(30)
            btn_send_one.clicked.connect(lambda _, idx=i: self.send_single_joint(idx))
            hl.addWidget(lbl); hl.addWidget(sld); hl.addWidget(lbl_ang); hl.addWidget(lbl_phys); hl.addWidget(btn_send_one)
            right_layout.addLayout(hl)
            self.joint_sliders[i]=sld; self.joint_labels[i]=lbl_ang; self.joint_physical_labels[i]=lbl_phys

        # === 터치스크린 위치 입력 ===
        right_layout.addWidget(QLabel("<b>터치스크린 위치/크기 설정</b>"))
        form = QFormLayout()
        self.ed_screenX0 = QLineEdit(str(self.screenX0))
        self.ed_screenY0 = QLineEdit(str(self.screenY0))
        self.ed_screenZ0 = QLineEdit(str(self.screenZ0))
        self.ed_screenW = QLineEdit(str(self.screenW))
        self.ed_screenH = QLineEdit(str(self.screenH))
        for ed in [self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0, self.ed_screenW, self.ed_screenH]:
            ed.setValidator(QDoubleValidator(-10000.0, 10000.0, 2))
            ed.setMaximumWidth(70)
        xyz_row = QHBoxLayout()
        for lbl, ed in [("X0", self.ed_screenX0), ("Y0", self.ed_screenY0), ("Z0", self.ed_screenZ0)]:
            xyz_row.addWidget(QLabel(lbl)); xyz_row.addWidget(ed)
        size_row = QHBoxLayout()
        for lbl, ed in [("W", self.ed_screenW), ("H", self.ed_screenH)]:
            size_row.addWidget(QLabel(lbl)); size_row.addWidget(ed)
        right_layout.addLayout(xyz_row)
        right_layout.addLayout(size_row)
        btn_apply = QPushButton("터치스크린 적용"); btn_apply.clicked.connect(self.apply_screen)
        right_layout.addWidget(btn_apply)

        # === 터치스크린 ===
        right_layout.addWidget(QLabel("<b>가상 터치스크린</b>"))
        self.touch_sim = TouchScreenSim(); self.touch_sim.parent_viewer=self
        right_layout.addWidget(self.touch_sim)
        self.lbl_click=QLabel("터치스크린 클릭 위치: (---,---) @Z=---"); right_layout.addWidget(self.lbl_click)

        # === 통신 버튼 ===
        comm_row = QHBoxLayout()
        btn_all_send = QPushButton("전체 각도 전송 (ALL)"); btn_all_send.clicked.connect(self.send_all_servos)
        btn_disable_all = QPushButton("모든 서보 토크 해제"); btn_disable_all.clicked.connect(self.disable_all_servos)
        comm_row.addWidget(btn_all_send); comm_row.addWidget(btn_disable_all)
        right_layout.addLayout(comm_row)

        self.lbl_status_recv = QLabel("상태회신: ---"); right_layout.addWidget(self.lbl_status_recv)
        right_layout.addStretch(1)
        main_layout.addWidget(right_frame, stretch=1)

    # ===== 기본 로직 =====
    def refresh_ports(self):
        ports = SerialComm.list_ports()
        self.port_combo.clear(); self.port_combo.addItems(ports if ports else ["포트 없음"])

    def connect_serial(self):
        port = self.port_combo.currentText()
        if self.serial.connect(port): self.lbl_status.setText(f"상태: {port} 연결됨")
        else: self.lbl_status.setText("상태: 연결 실패")

    def send_single_joint(self, idx):
        angle = int(self.robot.get_servo_angles()[idx])
        self.serial.send_command(f"{idx} {angle}")

    def send_all_servos(self):
        angs = [int(a) for a in self.robot.get_servo_angles()]
        self.serial.send_command("ALL " + " ".join(map(str, angs)))

    def disable_all_servos(self):
        self.serial.send_command("Disable All")

    def on_status_received(self, angles):
        print("📥 수신됨:", angles)
        if len(angles) != 5: return
        #for i, val in enumerate(angles):
        #    self.joint_sliders[i].setValue(val)
        #    self.joint_labels[i].setText(f"{val}°")
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

        u, v, z = tx, ty, self.screenZ0
        wx, wy, wz = self.robot.map_touch_to_world(u, v, z)
        self.clicked_xy = (u, v)
        # 터치 좌표 (u,v,z) → 로봇 월드 좌표 변환
        self.clicked_xy = (u, v)
        self.lbl_click.setText(
        f"터치 목표: ({u:.1f}, {v:.1f}, {z:.1f})"
        f"    → EE 실제: ({wx:.1f}, {wy:.1f}, {wz:.1f})"
        )

        # 변환된 좌표를 IK에 전달
        self.robot.inverse_kinematics(wx, wy, wz)
        self.update_all()


    def apply_screen(self):
        self.screenX0 = float(self.ed_screenX0.text())
        self.screenY0 = float(self.ed_screenY0.text())
        self.screenZ0 = float(self.ed_screenZ0.text())
        self.screenW = float(self.ed_screenW.text())
        self.screenH = float(self.ed_screenH.text())
        w = max(self.touch_sim.width(), 1)
        h = max(self.touch_sim.height(), 1)
        self.robot.set_screen_calibration(
            origin=[self.screenX0, self.screenY0, self.screenZ0],
            u_axis=[1, 0, 0],
            v_axis=[0, 1, 0],
            su=self.screenW / w,
            sv=self.screenH / h,
            lock_z_to_plane=True
        )
        # ✅ 디버그 출력: 스크린 평면 벡터와 보정 행렬 확인
        print("─" * 60)
        print(f"[apply_screen] Origin = ({self.screenX0:.2f}, {self.screenY0:.2f}, {self.screenZ0:.2f})")
        print(f"U_axis = {self.robot.screen_u}, V_axis = {self.robot.screen_v}")
        print(f"Normal = {self.robot.screen_normal}")
        print(f"R_screen_to_world =\n{self.robot.R_screen_to_world}")
        print(f"Scale su, sv = {self.robot.su:.4f}, {self.robot.sv:.4f}")
        print("─" * 60)
            # ✅ 화면 반영 (시각 갱신)
        self.update_view()

    def update_from_xyz(self):
        x = self.xyz_ema['X'].update(self.sliders['X'].value())
        y = self.xyz_ema['Y'].update(self.sliders['Y'].value())
        z = self.xyz_ema['Z'].update(self.sliders['Z'].value())
        self.robot.inverse_kinematics(x, y, z)
        self.update_all()

    def update_from_joints(self):
        for i,s in self.joint_sliders.items(): self.robot.joints[i]=s.value()
        self.robot.update_end_effector(); self.update_all()

    def set_view(self, mode):
        self.view_mode = mode; self.update_view()

    def update_view(self):
        self.ax.clear()
        # 베이스 축 표시
        origin = np.array([0, 0, 0])
        axes = np.eye(3) * 50
        colors = ['r', 'g', 'b']
        for i in range(3):
            self.ax.quiver(origin[0], origin[1], origin[2], axes[0,i], axes[1,i], axes[2,i], color=colors[i], linewidth=2)

        pts=self.robot.forward_kinematics(); xs,ys,zs=zip(*pts)
        self.ax.plot(xs,ys,zs,marker='o',linewidth=3,color='blue')
        j2_pos, j3_pos, j4_pos, ee_pos = self.robot.joint_positions()
        self.ax.scatter([j2_pos[0]],[j2_pos[1]],[j2_pos[2]], color='orange', s=50)
        self.ax.scatter([j3_pos[0]],[j3_pos[1]],[j3_pos[2]], color='purple', s=50)
        self.ax.scatter([j4_pos[0]],[j4_pos[1]],[j4_pos[2]], color='cyan', s=40)
        self.ax.scatter([ee_pos[0]],[ee_pos[1]],[ee_pos[2]], color='red', s=50)

        # 각 관절 각도 표시
        jdeg = self.robot.joints; servo = self.robot.get_servo_angles()
        bx, by, bz = self.robot.base        
        self.ax.text(bx, by, bz + 10, f"0 {jdeg[0]:.1f}° | S0 {servo[0]}°", fontsize=8)
        self.ax.text(j2_pos[0], j2_pos[1], j2_pos[2] + 8, f"1 {jdeg[1]:.1f}° | S1 {servo[1]}°", fontsize=8)
        self.ax.text(j3_pos[0], j3_pos[1], j3_pos[2] + 8, f"2 {jdeg[2]:.1f}° | S2 {servo[2]}°", fontsize=8)
        self.ax.text(j4_pos[0], j4_pos[1], j4_pos[2] + 8, f"3 {jdeg[3]:.1f}° | S3 {servo[3]}°", fontsize=8)
        self.ax.text(ee_pos[0], ee_pos[1], ee_pos[2] + 10, f"4 {jdeg[4]:.1f}° | S4 {servo[4]}°", fontsize=8)
        #self.ax.text(ee_pos[0], ee_pos[1], ee_pos[2] + 5, f"EE ({ee_pos[0]:.1f},{ee_pos[1]:.1f},{ee_pos[2]:.1f})", fontsize=8)

        x0,y0,z0,w,h=self.screenX0,self.screenY0,self.screenZ0,self.screenW,self.screenH
        verts=[[(x0,y0,z0),(x0+w,y0,z0),(x0+w,y0+h,z0),(x0,y0+h,z0)]]
        poly=Poly3DCollection(verts,alpha=0.25,facecolor='cyan',edgecolor='black')
        self.ax.add_collection3d(poly)
        if self.clicked_xy is not None:
            cx,cy=self.clicked_xy; self.ax.scatter([cx],[cy],[self.screenZ0],color='magenta',s=60)
        lim=200
        self.ax.set_xlim(-lim/2,lim/2); self.ax.set_ylim(-10,lim); self.ax.set_zlim(-10,lim)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        views={'xy':(90,-90),'yz':(0,0),'zx':(0,90),'3d':(30,45)}
        elev,azim=views[self.view_mode]; self.ax.view_init(elev=elev,azim=azim)

        self.robot.update_end_effector()  # EE 재계산
        ee = self.robot.end_effector
        x, y, z = ee
        self.lbl_ee.setText(f"EE: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        self.canvas.draw_idle()




    def update_all(self):
        self.block_signals(True)
        self.sync_joint_sliders(); self.sync_xyz_sliders(); self.update_servo_labels()
        self.block_signals(False); self.update_view()

    def block_signals(self,state):
        for s in list(self.sliders.values())+list(self.joint_sliders.values()): s.blockSignals(state)

    def sync_joint_sliders(self):
        for i in range(5):
            v=float(self.robot.joints[i]); sim_min,sim_max,_,_=self.robot.servo_map[i]
            v=min(max(v,sim_min),sim_max); self.joint_sliders[i].setValue(int(v))

    def sync_xyz_sliders(self):
        x,y,z=self.robot.end_effector
        self.sliders['X'].setValue(int(x)); self.sliders['Y'].setValue(int(y)); self.sliders['Z'].setValue(int(z))

    def update_servo_labels(self):
        angs=self.robot.get_servo_angles() if hasattr(self.robot,'get_servo_angles') else []
        for i in range(5):
            servo_ang=angs[i] if i<len(angs) else int(self.joint_sliders[i].value())
            phys_src=getattr(self.robot,'joints',None)
            phys_ang=float(phys_src[i]) if phys_src is not None else float(self.joint_sliders[i].value())
            self.joint_labels[i].setText(f"{servo_ang}°"); self.joint_physical_labels[i].setText(f"({phys_ang:.1f}°)")

    def closeEvent(self, event):
        if hasattr(self,'serial') and self.serial: self.serial.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    w = RobotArmViewer(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
