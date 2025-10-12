# main.py (251011 복원+튜닝+베이스 기둥 가시화) — 3D 내부 2x 스케일 & 좌표축 동기화 + Y-clip (Y < -25 숨김)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QLineEdit, QFormLayout, QFrame
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
        self.setGeometry(0, 0, 1200, 700)

        self.robot = RobotArm()
        self.serial = SerialComm()

        self.view_mode = '3d'
        self.screenX0, self.screenY0, self.screenZ0 = -50.0, 60.0, 10.0
        self.screenW, self.screenH = 155.0, 90.0
        self.clicked_xy = None

        self.xyz_ema = {k: EMA(0.3) for k in ['X','Y','Z']}

        # 내부 컨텐츠만 2배 확대(화이트 창 크기는 그대로)
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

        right_frame = QFrame(); right_frame.setFixedWidth(450)
        right_layout = QVBoxLayout(right_frame)

        # === View Buttons ===
        view_bar = QHBoxLayout()
        for text, mode in [("3D", '3d'), ("XY", 'xy'), ("YZ", 'yz'), ("ZX", 'zx')]:
            b = QPushButton(text)
            b.clicked.connect(lambda _, m=mode: self.set_view(m))
            view_bar.addWidget(b)
        right_layout.addLayout(view_bar)

        # === XYZ 슬라이더 ===
        xyz_label = QLabel("<b>엔드이펙터 XYZ 제어</b>"); right_layout.addWidget(xyz_label)
        self.sliders = {}
        for axis in ["X","Y","Z"]:
            hl=QHBoxLayout(); lbl=QLabel(f"{axis}:"); sld=QSlider(Qt.Horizontal)
            sld.setRange(-200,200); sld.setValue(0); sld.setTracking(True)
            sld.valueChanged.connect(self.update_from_xyz)
            sld.sliderReleased.connect(self.update_from_xyz)
            hl.addWidget(lbl); hl.addWidget(sld); right_layout.addLayout(hl)
            self.sliders[axis]=sld

        # === Joint 슬라이더 ===
        joint_label = QLabel("<b>Joint 제어 (서보 각도 표시)</b>"); right_layout.addWidget(joint_label)
        self.joint_sliders={}; self.joint_labels={}
        for i in range(5):
            hl=QHBoxLayout(); lbl=QLabel(f"J{i+1}:"); sld=QSlider(Qt.Horizontal)
            sld.setRange(-90,90); sld.setValue(0); sld.setTracking(True)
            sld.valueChanged.connect(self.update_from_joints)
            sld.sliderReleased.connect(self.update_from_joints)
            lbl_ang=QLabel("0°"); lbl_ang.setFixedWidth(60)
            hl.addWidget(lbl); hl.addWidget(sld); hl.addWidget(lbl_ang)
            right_layout.addLayout(hl); self.joint_sliders[i]=sld; self.joint_labels[i]=lbl_ang

        # === 스크린 입력창 ===
        form = QFormLayout()
        self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0 = QLineEdit(str(self.screenX0)), QLineEdit(str(self.screenY0)), QLineEdit(str(self.screenZ0))
        self.ed_screenW, self.ed_screenH = QLineEdit(str(self.screenW)), QLineEdit(str(self.screenH))
        for ed in [self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0, self.ed_screenW, self.ed_screenH]:
            ed.setValidator(QDoubleValidator(-10000.0, 10000.0, 2)); ed.setMaximumWidth(120)
        # XYZ 한 줄 배치
        xyz_row = QWidget()
        xyz_lay = QHBoxLayout(xyz_row); xyz_lay.setContentsMargins(0,0,0,0); xyz_lay.setSpacing(6)
        xyz_lay.addWidget(QLabel("X0")); xyz_lay.addWidget(self.ed_screenX0)
        xyz_lay.addWidget(QLabel("Y0")); xyz_lay.addWidget(self.ed_screenY0)
        xyz_lay.addWidget(QLabel("Z0")); xyz_lay.addWidget(self.ed_screenZ0)
        form.addRow(QLabel("Screen XYZ"), xyz_row)
        form.addRow("screenW", self.ed_screenW)
        form.addRow("screenH", self.ed_screenH)
        btn_apply = QPushButton("터치스크린 적용"); btn_apply.clicked.connect(self.apply_screen)
        right_layout.addLayout(form); right_layout.addWidget(btn_apply)

        # === 터치스크린 시뮬레이터 ===
        right_layout.addWidget(QLabel("<b>가상 터치스크린</b>"))
        self.touch_sim = TouchScreenSim(); self.touch_sim.parent_viewer=self
        right_layout.addWidget(self.touch_sim)

        self.lbl_click=QLabel("터치스크린 클릭 위치: (---,---) @Z=---"); right_layout.addWidget(self.lbl_click)
        btn_connect=QPushButton("시리얼 연결"); btn_connect.clicked.connect(self.serial.connect)
        btn_send=QPushButton("현재 각도 전송"); btn_send.clicked.connect(self.send_to_robot)
        right_layout.addWidget(btn_connect); right_layout.addWidget(btn_send)

        # EE 좌표 표시 라벨
        self.lbl_ee = QLabel("EE: (0.0, 0.0, 0.0)")
        right_layout.addWidget(self.lbl_ee)

        right_layout.addStretch(1)
        main_layout.addWidget(right_frame, stretch=1)

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
        for i in range(5): self.joint_sliders[i].setValue(int(self.robot.joints[i]))

    def sync_xyz_sliders(self):
        x, y, z = self.robot.end_effector
        self.sliders['X'].setValue(int(x))
        self.sliders['Y'].setValue(int(y))
        self.sliders['Z'].setValue(int(z))

    def update_servo_labels(self):
        angs=self.robot.get_servo_angles()
        for i,a in enumerate(angs): self.joint_labels[i].setText(f"{a}°")

    def send_to_robot(self): self.serial.send_angles(self.robot.get_servo_angles())

    def draw_touch_screen(self, half=None):
        x0,y0,z0,w,h=self.screenX0,self.screenY0,self.screenZ0,self.screenW,self.screenH
        if half is None:
            # 기본은 고정 크기, 줌 후에는 span 기반으로 호출 가능
            verts=[[(x0,y0,z0),(x0+w,y0,z0),(x0+w,y0+h,z0),(x0,y0+h,z0)]]
        else:
            # half가 주어지면 정사각 화면을 중심/half로 재구성 (예시)
            verts=[[(x0-half,y0-half,z0),(x0+half,y0-half,z0),(x0+half,y0+half,z0),(x0-half,y0+half,z0)]]
        poly=Poly3DCollection(verts,alpha=0.25,facecolor='cyan',edgecolor='black')
        self.ax.add_collection3d(poly)

    def update_view(self):
        self.ax.clear()

        # === 로봇/마커/라벨 등 기본 그리기 ===
        pts=self.robot.forward_kinematics(); xs,ys,zs=zip(*pts)
        self.ax.plot(xs,ys,zs,marker='o',linewidth=3,color='blue')
        j2_pos, j3_pos, j4_pos, ee_pos = self.robot.joint_positions()
        self.ax.scatter([j2_pos[0]],[j2_pos[1]],[j2_pos[2]], color='orange', s=50)
        self.ax.scatter([j3_pos[0]],[j3_pos[1]],[j3_pos[2]], color='purple', s=50)
        self.ax.scatter([j4_pos[0]],[j4_pos[1]],[j4_pos[2]], color='cyan', s=40)
        self.ax.scatter([ee_pos[0]],[ee_pos[1]],[ee_pos[2]], color='red', s=50)

        # 관절 각도 3D 라벨
        jdeg = self.robot.joints
        bx, by, bz = self.robot.base
        self.ax.text(bx,         by,         bz + 10, f"J1 {jdeg[0]:.1f}°", fontsize=8)
        self.ax.text(j2_pos[0],  j2_pos[1],  j2_pos[2] + 8,  f"J2 {jdeg[1]:.1f}°", fontsize=8)
        self.ax.text(j3_pos[0],  j3_pos[1],  j3_pos[2] + 8,  f"J3 {jdeg[2]:.1f}°", fontsize=8)
        self.ax.text(j4_pos[0],  j4_pos[1],  j4_pos[2] + 8,  f"J4 {jdeg[3]:.1f}°", fontsize=8)
        self.ax.text(ee_pos[0],  ee_pos[1],  ee_pos[2] + 12, f"J5 {jdeg[4]:.1f}°", fontsize=8)

        # EE 좌표 텍스트 + 패널 라벨 갱신
        self.ax.text(ee_pos[0], ee_pos[1], ee_pos[2] + 5, f"EE ({ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f})", fontsize=8)
        if hasattr(self, 'lbl_ee'):
            self.lbl_ee.setText(f"EE: ({ee_pos[0]:.1f}, {ee_pos[1]:.1f}, {ee_pos[2]:.1f})")

        # 가상 스크린 + 클릭 표시
        self.draw_touch_screen()
        if self.clicked_xy is not None:
            cx, cy = self.clicked_xy; self.ax.scatter([cx],[cy],[self.screenZ0],color='magenta',s=60)

        # === 기본 축 범위/레이블/시점 설정 ===
        lim=220
        self.ax.set_xlim(-lim,lim)
        self.ax.set_ylim(-lim/2,lim)   
        self.ax.set_zlim(-lim/2,lim)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        views={'xy':(90,-90),'yz':(0,0),'zx':(0,90),'3d':(30,45)}
        elev,azim=views[self.view_mode]; self.ax.view_init(elev=elev,azim=azim)

        # === 내부 줌 2x (화이트 창 그대로, 콘텐츠만 확대) — 한 번만 적용 ===
        zx = getattr(self, "view_zoom", 2.0)
        if zx and zx != 1.0:
            for getter, setter in [
                (self.ax.get_xlim3d, self.ax.set_xlim3d),
                (self.ax.get_ylim3d, self.ax.set_ylim3d),
                (self.ax.get_zlim3d, self.ax.set_zlim3d),
            ]:
                lo, hi = getter()
                c = (lo + hi) / 2.0
                half = (hi - lo) / (2.0 * zx)
                setter(c - half, c + half)

        # === 줌/클립 후 스팬 기준으로 헬퍼/그리드/사이즈 재계산 ===
        span = max(
            self.ax.get_xlim3d()[1]-self.ax.get_xlim3d()[0],
            self.ax.get_ylim3d()[1]-self.ax.get_ylim3d()[0],
            self.ax.get_zlim3d()[1]-self.ax.get_zlim3d()[0],
        )

        ymin, ymax = self.ax.get_ylim3d(); zmin, zmax = self.ax.get_zlim3d()
        if ymin < -10:
            self.ax.set_ylim(-10, max(ymax, -9))
        if zmin < -10:
            self.ax.set_zlim(-10, max(zmax, -9))
        axis_len = 0.2 * span     # 좌표축 길이(전체의 20%)
        s = 0.03 * span           # 베이스 상단 사각형 half-size(3%)

        # 베이스 기둥 + 상단 사각형(줌된 스케일 기준)
        bx, by, bz = self.robot.base
        self.ax.plot([bx, bx], [by, by], [0, bz], linewidth=4, color='k')
        xs = [bx-s, bx+s, bx+s, bx-s, bx-s]
        ys = [by-s, by-s, by+s, by+s, by-s]
        zs = [bz,   bz,   bz,   bz,   bz]
        self.ax.plot(xs, ys, zs, color='k')

        # 좌표축(helper) 길이도 span 비율로
        self.ax.quiver(bx,by,bz, axis_len,0,0, color='r', arrow_length_ratio=0.2)
        self.ax.quiver(bx,by,bz, 0,axis_len,0, color='g', arrow_length_ratio=0.2)
        self.ax.quiver(bx,by,bz, 0,0,axis_len, color='b', arrow_length_ratio=0.2)

        # (선택) 가상 스크린을 span 기반 half로 그리고 싶다면:
        # self.draw_touch_screen(half=0.35*span)

        self.canvas.draw_idle()

def main():
    app=QApplication(sys.argv)
    w=RobotArmViewer(); w.show()
    sys.exit(app.exec())

if __name__=='__main__':
    main()
