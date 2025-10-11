# main.py
# 베이스 축 표시 추가 버전

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
import sys


class TouchScreenSim(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: lightblue; border: 2px solid gray;")
        self.click_pos = None
        self.parent_viewer = None

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
        self.setWindowTitle("로봇팔 3D 뷰어 (베이스축 표시)")
        self.setGeometry(50, 50, 1200, 700)

        self.robot = RobotArm()
        self.serial = SerialComm()

        self.view_mode = '3d'
        self.screenX0, self.screenY0, self.screenZ0 = 50.0, 50.0, 20.0
        self.screenW, self.screenH = 155.0, 90.0
        self.clicked_xy = None

        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ===== 왼쪽: 3D 뷰 =====
        left_frame = QFrame(); left_layout = QVBoxLayout(left_frame)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas.mpl_connect('button_press_event', self.on_click)
        left_layout.addWidget(self.canvas)
        main_layout.addWidget(left_frame, stretch=2)

        # ===== 오른쪽: 컨트롤 패널 =====
        right_frame = QFrame(); right_frame.setFixedWidth(450)
        right_layout = QVBoxLayout(right_frame); right_layout.setAlignment(Qt.AlignTop)

        # 뷰 전환 버튼
        view_bar = QHBoxLayout()
        for text, mode in [("3D", '3d'), ("XY", 'xy'), ("YZ", 'yz'), ("ZX", 'zx')]:
            b = QPushButton(text)
            b.clicked.connect(lambda _, m=mode: self.set_view(m))
            view_bar.addWidget(b)
        right_layout.addLayout(view_bar)

        # XYZ 슬라이더
        xyz_bar = QVBoxLayout(); self.sliders = {}
        xyz_label = QLabel("<b>엔드이펙터 XYZ 제어</b>")
        right_layout.addWidget(xyz_label)
        for axis in ["X", "Y", "Z"]:
            hl = QHBoxLayout(); lbl = QLabel(f"{axis}:"); sld = QSlider(Qt.Horizontal)
            sld.setRange(-200, 200); sld.setValue(0)
            sld.sliderReleased.connect(self.update_from_xyz)
            hl.addWidget(lbl); hl.addWidget(sld)
            xyz_bar.addLayout(hl); self.sliders[axis] = sld
        right_layout.addLayout(xyz_bar)

        # Joint 슬라이더 + 각도 표시
        joint_label = QLabel("<b>Joint 제어 (서보 각도 표시)</b>")
        right_layout.addWidget(joint_label)
        joint_bar = QVBoxLayout(); self.joint_sliders = {}; self.joint_labels = {}
        for i in range(5):
            hl = QHBoxLayout(); lbl = QLabel(f"J{i+1}:"); sld = QSlider(Qt.Horizontal)
            sld.setRange(-90, 90); sld.setValue(0)
            sld.sliderReleased.connect(self.update_from_joints)
            lbl_angle = QLabel("0°"); lbl_angle.setFixedWidth(50)
            hl.addWidget(lbl); hl.addWidget(sld); hl.addWidget(lbl_angle)
            joint_bar.addLayout(hl)
            self.joint_sliders[i] = sld; self.joint_labels[i] = lbl_angle
        right_layout.addLayout(joint_bar)

        # 터치스크린 설정 입력창
        form = QFormLayout()
        self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0 = QLineEdit(str(self.screenX0)), QLineEdit(str(self.screenY0)), QLineEdit(str(self.screenZ0))
        self.ed_screenW, self.ed_screenH = QLineEdit(str(self.screenW)), QLineEdit(str(self.screenH))
        for ed in [self.ed_screenX0, self.ed_screenY0, self.ed_screenZ0, self.ed_screenW, self.ed_screenH]:
            ed.setValidator(QDoubleValidator(-10000.0, 10000.0, 2)); ed.setMaximumWidth(120)
        form.addRow("screenX0", self.ed_screenX0); form.addRow("screenY0", self.ed_screenY0)
        form.addRow("screenZ0", self.ed_screenZ0); form.addRow("screenW", self.ed_screenW)
        form.addRow("screenH", self.ed_screenH)
        btn_apply = QPushButton("터치스크린 적용"); btn_apply.clicked.connect(self.apply_screen)
        right_layout.addLayout(form); right_layout.addWidget(btn_apply)

        # 가상 터치스크린 창
        right_layout.addWidget(QLabel("<b>가상 터치스크린</b>"))
        self.touch_sim = TouchScreenSim(); self.touch_sim.parent_viewer = self
        right_layout.addWidget(self.touch_sim)

        # 클릭 좌표 및 시리얼
        self.lbl_click = QLabel("터치스크린 클릭 위치: (---, ---) @Z=---")
        right_layout.addWidget(self.lbl_click)
        btn_connect = QPushButton("시리얼 연결"); btn_connect.clicked.connect(self.serial.connect)
        btn_send = QPushButton("현재 각도 전송"); btn_send.clicked.connect(self.send_to_robot)
        right_layout.addWidget(btn_connect); right_layout.addWidget(btn_send)

        right_layout.addStretch(1)
        main_layout.addWidget(right_frame, stretch=1)

        self.update_view()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if (self.screenX0 <= x <= self.screenX0 + self.screenW) and (self.screenY0 <= y <= self.screenY0 + self.screenH):
            self.clicked_xy = (x, y)
            self.lbl_click.setText(f"3D 터치 클릭: ({x:.1f}, {y:.1f}) @Z={self.screenZ0:.1f}")
            self.robot.inverse_kinematics(x, y, self.screenZ0)
            self.update_all()

    def on_touchscreen_click(self, click_x, click_y, w, h):
        rel_x = click_x / w; rel_y = 1.0 - (click_y / h)
        tx = self.screenX0 + self.screenW * rel_x; ty = self.screenY0 + self.screenH * rel_y
        self.clicked_xy = (tx, ty)
        self.lbl_click.setText(f"가상 터치 클릭: ({tx:.1f}, {ty:.1f}) @Z={self.screenZ0:.1f}")
        self.robot.inverse_kinematics(tx, ty, self.screenZ0)
        self.update_all()

    def update_all(self):
        self.block_all_signals(True)
        self.sync_joint_sliders(); self.sync_xyz_sliders(); self.update_servo_labels()
        self.block_all_signals(False)
        self.update_view()

    def set_view(self, mode): self.view_mode = mode; self.update_view()

    def apply_screen(self):
        self.screenX0, self.screenY0, self.screenZ0 = float(self.ed_screenX0.text()), float(self.ed_screenY0.text()), float(self.ed_screenZ0.text())
        self.screenW, self.screenH = float(self.ed_screenW.text()), float(self.ed_screenH.text())
        self.update_view()

    def update_from_xyz(self):
        x, y, z = self.sliders['X'].value(), self.sliders['Y'].value(), self.sliders['Z'].value()
        self.robot.inverse_kinematics(x, y, z); self.update_all()

    def update_from_joints(self):
        for i, s in self.joint_sliders.items(): self.robot.joints[i] = s.value()
        self.robot.update_end_effector(); self.update_all()

    def sync_joint_sliders(self):
        for i in range(5): self.joint_sliders[i].setValue(int(self.robot.joints[i]))
    def sync_xyz_sliders(self):
        x, y, z = self.robot.end_effector
        self.sliders['X'].setValue(int(x)); self.sliders['Y'].setValue(int(y)); self.sliders['Z'].setValue(int(z))
    def update_servo_labels(self):
        angles = self.robot.get_servo_angles()
        for i in range(5): self.joint_labels[i].setText(f"{angles[i]}°")

    def block_all_signals(self, state: bool):
        for s in list(self.sliders.values()) + list(self.joint_sliders.values()):
            s.blockSignals(state)

    def send_to_robot(self): self.serial.send_angles(self.robot.get_servo_angles())

    def draw_touch_screen(self):
        x0, y0, z0, w, h = self.screenX0, self.screenY0, self.screenZ0, self.screenW, self.screenH
        verts = [[(x0, y0, z0), (x0+w, y0, z0), (x0+w, y0+h, z0), (x0, y0+h, z0)]]
        poly = Poly3DCollection(verts, alpha=0.25, facecolor='cyan', edgecolor='black')
        self.ax.add_collection3d(poly)

    def update_view(self):
        self.ax.clear()
        pts = self.robot.forward_kinematics(); xs, ys, zs = zip(*pts)
        self.ax.plot(xs, ys, zs, marker='o', linewidth=3, color='blue')
        ex, ey, ez = self.robot.end_effector
        self.ax.scatter([ex], [ey], [ez], color='red', s=50)
        self.draw_touch_screen()
        if self.clicked_xy is not None:
            cx, cy = self.clicked_xy; self.ax.scatter([cx], [cy], [self.screenZ0], color='magenta', s=60)

        # ===== 베이스 좌표축 표시 =====
        base = self.robot.base
        axis_len = 30
        self.ax.quiver(base[0], base[1], base[2], axis_len, 0, 0, color='r', arrow_length_ratio=0.2)
        self.ax.quiver(base[0], base[1], base[2], 0, axis_len, 0, color='g', arrow_length_ratio=0.2)
        self.ax.quiver(base[0], base[1], base[2], 0, 0, axis_len, color='b', arrow_length_ratio=0.2)

        lim = 220
        self.ax.set_xlim(-lim, lim); self.ax.set_ylim(-lim, lim); self.ax.set_zlim(0, lim)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        views = {'xy': (90, -90), 'yz': (0, 0), 'zx': (0, 90), '3d': (30, 45)}
        elev, azim = views[self.view_mode]; self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = RobotArmViewer(); w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
