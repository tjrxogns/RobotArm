# main.py (251011 튜닝 적용)
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QLineEdit, QFormLayout, QFrame
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
        self.setWindowTitle("로봇팔 3D 뷰어 (251011 튜닝 버전)")
        self.setGeometry(50, 50, 1200, 700)

        self.robot = RobotArm()
        self.serial = SerialComm()

        self.screenX0, self.screenY0, self.screenZ0 = 50.0, 50.0, 20.0
        self.screenW, self.screenH = 155.0, 90.0

        self.xyz_ema = {k: EMA(0.3) for k in ['X','Y','Z']}

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

        xyz_label = QLabel("<b>엔드이펙터 XYZ 제어</b>"); right_layout.addWidget(xyz_label)
        self.sliders = {}
        for axis in ["X","Y","Z"]:
            hl=QHBoxLayout(); lbl=QLabel(f"{axis}:"); sld=QSlider(Qt.Horizontal)
            sld.setRange(-200,200); sld.setValue(0)
            sld.sliderReleased.connect(self.update_from_xyz)
            hl.addWidget(lbl); hl.addWidget(sld); right_layout.addLayout(hl)
            self.sliders[axis]=sld

        joint_label = QLabel("<b>Joint 제어 (서보 각도 표시)</b>"); right_layout.addWidget(joint_label)
        self.joint_sliders={}; self.joint_labels={}
        for i in range(5):
            hl=QHBoxLayout(); lbl=QLabel(f"J{i+1}:"); sld=QSlider(Qt.Horizontal)
            sld.setRange(-90,90); sld.setValue(0)
            sld.sliderReleased.connect(self.update_from_joints)
            lbl_ang=QLabel("0°"); lbl_ang.setFixedWidth(50)
            hl.addWidget(lbl); hl.addWidget(sld); hl.addWidget(lbl_ang)
            right_layout.addLayout(hl); self.joint_sliders[i]=sld; self.joint_labels[i]=lbl_ang

        right_layout.addWidget(QLabel("<b>가상 터치스크린</b>"))
        self.touch_sim = TouchScreenSim(); self.touch_sim.parent_viewer=self
        right_layout.addWidget(self.touch_sim)

        self.lbl_click=QLabel("터치스크린 클릭 위치: (---,---)"); right_layout.addWidget(self.lbl_click)
        btn_send=QPushButton("현재 각도 전송"); btn_send.clicked.connect(self.send_to_robot)
        right_layout.addWidget(btn_send)

        main_layout.addWidget(right_frame, stretch=1)

    def on_touchscreen_click(self, x, y, w, h):
        rel_x=x/w; rel_y=1-(y/h)
        tx=self.screenX0+self.screenW*rel_x; ty=self.screenY0+self.screenH*rel_y
        tz=self.screenZ0
        self.lbl_click.setText(f"터치: ({tx:.1f},{ty:.1f}) Z={tz:.1f}")
        self.robot.inverse_kinematics(tx,ty,tz)
        self.update_all()

    def update_from_xyz(self):
        x=self.xyz_ema['X'].update(self.sliders['X'].value())
        y=self.xyz_ema['Y'].update(self.sliders['Y'].value())
        z=self.xyz_ema['Z'].update(self.sliders['Z'].value())
        self.robot.inverse_kinematics(x,y,z)
        self.update_all()

    def update_from_joints(self):
        for i,s in self.joint_sliders.items(): self.robot.joints[i]=s.value()
        self.robot.update_end_effector(); self.update_all()

    def update_all(self):
        self.block_signals(True)
        self.sync_joint_sliders(); self.update_servo_labels()
        self.block_signals(False); self.update_view()

    def block_signals(self,state):
        for s in list(self.sliders.values())+list(self.joint_sliders.values()): s.blockSignals(state)

    def sync_joint_sliders(self):
        for i in range(5): self.joint_sliders[i].setValue(int(self.robot.joints[i]))

    def update_servo_labels(self):
        angs=self.robot.get_servo_angles()
        for i,a in enumerate(angs): self.joint_labels[i].setText(f"{a}°")

    def send_to_robot(self): self.serial.send_angles(self.robot.get_servo_angles())

    def draw_touch_screen(self):
        x0,y0,z0,w,h=self.screenX0,self.screenY0,self.screenZ0,self.screenW,self.screenH
        verts=[[(x0,y0,z0),(x0+w,y0,z0),(x0+w,y0+h,z0),(x0,y0+h,z0)]]
        poly=Poly3DCollection(verts,alpha=0.25,facecolor='cyan',edgecolor='black')
        self.ax.add_collection3d(poly)

    def update_view(self):
        self.ax.clear()
        pts=self.robot.forward_kinematics(); xs,ys,zs=zip(*pts)
        self.ax.plot(xs,ys,zs,marker='o',linewidth=3,color='blue')
        ex,ey,ez=self.robot.end_effector; self.ax.scatter([ex],[ey],[ez],color='red',s=50)
        self.draw_touch_screen()
        self.ax.set_xlim(-220,220); self.ax.set_ylim(-220,220); self.ax.set_zlim(0,220)
        self.canvas.draw_idle()

def main():
    app=QApplication(sys.argv)
    w=RobotArmViewer(); w.show()
    sys.exit(app.exec())

if __name__=='__main__':
    main()
