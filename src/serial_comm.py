import serial
import time
import serial.tools.list_ports

class SerialComm:
    def __init__(self, port=None, baudrate=9600, timeout=1):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def connect(self, port=None):
        if port:
            self.port = port
        if not self.port:
            print("[Serial] 포트가 지정되지 않았습니다.")
            return False
        try:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
            time.sleep(2)
            print(f"[Serial] Connected to {self.port} ({self.baudrate}bps)")
            return True
        except serial.SerialException as e:
            print(f"[Serial] 연결 실패: {e}")
            self.ser = None
            return False

    def send_angle(self, servo_id: int, angle: int):
        if not self.ser:
            print("[Serial] 포트 미연결 상태")
            return
        cmd = f"{servo_id} {angle}\n"
        self.ser.write(cmd.encode())
        print(f"[TX] {cmd.strip()}")

    def send_angles(self, angles):
        if not self.ser:
            print("[Serial] 포트 미연결 상태")
            return
        for i, ang in enumerate(angles, start=1):
            self.send_angle(i, int(ang))
            time.sleep(0.05)
        print("[Serial] 모든 서보 전송 완료")

    def request_status(self):
        if self.ser:
            self.ser.write(b'P\n')
            print("[TX] P (상태 요청)")

    def read_line(self):
        if self.ser and self.ser.in_waiting:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"[RX] {line}")
            return line
        return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[Serial] Closed")

    @staticmethod
    def list_ports():
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
