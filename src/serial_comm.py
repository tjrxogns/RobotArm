from PySide6.QtCore import QObject, Signal
import serial, serial.tools.list_ports, time, threading

class SerialComm(QObject):
    # ✅ 상태회신 시그널 정의
    status_received = Signal(list)  # e.g. [90,45,130,85,95]

    def __init__(self):
        super().__init__()
        self.ser = None
        self.running = False
        self.thread = None

    @staticmethod
    def list_ports():
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=0.05)
            self.running = True
            # ✅ Python thread로 read_loop 실행
            self.thread = threading.Thread(target=self.read_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print("Serial connect error:", e)
            return False

    def read_loop(self):
        buffer = ""
        print("🔵 Serial read loop started")
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    ch = self.ser.read().decode(errors="ignore")
                    if ch == "\n":
                        line = buffer.strip()
                        buffer = ""
                        if line:
                            print("RX:", line)
                            if line.startswith("@"):
                                try:
                                    vals = [int(v) for v in line[1:].split(",")]
                                    print("✅ 상태회신 수신:", vals)
                                    self.status_received.emit(vals)
                                except Exception as e:
                                    print("Parse error:", e)
                    else:
                        buffer += ch
                else:
                    time.sleep(0.01)
            except Exception as e:
                print("Serial read error:", e)
                time.sleep(0.1)

    def send_command(self, text):
        if self.ser and self.ser.is_open:
            msg = text.strip() + "\n"
            self.ser.write(msg.encode())
            print("TX:", msg.strip())

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
