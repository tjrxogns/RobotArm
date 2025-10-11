# serial_comm.py
# 로봇 제어용 시리얼 통신

import serial_comm

class SerialComm:
    def __init__(self, port='COM3', baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        try:
            if self.ser and self.ser.is_open:
                print("[INFO] 이미 연결됨")
                return True
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"[INFO] 시리얼 포트 연결됨: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[ERROR] 시리얼 연결 실패: {e}")
            return False

    def send_angles(self, angles):
        if not self.ser or not self.ser.is_open:
            print("[WARN] 포트가 열려있지 않습니다.")
            return False
        try:
            cmd = 'A,' + ','.join(map(str, angles)) + '\n'
            self.ser.write(cmd.encode('utf-8'))
            print(f"[SEND] {cmd.strip()}")
            return True
        except Exception as e:
            print(f"[ERROR] 전송 실패: {e}")
            return False

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("[INFO] 시리얼 포트 닫힘")
        except Exception as e:
            print(f"[ERROR] 닫기 실패: {e}")
