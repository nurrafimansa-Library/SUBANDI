import serial
import time
from datetime import datetime

# Konfigurasi port dan baudrate
PORT = "COM5"       # ganti sesuai port
BAUD = 115200
FILENAME = "data_NBS2.csv"

# Buka koneksi serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # tunggu ESP32 siap

print("Mulai logging... Tekan Ctrl+C untuk berhenti.")
print(f"Menyimpan data ke: {FILENAME}")

# Buka file CSV dan tulis header
with open(FILENAME, "w") as f:
    f.write("timestamp,hum,suhu,label\n")

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{ts},{line}")
                f.write(f"{ts},{line}\n")
                f.flush()

    except KeyboardInterrupt:
        print("\nLogging dihentikan oleh user.")

ser.close()
