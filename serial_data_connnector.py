import serial
import time

SERIAL_PORT = 'COM3'  # Change as needed (Linux: "/dev/ttyUSB0")
BAUD_RATE = 9600

def read_serial_data():
    """Reads data from the serial port."""
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)  # Allow Arduino to send data
            ser.flush()  # Clear buffer
            raw_data = ser.readline().decode('utf-8').strip()  # Read a line

            if raw_data:
                print("Raw Serial Data:", raw_data)
                # Parsing the sensor values
                parts = raw_data.split("|")
                parsed_data = {}
                for part in parts:
                    key, value = part.split(":")
                    parsed_data[key.strip()] = float(value.strip())
                return parsed_data
            else:
                return None
    except Exception as e:
        print(f"Error Reading Serial Data: {e}")
        return None