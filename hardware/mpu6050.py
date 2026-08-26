"""
This program handles the communication over I2C
between a Raspberry Pi and a MPU-6050 Gyroscope / Accelerometer combo.

Released under the MIT License
Copyright (c) 2015, 2016, 2017, 2021 Martijn (martijn@mrtijn.nl) and contributers

https://github.com/m-rtijn/mpu6050
"""

import math
import platform
import time

if platform.system() == 'Windows':
    # Mock implementation for Windows
    class SMBusMock:
        def __init__(self, bus):
            self.bus = bus
        def write_byte_data(self, addr, reg, val):
            pass
        def read_byte_data(self, addr, reg):
            return 0x68  # WHO_AM_I response
        def read_i2c_block_data(self, addr, reg, length):
            return [0] * length
        # Add any other methods you use

    smbus = type('smbus', (), {'SMBus': SMBusMock})()
else:
    import smbus2 as smbus # type: ignore

from collections import deque

class mpu6050:

    # Global Variables
    GRAVITIY_MS2 = 9.80665
    address = None
    bus = None

    # Scale Modifiers
    ACCEL_SCALE_MODIFIER_2G = 16384.0
    ACCEL_SCALE_MODIFIER_4G = 8192.0
    ACCEL_SCALE_MODIFIER_8G = 4096.0
    ACCEL_SCALE_MODIFIER_16G = 2048.0

    GYRO_SCALE_MODIFIER_250DEG = 131.0
    GYRO_SCALE_MODIFIER_500DEG = 65.5
    GYRO_SCALE_MODIFIER_1000DEG = 32.8
    GYRO_SCALE_MODIFIER_2000DEG = 16.4

    # Pre-defined ranges
    ACCEL_RANGE_2G = 0x00
    ACCEL_RANGE_4G = 0x08
    ACCEL_RANGE_8G = 0x10
    ACCEL_RANGE_16G = 0x18

    GYRO_RANGE_250DEG = 0x00
    GYRO_RANGE_500DEG = 0x08
    GYRO_RANGE_1000DEG = 0x10
    GYRO_RANGE_2000DEG = 0x18

    FILTER_BW_256=0x00
    FILTER_BW_188=0x01
    FILTER_BW_98=0x02
    FILTER_BW_42=0x03
    FILTER_BW_20=0x04
    FILTER_BW_10=0x05
    FILTER_BW_5=0x06

    # MPU-6050 Registers
    PWR_MGMT_1 = 0x6B
    PWR_MGMT_2 = 0x6C
    FIFO_EN = 0x23
    FIFO_COUNT = 0X72
    FIFO_R_W = 0X74

    ACCEL_XOUT0 = 0x3B
    ACCEL_YOUT0 = 0x3D
    ACCEL_ZOUT0 = 0x3F

    TEMP_OUT0 = 0x41

    GYRO_XOUT0 = 0x43
    GYRO_YOUT0 = 0x45
    GYRO_ZOUT0 = 0x47

    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG = 0x1B
    MPU_CONFIG = 0x1A

    INT_ENABLE = 0X38
    WHO_AM_I = 0X75

    def __init__(self, address, bus=1):
        self.address = address
        self.bus = smbus.SMBus(bus) # type: ignore
        # Wake up the MPU-6050 since it starts in sleep mode
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)

        self._initialize_device()
        self.filter = MovingAverageFilter(window_size=5)
        self.low_pass_filter = LowPassFilter(alpha=0.2)

    def _initialize_device(self):
        """Initialize the device with proper error handling"""
        try:
            # Check if device is connected
            if not self.test_connection():
                raise IOError("MPU6050 not found at address 0x{:02X}".format(self.address))
            
            # Wake up the MPU-6050 since it starts in sleep mode
            assert self.bus is not None
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
            time.sleep(0.1)
            
            # Enable all sensors
            self.enable_sensors()
            
        except IOError as e:
            print(f"Error initializing MPU6050: {e}")
            raise

    def configure_filter(self, filter_setting=FILTER_BW_256):
        """Configure the digital low-pass filter"""
        # Set accelerometer and gyroscope bandwidth
        self.set_filter_range(filter_setting)

    def set_sample_rate(self, rate=100):
        """Set the sample rate divider with validation"""
        if rate < 4 or rate > 1000:
            raise ValueError("Sample rate must be between 4 and 1000 Hz")

        div = int(1000 / rate - 1)  # Assuming 1kHz gyro output rate
        assert self.bus is not None
        self.bus.write_byte_data(self.address, 0x19, div)

    def test_connection(self):
        """Test if the device is connected and responding"""
        try:
            # Read WHO_AM_I register (should return 0x68 or 0x71 for MPU6050)
            assert self.bus is not None
            who_am_i = self.bus.read_byte_data(self.address, self.WHO_AM_I)
            return who_am_i in [0x68, 0x71]  # Some MPU6050 clones return 0x71
        except IOError:
            return False

    # I2C communication methods
    def read_i2c_word(self, register):
        """Read two i2c registers and combine them.

        register -- the first register to read from.
        Returns the combined read results.
        """
        try:
            # Read the data from the registers
            assert self.bus is not None
            high = self.bus.read_byte_data(self.address, register)
            low = self.bus.read_byte_data(self.address, register + 1)

            value = (high << 8) + low

            if (value >= 0x8000):
                return -((65535 - value) + 1)
            else:
                return value
        except IOError as e:
            print(f"I2C read error: {e}")
            return 0

    # MPU-6050 Methods
    def enable_sensors(self):
        """Ensure all sensors are enabled with proper settings"""
        # Wake up the device and ensure all sensors are enabled
        assert self.bus is not None
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
        
        # Enable accelerometer and gyroscope
        self.bus.write_byte_data(self.address, self.PWR_MGMT_2, 0x00)

    def get_temp(self):
        """Reads the temperature from the onboard temperature sensor of the MPU-6050.

        Returns the temperature in degrees Celcius.
        """
        try:
            raw_temp = self.read_i2c_word(self.TEMP_OUT0)

            # Get the actual temperature using the formule given in the
            # MPU-6050 Register Map and Descriptions revision 4.2, page 30
            actual_temp = (raw_temp / 340.0) + 36.53
            return actual_temp
        except IOError as e:
            print(f"Error reading temperature: {e}")
            return 0

    def get_temperature_compensated_gyro(self):
        """Get gyro data with temperature compensation"""
        temp = self.get_temp()
        gyro_data = self.get_gyro_data()
        
        # Simple temperature compensation (adjust coefficients based on your testing)
        temp_factor = 1.0 + 0.01 * (temp - 25)  # 1% change per degree from 25°C
        return {
            'x': gyro_data['x'] * temp_factor,
            'y': gyro_data['y'] * temp_factor,
            'z': gyro_data['z'] * temp_factor
        }
    
    def set_accel_range(self, accel_range):
        """Sets the range of the accelerometer to range.

        accel_range -- the range to set the accelerometer to. Using a
        pre-defined range is advised.
        """
        # First change it to 0x00 to make sure we write the correct value later
        assert self.bus is not None
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)

        # Write the new range to the ACCEL_CONFIG register
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, accel_range)

    def set_gyro_range(self, gyro_range):
        """Sets the range of the gyroscope to range"""
        # First change it to 0x00 to make sure we write the correct value later
        assert self.bus is not None
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
        # Write the new range to the GYRO_CONFIG register
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, gyro_range)

    def read_accel_range(self, raw = False):
        """Reads the range the accelerometer is set to.

        If raw is True, it will return the raw value from the ACCEL_CONFIG
        register
        If raw is False, it will return an integer: -1, 2, 4, 8 or 16. When it
        returns -1 something went wrong.
        """
        assert self.bus is not None
        raw_data = self.bus.read_byte_data(self.address, self.ACCEL_CONFIG)

        if raw is True:
            return raw_data
        elif raw is False:
            if raw_data == self.ACCEL_RANGE_2G:
                return 2
            elif raw_data == self.ACCEL_RANGE_4G:
                return 4
            elif raw_data == self.ACCEL_RANGE_8G:
                return 8
            elif raw_data == self.ACCEL_RANGE_16G:
                return 16
            else:
                return -1

    def get_accel_data(self, g = False):
        """Gets and returns the X, Y and Z values from the accelerometer.

        If g is True, it will return the data in g
        If g is False, it will return the data in m/s^2
        Returns a dictionary with the measurement results.
        """
        x = self.read_i2c_word(self.ACCEL_XOUT0)
        y = self.read_i2c_word(self.ACCEL_YOUT0)
        z = self.read_i2c_word(self.ACCEL_ZOUT0)

        accel_scale_modifier = None
        accel_range = self.read_accel_range(True)

        if accel_range == self.ACCEL_RANGE_2G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G
        elif accel_range == self.ACCEL_RANGE_4G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_4G
        elif accel_range == self.ACCEL_RANGE_8G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_8G
        elif accel_range == self.ACCEL_RANGE_16G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_16G
        else:
            print("Unkown range - accel_scale_modifier set to self.ACCEL_SCALE_MODIFIER_2G")
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G

        x = x / accel_scale_modifier
        y = y / accel_scale_modifier
        z = z / accel_scale_modifier

        if g is True:
            return {'x': x, 'y': y, 'z': z}
        elif g is False:
            x = x * self.GRAVITIY_MS2
            y = y * self.GRAVITIY_MS2
            z = z * self.GRAVITIY_MS2
            return {'x': x, 'y': y, 'z': z}

    def calibrate_accel(self, samples=500):
        """Calibrate accelerometer by calculating average offset"""
        print("Calibrating accelerometer... keep sensor level")
        x_offset, y_offset, z_offset = 0, 0, 0
        
        valid_samples = 0
        for _ in range(samples):
            data = self.get_accel_data(g=True)  # Get data in g units
            if data is None:
                continue

            x_offset += data['x']
            y_offset += data['y']
            z_offset += data['z'] - 1.0  # Z should be 1g when level
            valid_samples += 1

        if valid_samples == 0:
            raise RuntimeError("Unable to read accelerometer data during calibration")
        
        self.accel_offset = {
            'x': x_offset / valid_samples,
            'y': y_offset / valid_samples,
            'z': z_offset / valid_samples
        }
        print(f"Accel offsets: {self.accel_offset}")
    
    def get_calibrated_accel_data(self, g=False):
        """Return accelerometer data with offset compensation"""
        raw_data = self.get_accel_data(g=g)
        if raw_data is None:
            return None

        if g:
            # For g units, subtract the offset directly
            return {
                'x': raw_data['x'] - self.accel_offset['x'],
                'y': raw_data['y'] - self.accel_offset['y'],
                'z': raw_data['z'] - self.accel_offset['z']
            }
        else:
            # For m/s², convert offset to m/s² first
            return {
                'x': raw_data['x'] - (self.accel_offset['x'] * self.GRAVITIY_MS2),
                'y': raw_data['y'] - (self.accel_offset['y'] * self.GRAVITIY_MS2),
                'z': raw_data['z'] - (self.accel_offset['z'] * self.GRAVITIY_MS2)
            }

    def calibrate_gyro(self, samples=500):
        """Calibrate gyroscope by calculating average offset"""
        print("Calibrating gyroscope... keep sensor stationary")
        x_offset, y_offset, z_offset = 0, 0, 0
        
        for _ in range(samples):
            data = self.get_gyro_data()
            x_offset += data['x']
            y_offset += data['y']
            z_offset += data['z']
        
        self.gyro_offset = {
            'x': x_offset / samples,
            'y': y_offset / samples,
            'z': z_offset / samples
        }
        print(f"Gyro offsets: {self.gyro_offset}")
        
    def get_calibrated_gyro_data(self):
        """Return gyro data with offset compensation"""
        raw_data = self.get_gyro_data()
        return {
            'x': raw_data['x'] - self.gyro_offset['x'],
            'y': raw_data['y'] - self.gyro_offset['y'],
            'z': raw_data['z'] - self.gyro_offset['z']
        }

    def set_filter_range(self, filter_range=FILTER_BW_256):
        """Sets the low-pass bandpass filter frequency"""
        # Keep the current EXT_SYNC_SET configuration in bits 3, 4, 5 in the MPU_CONFIG register
        assert self.bus is not None
        EXT_SYNC_SET = self.bus.read_byte_data(self.address, self.MPU_CONFIG) & 0b00111000
        return self.bus.write_byte_data(self.address, self.MPU_CONFIG,  EXT_SYNC_SET | filter_range)


    def read_gyro_range(self, raw = False):
        """Reads the range the gyroscope is set to.

        If raw is True, it will return the raw value from the GYRO_CONFIG
        register.
        If raw is False, it will return 250, 500, 1000, 2000 or -1. If the
        returned value is equal to -1 something went wrong.
        """
        assert self.bus is not None
        raw_data = self.bus.read_byte_data(self.address, self.GYRO_CONFIG)

        if raw is True:
            return raw_data
        elif raw is False:
            if raw_data == self.GYRO_RANGE_250DEG:
                return 250
            elif raw_data == self.GYRO_RANGE_500DEG:
                return 500
            elif raw_data == self.GYRO_RANGE_1000DEG:
                return 1000
            elif raw_data == self.GYRO_RANGE_2000DEG:
                return 2000
            else:
                return -1

    def get_gyro_data(self):
        """Gets and returns the X, Y and Z values from the gyroscope.

        Returns the read values in a dictionary.
        """
        x = self.read_i2c_word(self.GYRO_XOUT0)
        y = self.read_i2c_word(self.GYRO_YOUT0)
        z = self.read_i2c_word(self.GYRO_ZOUT0)

        gyro_scale_modifier = None
        gyro_range = self.read_gyro_range(True)

        if gyro_range == self.GYRO_RANGE_250DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_2000DEG
        else:
            print("Unkown range - gyro_scale_modifier set to self.GYRO_SCALE_MODIFIER_250DEG")
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG

        x = x / gyro_scale_modifier
        y = y / gyro_scale_modifier
        z = z / gyro_scale_modifier

        return {'x': x, 'y': y, 'z': z}
    
    def get_all_data(self):
        """Reads and returns all the available data."""
        temp = self.get_temp()
        accel = self.get_accel_data()
        gyro = self.get_gyro_data()

        return [accel, gyro, temp]

    def get_filtered_data(self):
        raw_data = self.get_all_data()
        return self.filter.filter(raw_data[0]), self.filter.filter(raw_data[1]), raw_data[2]
    
    def safe_read_i2c_word(self, register, retries=3):
        """Read with retry logic for unreliable I2C connections"""
        for attempt in range(retries):
            try:
                return self.read_i2c_word(register)
            except IOError:
                if attempt == retries - 1:
                    raise
                time.sleep(0.01)  # Short delay before retry
    
    def is_data_valid(self, accel_data, gyro_data):
        """Check if sensor data is within expected ranges"""
        # Check accelerometer data (should be within ±16g based on your setting)
        for axis in ['x', 'y', 'z']:
            if abs(accel_data[axis]) > 16 * self.GRAVITIY_MS2:  # Adjust based on your range setting
                return False
        
        # Check gyroscope data (should be within ±2000°/s based on your setting)
        for axis in ['x', 'y', 'z']:
            if abs(gyro_data[axis]) > 2000:  # Adjust based on your range setting
                return False
                
        return True
    
    def get_timestamped_data(self):
        """Get sensor data with precise timestamp"""
        timestamp = time.time()
        data = self.get_all_data()
        return {
            'timestamp': timestamp,
            'accel': data[0],
            'gyro': data[1],
            'temp': data[2]
        }
    
    def read_all_data_block(self):
        """Read all sensor data in a single I2C transaction"""
        try:
            # Read 14 bytes starting from ACCEL_XOUT0
            assert self.bus is not None
            data = self.bus.read_i2c_block_data(self.address, self.ACCEL_XOUT0, 14)

            # Convert data to signed values
            accel_x = self._to_signed_int(data[0], data[1])
            accel_y = self._to_signed_int(data[2], data[3])
            accel_z = self._to_signed_int(data[4], data[5])
            temp = self._to_signed_int(data[6], data[7])
            gyro_x = self._to_signed_int(data[8], data[9])
            gyro_y = self._to_signed_int(data[10], data[11])
            gyro_z = self._to_signed_int(data[12], data[13])

            # Convert to proper units
            accel_scale = self._get_accel_scale()
            gyro_scale = self._get_gyro_scale()

            accel_data = {
                'x': accel_x / accel_scale * self.GRAVITIY_MS2,
                'y': accel_y / accel_scale * self.GRAVITIY_MS2,
                'z': accel_z / accel_scale * self.GRAVITIY_MS2
            }

            gyro_data = {
                'x': gyro_x / gyro_scale,
                'y': gyro_y / gyro_scale,
                'z': gyro_z / gyro_scale
            }

            temp_data = (temp / 340.0) + 36.53

            return accel_data, gyro_data, temp_data

        except IOError as e:
            print(f"I2C read error: {e}")
            # Fall back to individual reads
            return self.get_accel_data(), self.get_gyro_data(), self.get_temp()

    def enable_fifo(self, accel=True, gyro=True, temp=True):
        """Enable FIFO buffer for specified sensors"""
        fifo_en = 0x00
        if temp:
            fifo_en |= 0x80  # TEMP_FIFO_EN
        if gyro:
            fifo_en |= 0x70  # All gyro axes
        if accel:
            fifo_en |= 0x08  # ACCEL_FIFO_EN
            
        assert self.bus is not None
        self.bus.write_byte_data(self.address, self.FIFO_EN, fifo_en)
        
        # Enable FIFO
        self.bus.write_byte_data(self.address, 0x6A, 0x40)

    def get_fifo_count(self):
        """Get number of bytes in FIFO buffer"""
        try:
            assert self.bus is not None
            high = self.bus.read_byte_data(self.address, self.FIFO_COUNT)
            low = self.bus.read_byte_data(self.address, self.FIFO_COUNT + 1)
            return (high << 8) + low
        except IOError:
            return 0

    def read_fifo_data(self, count):
        """Read data from FIFO buffer"""
        try:
            assert self.bus is not None
            return self.bus.read_i2c_block_data(self.address, self.FIFO_R_W, count)
        except IOError as e:
            print(f"Error reading FIFO data: {e}")
            return []

    def _to_signed_int(self, high, low):
        """Convert two bytes to a signed integer"""
        value = (high << 8) + low
        return value if value < 32768 else value - 65536

    def _get_accel_scale(self):
        """Get the current accelerometer scale modifier"""
        accel_range = self.read_accel_range(True)
        if accel_range == self.ACCEL_RANGE_2G:
            return self.ACCEL_SCALE_MODIFIER_2G
        elif accel_range == self.ACCEL_RANGE_4G:
            return self.ACCEL_SCALE_MODIFIER_4G
        elif accel_range == self.ACCEL_RANGE_8G:
            return self.ACCEL_SCALE_MODIFIER_8G
        elif accel_range == self.ACCEL_RANGE_16G:
            return self.ACCEL_SCALE_MODIFIER_16G
        else:
            return self.ACCEL_SCALE_MODIFIER_2G

    def _get_gyro_scale(self):
        """Get the current gyroscope scale modifier"""
        gyro_range = self.read_gyro_range(True)
        if gyro_range == self.GYRO_RANGE_250DEG:
            return self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            return self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            return self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            return self.GYRO_SCALE_MODIFIER_2000DEG
        else:
            return self.GYRO_SCALE_MODIFIER_250DEG
    
    def reset_fifo(self):
        """Reset the FIFO buffer"""
        # Reset FIFO and I2C Master
        assert self.bus is not None
        self.bus.write_byte_data(self.address, 0x6A, 0x04)
        # Enable FIFO
        self.bus.write_byte_data(self.address, 0x6A, 0x40)

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = {'x': deque(maxlen=window_size), 
                      'y': deque(maxlen=window_size), 
                      'z': deque(maxlen=window_size)}
        
    def filter(self, new_data):
        filtered = {}
        for axis in ['x', 'y', 'z']:
            self.values[axis].append(new_data[axis])
            filtered[axis] = sum(self.values[axis]) / len(self.values[axis])
        return filtered


class LowPassFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_values = {'x': 0, 'y': 0, 'z': 0}
        self.initialized = False
        
    def filter(self, new_data):
        filtered = {}
        for axis in ['x', 'y', 'z']:
            if not self.initialized:
                self.prev_values[axis] = new_data[axis]
                filtered[axis] = new_data[axis]
            else:
                filtered[axis] = self.alpha * new_data[axis] + (1 - self.alpha) * self.prev_values[axis]
                self.prev_values[axis] = filtered[axis]
                
        self.initialized = True
        return filtered


__all__ = [
    "mpu6050",
    "MovingAverageFilter",
    "LowPassFilter",
]


if __name__ == "__main__":
    mpu = mpu6050(0x68)
    
    # Test connection
    if not mpu.test_connection():
        print("MPU6050 not found")
        exit(1)
    
    # Configure settings
    mpu.set_accel_range(mpu.ACCEL_RANGE_4G)
    mpu.set_gyro_range(mpu.GYRO_RANGE_500DEG)
    mpu.configure_filter(mpu.FILTER_BW_42)
    mpu.set_sample_rate(100)
    mpu.enable_sensors()
    
    # Calibrate sensors
    print("Calibrating sensors... keep device stationary")
    mpu.calibrate_gyro(1000)
    mpu.calibrate_accel(1000)
    
    # Main loop
    try:
        while True:
            # Get timestamped, calibrated data using block read
            data = mpu.read_all_data_block()
            timestamp = time.time()
            
            calibrated_accel = mpu.get_calibrated_accel_data()
            calibrated_gyro = mpu.get_calibrated_gyro_data()
            
            # Apply moving average filter
            filtered_accel = mpu.filter.filter(calibrated_accel)
            filtered_gyro = mpu.filter.filter(calibrated_gyro)
            
            # Check data validity
            if mpu.is_data_valid(filtered_accel, filtered_gyro):
                print(f"Time: {timestamp:.3f}")
                print(f"Temp: {data[2]:.2f}°C")
                print(f"Accel: X={filtered_accel['x']:.2f}, Y={filtered_accel['y']:.2f}, Z={filtered_accel['z']:.2f} m/s²")
                print(f"Gyro: X={filtered_gyro['x']:.2f}, Y={filtered_gyro['y']:.2f}, Z={filtered_gyro['z']:.2f} °/s")
                print("---")
            
            time.sleep(0.01)  # ~100Hz
            
    except KeyboardInterrupt:
        print("Exiting...")