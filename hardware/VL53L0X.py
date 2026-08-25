#!/usr/bin/python

# MIT License
#
# Copyright (c) 2017 John Bryan Moore
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from ctypes import CDLL, CFUNCTYPE, POINTER, c_int, c_uint, pointer, c_ubyte, c_uint8, c_uint32

import sysconfig
import time
import pkg_resources

SMBUS='smbus'
for dist in pkg_resources.working_set:
    #print(dist.project_name, dist.version)
    if dist.project_name == 'smbus':
        break
    if dist.project_name == 'smbus2':
        SMBUS='smbus2'
        break
if SMBUS == 'smbus':
    import smbus
elif SMBUS == 'smbus2':
    import smbus2 as smbus
import site


class Vl53l0xError(RuntimeError):
    pass


class Vl53l0xAccuracyMode:
    GOOD = 0        # 33 ms timing budget 1.2m range
    BETTER = 1      # 66 ms timing budget 1.2m range
    BEST = 2        # 200 ms 1.2m range
    LONG_RANGE = 3  # 33 ms timing budget 2m range
    HIGH_SPEED = 4  # 20 ms timing budget 1.2m range


class Vl53l0xDeviceMode:
    SINGLE_RANGING = 0
    CONTINUOUS_RANGING = 1
    SINGLE_HISTOGRAM = 2
    CONTINUOUS_TIMED_RANGING = 3
    SINGLE_ALS = 10
    GPIO_DRIVE = 20
    GPIO_OSC = 21


class Vl53l0xGpioAlarmType:
    OFF = 0
    THRESHOLD_CROSSED_LOW = 1
    THRESHOLD_CROSSED_HIGH = 2
    THRESHOLD_CROSSED_OUT = 3
    NEW_MEASUREMENT_READY = 4


class Vl53l0xInterruptPolarity:
    LOW = 0
    HIGH = 1


# Read/write function pointer types.
_I2C_READ_FUNC = CFUNCTYPE(c_int, c_ubyte, c_ubyte, POINTER(c_ubyte), c_ubyte)
_I2C_WRITE_FUNC = CFUNCTYPE(c_int, c_ubyte, c_ubyte, POINTER(c_ubyte), c_ubyte)

# Load VL53L0X shared lib
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix is None:
    suffix = ".so"
_POSSIBLE_LIBRARY_LOCATIONS = ['../bin'] + site.getsitepackages() + [site.getusersitepackages()]
for lib_location in _POSSIBLE_LIBRARY_LOCATIONS:
    try:
        _TOF_LIBRARY = CDLL(lib_location + '/vl53l0x_python' + suffix)
        break
    except OSError:
        pass
else:
    raise OSError('Could not find vl53l0x_python' + suffix)


class VL53L0X:
    """VL53L0X ToF."""
    def __init__(self, i2c_bus=1, i2c_address=0x29, tca9548a_num=255, tca9548a_addr=0):
        """Initialize the VL53L0X ToF Sensor from ST"""
        self._i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self._tca9548a_num = tca9548a_num
        self._tca9548a_addr = tca9548a_addr
        self._i2c = smbus.SMBus()
        self._dev = None
        # Resgiter Address
        self.ADDR_UNIT_ID_HIGH = 0x16 # Serial number high byte
        self.ADDR_UNIT_ID_LOW = 0x17 # Serial number low byte
        self.ADDR_I2C_ID_HIGH = 0x18 # Write serial number high byte for I2C address unlock
        self.ADDR_I2C_ID_LOW = 0x19 # Write serial number low byte for I2C address unlock
        self.ADDR_I2C_SEC_ADDR = 0x8a # Write new I2C address after unlock

    def open(self):
        self._i2c.open(bus=self._i2c_bus)
        self._configure_i2c_library_functions()
        self._dev = _TOF_LIBRARY.initialise(self.i2c_address, self._tca9548a_num, self._tca9548a_addr)

    def close(self):
        if self._dev is not None:
            self.stop_ranging()
        self._i2c.close()
        self._dev = None

    def _configure_i2c_library_functions(self):
        # I2C bus read callback for low level library.
        def _i2c_read(address, reg, data_p, length):
            for attempt in range(3):
                try:
                    result = self._i2c.read_i2c_block_data(address, reg, length)
                    for index in range(length):
                        data_p[index] = result[index]
                    return 0
                except IOError:
                    if attempt == 2:
                        return -1
                    time.sleep(0.001)
            return -1

        # I2C bus write callback for low level library.
        def _i2c_write(address, reg, data_p, length):
            for attempt in range(3):
                try:
                    data = []
                    for index in range(length):
                        data.append(data_p[index])
                    self._i2c.write_i2c_block_data(address, reg, data)
                    return 0
                except IOError:
                    if attempt == 2:
                        return -1
                    time.sleep(0.001)
            return -1

        # Pass i2c read/write function pointers to VL53L0X library.
        self._i2c_read_func = _I2C_READ_FUNC(_i2c_read)
        self._i2c_write_func = _I2C_WRITE_FUNC(_i2c_write)
        _TOF_LIBRARY.VL53L0X_set_i2c(self._i2c_read_func, self._i2c_write_func)

    def save_configuration(self):
        """Save current configuration for later restoration"""
        # This would need to be implemented based on your specific needs
        # and the registers you want to save
        return {"address": self.i2c_address, "mode": self._current_mode}
    
    def restore_configuration(self, config):
        """Restore previously saved configuration"""
        if config.get("address") != self.i2c_address:
            self.change_address(config["address"])
        if config.get("mode"):
            self.start_ranging(config["mode"])

    def start_ranging(self, mode=Vl53l0xAccuracyMode.GOOD):
        """Start VL53L0X ToF Sensor Ranging"""
        if self._dev is None:
            raise Vl53l0xError("Device not initialized. Call open() first.")
        
        _TOF_LIBRARY.startRanging(self._dev, mode)

    def stop_ranging(self):
        """Stop VL53L0X ToF Sensor Ranging"""
        if self._dev is not None:
            _TOF_LIBRARY.stopRanging(self._dev)

    def get_distance(self):
        """Get distance from VL53L0X ToF Sensor"""
        if self._dev is None:
            raise Vl53l0xError("Device not initialized. Call open() first.")
        
        return _TOF_LIBRARY.getDistance(self._dev)
    
    def get_distance_with_retry(self, retries=3):
        """Get distance with retry logic for more reliable readings"""
        for attempt in range(retries):
            try:
                distance = self.get_distance()
                # Basic range validation (adjust based on your needs)
                if 20 <= distance <= 2000:
                    return distance
            except Exception:
                if attempt == retries - 1:
                    raise
            time.sleep(0.01)
        raise Vl53l0xError("Failed to get valid distance after retries")

    # This function included to show how to access the ST library directly
    # from python instead of through the simplified interface
    def get_timing(self):
        if self._dev is None:
            raise Vl53l0xError("Device not initialized. Call open() first.")
            
        budget = c_uint(0)
        budget_p = pointer(budget)
        status = _TOF_LIBRARY.VL53L0X_GetMeasurementTimingBudgetMicroSeconds(self._dev, budget_p)
        if status == 0:
            return budget.value + 1000
        else:
            return 0

    def start_continuous_monitoring(self, callback, interval_ms=100):
        """Start continuous monitoring with callback"""
        self.stop_ranging()
        self.start_ranging(Vl53l0xAccuracyMode.HIGH_SPEED)
        
        def monitor_loop():
            while self._dev is not None:
                try:
                    distance = self.get_distance_with_retry()
                    callback(distance)
                except Vl53l0xError:
                    pass
                time.sleep(interval_ms / 1000)
        
        import threading
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def factory_reset(self):
        """Reset sensor to factory defaults"""
        if self._dev is not None:
            self.stop_ranging()
        
        # Reset through I2C
        try:
            self._i2c.write_byte_data(self.i2c_address, 0x00, 0x00)
            time.sleep(0.1)
        except:
            pass
        
        # Reinitialize
        if self._dev is not None:
            self.open()

    def configure_gpio_interrupt(
            self, proximity_alarm_type=Vl53l0xGpioAlarmType.THRESHOLD_CROSSED_LOW,
            interrupt_polarity=Vl53l0xInterruptPolarity.HIGH, threshold_low_mm=250, threshold_high_mm=500):
        """
        Configures a GPIO interrupt from device, be sure to call "clear_interrupt" after interrupt is processed.
        """
        if self._dev is None:
            raise Vl53l0xError("Device not initialized. Call open() first.")
            
        pin = c_uint8(0)  # 0 is only GPIO pin.
        device_mode = c_uint8(Vl53l0xDeviceMode.CONTINUOUS_RANGING)
        functionality = c_uint8(proximity_alarm_type)
        polarity = c_uint8(interrupt_polarity)
        status = _TOF_LIBRARY.VL53L0X_SetGpioConfig(self._dev, pin, device_mode, functionality, polarity)
        if status != 0:
            raise Vl53l0xError('Error setting VL53L0X GPIO config')

        threshold_low = c_uint32(threshold_low_mm << 16)
        threshold_high = c_uint32(threshold_high_mm << 16)
        status = _TOF_LIBRARY.VL53L0X_SetInterruptThresholds(self._dev, device_mode, threshold_low, threshold_high)
        if status != 0:
            raise Vl53l0xError('Error setting VL53L0X thresholds')

        # Ensure any pending interrupts are cleared.
        self.clear_interrupt()

    def clear_interrupt(self):
        if self._dev is None:
            raise Vl53l0xError("Device not initialized. Call open() first.")
            
        mask = c_uint32(0)
        status = _TOF_LIBRARY.VL53L0X_ClearInterruptMask(self._dev, mask)
        if status != 0:
            raise Vl53l0xError('Error clearing VL53L0X interrupt')

    def change_address(self, new_address):
        if self._dev is not None:
            raise Vl53l0xError('Error changing VL53L0X address')

        self._i2c.open(bus=self._i2c_bus)

        if new_address is None or new_address == self.i2c_address:
            return

        # read value from 0x16,0x17
        high = self._i2c.read_byte_data(self.i2c_address, self.ADDR_UNIT_ID_HIGH)
        low = self._i2c.read_byte_data(self.i2c_address, self.ADDR_UNIT_ID_LOW)

        # write value to 0x18,0x19
        self._i2c.write_byte_data(self.i2c_address, self.ADDR_I2C_ID_HIGH, high)
        self._i2c.write_byte_data(self.i2c_address, self.ADDR_I2C_ID_LOW, low)

        # write new_address to 0x1a
        self._i2c.write_byte_data(self.i2c_address, self.ADDR_I2C_SEC_ADDR, new_address)

        self.i2c_address = new_address

        self._i2c.close()
        
    def __del__(self):
        """Ensure proper cleanup"""
        try:
            self.close()
        except:
            pass

    # ----------------- #
    # ----- EXTRA ----- #
    # ----------------- #

    def perform_calibration(self, target_distance_mm=100):
        # Placeholder for actual calibration routine
        # Refer to VL53L0X API manual for implementation details
        print("Perform calibration at", target_distance_mm, "mm")

    def get_temperature_compensated_distance(self, actual_temp=None):
        """Apply temperature compensation to distance readings"""
        distance = self.get_distance_with_retry()
        
        if actual_temp is None:
            # Use placeholder if no temperature sensor available
            actual_temp = 25
        
        # Simple temperature compensation (adjust based on your calibration)
        temp_diff = actual_temp - 25  # Difference from reference temperature
        compensation = temp_diff * 0.5  # 0.5mm per degree C (adjust based on your testing)
        
        return max(0, distance + compensation)
    
    def check_sensor_health(self):
        """Check if sensor is responding properly"""
        try:
            # Try to read a known register
            test_val = self._i2c.read_byte_data(self.i2c_address, 0xC0)
            return test_val == 0xEE  # Known value from datasheet
        except:
            return False

    def adjust_timing_budget(self, environment_conditions):
        """Adjust timing budget based on environmental conditions"""
        if environment_conditions.get('high_ambient_light', False):
            self.start_ranging(Vl53l0xAccuracyMode.BETTER)
        elif environment_conditions.get('long_range', False):
            self.start_ranging(Vl53l0xAccuracyMode.LONG_RANGE)
        else:
            self.start_ranging(Vl53l0xAccuracyMode.GOOD)

    def set_power_mode(self, high_performance=True):
        if high_performance:
            # Use higher accuracy modes
            self.start_ranging(Vl53l0xAccuracyMode.BEST)
        else:
            # Use power-saving mode
            self.start_ranging(Vl53l0xAccuracyMode.HIGH_SPEED)

    def get_filtered_distance(self, samples=5, max_deviation=50):
        readings = []
        for _ in range(samples):
            readings.append(self.get_distance_with_retry())
            time.sleep(0.01)

        # Remove outliers
        median = sorted(readings)[len(readings)//2]
        filtered = [r for r in readings if abs(r - median) <= max_deviation]
        
        return sum(filtered) // len(filtered) if filtered else median
    
    def get_signal_strength(self):
        """Get signal strength information if supported by library"""
        try:
            # This would require extending the C library
            signal = c_uint(0)
            signal_p = pointer(signal)
            status = _TOF_LIBRARY.VL53L0X_GetSignalRate(self._dev, signal_p)
            if status == 0:
                return signal.value
        except:
            pass
        return 0