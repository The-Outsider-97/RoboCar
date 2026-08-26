#!/usr/bin/env python3
"""
AS5600 class for Raspberry Pi
Based on the original Arduino library by Tom Denton
Adapted for Raspberry Pi with smbus2
"""

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
    from smbus2 import SMBus # type: ignore

class AS5600:
    # I2C address
    _ams5600_address = 0x36
    
    # Single byte registers
    _addr_status = 0x0b    # magnet status
    _addr_agc = 0x1a       # automatic gain control
    _addr_burn = 0xff      # permanent burning of configs (zpos, mpos, mang, conf)
    _addr_zmco = 0x00      # number of times zpos/mpos has been permanently burned
    
    # Double byte registers (lower address, higher byte data)
    _addr_zpos = 0x01      # zero position (start) - 0x02 is lower byte
    _addr_mpos = 0x03      # maximum position (stop) - 0x04 is lower byte
    _addr_mang = 0x05      # maximum angle - 0x06 is lower byte
    _addr_conf = 0x07      # configuration - 0x08 is lower byte
    _addr_raw_angle = 0x0c # raw angle - 0x0d is lower byte
    _addr_angle = 0x0e     # mapped angle - 0x0f is lower byte
    _addr_magnitude = 0x1b # magnitude of internal CORDIC - 0x1c is lower byte
    
    def __init__(self, i2c_bus=1):
        """
        Initialize the AS5600 sensor
        :param i2c_bus: I2C bus number (default: 1 for Raspberry Pi)
        """
        self.bus = smbus.SMBus(i2c_bus) # type: ignore
    
    def get_address(self):
        """
        Get I2C address of AS5600
        :return: I2C address
        """
        return self._ams5600_address
    
    def set_output(self, mode):
        """
        Set output mode in CONF register
        :param mode: 0 for digital PWM, 1 for analog (full range), 2 for analog (reduced range)
        """
        conf_lo = self._addr_conf + 1  # lower byte address
        config_status = self._read_byte(conf_lo)
        config_status &= 0b11001111  # bits 5:4 = 00, default
        
        if mode == 0:
            config_status |= 0b100000  # bits 5:4 = 10
        elif mode == 2:
            config_status |= 0b010000  # bits 5:4 = 01
            
        self._write_byte(conf_lo, config_status)
    
    def set_max_angle(self, new_max_angle=None):
        """
        Set maximum angle
        :param new_max_angle: new maximum angle to set or None to use current raw angle
        :return: value of max angle register
        """
        if new_max_angle is None:
            max_angle = self.get_raw_angle()
        else:
            max_angle = new_max_angle
            
        self._write_byte(self._addr_mang, (max_angle >> 8) & 0xFF)  # high byte
        time.sleep(0.002)
        self._write_byte(self._addr_mang + 1, max_angle & 0xFF)     # low byte
        time.sleep(0.002)
        
        return self.get_max_angle()
    
    def get_max_angle(self):
        """
        Get maximum angle
        :return: value of max angle register
        """
        return self._read_two_bytes_separately(self._addr_mang)
    
    def set_start_position(self, start_angle=None):
        """
        Set start position
        :param start_angle: new start angle or None to use current raw angle
        :return: value of start position register
        """
        if start_angle is None:
            raw_start_angle = self.get_raw_angle()
        else:
            raw_start_angle = start_angle
            
        self._write_byte(self._addr_zpos, (raw_start_angle >> 8) & 0xFF)  # high byte
        time.sleep(0.002)
        self._write_byte(self._addr_zpos + 1, raw_start_angle & 0xFF)     # low byte
        time.sleep(0.002)
        
        return self.get_start_position()
    
    def get_start_position(self):
        """
        Get start position
        :return: value of start position register
        """
        return self._read_two_bytes_separately(self._addr_zpos)
    
    def set_end_position(self, end_angle=None):
        """
        Set end position
        :param end_angle: new end angle or None to use current raw angle
        :return: value of end position register
        """
        if end_angle is None:
            raw_end_angle = self.get_raw_angle()
        else:
            raw_end_angle = end_angle
            
        self._write_byte(self._addr_mpos, (raw_end_angle >> 8) & 0xFF)  # high byte
        time.sleep(0.002)
        self._write_byte(self._addr_mpos + 1, raw_end_angle & 0xFF)     # low byte
        time.sleep(0.002)
        
        return self.get_end_position()
    
    def get_end_position(self):
        """
        Get end position
        :return: value of end position register
        """
        return self._read_two_bytes_separately(self._addr_mpos)
    
    def get_raw_angle(self):
        """
        Get raw angle
        :return: value of raw angle register
        """
        return self._read_two_bytes_together(self._addr_raw_angle)
    
    def get_scaled_angle(self):
        """
        Get scaled angle
        :return: value of scaled angle register
        """
        return self._read_two_bytes_together(self._addr_angle)
    
    def detect_magnet(self):
        """
        Detect if magnet is present
        :return: 1 if magnet is detected, 0 if not
        """
        # Status bits: 0 0 MD ML MH 0 0 0 
        # MD high = magnet detected
        mag_status = self._read_byte(self._addr_status)
        return 1 if (mag_status & 0x20) else 0
    
    def get_magnet_strength(self):
        """
        Get magnet strength
        :return: 0 if magnet not detected, 1 if too weak, 2 if just right, 3 if too strong
        """
        # Status bits: 0 0 MD ML MH 0 0 0 
        # MD high = magnet detected  
        # ML high = AGC maximum overflow, magnet too weak
        # MH high = AGC minimum overflow, magnet too strong
        mag_status = self._read_byte(self._addr_status)
        ret_val = 0  # no magnet
        
        if mag_status & 0x20:
            ret_val = 2  # magnet detected
            if mag_status & 0x10:
                ret_val = 1  # too weak
            elif mag_status & 0x08:
                ret_val = 3  # too strong
                
        return ret_val
    
    def get_agc(self):
        """
        Get AGC value
        :return: value of AGC register
        """
        return self._read_byte(self._addr_agc)
    
    def get_magnitude(self):
        """
        Get magnitude
        :return: value of magnitude register
        """
        return self._read_two_bytes_together(self._addr_magnitude)
    
    def get_conf(self):
        """
        Get configuration
        :return: value of CONF register
        """
        return self._read_two_bytes_separately(self._addr_conf)
    
    def set_conf(self, conf):
        """
        Set configuration
        :param conf: value to set in CONF register
        """
        self._write_byte(self._addr_conf, (conf >> 8) & 0xFF)  # high byte
        time.sleep(0.002)
        self._write_byte(self._addr_conf + 1, conf & 0xFF)     # low byte
        time.sleep(0.002)
    
    def get_burn_count(self):
        """
        Get burn count
        :return: value of zmco register
        """
        return self._read_byte(self._addr_zmco)
    
    def burn_angle(self):
        """
        Burn angle settings
        :return: 1 success, -1 no magnet, -2 burn limit exceeded, -3 start/end positions not set
        """
        z_position = self.get_start_position()
        m_position = self.get_end_position()
        
        ret_val = 1
        if self.detect_magnet() == 1:
            if self.get_burn_count() < 3:
                if z_position == 0 and m_position == 0:
                    ret_val = -3
                else:
                    self._write_byte(self._addr_burn, 0x80)
            else:
                ret_val = -2
        else:
            ret_val = -1
            
        return ret_val
    
    def burn_max_angle_and_config(self):
        """
        Burn max angle and config
        :return: 1 success, -1 burn limit exceeded, -2 max angle too small
        """
        max_angle = self.get_max_angle()
        
        ret_val = 1
        if self.get_burn_count() == 0:
            if max_angle * 0.087 < 18:
                ret_val = -2
            else:
                self._write_byte(self._addr_burn, 0x40)
        else:
            ret_val = -1
            
        return ret_val
    
    def _read_byte(self, register):
        """
        Read one byte from I2C
        :param register: register to read
        :return: data read from I2C
        """
        return self.bus.read_byte_data(self._ams5600_address, register)
    
    def _read_two_bytes_together(self, register):
        """
        Read two bytes from I2C with auto-increment (for angle registers)
        :param register: register to read
        :return: combined data as word
        """
        # Read two bytes in one transaction
        data = self.bus.read_i2c_block_data(self._ams5600_address, register, 2)
        return (data[0] << 8) | data[1]
    
    def _read_two_bytes_separately(self, register):
        """
        Read two bytes from I2C separately
        :param register: register to read
        :return: combined data as word
        """
        high_byte = self._read_byte(register)
        low_byte = self._read_byte(register + 1)
        return (high_byte << 8) | low_byte
    
    def _write_byte(self, register, data):
        """
        Write one byte to I2C
        :param register: register to write to
        :param data: data to write
        """
        self.bus.write_byte_data(self._ams5600_address, register, data)


__all__ = [
    "AS5600",
]


# Example usage
if __name__ == "__main__":
    # Initialize the sensor
    sensor = AS5600(i2c_bus=1)
    
    # Check if magnet is detected
    if sensor.detect_magnet():
        print("Magnet detected")
        
        # Get magnet strength
        strength = sensor.get_magnet_strength()
        strength_str = ["No magnet", "Too weak", "Just right", "Too strong"][strength]
        print(f"Magnet strength: {strength_str}")
        
        # Read and display angles
        raw_angle = sensor.get_raw_angle()
        scaled_angle = sensor.get_scaled_angle()
        
        print(f"Raw angle: {raw_angle}")
        print(f"Scaled angle: {scaled_angle}")
        
        # Read other values
        print(f"AGC: {sensor.get_agc()}")
        print(f"Magnitude: {sensor.get_magnitude()}")
    else:
        print("No magnet detected")