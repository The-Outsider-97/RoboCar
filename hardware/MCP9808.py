#!/usr/bin/env python3
"""
MCP9808 class for Raspberry Pi
Based on the Seeed Studio Arduino library
Adapted for Raspberry Pi with smbus2
"""

import time
from smbus2 import SMBus

class MCP9808:
    # Register addresses
    SET_CONFIG_ADDR = 0x01
    SET_UPPER_LIMIT_ADDR = 0x02
    SET_LOWER_LIMIT_ADDR = 0x03
    SET_CRITICAL_LIMIT_ADDR = 0x04
    AMBIENT_TEMPERATURE_ADDR = 0x05
    SET_RESOLUTION_ADDR = 0x08
    
    # Resolution values
    RESOLUTION_0_5_DEGREE = 0
    RESOLUTION_0_25_DEGREE = 0x01
    RESOLUTION_0_125_DEGREE = 0x02
    RESOLUTION_0_0625_DEGREE = 0x03
    
    # Sign bit for temperature calculation
    SIGN_BIT = 0x10
    
    def __init__(self, i2c_addr=0x18, i2c_bus=1):
        """
        Initialize the MCP9808 sensor
        :param i2c_addr: I2C address (default: 0x18)
        :param i2c_bus: I2C bus number (default: 1 for Raspberry Pi)
        """
        self._iic_addr = i2c_addr
        self.bus = SMBus(i2c_bus)
    
    def set_config(self, cfg):
        """
        Set configuration
        :param cfg: configuration value
        :return: 0 if success
        """
        return self._write_16bit(self.SET_CONFIG_ADDR, cfg)
    
    def set_upper_limit(self, cfg):
        """
        Set upper temperature limit
        :param cfg: limit value
        :return: 0 if success
        """
        return self._write_16bit(self.SET_UPPER_LIMIT_ADDR, cfg)
    
    def set_lower_limit(self, cfg):
        """
        Set lower temperature limit
        :param cfg: limit value
        :return: 0 if success
        """
        return self._write_16bit(self.SET_LOWER_LIMIT_ADDR, cfg)
    
    def set_critical_limit(self, cfg):
        """
        Set critical temperature limit
        :param cfg: limit value
        :return: 0 if success
        """
        return self._write_16bit(self.SET_CRITICAL_LIMIT_ADDR, cfg)
    
    def set_resolution(self, resolution):
        """
        Set temperature resolution
        :param resolution: resolution value
        :return: 0 if success
        """
        return self._write_byte(self.SET_RESOLUTION_ADDR, resolution)
    
    def read_temp_reg(self):
        """
        Read temperature register
        :return: 16-bit temperature value
        """
        return self._read_16bit(self.AMBIENT_TEMPERATURE_ADDR)
    
    def calculate_temp(self, temp_value):
        """
        Calculate temperature from 16-bit value
        :param temp_value: 16-bit temperature value
        :return: temperature in Celsius as float
        """
        temp_upper = (temp_value >> 8) & 0xFF
        temp_lower = temp_value & 0xFF
        
        if temp_upper & self.SIGN_BIT:
            # Negative temperature
            temp_upper &= 0x0F  # Clear flag bits
            temp = 256 - (temp_upper * 16 + temp_lower * 0.0625)
            temp *= -1
        else:
            # Positive temperature
            temp_upper &= 0x0F  # Clear flag bits
            temp = temp_upper * 16 + temp_lower * 0.0625
        
        return temp
    
    def get_temp(self):
        """
        Get temperature
        :return: temperature in Celsius as float
        """
        temp_value = self.read_temp_reg()
        return self.calculate_temp(temp_value)
    
    def init(self, resolution=RESOLUTION_0_0625_DEGREE):
        """
        Initialize sensor with specified resolution
        :param resolution: resolution value (default: 0.0625°C)
        :return: 0 if success, -1 if error
        """
        try:
            # Set resolution
            if self.set_resolution(resolution):
                return -1
            return 0
        except:
            return -1
    
    def _write_byte(self, reg, data):
        """
        Write one byte to I2C
        :param reg: register address
        :param data: data to write
        :return: 0 if success
        """
        self.bus.write_byte_data(self._iic_addr, reg, data)
        return 0
    
    def _write_16bit(self, reg, data):
        """
        Write 16-bit value to I2C
        :param reg: register address
        :param data: 16-bit data to write
        :return: 0 if success
        """
        # Write high byte first
        self.bus.write_byte_data(self._iic_addr, reg, (data >> 8) & 0xFF)
        # Write low byte
        self.bus.write_byte_data(self._iic_addr, reg + 1, data & 0xFF)
        return 0
    
    def _read_16bit(self, reg):
        """
        Read 16-bit value from I2C
        :param reg: register address
        :return: 16-bit value
        """
        # Read two bytes
        data = self.bus.read_i2c_block_data(self._iic_addr, reg, 2)
        return (data[0] << 8) | data[1]


# Example usage
if __name__ == "__main__":
    # Initialize the sensor
    sensor = MCP9808(i2c_addr=0x18, i2c_bus=1)
    
    # Initialize with high resolution
    if sensor.init(sensor.RESOLUTION_0_0625_DEGREE) == 0:
        print("Sensor initialized successfully")
        
        # Read and display temperature
        temperature = sensor.get_temp()
        print(f"Temperature: {temperature:.4f} °C")
        
        # Read raw temperature value
        raw_temp = sensor.read_temp_reg()
        print(f"Raw temperature value: 0x{raw_temp:04X}")
    else:
        print("Failed to initialize sensor")