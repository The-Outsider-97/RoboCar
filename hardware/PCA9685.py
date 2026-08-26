# Copyright (c) 2025 Remy 3Design
# Author: Jean-Erolle A. Remy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import division
import math
import time

from machine import Pin, I2C # type: ignore
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("PCA9685")
printer = PrettyPrinter()

def software_reset(self, i2c=None, **kwargs):
    """Sends a software reset (SWRST) command to all servo drivers on the bus."""
    # Setup I2C interface for device 0x00 to talk to all of them.
    i2c = I2C
    self._device = i2c.get_i2c_device(0x00, **kwargs)
    self._device.writeRaw8(0x06)  # SWRST

class PCA9685(object):
    """PCA9685 16-channel 12-bit PWM LED/servo controller."""

    # Register addresses
    PCA9685_ADDRESS     = 0x40
    MODE1               = 0x00
    MODE2               = 0x01
    SUBADR1             = 0x02
    SUBADR2             = 0x03
    SUBADR3             = 0x04
    ALLCALLADR          = 0x05
    LED0_ON_L           = 0x06
    LED0_ON_H           = 0x07
    LED0_OFF_L          = 0x08
    LED0_OFF_H          = 0x09
    ALL_LED_ON_L        = 0xFA
    ALL_LED_ON_H        = 0xFB
    ALL_LED_OFF_L       = 0xFC
    ALL_LED_OFF_H       = 0xFD
    PRESCALE            = 0xFE

    # Mode1 register bits
    MODE1_ALLCALL       = 0x01
    MODE1_SUB3          = 0x02
    MODE1_SUB2          = 0x04
    MODE1_SUB1          = 0x08
    MODE1_SLEEP         = 0x10
    MODE1_AI            = 0x20
    MODE1_EXTCLK        = 0x40
    MODE1_RESTART       = 0x80
    
    # Mode2 register bits
    MODE2_OUTNE_0       = 0x01
    MODE2_OUTNE_1       = 0x02
    MODE2_OUTDRV        = 0x04
    MODE2_OCH           = 0x08
    MODE2_INVRT         = 0x10

    # Software reset address
    SWRST_ADDRESS       = 0x00
    SWRST_COMMAND       = 0x06

    def __init__(self, i2c_bus=1, address=0x40, sda_pin=2, scl_pin=3, freq=50):
        """
        Initialize PCA9685
        
        Args:
            i2c_bus: I2C bus number (default 1 for Raspberry Pi)
            address: I2C address of PCA9685 (default 0x40)
            sda_pin: SDA pin number (default GPIO2 for Pi 5)
            scl_pin: SCL pin number (default GPIO3 for Pi 5)
            freq: PWM frequency in Hz (default 50 for servos)
        """
        self.i2c = I2C(i2c_bus, sda=Pin(sda_pin), scl=Pin(scl_pin))
        self.address = address
        self.frequency = freq

        # Initialize device
        self.reset()
        self.set_pwm_freq(freq)

    def reset(self):
        """Reset the PCA9685 to default state"""
        self.write_byte(self.MODE1, self.MODE1_RESTART | self.MODE1_AI)
        self.write_byte(self.MODE2, self.MODE2_OUTDRV)
        time.sleep(0.01)

    @classmethod
    def software_reset(cls, i2c_bus=1, sda_pin=2, scl_pin=3):
        """
        Sends a software reset (SWRST) command to all PCA9685 devices on the bus.
        This resets all PCA9685 devices to their power-on state.
        
        Args:
            i2c_bus: I2C bus number (default 1 for Raspberry Pi)
            sda_pin: SDA pin number (default GPIO2 for Pi 5)
            scl_pin: SCL pin number (default GPIO3 for Pi 5)
        """
        try:
            # Create temporary I2C connection
            i2c = I2C(i2c_bus, sda=Pin(sda_pin), scl=Pin(scl_pin))
            
            # Send software reset command to general call address
            i2c.writeto(cls.SWRST_ADDRESS, bytes([cls.SWRST_COMMAND]))
            
            # Wait for devices to reset
            time.sleep(0.01)
            print("Software reset command sent to all PCA9685 devices")
            
        except Exception as e:
            print(f"Software reset failed: {e}")

    def sleep(self):
        """Put PCA9685 to sleep (oscillator off)"""
        mode1 = self.read_byte(self.MODE1)
        self.write_byte(self.MODE1, (mode1 & ~self.MODE1_RESTART) | self.MODE1_SLEEP)
        
    def wake(self):
        """Wake up PCA9685 (oscillator on)"""
        mode1 = self.read_byte(self.MODE1)
        self.write_byte(self.MODE1, mode1 & ~self.MODE1_SLEEP)
        time.sleep(0.005)  # Wait for oscillator to stabilize
        
    def set_pwm_freq(self, freq):
        """
        Set PWM frequency
        
        Args:
            freq: Frequency in Hz (24-1526 Hz)
        """
        self.frequency = freq
        
        # Calculate prescale value
        prescaleval = 25000000.0  # 25MHz
        prescaleval /= 4096.0     # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        prescale = int(math.floor(prescaleval + 0.5))
        
        # Clamp to valid range
        if prescale < 3:
            prescale = 3
        elif prescale > 255:
            prescale = 255
        
        # Set prescale (must be in sleep mode)
        old_mode = self.read_byte(self.MODE1)
        new_mode = (old_mode & ~self.MODE1_RESTART) | self.MODE1_SLEEP
        self.write_byte(self.MODE1, new_mode)
        self.write_byte(self.PRESCALE, prescale)
        self.write_byte(self.MODE1, old_mode)
        
        # Wait for oscillator to stabilize
        time.sleep(0.005)
        
        # Restart PWM
        self.write_byte(self.MODE1, old_mode | self.MODE1_RESTART)
        
    def set_pwm(self, channel, on, off):
        """
        Set PWM output on specific channel
        
        Args:
            channel: Channel number (0-15)
            on: Tick when signal turns on (0-4095)
            off: Tick when signal turns off (0-4095)
        """
        if channel < 0 or channel > 15:
            raise ValueError("Channel must be between 0 and 15")
            
        on = max(0, min(4095, on))
        off = max(0, min(4095, off))
        
        # Write to LED registers
        self.write_byte(self.LED0_ON_L + 4 * channel, on & 0xFF)
        self.write_byte(self.LED0_ON_H + 4 * channel, on >> 8)
        self.write_byte(self.LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write_byte(self.LED0_OFF_H + 4 * channel, off >> 8)
        
    def set_pulse_width(self, channel, pulse_width_us):
        """
        Set pulse width in microseconds
        
        Args:
            channel: Channel number (0-15)
            pulse_width_us: Pulse width in microseconds
        """
        # Calculate ticks (4096 steps per period)
        period_us = 1000000.0 / self.frequency  # Period in microseconds
        ticks = int((pulse_width_us / period_us) * 4096)
        self.set_pwm(channel, 0, ticks)
        
    def set_duty_cycle(self, channel, duty_cycle):
        """
        Set duty cycle as percentage (0-100)
        
        Args:
            channel: Channel number (0-15)
            duty_cycle: Duty cycle percentage (0-100)
        """
        duty_cycle = max(0, min(100, duty_cycle))
        ticks = int((duty_cycle / 100.0) * 4095)
        self.set_pwm(channel, 0, ticks)
        
    def set_servo_angle(self, channel, angle, min_pulse=1000, max_pulse=2000):
        """
        Set servo angle in degrees
        
        Args:
            channel: Channel number (0-15)
            angle: Angle in degrees (typically 0-180)
            min_pulse: Minimum pulse width in microseconds (default 1000)
            max_pulse: Maximum pulse width in microseconds (default 2000)
        """
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        
        # Calculate pulse width
        pulse_width = min_pulse + (angle / 180.0) * (max_pulse - min_pulse)
        self.set_pulse_width(channel, int(pulse_width))
        
    def set_motor_speed(self, channel, speed, min_pulse=1000, max_pulse=2000, neutral=1500):
        """
        Set motor speed for continuous rotation servo
        
        Args:
            channel: Channel number (0-15)
            speed: Speed (-100 to 100, where 0 is stop)
            min_pulse: Minimum pulse width in microseconds (default 1000)
            max_pulse: Maximum pulse width in microseconds (default 2000)
            neutral: Neutral/stop pulse width in microseconds (default 1500)
        """
        # Clamp speed to valid range
        speed = max(-100, min(100, speed))
        
        if speed == 0:
            pulse_width = neutral
        elif speed > 0:
            # Forward
            pulse_width = neutral + (speed / 100.0) * (max_pulse - neutral)
        else:
            # Reverse
            pulse_width = neutral + (speed / 100.0) * (neutral - min_pulse)
            
        self.set_pulse_width(channel, int(pulse_width))
        
    def set_pin(self, channel, value):
        """
        Set channel fully on or off
        
        Args:
            channel: Channel number (0-15)
            value: 0 for fully off, 1 for fully on
        """
        if value:
            self.set_pwm(channel, 4096, 0)  # Fully on
        else:
            self.set_pwm(channel, 0, 4096)  # Fully off

    # LED-specific methods
    def set_led_brightness(self, channel, brightness):
        """
        Set LED brightness (0-100%)
        
        Args:
            channel: Channel number (0-15)
            brightness: Brightness percentage (0-100)
        """
        self.set_duty_cycle(channel, brightness)
        
    def led_on(self, channel):
        """
        Turn LED fully on
        
        Args:
            channel: Channel number (0-15)
        """
        self.set_pin(channel, 1)
        
    def led_off(self, channel):
        """
        Turn LED fully off
        
        Args:
            channel: Channel number (0-15)
        """
        self.set_pin(channel, 0)
        
    def fade_led(self, channel, start_brightness, end_brightness, duration=1.0, steps=50):
        """
        Fade LED smoothly between brightness levels
        
        Args:
            channel: Channel number (0-15)
            start_brightness: Starting brightness (0-100)
            end_brightness: Ending brightness (0-100)
            duration: Fade duration in seconds
            steps: Number of steps for smooth fade
        """
        step_delay = duration / steps
        brightness_step = (end_brightness - start_brightness) / steps
        
        current_brightness = start_brightness
        for _ in range(steps):
            self.set_led_brightness(channel, current_brightness)
            current_brightness += brightness_step
            time.sleep(step_delay)
        
        # Ensure final brightness is set exactly
        self.set_led_brightness(channel, end_brightness)
        
    def blink_led(self, channel, times=3, on_time=0.5, off_time=0.5, brightness=100):
        """
        Blink LED specified number of times
        
        Args:
            channel: Channel number (0-15)
            times: Number of blinks
            on_time: Time LED is on in seconds
            off_time: Time LED is off in seconds
            brightness: Brightness during on state (0-100)
        """
        for _ in range(times):
            self.set_led_brightness(channel, brightness)
            time.sleep(on_time)
            self.led_off(channel)
            time.sleep(off_time)

    def set_all_leds_brightness(self, brightness):
        """
        Set all LEDs to the same brightness
        
        Args:
            brightness: Brightness percentage (0-100)
        """
        ticks = int((brightness / 100.0) * 4095)

        # Use ALL_LED registers to set all channels at once
        self.write_byte(self.ALL_LED_ON_L, 0)
        self.write_byte(self.ALL_LED_ON_H, 0)
        self.write_byte(self.ALL_LED_OFF_L, ticks & 0xFF)
        self.write_byte(self.ALL_LED_OFF_H, ticks >> 8)

    def all_leds_on(self):
        """Turn all LEDs fully on"""
        self.write_byte(self.ALL_LED_ON_L, 0)
        self.write_byte(self.ALL_LED_ON_H, 0x10)  # Set full on bit
        self.write_byte(self.ALL_LED_OFF_L, 0)
        self.write_byte(self.ALL_LED_OFF_H, 0)

    def all_leds_off(self):
        """Turn all LEDs fully off"""
        self.write_byte(self.ALL_LED_ON_L, 0)
        self.write_byte(self.ALL_LED_ON_H, 0)
        self.write_byte(self.ALL_LED_OFF_L, 0)
        self.write_byte(self.ALL_LED_OFF_H, 0x10)  # Set full off bit

    def write_byte(self, reg, value):
        """Write byte to register"""
        self.i2c.writeto_mem(self.address, reg, bytes([value]))
        
    def read_byte(self, reg):
        """Read byte from register"""
        return self.i2c.readfrom_mem(self.address, reg, 1)[0]
        
    def write_word(self, reg, value):
        """Write word (2 bytes) to register"""
        data = bytes([value & 0xFF, (value >> 8) & 0xFF])
        self.i2c.writeto_mem(self.address, reg, data)
        
    def read_word(self, reg):
        """Read word (2 bytes) from register"""
        data = self.i2c.readfrom_mem(self.address, reg, 2)
        return data[0] | (data[1] << 8)


__all__ = [
    "software_reset",
    "PCA9685",
]


# Example usage
if __name__ == "__main__":
    # Initialize PCA9685
    pwm = PCA9685(i2c_bus=1, address=0x40, sda_pin=2, scl_pin=3, freq=50)
    
    # Example: Control steering servo on channel 0
    pwm.set_servo_angle(0, 90)  # Center position
    
    # Example: Control motor on channel 1
    pwm.set_motor_speed(1, 0)   # Stop

    # Example: LED control on channel 2
    pwm.set_led_brightness(2, 50)   # 50% brightness
    pwm.fade_led(2, 0, 100, 2.0)    # Fade from 0 to 100% over 2 seconds
    pwm.blink_led(2, times=5)       # Blink 5 times
    
    # Example: Control all LEDs
    pwm.set_all_leds_brightness(25)  # All LEDs at 25% brightness
    
    # Example: Software reset (can be called without instance)
    # PCA9685.software_reset()
    
    print("PCA9685 initialized successfully!")