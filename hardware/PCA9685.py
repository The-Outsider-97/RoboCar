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

from ..modules.machine import Pin, I2C
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("PCA9685")
printer = PrettyPrinter()


class PCA9685(object):
    """PCA9685 16-channel 12-bit PWM controller for RoboCar.

    Channel numbering in this class is zero-based.  Therefore the physical
    channel labels printed on the board map as follows::

        board channel 1 -> index 0 -> ESC
        board channel 2 -> index 1 -> steering servo
        board channel 5 -> index 4 -> blue severity LED
        board channel 6 -> index 5 -> green severity LED
        board channel 7 -> index 6 -> yellow severity LED
        board channel 8 -> index 7 -> red severity LED

    The PCA9685 has one shared PWM frequency for all sixteen outputs.  RoboCar
    therefore retains 50 Hz for the ESC and steering servo and performs the
    requested 3-pulse-per-second severity pattern in software.
    """

    CHANNEL_COUNT        = 16

    # RoboCar actuator allocation (board channels 1 and 2).
    ESC_CHANNEL          = 0
    STEERING_CHANNEL     = 1

    # RoboCar severity-light allocation (board channels 5 through 8).
    BLUE_LED_CHANNEL     = 4
    GREEN_LED_CHANNEL    = 5
    YELLOW_LED_CHANNEL   = 6
    RED_LED_CHANNEL      = 7

    SEVERITY_LED_CHANNELS = {
        "blue": BLUE_LED_CHANNEL,
        "green": GREEN_LED_CHANNEL,
        "yellow": YELLOW_LED_CHANNEL,
        "red": RED_LED_CHANNEL,
    }
    SEVERITY_LEVEL_COLORS = {
        1: "blue",
        2: "green",
        3: "yellow",
        4: "red",
    }
    DEFAULT_SEVERITY_RATE_HZ = 3.0

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

    def __init__(self, i2c_bus=1, address=0x40, sda_pin=2, scl_pin=3, freq=50, i2c=None):
        """
        Initialize PCA9685

        Args:
            i2c_bus: I2C bus number (default 1 for Raspberry Pi)
            address: I2C address of PCA9685 (default 0x40)
            sda_pin: SDA pin number (default GPIO2 for Pi 5)
            scl_pin: SCL pin number (default GPIO3 for Pi 5)
            freq: PWM frequency in Hz (default 50 for servos)
            i2c: optional pre‑configured I2C instance (for testing / simulation)
        """
        if i2c is None:
            self.i2c = I2C(i2c_bus, sda=Pin(sda_pin), scl=Pin(scl_pin))
        else:
            self.i2c = i2c
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
            on: Tick when signal turns on (0-4095), or 4096 for full on
            off: Tick when signal turns off (0-4095), or 4096 for full off
        """
        if channel < 0 or channel >= self.CHANNEL_COUNT:
            raise ValueError("Channel must be between 0 and 15")

        on = int(on)
        off = int(off)
        if not 0 <= on <= 4096 or not 0 <= off <= 4096:
            raise ValueError("PWM on/off values must be between 0 and 4096")
        if on == 4096 and off == 4096:
            raise ValueError("A PCA9685 channel cannot be both fully on and fully off")

        # Bit 4 of LEDn_ON_H / LEDn_OFF_H is the PCA9685 full-on/full-off bit.
        # Handling the 4096 sentinel explicitly is essential: clamping it to
        # 4095 makes led_off() produce an almost fully-on output.
        base = self.LED0_ON_L + 4 * channel
        if on == 4096:
            on_l, on_h = 0x00, 0x10
            off_l, off_h = 0x00, 0x00
        elif off == 4096:
            on_l, on_h = 0x00, 0x00
            off_l, off_h = 0x00, 0x10
        else:
            on_l, on_h = on & 0xFF, (on >> 8) & 0x0F
            off_l, off_h = off & 0xFF, (off >> 8) & 0x0F

        self.write_byte(base, on_l)
        self.write_byte(base + 1, on_h)
        self.write_byte(base + 2, off_l)
        self.write_byte(base + 3, off_h)
        
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
        duty_cycle = max(0.0, min(100.0, float(duty_cycle)))
        if duty_cycle == 0.0:
            self.set_pin(channel, 0)
        elif duty_cycle == 100.0:
            self.set_pin(channel, 1)
        else:
            ticks = int(round((duty_cycle / 100.0) * 4096.0))
            self.set_pwm(channel, 0, min(4095, ticks))
        
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
        Set bidirectional ESC/motor command from -100 to 100.
        
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
    def _validate_led_channel(self, channel):
        """Reject actuator or unallocated channels passed to an LED method."""
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise ValueError("LED channel must be an integer")
        if channel not in self.SEVERITY_LED_CHANNELS.values():
            allowed = ", ".join(
                str(item) for item in self.SEVERITY_LED_CHANNELS.values()
            )
            raise ValueError(
                "LED channel %r is not allocated to a severity LED; expected "
                "one of the zero-based channels: %s" % (channel, allowed)
            )
        return channel

    @staticmethod
    def _validate_brightness(brightness):
        """Return a finite LED brightness percentage in the closed 0-100 range."""
        try:
            value = float(brightness)
        except (TypeError, ValueError):
            raise ValueError("LED brightness must be a finite number from 0 to 100")
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError("LED brightness must be a finite number from 0 to 100")
        return value

    def set_led_brightness(self, channel, brightness):
        """
        Set LED brightness (0-100%)
        
        Args:
            channel: Channel number (0-15)
            brightness: Brightness percentage (0-100)
        """
        self.set_duty_cycle(
            self._validate_led_channel(channel),
            self._validate_brightness(brightness),
        )
        
    def led_on(self, channel):
        """
        Turn LED fully on
        
        Args:
            channel: Channel number (0-15)
        """
        self.set_pin(self._validate_led_channel(channel), 1)
        
    def led_off(self, channel):
        """
        Turn LED fully off
        
        Args:
            channel: Channel number (0-15)
        """
        self.set_pin(self._validate_led_channel(channel), 0)
        
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

    def clear_severity_leds(self):
        """Turn off only the four configured severity LEDs.

        The ESC and steering channels are deliberately excluded.  Global
        ALL_LED register writes are unsafe when actuators and LEDs share one
        PCA9685 because they would overwrite the ESC and servo waveforms.
        """
        for channel in self.SEVERITY_LED_CHANNELS.values():
            self.led_off(channel)

    def set_severity_led(self, color, brightness=100):
        """Show one severity color continuously and turn the other three off.

        Args:
            color: One of ``blue``, ``green``, ``yellow``, or ``red``.
            brightness: Active LED brightness from 0 to 100 percent.

        Returns:
            The zero-based PCA9685 channel used for the selected LED.
        """
        normalized = str(color).strip().lower()
        if normalized not in self.SEVERITY_LED_CHANNELS:
            allowed = ", ".join(self.SEVERITY_LED_CHANNELS)
            raise ValueError("Unknown severity color %r; expected one of: %s" % (color, allowed))

        channel = self.SEVERITY_LED_CHANNELS[normalized]
        self.clear_severity_leds()
        self.set_led_brightness(channel, brightness)
        return channel

    def set_severity_level(self, level, brightness=100):
        """Show one of the four ordered severity levels continuously.

        The order follows the supplied physical color order exactly:
        level 1 is blue, level 2 green, level 3 yellow, and level 4 red.
        No application-specific meaning is imposed on those four levels here.
        """
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError("severity level must be an integer from 1 through 4")
        color = self.SEVERITY_LEVEL_COLORS.get(level)
        if color is None:
            raise ValueError("severity level must be an integer from 1 through 4")
        return self.set_severity_led(color, brightness=brightness)

    def blink_severity_led(
        self,
        color,
        pulses=3,
        rate_hz=DEFAULT_SEVERITY_RATE_HZ,
        brightness=100,
    ):
        """Blink one severity LED at an exact pulse rate.

        A pulse consists of one on phase plus one off phase.  At the RoboCar
        default of 3 Hz, each phase lasts 1/6 second and three complete pulses
        take one second.  This method is intentionally finite and blocking; it
        never pulses the ESC channel.

        Args:
            color: One of ``blue``, ``green``, ``yellow``, or ``red``.
            pulses: Number of complete on/off pulses; must be a positive int.
            rate_hz: Complete pulses per second; must be greater than zero.
            brightness: Active LED brightness from 0 to 100 percent.

        Returns:
            The zero-based PCA9685 channel used for the selected LED.
        """
        if isinstance(pulses, bool) or not isinstance(pulses, int) or pulses <= 0:
            raise ValueError("pulses must be a positive integer")

        rate_hz = float(rate_hz)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be a finite value greater than zero")

        normalized = str(color).strip().lower()
        if normalized not in self.SEVERITY_LED_CHANNELS:
            allowed = ", ".join(self.SEVERITY_LED_CHANNELS)
            raise ValueError("Unknown severity color %r; expected one of: %s" % (color, allowed))

        channel = self.SEVERITY_LED_CHANNELS[normalized]
        half_period = 0.5 / rate_hz
        self.clear_severity_leds()
        try:
            self.blink_led(
                channel,
                times=pulses,
                on_time=half_period,
                off_time=half_period,
                brightness=brightness,
            )
        finally:
            # A completed, interrupted, or failed finite pattern always leaves
            # the severity bank in a deterministic off state.
            self.clear_severity_leds()
        return channel

    def blink_severity_level(
        self,
        level,
        pulses=3,
        rate_hz=DEFAULT_SEVERITY_RATE_HZ,
        brightness=100,
    ):
        """Blink severity level 1-4 using its assigned color channel."""
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError("severity level must be an integer from 1 through 4")
        color = self.SEVERITY_LEVEL_COLORS.get(level)
        if color is None:
            raise ValueError("severity level must be an integer from 1 through 4")
        return self.blink_severity_led(
            color,
            pulses=pulses,
            rate_hz=rate_hz,
            brightness=brightness,
        )

    def set_all_leds_brightness(self, brightness):
        """
        Set the four RoboCar severity LEDs to the same brightness.

        This compatibility method no longer writes the PCA9685 ALL_LED
        registers because channels 0 and 1 control physical actuators.
        
        Args:
            brightness: Brightness percentage (0-100)
        """
        for channel in self.SEVERITY_LED_CHANNELS.values():
            self.set_led_brightness(channel, brightness)

    def all_leds_on(self):
        """Turn all four RoboCar severity LEDs fully on."""
        for channel in self.SEVERITY_LED_CHANNELS.values():
            self.led_on(channel)

    def all_leds_off(self):
        """Turn all four RoboCar severity LEDs fully off."""
        self.clear_severity_leds()

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
    "PCA9685",
]


if __name__ == "__main__":
    print("\n=== Running PCA9685 Hardware Test ===\n")
    printer.status("TEST", "Starting PCA9685 channel-allocation test", "info")

    pwm = None
    test_succeeded = False
    try:
        pwm = PCA9685(
            i2c_bus=1,
            address=0x40,
            sda_pin=2,
            scl_pin=3,
            freq=50,
        )

        # Safety first: the test never commands non-zero motor throttle.
        pwm.set_motor_speed(PCA9685.ESC_CHANNEL, 0)
        printer.status(
            "TEST",
            "ESC neutral confirmed on board channel 1 (index 0)",
            "info",
        )

        pwm.set_servo_angle(PCA9685.STEERING_CHANNEL, 90)
        printer.status(
            "TEST",
            "Steering centered on board channel 2 (index 1)",
            "info",
        )

        # Each color produces three complete pulses in one second: 3 pulses/s.
        for level, color in PCA9685.SEVERITY_LEVEL_COLORS.items():
            channel = PCA9685.SEVERITY_LED_CHANNELS[color]
            printer.status(
                "TEST",
                "Level %d / %s LED: board channel %d (index %d), 3 pulses/s"
                % (level, color.capitalize(), channel + 1, channel),
                "info",
            )
            pwm.blink_severity_level(
                level,
                pulses=3,
                rate_hz=PCA9685.DEFAULT_SEVERITY_RATE_HZ,
                brightness=100,
            )

        test_succeeded = True
        printer.status("TEST", "PCA9685 channel test completed", "success")

    except Exception as exc:
        printer.status(
            "TEST",
            "PCA9685 hardware test failed: %s: %s"
            % (type(exc).__name__, exc),
            "error",
        )
        raise

    finally:
        if pwm is not None:
            # Leave the car in a deterministic safe state even when the test is
            # interrupted: motor neutral, steering centered, severity LEDs off.
            try:
                pwm.set_motor_speed(PCA9685.ESC_CHANNEL, 0)
            except Exception as exc:
                printer.status("TEST", "ESC cleanup failed: %s" % exc, "error")
            try:
                pwm.set_servo_angle(PCA9685.STEERING_CHANNEL, 90)
            except Exception as exc:
                printer.status("TEST", "Steering cleanup failed: %s" % exc, "error")
            try:
                pwm.clear_severity_leds()
            except Exception as exc:
                printer.status("TEST", "LED cleanup failed: %s" % exc, "error")

    if test_succeeded:
        print("\n=== Test ran successfully ===\n")
