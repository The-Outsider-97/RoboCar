"""
This MicroPython library is designed for Raspberry Pi Pico to make it easy to use with ultrasonic sensor. It is easy to use for not only beginners but also experienced users... 

It is created by DIYables to work with DIYables products, but also work with products from other brands. Please consider purchasing products from [DIYables Store on Amazon](https://amazon.com/diyables) from to support our work.

Product Link:
- Ultrasonic Sensor: https://diyables.io/products/ultrasonic-sensor
- Sensor Kit: https://diyables.io/products/sensor-kit


Copyright (c) 2024, DIYables.io. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

- Neither the name of the DIYables.io nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY DIYABLES.IO "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL DIYABLES.IO BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

# DIYables_Pico_Ultrasonic_Sensor.py

from machine import Pin
import time
import utime
import machine

class UltrasonicSensor:
    def __init__(self, trig_pin, echo_pin, vsys_pin=None):
        self.trig = Pin(trig_pin, Pin.OUT)
        self.echo = Pin(echo_pin, Pin.IN)
        self.detection_threshold = 500 # float('inf')  # Initially set to infinity
        self.filter_enabled = False  # Filter is disabled by default
        self.filter_mode = 'median'
        self.filter_window = 5
        self.num_samples = 5  # Default number of samples to 5 when filter is disabled
        self.distances = []  # Store measurements
        self.distance_history = []   # For advanced filtering
        self.error_count = 0
        self.last_valid_distance = None
        self.timeout_us = 38000  # 38ms timeout for maximum range
        self.cm_per_us = 0.017  # Default at 20°C
        self.vsys_pin = vsys_pin
        if vsys_pin:
            self.adc = machine.ADC(machine.Pin(vsys_pin))

        # Initialize trigger to low
        self.trig.low()
        utime.sleep_us(5)

    def read_voltage(self):
        """Read system voltage for compensation"""
        if self.vsys_pin:
            reading = self.adc.read_u16()
            voltage = reading * 3.3 / 65535
            return voltage * 3  # VSYS is divided by 3
        return 3.3  # Assume nominal voltage

    def set_temperature(self, celsius):
        """Adjust speed of sound based on temperature in Celsius"""
        # Speed of sound = 331.3 + (0.606 * temperature) m/s
        self.speed_of_sound = 331.3 + (0.606 * celsius)
        # Update conversion factor (cm per microsecond)
        self.cm_per_us = (self.speed_of_sound * 100) / 1000000  # Convert to cm/μs

    def loop(self):
        """Perform a measurement cycle and update the list of distances."""
        # Ensure the trigger pin is low for a clean pulse
        self.trig.low()
        time.sleep_us(2)
        
        # Send a 12 microsecond pulse to start the measurement
        self.trig.high()
        time.sleep_us(12)
        self.trig.low()

        # Wait for the echo to start
        timeout_start = utime.ticks_us()
        while self.echo.value() == 0:
            if utime.ticks_diff(utime.ticks_us(), timeout_start) > self.timeout_us:
                self.error_count += 1
                return None  # Return None or some error indication on timeout

        # Record the start time of the echo
        signal_off = utime.ticks_us()

        # Wait for the echo to end
        while self.echo.value() == 1:
            if utime.ticks_diff(utime.ticks_us(), signal_off) > self.timeout_us:
                self.error_count += 1
                return None  # Return None or some error indication on timeout

        # Record the end time of the echo
        signal_on = utime.ticks_us()
            
        # Calculate the duration of the echo pulse
        time_passed = utime.ticks_diff(signal_on, signal_off)

        # Apply voltage compensation if available
        voltage_factor = 1.0
        if self.vsys_pin:
            voltage = self.read_voltage()
            voltage_factor = 3.3 / voltage  # Compensate for voltage drops

        # Calculate the distance in centimeters
        distance = (time_passed * self.cm_per_us) * voltage_factor

        # Validate reading
        if 2 <= distance <= 500:  # Valid range per datasheet
            self.distances.append(distance)
            self.distance_history.append(distance)
            self.last_valid_distance = distance
            
            # Maintain sample size limits
            if len(self.distances) > self.num_samples:
                self.distances.pop(0)
            if len(self.distance_history) > self.filter_window:
                self.distance_history.pop(0)
        else:
            self.error_count += 1
        
    def calibrate(self, known_distance, num_readings=10):
        """Auto-calibrate by taking multiple readings with a known distance"""
        calibration_values = []
        for _ in range(num_readings):
            self.loop()
            utime.sleep_ms(50)  # Wait between readings
            if self.distances and self.distances[-1] is not None:
                calibration_values.append(self.distances[-1])
        
        if calibration_values:
            avg_distance = sum(calibration_values) / len(calibration_values)
            self.calibration_offset = avg_distance - known_distance
            # Apply offset to all future readings
            self.distances = [d - self.calibration_offset for d in self.distances]
            self.distance_history = [d - self.calibration_offset for d in self.distance_history]

    def clear_calibration(self):
        """Remove any applied calibration offset"""
        if hasattr(self, 'calibration_offset'):
            del self.calibration_offset

    def get_distance(self):
        """Return the calculated distance based on current measurements."""
        if not self.distances:
            return None

        # Apply the selected filter
        if self.filter_enabled:
            if self.filter_mode == 'median' and len(self.distance_history) >= 3:
                # Median filter
                sorted_distances = sorted(self.distance_history)
                calculated_distance = sorted_distances[len(sorted_distances) // 2]
            elif self.filter_mode == 'moving_average' and len(self.distance_history) > 0:
                # Moving average filter
                calculated_distance = sum(self.distance_history) / len(self.distance_history)
            else:
                # Default to middle-range average if filter requirements not met
                if len(self.distances) >= self.num_samples:
                    sorted_distances = sorted(self.distances)
                    mid_index_start = len(sorted_distances) // 4
                    mid_index_end = len(sorted_distances) * 3 // 4
                    mid_distances = sorted_distances[mid_index_start:mid_index_end]
                    calculated_distance = sum(mid_distances) / len(mid_distances)
                else:
                    calculated_distance = self.distances[-1]
        else:
            # No filtering
            calculated_distance = self.distances[-1]

        # Apply calibration offset if it exists
        if hasattr(self, 'calibration_offset'):
            calculated_distance -= self.calibration_offset

        # Check against detection threshold
        if calculated_distance > self.detection_threshold:
            return False  # No object detected within threshold
        return calculated_distance

    def set_detection_threshold(self, distance):
        """Set the maximum distance beyond which no object is considered detected."""
        self.detection_threshold = distance

    def set_filter_mode(self, mode='median', window_size=5):
        """Set filtering mode: 'median' or 'moving_average'"""
        if mode in ['median', 'moving_average']:
            self.filter_mode = mode
            self.filter_window = window_size
            self.distance_history = self.distance_history[-window_size:]  # Truncate history

    def enable_filter(self, num_samples=20):
        """Enable filtering of measurements and set number of samples for filtering."""
        if num_samples > 0:
            self.num_samples = num_samples
            self.filter_enabled = True
        else:
            raise ValueError("Number of samples must be greater than 0")

    def disable_filter(self):
        """Disable filtering of measurements and reset number of samples to 1."""
        self.filter_enabled = False
        # self.num_samples = 1  # Reset to default sample count

    def enable_power_save(self, enable=True):
        """Enable/disable power-saving features"""
        self.power_save = enable
        if enable:
            # Reduce sampling rate and disable filtering when in power save
            self.num_samples = 3
            self.filter_enabled = False