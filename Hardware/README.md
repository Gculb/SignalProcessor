# Hardware Signal Processor

This folder contains the Arduino-facing hardware implementation of the signal processor.

## Files

- [ArduinoSignalProcessor.ino](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware/ArduinoSignalProcessor.ino)
- [SignalProcessor.h](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware/SignalProcessor.h)
- [SignalProcessor.cpp](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware/SignalProcessor.cpp)

## Signal path

The processor accepts:

- the main signal sample
- optional microphone noise reference input
- optional sensor noise reference input
- optional random-noise estimate

It then applies:

1. DC offset removal
2. microphone noise subtraction
3. sensor noise subtraction
4. high-pass filtering
5. low-pass filtering
6. notch filtering
7. random-noise suppression with smoothing and a noise gate

## Arduino notes

- The sketch includes `Arduino.h` so the usual Arduino symbols are available.
- `SignalProcessor.cpp` is C++-safe for the Arduino build system.
- If you only have one analog input, call `signal_processor_process_sample(...)` instead of passing references.
- Change the notch from `60.0f` to `50.0f` if your hardware environment uses 50 Hz mains power.
- The demo loop assumes an 8 kHz sample cadence with `delayMicroseconds(125)`.
