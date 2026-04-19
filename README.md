# Signal Processor

## Overview

Signal Processor is split into two connected parts:

- [Digital](/C:/Users/Gculb/Desktop/SignalProcessor/Digital), a Python prototype for offline signal analysis, WAV inspection, filtering, and experimentation
- [Hardware](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware), an Arduino-oriented implementation for running a simplified signal-cleaning pipeline on embedded hardware

The purpose of the project is to prototype a DSP pipeline in software first, learn what kinds of noise are present in the recordings, and then carry a practical version of that cleanup path onto hardware.

## Project Split

### Digital

The `Digital` folder is the analysis and prototyping side of the project. It is used to:

- load and inspect sample WAV files
- generate synthetic test signals
- analyze signals in the time and frequency domains
- apply DSP filters such as high-pass, low-pass, band-pass, and notch filters
- test denoising ideas before moving them to hardware
- save cleaned output files for comparison

This side of the repo is where we determine what the real recordings contain and what filter settings are worth deploying.

### Hardware

The `Hardware` folder is the embedded side of the project. It is used to:

- process live samples from a microphone or sensor
- subtract optional microphone-noise and sensor-noise reference inputs
- reduce random noise on-device
- run a hardware-friendly cleanup chain with DC removal, filtering, notch suppression, smoothing, and gating
- provide an Arduino sketch and reusable signal processor implementation

This side is not a direct port of the Python stack. It is a practical embedded implementation built from the lessons learned in `Digital`.

## Current Capabilities

### Digital capabilities

- `SignalProcessor` class for core DSP operations
- FFT-based spectrum inspection
- time-domain and frequency-domain plotting
- synthetic sine and noisy signal generation
- WAV loading and saving
- sample pipelines for inspecting and cleaning recorded files

### Hardware capabilities

- reusable Arduino-compatible signal processor implementation
- support for:
  - main signal input
  - microphone noise reference input
  - sensor noise reference input
  - random noise estimate input
- filtering stages suitable for embedded use

## Repository Layout

```text
SignalProcessor/
|
+-- Digital/
|   +-- core/
|   |   +-- __init__.py
|   |   +-- signal_processor.py
|   +-- io/
|   |   +-- __init__.py
|   |   +-- file_loader.py
|   |   +-- file_saver.py
|   |   +-- signal_generator.py
|   +-- example/
|   |   +-- sample_pipeline.py
|   |   +-- sample_pipeline2.py
|   +-- sample_files/
|   +-- processed_files/
|
+-- Hardware/
|   +-- ArduinoSignalProcessor.ino
|   +-- SignalProcessor.h
|   +-- SignalProcessor.cpp
|   +-- README.md
|
+-- README.md
```

## How The Two Sides Work Together

The intended workflow is:

1. Use `Digital` to inspect sample files and identify what noise is actually present.
2. Tune filtering and denoising behavior in Python where iteration is faster.
3. Recreate the useful parts of that pipeline in `Hardware` using embedded-friendly code.
4. Deploy the hardware version to a board for live signal cleanup.

## Digital Usage

The Python side depends on packages such as `numpy`, `scipy`, `matplotlib`, and `soundfile`.

Example workflow:

```python
from Digital.core.signal_processor import SignalProcessor
from Digital.io.signal_generator import generate_noisy_signal

sample_rate = 1000
duration = 2

signal = generate_noisy_signal(freq=50, duration=duration, sample_rate=sample_rate)
processor = SignalProcessor(signal, sample_rate)

processor.apply_lowpass(100)
processor.apply_highpass(10)
processor.normalize()
processor.plot_time_domain()
processor.plot_frequency_domain()
```

For recorded files, the sample pipeline in [sample_pipeline2.py](/C:/Users/Gculb/Desktop/SignalProcessor/Digital/example/sample_pipeline2.py) inspects the WAV files in `Digital/sample_files` and writes cleaned output into `Digital/processed_files`.

## Hardware Usage

The Arduino-facing implementation lives in:

- [ArduinoSignalProcessor.ino](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware/ArduinoSignalProcessor.ino)
- [SignalProcessor.h](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware/SignalProcessor.h)
- [SignalProcessor.cpp](/C:/Users/Gculb/Desktop/SignalProcessor/Hardware/SignalProcessor.cpp)

The embedded processor is designed for real-time use and accepts:

- the main sample
- optional microphone noise reference
- optional sensor noise reference
- optional random-noise estimate

This makes the hardware implementation suitable for live microphone-plus-sensor setups where noise can come from multiple sources.

## Goals

- improve the digital analysis pipeline so it better identifies useful signal versus noise
- continue tuning the hardware pipeline based on real recordings
- support continuous signal processing and frequency tracking
- strengthen the bridge between offline WAV analysis and real-time embedded deployment

## License

MIT License.
