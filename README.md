Signal Processor 
Overview

The Signal Processor is a modular Python project designed to demonstrate fundamental digital signal processing (DSP) concepts. This project provides a software-based framework to:

Process and manipulate audio or synthetic signals

Apply common DSP filters (highpass, lowpass, bandpass)

Analyze signals in the frequency domain using FFT

Visualize signals in both time and frequency domains

This project serves as a software prototype for a signal processing pipeline, which can later be extended to real-time hardware (Arduino, microcontrollers) or API-driven backends.

Features

Core DSP Engine (SignalProcessor)

Normalize signals

Apply highpass, lowpass, and bandpass filters

Compute and visualize frequency spectrum

Visualize time-domain signals

Signal Generation

Synthetic sine waves

Noisy signals for testing and demonstration

Extensible Architecture

Layered structure separating core DSP, I/O, and examples

Easily expandable for hardware input, streaming, or API integration

Folder Structure
signal-processor/
│
├── core/
│   ├── __init__.py         # Package initializer
│   └── signalProcessor.py  # DSP core class
│
├── io/
│   ├── __init__.py         # Exposes generators/loaders
│   ├── signal_generator.py # Synthetic signal creation
│   └── file_loader.py      # Audio file loading
│
├── examples/
│   └── demo_pipeline.py    # Example usage of DSP pipeline
│
├── tests/
│   └── test_signal_processor.py # Unit tests for DSP class
│
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
Installation

Clone the repository:

git clone https://github.com/yourusername/signal-processor.git
cd signal-processor

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
Usage Example
from core.signalProcessor import SignalProcessor
from io.signal_generator import generate_noisy_signal

SAMPLE_RATE = 1000  # Hz
DURATION = 2        # seconds

# Generate synthetic signal
signal = generate_noisy_signal(freq=50, duration=DURATION, sample_rate=SAMPLE_RATE)

# Create processor
sp = SignalProcessor(signal, SAMPLE_RATE)

# Apply processing
sp.apply_lowpass(100)
sp.apply_highpass(10)
sp.normalize()

# Visualize
sp.plot_time_domain()
sp.plot_frequency_domain()
Future Extensions

Real-time audio input or streaming

API integration for remote signal processing

Hardware integration (Arduino or other microcontrollers)

Advanced DSP operations (filters, windowing, spectral analysis)

Dependencies

numpy – numerical computing

scipy – signal processing functions

matplotlib – plotting and visualization

License

MIT License – Free to use, modify, and distribute.