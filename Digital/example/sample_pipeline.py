

from core.signal_processor import SignalProcessor
from io.signal_generator import generate_noisy_signal


SAMPLE_RATE = 1000
DURATION = 2

signal = generate_noisy_signal(freq=50, duration=DURATION, sample_rate=SAMPLE_RATE)
sp = SignalProcessor(signal, SAMPLE_RATE)
sp.apply_lowpass(100)
sp.normalize()
sp.plot_time_domain() 