

from Digital.core.signal_processor import SignalProcessor
from Digital.io.signal_generator import generate_noisy_signal 
import matplotlib.pyplot as plt

def main():
    SAMPLE_RATE = 1000
    DURATION = 2
    FREQUENCY = 50

    signal = generate_noisy_signal(freq=FREQUENCY, duration=DURATION, sample_rate=SAMPLE_RATE)
    sp = SignalProcessor(signal, SAMPLE_RATE)
    sp.apply_bandpass(48, 52)
    sp.plot_time_domain() 
    sp.plot_frequency_domain(xlim=200)

if __name__ == "__main__":
    main() 