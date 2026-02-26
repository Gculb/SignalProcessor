import numpy as np

def generate_sine(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def generate_noisy_signal(freq, duration, sample_rate, noise_level=0.3):
    signal = generate_sine(freq, duration, sample_rate)
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise 