import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class SignalProcessor:
    def __init__(self,  signal_data, sample_rate):
        self.signal = signal_data
        self.sample_rate = sample_rate 
    def normalize(self):
        max_val = np.max(np.abs(self.signal))
        if max_val == 0:
            return self.signal
        self.signal = self.signal / max_val
        return self.signal
    def compute_fft(self):
        fft_vals = np.fft.rfft(self.signal)
        freqs = np.fft.rfftfreq(len(self.signal), 1/self.sample_rate)
        return freqs, np.abs(fft_vals)
    def apply_highpass(self, cutoff, order=4):
        self._validate_cutoff(cutoff)
        b, a = signal.butter(order, cutoff / (0.5 * self.sample_rate), btype='high')
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal

    def apply_lowpass(self, cutoff, order=4):
        self._validate_cutoff(cutoff)
        b, a = signal.butter(order, cutoff / (0.5 * self.sample_rate), btype='low')
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal
    
    def apply_bandpass(self, low_cutoff, high_cutoff):
        self._validate_cutoff(low_cutoff)
        self._validate_cutoff(high_cutoff)
        b, a = signal.butter(4, [low_cutoff / (0.5 * self.sample_rate), high_cutoff / (0.5 * self.sample_rate)], btype='band')
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal
    def plot_frequency_domain(self, xlim=None):
        freqs, mags = self.compute_fft()
        plt.plot(freqs, mags)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Frequency Domain (FFT)")
        
        if xlim is not None:
            plt.xlim(0, xlim)
            
        plt.show()
    def plot_time_domain(self):
        t = np.arange(len(self.signal)) / self.sample_rate
        plt.plot(t, self.signal)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Time Domain Signal")
        plt.show()
    def _validate_cutoff(self, cutoff):
        nyquist = self.sample_rate / 2
        if cutoff <= 0 or cutoff >= nyquist:
            raise ValueError(f"Cutoff must be between 0 and {nyquist} Hz")
