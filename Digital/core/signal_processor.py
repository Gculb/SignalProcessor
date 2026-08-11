import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class SignalProcessor:
    def __init__(self,  signal_data, sample_rate):
        self.signal = np.asarray(signal_data, dtype=np.float32)
        self.sample_rate = sample_rate 

    def to_mono(self, strategy="average"):
        if self.signal.ndim == 1:
            return self.signal

        if strategy == "average":
            self.signal = np.mean(self.signal, axis=1)
            return self.signal

        if strategy == "best_speech_band":
            best_channel = 0
            best_score = -np.inf

            for channel_index in range(self.signal.shape[1]):
                channel = self.signal[:, channel_index]
                score = self._speech_band_score(channel)
                if score > best_score:
                    best_score = score
                    best_channel = channel_index

            self.signal = self.signal[:, best_channel]
            return self.signal

        raise ValueError(f"Unsupported mono strategy: {strategy}")

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
    def apply_notch(self, notch_freq, quality_factor=30):
        self._validate_cutoff(notch_freq)
        b, a = signal.iirnotch(notch_freq / (0.5 * self.sample_rate), quality_factor)
        self.signal = signal.filtfilt(b, a, self.signal)
        return self.signal

    def detect_noise_peaks(
        self,
        min_freq=40,
        max_freq=1000,
        percentile=20,
        max_peaks=4,
        prominence_ratio=6.0,
        return_details=False,
    ):
        analysis_signal = self._get_quiet_frames(percentile=percentile)
        if analysis_signal.size == 0:
            return [] if not return_details else []

        frequencies, spectrogram = self._compute_spectrogram(
            analysis_signal,
            frame_seconds=0.08,
            hop_seconds=0.04,
        )

        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies = frequencies[mask]
        spectrogram = spectrogram[mask, :]
        if frequencies.size < 3 or spectrogram.shape[1] == 0:
            return [] if not return_details else []

        average_power = np.mean(spectrogram, axis=1)
        power_floor = np.percentile(average_power, 10)
        power_floor = max(power_floor, 1e-12)

        local_noise_floor = np.empty_like(average_power)
        for index in range(average_power.size):
            start = max(0, index - 3)
            end = min(average_power.size, index + 4)
            neighbors = np.concatenate((average_power[start:index], average_power[index + 1:end]))
            local_noise_floor[index] = np.median(neighbors) if neighbors.size else average_power[index]
        local_noise_floor = np.maximum(local_noise_floor, power_floor)

        peak_indices, _ = signal.find_peaks(average_power)
        if peak_indices.size == 0:
            return [] if not return_details else []

        candidate_indices = [
            int(index)
            for index in peak_indices
            if average_power[index] >= local_noise_floor[index] * prominence_ratio
        ]

        if not candidate_indices:
            return [] if not return_details else []

        details = []
        global_threshold = np.percentile(average_power, 50) * 1.5
        for index in candidate_indices:
            frame_threshold = max(local_noise_floor[index] * 1.5, global_threshold)
            persistence = float(np.mean(spectrogram[index, :] >= frame_threshold))

            left = index
            while left > 0 and average_power[left] >= local_noise_floor[index] * 1.5:
                left -= 1
            right = index
            while right + 1 < average_power.size and average_power[right] >= local_noise_floor[index] * 1.5:
                right += 1

            bandwidth = float(frequencies[right] - frequencies[left]) if right > left else float(frequencies[1] - frequencies[0])
            details.append(
                {
                    "frequency": float(frequencies[index]),
                    "ratio": round(float(average_power[index] / local_noise_floor[index]), 2),
                    "persistence": round(persistence, 3),
                    "bandwidthHz": round(bandwidth, 2),
                }
            )

        ranked = sorted(
            details,
            key=lambda item: (item["ratio"], item["persistence"]),
            reverse=True,
        )

        selected = []
        selected_details = []
        for detail in ranked:
            if all(abs(detail["frequency"] - existing) > 20 for existing in selected):
                selected.append(detail["frequency"])
                selected_details.append(detail)
            if len(selected) >= max_peaks:
                break

        harmonic_peaks = self._filter_harmonic_noise_peaks(selected)
        filtered_frequencies = harmonic_peaks if harmonic_peaks else selected

        if return_details:
            return [detail for detail in selected_details if detail["frequency"] in filtered_frequencies]

        return filtered_frequencies

    def _compute_spectrogram(self, samples, frame_seconds=0.08, hop_seconds=0.04):
        frame_length = max(64, int(frame_seconds * self.sample_rate))
        hop_length = max(1, int(hop_seconds * self.sample_rate))
        if frame_length > samples.size:
            frame_length = samples.size
            hop_length = max(1, frame_length // 2)

        window = np.hanning(frame_length)
        frames = []
        for start in range(0, samples.size - frame_length + 1, hop_length):
            frame = samples[start : start + frame_length] * window
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            frames.append(spectrum)

        if not frames:
            return np.array([]), np.array([[]])

        frequencies = np.fft.rfftfreq(frame_length, 1 / self.sample_rate)
        return frequencies, np.stack(frames, axis=1)

    def apply_notch_series(self, notch_freqs, quality_factor=30):
        for notch_freq in notch_freqs:
            if 0 < notch_freq < self.sample_rate / 2:
                self.apply_notch(notch_freq, quality_factor=quality_factor)
        return self.signal

    def denoise_speech(
        self,
        highpass=80,
        lowpass=3400,
        detect_noise=True,
        notch_quality_factor=30,
        percentile=20,
    ):
        self.apply_highpass(highpass)
        self.apply_lowpass(lowpass)

        detected_peaks = []
        if detect_noise:
            detected_peaks = self.detect_noise_peaks(
                min_freq=max(40, highpass),
                max_freq=min(1000, lowpass),
                percentile=percentile,
            )
            detected_peaks = self._filter_harmonic_noise_peaks(detected_peaks)
            self.apply_notch_series(
                detected_peaks,
                quality_factor=notch_quality_factor,
            )

        self.normalize()
        return detected_peaks

    def inspect_signal(self):
        if self.signal.ndim == 1:
            analysis_processor = self
            channels = 1
        else:
            mono_preview = np.mean(self.signal, axis=1)
            analysis_processor = SignalProcessor(mono_preview, self.sample_rate)
            channels = self.signal.shape[1]

        dominant_frequency = analysis_processor.get_dominant_frequency()
        noise_peaks = analysis_processor.detect_noise_peaks()

        return {
            "sample_rate": self.sample_rate,
            "channels": channels,
            "duration_seconds": len(self.signal) / self.sample_rate,
            "dominant_frequency_hz": float(dominant_frequency),
            "noise_peaks_hz": noise_peaks,
        }

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
        
    def get_frequency_spectrum(self):
        N = len(self.signal)

        fft_vals = np.fft.fft(self.signal)
        fft_vals = np.abs(fft_vals)[:N // 2]

        freqs = np.fft.fftfreq(N, 1 / self.sample_rate)
        freqs = freqs[:N // 2]

        return freqs, fft_vals

    def get_dominant_frequency(self):
        freqs, spectrum = self.get_frequency_spectrum()
        peak_index = np.argmax(spectrum)
        return freqs[peak_index]

    def _speech_band_score(self, channel):
        freqs, psd = signal.welch(
            channel,
            fs=self.sample_rate,
            nperseg=min(4096, len(channel)),
        )
        speech_mask = (freqs >= 120) & (freqs <= 3400)
        noise_mask = (freqs >= 0) & (freqs < 120)

        speech_energy = np.sum(psd[speech_mask])
        noise_energy = np.sum(psd[noise_mask]) + 1e-12
        return speech_energy / noise_energy

    def _filter_harmonic_noise_peaks(self, peaks, tolerance_hz=4.0):
        harmonic_matches = {}
        for base_frequency in (50.0, 60.0):
            matches = []
            harmonic = 1
            while harmonic * base_frequency <= 1000:
                target = harmonic * base_frequency
                for peak in peaks:
                    if abs(peak - target) <= tolerance_hz:
                        matches.append(peak)
                harmonic += 1

            if len(matches) >= 2:
                harmonic_matches[base_frequency] = sorted(set(matches))

        if not harmonic_matches:
            return []

        best_base = max(harmonic_matches, key=lambda key: len(harmonic_matches[key]))
        return harmonic_matches[best_base]

    def _get_quiet_frames(self, frame_seconds=0.03, hop_seconds=0.01, percentile=20):
        if self.signal.ndim != 1:
            raise ValueError("Noise analysis expects a mono signal.")

        frame_length = max(1, int(frame_seconds * self.sample_rate))
        hop_length = max(1, int(hop_seconds * self.sample_rate))

        frames = []
        rms_values = []
        for start in range(0, len(self.signal) - frame_length + 1, hop_length):
            frame = self.signal[start:start + frame_length]
            frames.append(frame)
            rms_values.append(np.sqrt(np.mean(np.square(frame))))

        if not frames:
            return self.signal

        rms_values = np.asarray(rms_values)
        threshold = np.percentile(rms_values, percentile)
        quiet_frames = [frame for frame, rms in zip(frames, rms_values) if rms <= threshold]

        if not quiet_frames:
            return self.signal

        return np.concatenate(quiet_frames)
