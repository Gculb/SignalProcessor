from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


class SignalClassifier:
    def __init__(self) -> None:
        self.model = self._train_model() if SKLEARN_AVAILABLE else None

    def _train_model(self) -> Any:
        rng = np.random.default_rng(42)
        X, y = self._build_synthetic_dataset(rng)

        self.class_names = ["noise", "speech_like", "ecg_like"]
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=500, solver="liblinear", multi_class="ovr"),
                ),
            ]
        )
        model.fit(X, y)
        return model

    def _build_synthetic_dataset(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        examples = []
        labels = []

        for _ in range(150):
            sample_rate = int(rng.choice([16000, 22050, 44100]))
            examples.append(self.extract_features(self._make_speech_like_signal(sample_rate, rng), sample_rate))
            labels.append(1)

        for _ in range(150):
            sample_rate = int(rng.choice([16000, 22050, 44100]))
            examples.append(self.extract_features(self._make_ecg_signal(sample_rate, rng), sample_rate))
            labels.append(2)

        for _ in range(150):
            sample_rate = int(rng.choice([16000, 22050, 44100]))
            examples.append(self.extract_features(self._make_noise_signal(sample_rate, rng), sample_rate))
            labels.append(0)

        return np.vstack(examples), np.asarray(labels, dtype=np.int64)

    def _make_speech_like_signal(self, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
        duration = rng.uniform(0.2, 0.6)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        frequency = float(rng.uniform(200, 2800))
        carrier = np.sin(2 * np.pi * frequency * t)
        noise = rng.normal(scale=0.03, size=carrier.shape)
        envelope = np.clip(np.sin(2 * np.pi * 2.5 * t) * 0.5 + 0.5, 0.1, 1.0)
        signal = carrier * envelope + noise
        return signal.astype(np.float32)

    def _make_ecg_signal(self, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
        heart_rate = rng.uniform(40.0, 120.0) / 60.0
        duration = rng.uniform(2.0, 4.0)
        shape = int(sample_rate * duration)
        t = np.linspace(0.0, duration, shape, endpoint=False)
        period = 1.0 / heart_rate

        ecg = np.zeros(shape, dtype=np.float32)
        qrs_width = int(max(1, sample_rate * 0.08))
        for beat_center in np.arange(0, duration, period):
            center_index = int(min(shape - 1, beat_center * sample_rate))
            left = max(0, center_index - qrs_width)
            right = min(shape, center_index + qrs_width)
            window = np.hamming(right - left)
            ecg[left:right] += window * rng.uniform(0.6, 1.0)

        baseline = 0.05 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t)
        noise = rng.normal(scale=0.02, size=shape)
        return np.asarray(ecg + baseline + noise, dtype=np.float32)

    def _make_noise_signal(self, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
        duration = rng.uniform(0.2, 0.6)
        shape = int(sample_rate * duration)
        noise = rng.normal(scale=rng.uniform(0.15, 0.4), size=shape)
        band = self._pink_noise(shape, rng) * rng.uniform(0.2, 0.6)
        pulse = np.zeros(shape, dtype=np.float32)
        for _ in range(rng.integers(1, 4)):
            center = rng.integers(0, shape)
            width = rng.integers(40, min(400, shape // 3))
            amplitude = rng.uniform(0.4, 0.9)
            start = max(0, center - width // 2)
            end = min(shape, start + width)
            pulse[start:end] += amplitude * np.hamming(end - start)

        return np.asarray(noise + band + pulse, dtype=np.float32)

    def _pink_noise(self, length: int, rng: np.random.Generator) -> np.ndarray:
        freqs = np.fft.rfftfreq(length, d=1.0)
        freqs[0] = 1.0
        spectrum = rng.normal(size=freqs.shape) / np.sqrt(freqs)
        signal = np.fft.irfft(spectrum, n=length)
        return signal.astype(np.float32)

    def extract_features(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return np.zeros(8, dtype=np.float32)

        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        dbfs = 20 * math.log10(max(rms, 1e-6))
        zero_crossings = int(np.count_nonzero(np.diff(np.signbit(samples))))
        zcr = float(zero_crossings / max(samples.size - 1, 1))

        windowed = samples * np.hanning(samples.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(samples.size, 1.0 / sample_rate)
        total_energy = float(np.sum(spectrum)) + 1e-12

        speech_mask = (freqs >= 120) & (freqs <= 3400)
        speech_energy_ratio = float(np.sum(spectrum[speech_mask]) / total_energy)

        ecg_mask = (freqs >= 0.5) & (freqs <= 40)
        low_freq_energy_ratio = float(np.sum(spectrum[ecg_mask]) / total_energy)

        dominant_index = int(np.argmax(spectrum)) if spectrum.size else 0
        dominant_frequency = float(freqs[dominant_index]) if spectrum.size else 0.0

        positive = spectrum[spectrum > 1e-12]
        spectral_flatness = float(
            math.exp(np.mean(np.log(positive))) / (np.mean(positive) + 1e-12)
            if positive.size
            else 0.0
        )

        return np.asarray(
            [
                rms,
                peak,
                dbfs,
                zcr,
                speech_energy_ratio,
                low_freq_energy_ratio,
                spectral_flatness,
                dominant_frequency,
            ],
            dtype=np.float32,
        )

    def predict(
        self,
        samples: np.ndarray,
        sample_rate: int,
        signal_type: str = "generic",
        device_type: str = "generic",
    ) -> dict[str, Any]:
        features = self.extract_features(samples, sample_rate).reshape(1, -1)

        if self.model is not None:
            probabilities = self.model.predict_proba(features)[0]
            class_index = int(np.argmax(probabilities))
            label = self.class_names[class_index]
            return {
                "label": label,
                "confidence": round(float(probabilities[class_index]), 4),
                "signalTypeHint": signal_type,
                "deviceTypeHint": device_type,
            }

        speech_ratio = float(features[0, 4])
        low_freq_ratio = float(features[0, 5])
        spectral_flatness = float(features[0, 6])
        dominant_frequency = float(features[0, 7])
        device_type_lower = device_type.lower()
        signal_type_lower = signal_type.lower()

        if "speech" in signal_type_lower or "microphone" in device_type_lower:
            if speech_ratio > 0.25 or dominant_frequency > 100:
                label = "speech_like"
                confidence = min(max(0.4 + speech_ratio, 0.0), 1.0)
            else:
                label = "noise"
                confidence = min(max(0.6 - spectral_flatness, 0.0), 1.0)
        elif "ecg" in signal_type_lower or "ecg" in device_type_lower:
            label = "ecg_like"
            confidence = min(max(0.5 + low_freq_ratio, 0.0), 1.0)
        elif "accelerometer" in device_type_lower:
            label = "other"
            confidence = min(max(0.4 + low_freq_ratio * 0.2, 0.0), 1.0)
        else:
            if speech_ratio > 0.35 and spectral_flatness < 0.6:
                label = "speech_like"
                confidence = min(max(0.35 + speech_ratio, 0.0), 1.0)
            elif low_freq_ratio > 0.35 and dominant_frequency < 60:
                label = "ecg_like"
                confidence = min(max(0.45 + low_freq_ratio, 0.0), 1.0)
            elif spectral_flatness > 0.7:
                label = "noise"
                confidence = min(max(0.7 - spectral_flatness, 0.0), 1.0)
            else:
                label = "speech_like"
                confidence = min(max(0.35 + speech_ratio, 0.0), 1.0)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "signalTypeHint": signal_type,
            "deviceTypeHint": device_type,
        }
