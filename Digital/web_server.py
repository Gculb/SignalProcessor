from __future__ import annotations

import base64
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from Digital.core.signal_processor import SignalProcessor

STATIC_ROOT = PROJECT_ROOT / "website"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_SAMPLES = 96_000


class SignalRequestHandler(BaseHTTPRequestHandler):
    server_version = "DigitalSignalServer/1.0"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_static_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return

        if self.path == "/app.js":
            self._send_static_file(STATIC_ROOT / "app.js", "text/javascript; charset=utf-8")
            return

        if self.path == "/styles.css":
            self._send_static_file(STATIC_ROOT / "styles.css", "text/css; charset=utf-8")
            return

        if self.path == "/api/health":
            self._send_json({"ok": True, "service": "digital-signal-processor"})
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/process":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
            sample_rate = int(payload.get("sampleRate", 0))
            samples = decode_samples(payload.get("samples"))
            settings = payload.get("settings", {})

            if sample_rate <= 0:
                raise ValueError("sampleRate must be a positive integer.")
            if samples.size == 0:
                raise ValueError("samples must contain at least one value.")
            if samples.size > MAX_SAMPLES:
                samples = samples[-MAX_SAMPLES:]

            result = process_signal(samples, sample_rate, settings)
            self._send_json(result)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:  # Keep the browser session alive while surfacing server failures.
            self._send_json({"error": f"Processing failed: {error}"}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is required.")
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def _send_static_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": f"Missing static file: {path.name}"}, status=404)
            return

        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def decode_samples(raw_samples: Any) -> np.ndarray:
    if isinstance(raw_samples, str):
        binary = base64.b64decode(raw_samples)
        return np.frombuffer(binary, dtype=np.float32).astype(np.float32, copy=True)

    if isinstance(raw_samples, list):
        return np.asarray(raw_samples, dtype=np.float32)

    raise ValueError("samples must be a base64 float32 buffer or an array of numbers.")


def process_signal(samples: np.ndarray, sample_rate: int, settings: dict[str, Any]) -> dict[str, Any]:
    raw_signal = np.asarray(samples, dtype=np.float32)
    raw_signal = np.nan_to_num(raw_signal, nan=0.0, posinf=1.0, neginf=-1.0)

    raw_metrics = compute_metrics(raw_signal, sample_rate)
    processor = SignalProcessor(raw_signal.copy(), sample_rate)

    highpass = float(settings.get("highpass", 80))
    lowpass = float(settings.get("lowpass", min(3400, sample_rate / 2 - 1)))
    detect_noise = bool(settings.get("detectNoise", True))

    applied_filters: list[str] = []
    detected_noise: list[float] = []

    try:
        if 0 < highpass < sample_rate / 2:
            processor.apply_highpass(highpass)
            applied_filters.append(f"highpass {highpass:g} Hz")
        if highpass < lowpass < sample_rate / 2:
            processor.apply_lowpass(lowpass)
            applied_filters.append(f"lowpass {lowpass:g} Hz")
        if detect_noise:
            detected_noise = processor.detect_noise_peaks(
                min_freq=max(40, highpass),
                max_freq=min(1000, lowpass),
            )
            processor.apply_notch_series(detected_noise)
            if detected_noise:
                applied_filters.append("adaptive notch")
        processor.normalize()
        processed_signal = np.asarray(processor.signal, dtype=np.float32)
    except ValueError:
        processed_signal = raw_signal

    processed_metrics = compute_metrics(processed_signal, sample_rate)
    spectrum = compute_spectrum(processed_signal, sample_rate)

    return {
        "sampleRate": sample_rate,
        "sampleCount": int(raw_signal.size),
        "durationSeconds": raw_signal.size / sample_rate,
        "filters": applied_filters,
        "noisePeaksHz": [round(float(peak), 2) for peak in detected_noise],
        "raw": raw_metrics,
        "processed": processed_metrics,
        "waveform": downsample(processed_signal, 256),
        "spectrum": spectrum,
    }


def compute_metrics(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    if samples.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "dbfs": -120.0,
            "dominantFrequencyHz": 0.0,
            "zeroCrossingRate": 0.0,
        }

    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    dbfs = 20 * math.log10(max(rms, 1e-6))
    dominant_frequency = dominant_frequency_hz(samples, sample_rate)
    zero_crossings = np.count_nonzero(np.diff(np.signbit(samples)))
    zero_crossing_rate = float(zero_crossings / max(samples.size - 1, 1))

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "dbfs": round(dbfs, 2),
        "dominantFrequencyHz": round(dominant_frequency, 2),
        "zeroCrossingRate": round(zero_crossing_rate, 5),
    }


def dominant_frequency_hz(samples: np.ndarray, sample_rate: int) -> float:
    if samples.size < 2:
        return 0.0

    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    if spectrum.size <= 1:
        return 0.0

    peak_index = int(np.argmax(spectrum[1:]) + 1)
    return float(frequencies[peak_index])


def compute_spectrum(samples: np.ndarray, sample_rate: int, bins: int = 96) -> list[dict[str, float]]:
    if samples.size < 2:
        return []

    windowed = samples * np.hanning(samples.size)
    magnitudes = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    max_frequency = min(5000, sample_rate / 2)
    mask = (frequencies >= 0) & (frequencies <= max_frequency)

    selected_freqs = frequencies[mask]
    selected_magnitudes = magnitudes[mask]
    if selected_magnitudes.size == 0:
        return []

    frequency_groups = np.array_split(selected_freqs, min(bins, selected_freqs.size))
    magnitude_groups = np.array_split(selected_magnitudes, min(bins, selected_magnitudes.size))
    max_magnitude = float(np.max(selected_magnitudes)) or 1.0

    return [
        {
            "frequency": round(float(np.mean(freq_group)), 2),
            "magnitude": round(float(np.max(mag_group) / max_magnitude), 5),
        }
        for freq_group, mag_group in zip(frequency_groups, magnitude_groups)
    ]


def downsample(samples: np.ndarray, points: int) -> list[float]:
    if samples.size <= points:
        return [round(float(value), 5) for value in samples]

    groups = np.array_split(samples, points)
    return [round(float(np.mean(group)), 5) for group in groups]


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), SignalRequestHandler)
    print(f"Signal Processor website running at http://{host}:{port}")
    print("Use Ctrl+C to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    run()
