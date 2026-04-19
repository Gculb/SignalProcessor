from pathlib import Path
import math

import numpy as np
from scipy import signal

from Digital.core.signal_processor import SignalProcessor
from Digital.io.file_loader import load_wav


EPSILON = 1e-12


def to_db(value):
    return 20.0 * math.log10(max(float(value), EPSILON))


def load_as_analysis_mono(path):
    sample_rate, data = load_wav(path)
    processor = SignalProcessor(data, sample_rate)

    if processor.signal.ndim > 1:
        processor.to_mono(strategy="best_speech_band")

    return sample_rate, np.asarray(processor.signal, dtype=np.float64)


def speech_band_rms(signal_data, sample_rate, low_cutoff=120.0, high_cutoff=3400.0):
    nyquist = 0.5 * sample_rate
    high_cutoff = min(high_cutoff, nyquist - 1.0)
    if high_cutoff <= low_cutoff:
        return np.sqrt(np.mean(np.square(signal_data)))

    b, a = signal.butter(4, [low_cutoff / nyquist, high_cutoff / nyquist], btype="band")
    filtered = signal.filtfilt(b, a, signal_data)
    return np.sqrt(np.mean(np.square(filtered)))


def level_match_to_original(original_signal, cleaned_signal, sample_rate):
    original_speech_rms = speech_band_rms(original_signal, sample_rate)
    cleaned_speech_rms = speech_band_rms(cleaned_signal, sample_rate)
    if cleaned_speech_rms <= EPSILON:
        return cleaned_signal

    gain = original_speech_rms / cleaned_speech_rms
    return cleaned_signal * gain


def estimate_noise_floor(signal_data, sample_rate, frame_seconds=0.03, hop_seconds=0.01, percentile=20):
    frame_length = max(1, int(frame_seconds * sample_rate))
    hop_length = max(1, int(hop_seconds * sample_rate))

    frames = []
    rms_values = []
    for start in range(0, len(signal_data) - frame_length + 1, hop_length):
        frame = signal_data[start:start + frame_length]
        frames.append(frame)
        rms_values.append(np.sqrt(np.mean(np.square(frame))))

    if not frames:
        rms = np.sqrt(np.mean(np.square(signal_data)))
        return rms, signal_data

    rms_values = np.asarray(rms_values)
    threshold = np.percentile(rms_values, percentile)
    quiet_frames = [frame for frame, rms in zip(frames, rms_values) if rms <= threshold]
    quiet_signal = np.concatenate(quiet_frames) if quiet_frames else signal_data
    quiet_rms = np.sqrt(np.mean(np.square(quiet_signal)))
    return quiet_rms, quiet_signal


def band_energy_ratio_db(signal_data, sample_rate, speech_band=(120, 3400), noise_band=(0, 120)):
    freqs, psd = signal.welch(signal_data, fs=sample_rate, nperseg=min(4096, len(signal_data)))
    speech_mask = (freqs >= speech_band[0]) & (freqs <= speech_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs < noise_band[1])

    speech_energy = np.sum(psd[speech_mask])
    noise_energy = np.sum(psd[noise_mask])
    return 10.0 * math.log10((speech_energy + EPSILON) / (noise_energy + EPSILON))


def peak_power_near(freqs, psd, target_hz, tolerance_hz=3.0):
    mask = (freqs >= (target_hz - tolerance_hz)) & (freqs <= (target_hz + tolerance_hz))
    if not np.any(mask):
        return EPSILON
    return float(np.max(psd[mask]))


def interference_attenuation_db(original_signal, cleaned_signal, sample_rate, candidates=(50, 60, 100, 120)):
    freqs_original, psd_original = signal.welch(
        original_signal,
        fs=sample_rate,
        nperseg=min(4096, len(original_signal)),
    )
    freqs_cleaned, psd_cleaned = signal.welch(
        cleaned_signal,
        fs=sample_rate,
        nperseg=min(4096, len(cleaned_signal)),
    )

    best_target = None
    best_original_power = -1.0
    for target in candidates:
        power = peak_power_near(freqs_original, psd_original, target)
        if power > best_original_power:
            best_original_power = power
            best_target = target

    cleaned_power = peak_power_near(freqs_cleaned, psd_cleaned, best_target)
    attenuation_db = 10.0 * math.log10((best_original_power + EPSILON) / (cleaned_power + EPSILON))
    return best_target, attenuation_db


def benchmark_pair(original_path, cleaned_path):
    original_sr, original_signal = load_as_analysis_mono(original_path)
    cleaned_sr, cleaned_signal = load_as_analysis_mono(cleaned_path)

    if original_sr != cleaned_sr:
        raise ValueError(f"Sample rate mismatch for {original_path.name}: {original_sr} vs {cleaned_sr}")

    limit = min(len(original_signal), len(cleaned_signal))
    original_signal = original_signal[:limit]
    cleaned_signal = cleaned_signal[:limit]
    cleaned_signal = level_match_to_original(original_signal, cleaned_signal, original_sr)

    original_rms = np.sqrt(np.mean(np.square(original_signal)))
    cleaned_rms = np.sqrt(np.mean(np.square(cleaned_signal)))

    original_noise_floor, _ = estimate_noise_floor(original_signal, original_sr)
    cleaned_noise_floor, _ = estimate_noise_floor(cleaned_signal, cleaned_sr)

    original_ratio_db = band_energy_ratio_db(original_signal, original_sr)
    cleaned_ratio_db = band_energy_ratio_db(cleaned_signal, cleaned_sr)

    interference_target, interference_drop_db = interference_attenuation_db(
        original_signal,
        cleaned_signal,
        original_sr,
    )

    return {
        "file": original_path.name,
        "sample_rate_hz": original_sr,
        "duration_s": limit / original_sr,
        "original_rms_dbfs": to_db(original_rms),
        "cleaned_rms_dbfs": to_db(cleaned_rms),
        "original_noise_floor_dbfs": to_db(original_noise_floor),
        "cleaned_noise_floor_dbfs": to_db(cleaned_noise_floor),
        "noise_floor_improvement_db": to_db(original_noise_floor) - to_db(cleaned_noise_floor),
        "original_speech_to_low_noise_db": original_ratio_db,
        "cleaned_speech_to_low_noise_db": cleaned_ratio_db,
        "speech_to_low_noise_improvement_db": cleaned_ratio_db - original_ratio_db,
        "interference_target_hz": interference_target,
        "interference_attenuation_db": interference_drop_db,
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    sample_dir = project_root / "Digital" / "sample_files"
    processed_dir = project_root / "Digital" / "processed_files"

    cleaned_files = sorted(processed_dir.glob("*_cleaned.wav"))
    if not cleaned_files:
        raise FileNotFoundError(f"No cleaned WAV files found in {processed_dir}")

    print("WAV benchmark results")
    print("=====================")

    for cleaned_path in cleaned_files:
        original_name = cleaned_path.name.replace("_cleaned.wav", ".wav")
        original_path = sample_dir / original_name
        if not original_path.exists():
            print(f"\nSkipping {cleaned_path.name}: no matching original file found.")
            continue

        metrics = benchmark_pair(original_path, cleaned_path)
        print(f"\nFile: {metrics['file']}")
        print(
            f"  sample_rate={metrics['sample_rate_hz']} Hz, "
            f"duration={metrics['duration_s']:.3f} s"
        )
        print(
            f"  rms_dbfs: original={metrics['original_rms_dbfs']:.2f}, "
            f"cleaned={metrics['cleaned_rms_dbfs']:.2f}"
        )
        print(
            f"  noise_floor_dbfs: original={metrics['original_noise_floor_dbfs']:.2f}, "
            f"cleaned={metrics['cleaned_noise_floor_dbfs']:.2f}, "
            f"improvement={metrics['noise_floor_improvement_db']:.2f} dB"
        )
        print(
            f"  speech_to_low_noise_db: original={metrics['original_speech_to_low_noise_db']:.2f}, "
            f"cleaned={metrics['cleaned_speech_to_low_noise_db']:.2f}, "
            f"improvement={metrics['speech_to_low_noise_improvement_db']:.2f} dB"
        )
        print(
            f"  strongest_mains_candidate={metrics['interference_target_hz']} Hz, "
            f"attenuation={metrics['interference_attenuation_db']:.2f} dB"
        )


if __name__ == "__main__":
    main()
