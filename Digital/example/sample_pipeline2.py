from pathlib import Path

from Digital.core.signal_processor import SignalProcessor
from Digital.io.file_loader import load_wav
from Digital.io.file_saver import save_wav


def inspect_and_process_file(wav_path, output_dir):
    sample_rate, data = load_wav(wav_path)
    processor = SignalProcessor(data, sample_rate)

    metadata = processor.inspect_signal()
    print(f"\nInspecting: {wav_path.name}")
    print(
        "  "
        f"sample_rate={metadata['sample_rate']} Hz, "
        f"channels={metadata['channels']}, "
        f"duration={metadata['duration_seconds']:.3f} s, "
        f"dominant_frequency={metadata['dominant_frequency_hz']:.1f} Hz"
    )
    print(f"  detected_noise_peaks={metadata['noise_peaks_hz']}")

    if metadata["channels"] > 1:
        processor.to_mono(strategy="best_speech_band")
        print("  selected mono strategy=best_speech_band")

    detected_noise = processor.denoise_speech()
    output_path = output_dir / f"{wav_path.stem}_cleaned.wav"
    save_wav(output_path, processor.sample_rate, processor.signal)

    print(
        "  "
        f"applied highpass=80 Hz, lowpass=3400 Hz, "
        f"notches={detected_noise}, saved={output_path.name}"
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "sample_files"
    output_dir = project_root / "processed_files"

    wav_files = sorted(sample_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {sample_dir}")

    for wav_path in wav_files:
        inspect_and_process_file(wav_path, output_dir)


if __name__ == "__main__":
    main()
