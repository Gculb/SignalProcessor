from Digital.core.signal_processor import SignalProcessor
from Digital.io.file_loader import load_wav
from Digital.io.file_saver import save_wav
import numpy as np
from pathlib import Path

#this loads a sample file and processes it using the SignalProcessor class. It demonstrates how to read a WAV file, normalize the audio data, and visualize both the time domain and frequency domain representations of the signal. Additionally, it applies a bandpass filter to isolate frequencies between 200 and 300 Hz, normalizes the filtered signal, and visualizes the results again. The program also prints the dominant frequency in the original signal before filtering.
def main():

    
    project_root = Path(__file__).resolve().parents[2]

    wav_path = project_root / "Digital" / "sample_files" / "M1F1-Alaw-AFsp.wav"

    sample_rate, data = load_wav(wav_path)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    sp = SignalProcessor(data, sample_rate)

    sp.plot_time_domain()
    sp.plot_frequency_domain(xlim=2000)

    print("Dominant frequency:", sp.get_dominant_frequency())
    sp.apply_highpass(80)
    sp.apply_lowpass(3900)
    sp.plot_time_domain()
    sp.plot_frequency_domain(xlim=2000)

    save_path = project_root / "Digital" / "processed_files" / "M1F1-Alaw-AFsp_processed.wav"
    save_wav(save_path, sp.sample_rate, sp.signal)


if __name__ == "__main__":
    main()