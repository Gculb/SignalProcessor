import numpy as np
import soundfile as sf
from pathlib import Path

def save_wav(path, sample_rate, data):

    path = Path(path)

 
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data)

    if np.max(np.abs(data)) != 0:
        data = data / np.max(np.abs(data))

    data = np.int16(data * 32767)

    # Reshape mono to (N, 1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    userInput = input("Do you want to save the file? (y/n): ")

    if userInput.lower() == "y":
        sf.write(str(path), data, sample_rate)
        print(f"File saved to {path}")
    else:
        print("File not saved.")