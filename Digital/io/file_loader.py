from scipy.io import wavfile

def load_wav(path):
    sample_rate, data = wavfile.read(path)
    return sample_rate, data 