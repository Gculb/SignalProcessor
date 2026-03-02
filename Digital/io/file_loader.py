import soundfile as sf

def load_wav(path):
    data, sample_rate = sf.read(path)
    return sample_rate, data