from scipy.io import wavfile 

def save_wav(path, sample_rate, data):
    userInput = input('Do you want to save the file? (y/n): ')
    if userInput.lower() == 'y':
        wavfile.write(path, sample_rate, data)
        print(f"File saved to {path}")
    else:
        print("File not saved.")