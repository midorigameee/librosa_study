import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# setting veriable
FILENAME = "data_1s.wav"
SAMPLING_RATE = 44100
FIGSIZE = (12, 8)


if __name__ == "__main__":
    data, sr = librosa.load(FILENAME, SAMPLING_RATE)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(FIGSIZE))
    
    # linear spectrogram
    plt.subplot(2,1,1)
    librosa.display.specshow(D, y_axis="linear", sr=sr, hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("linear spectrogram")

    # log spectrogram
    plt.subplot(2,1,2)
    librosa.display.specshow(D, y_axis="log", sr=sr, hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("log spectrogram")

    plt.show()