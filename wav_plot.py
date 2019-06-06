# wav_plot.py
import librosa
from librosa import display
import matplotlib.pyplot as plt


filename = "recorded_data\\Olympus\\direct_hajimemashite.wav"
wav, sr = librosa.load(filename, sr=44100)

plt.figure(figsize=(16, 9))
librosa.display.waveplot(wav, sr)
plt.show()