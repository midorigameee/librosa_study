import librosa

filename = "data_1s.wav"


wav, sr = librosa.load(path=filename)

savename = filename[:-4] + "_output.wav"
librosa.output.write_wav(y=wav, path=savename, sr=sr)