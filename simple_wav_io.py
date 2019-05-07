import librosa


filename = "test.wav"
savename = filename[:-4] + "_output.wav"
sr = 44100

# サンプリング周波数は引数srで指定しないと
# 勝手に22,050Hzになる

# wavを読み込む関数
wav, sr = librosa.load(path=filename, sr=sr)

# wavを出力する関数
librosa.output.write_wav(y=wav, path=savename, sr=sr)