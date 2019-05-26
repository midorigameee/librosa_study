import librosa
import librosa.display


filename = "test.wav"
sr = 44100

# サンプリング周波数は引数srで指定しないと
# 勝手に22,050Hzになる

# wavを読み込む関数
wav, sr = librosa.load(path=filename, sr=sr)

librosa.wavplot
