import pyaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2


CHUNK = 1024
TIME = 3
RATE = 44100

p = pyaudio.PyAudio()

stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

# グラフの大きさ
plt.figure(figsize=(10, 4))
# グラフの縦軸の上限を決める
# ax = plt.subplot()
# ax.set_ylim([min,max])


# 時間　＊　サンプリング周波数(T=1/s)
wav = np.zeros((TIME*RATE))

while stream.is_active():
    """
    # escが押されたら終了
    k = cv2.waitKey(1)
    print("k:", k)
    if k == 27:
        break
    """

    # マイクからchunkの数ずつ読み込み
    input = stream.read(CHUNK)

    # バッファをnumpyの形に変換
    y = np.frombuffer(input, dtype=np.int16)

    # 前回のデータを更新していく
    wav = np.concatenate([wav[CHUNK:], y], axis=0)

    # 描画していく

    ## 時間領域で表示
    # librosa.display.waveplot(wav, sr=RATE)

    ## FBANKで表示
    # 周波数領域をメル周波数領域に変換
    mel = librosa.feature.melspectrogram(y=wav,
                                        sr=RATE,
                                        n_fft=1024,
                                        hop_length=512,
                                        power=2,
                                        n_mels=40)
    
    # 対数を取る
    log_mel=librosa.power_to_db(mel, ref=np.max)
    
    # 対数パワースペクトルを表示
    librosa.display.specshow(log_mel,
                            y_axis="mel",
                            x_axis="time",
                            sr=RATE,
                            fmax=RATE/2,
                            hop_length=512)

    # 画面に表示する間隔
    # plt.show()を使うと毎回描画を消さないといけない
    plt.pause(0.001)
    plt.clf()

stream.stop_stream()
stream.close()
p.terminate()

print("Stop Streaming")