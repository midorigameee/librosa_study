import pyaudio
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np


CHUNK = 1024
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
# wav = np.zeros((TIME*RATE))

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
    y = np.frombuffer(input, dtype=np.int16) / 32768.0

    # 前回のデータを更新していく
    # wav = np.concatenate([wav[CHUNK:], y], axis=0)

    ## 描画していく
    # マイクから受け取った値をチャンク幅でFFT
    fft = scipy.fftpack.fft(y)
    # サンプリング周波数とチャンクからグラフの横軸を作成
    freqList = scipy.fftpack.fftfreq(CHUNK, d=1.0/ RATE) 

    # 振幅スペクトルにする→2乗しているからパワーじゃない？
    amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft]

    # プロット
    plt.plot(freqList, amplitudeSpectrum, marker= 'o', linestyle='-')
    plt.axis([0, RATE/2, 0, 50])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude spectrum")

    # 画面に表示する間隔
    # plt.show()を使うと毎回描画を消さないといけない
    plt.pause(0.001)
    plt.clf()

stream.stop_stream()
stream.close()
p.terminate()

print("Stop Streaming")