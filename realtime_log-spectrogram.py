import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math


# setting veriable
FILENAME = "170725_1.wav"
SAMPLING_RATE = 44100
FIGSIZE = (12, 8)

chunk = 1024


def _update(frame, x, y, fft, f_bin):
    """グラフを更新するための関数"""
    # 現在のグラフを消去する
    plt.cla()
    # データを更新 (追加) する
    # x.append(frame)
    # y.append(fft[frame])
    x = np.arange(f_bin)
    y = fft[frame] 

    # 折れ線グラフを再描画する
    plt.plot(x, y)

    ax=plt.subplot()
    ax.set_ylim([-100,0])


def main():
    data, sr = librosa.load(FILENAME, SAMPLING_RATE)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")

    power_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fft_spectrum = power_stft.T
    print(fft_spectrum.shape)

    f_max = sr/2
    f_bin = fft_spectrum.shape[1]


    # 描画領域
    fig = plt.figure(figsize=(10, 12))  # (横，縦)

    ## 動的なグラフの表示
    # 描画するデータ (最初は空っぽ)
    dynamic_x = []
    dynamic_y = []

    plt.title("Dynamic fft-spectrum")

    params = {
        'fig': fig,
        'func': _update,  # グラフを更新する関数
        'fargs': (dynamic_x, dynamic_y, fft_spectrum, f_bin),  # 関数の引数 (フレーム番号を除く)
        'interval': 0.1,  # 更新間隔 (ミリ秒)
        'frames': np.arange(0, fft_spectrum.shape[0], 1),  # フレーム番号を生成するイテレータ
        'repeat': False,  # 繰り返さない
    }
    # **は辞書を分解して引数として渡すという意味
    # *はリストを分解して引数として渡すという意味
    anime = animation.FuncAnimation(**params)

    # グラフを保存する
    # ffmpegが必要みたい（結構めんどくさそう）
    # anime.save('sin.gif', writer='pillow')

    # グラフを表示する
    plt.show()


"""
def fft_movie(fft_spectrum, sr=44100):
    # fft_spectrumは（フレーム数、fftbinの数）であることが前提

    # ナイキスト周波数の関係で
    f_max = sr/2
    f_bin = fft_spectrum.shape[1]

    plt.figure(figsize=(FIGSIZE))

    print("fft_spectrum.shape:{}" .format(fft_spectrum.shape))

    for n_frame in range(fft_spectrum.shape[0]):
        # linspace(start, stop, num)
        x = np.linspace(0, f_max, f_bin)
        y = np.power(fft_spectrum[n_frame])

        print("x.shape:{}, y.shape:{}" .format(x.shape, y.shape))

        plt.plot(x, y)

        plt.show()

        plt.clf()


def main():
    data, sr = librosa.load(FILENAME, SAMPLING_RATE)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")

    print("stft.shape:{}" .format(stft.shape))

    fft_movie(stft.T)

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
"""

if __name__ == "__main__":
    main()
