"""
時間領域の音源
↓（STFT）
周波数領域のスペクトル
↓（2乗）
周波数領域のパワースペクトル
↓（メルフィルタバンク）
メル周波数領域のパワースペクトル
↓（対数を取る）
メル周波数領域の対数パワースペクトル（FBANK）

"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# setting veriable
# TARGET_TYPE = "directory" # directory or file
# TARGET_PATH = "recorded_data"
SAMPLING_RATE = 44100
FIGSIZE = (12, 10)


def create_spectrogram(filename, sampling_rate):
    data, sr = librosa.load(filename, sampling_rate)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(FIGSIZE))

    ### パワースペクトログラムの表示
    plt.subplot(3, 1, 1)
    librosa.display.specshow(D,
                            y_axis="hz",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("power spectrogram")

    ### 対数パワースペクトログラムの表示
    plt.subplot(3, 1, 2)
    librosa.display.specshow(D,
                            y_axis="log",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("log-power spectrogram")

    ### FBANKのスペクトログラムの表示
    plt.subplot(3, 1, 3)
    mel = librosa.feature.melspectrogram(y=data,
                                        sr=sr,
                                        n_fft=1024,
                                        hop_length=512,
                                        power=1,
                                        n_mels=40)

    # 対数を取る
    log_mel = librosa.amplitude_to_db(mel, ref=np.max)

    # 対数パワースペクトルを表示
    librosa.display.specshow(log_mel,
                            y_axis="mel",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("mel spectrogram")

    # レイアウトを整える
    plt.tight_layout()

    # ファイル名は元々”.wav”が入っているのでそれを取り除く
    savename = filename[:-4] + ".jpg"
    savedir = "spectrogram"

    if os.path.exists(savedir) is not True:
        os.mkdir(savedir)

    savepath = os.path.join(savedir, savename)

    plt.savefig(savepath)


def directory_to_spectrogram(target_path, sampling_rate):
    # 指定ディレクトリ内の全てのファイルをリスト化
    target_list = os.listdir(target_path)

    # ディレクトリ移動
    os.chdir(target_path)

    # 指定ディレクトリ内の全てのファイルを見る
    for target_name in target_list:
        # wav以外のファイルは無視してスペクトログラムスペクトログラム作成
        if check_audio(target_name):
            create_spectrogram(target_name, sampling_rate)
        else:
            pass


def check_audio(filename):
    # ファイル名の後ろから3文字を参照
    extension = filename[-3:]

    # 後ろから3文字は拡張子なのでwavかどうか判断できる
    if extension == "wav":
        return True
    else:
        return False


def main():
    args = sys.argv

    print("target_type : {}" .format(args[1]))
    print("target_path : {}" .format(args[2]))

    target_type = args[1]
    target_path = args[2]

    if len(args) >= 3:
        if target_type == "d":
            directory_to_spectrogram(target_path, SAMPLING_RATE)
        elif target_type == "f":
            create_spectrogram(target_path, SAMPLING_RATE)
        else:
            print("Arguments are not suitable.")
    else:
        print("Arguments are too short.")


if __name__ == "__main__":
    main()


"""
# 今までのスペクトログラム
if __name__ == "__main__":
    data, sr = librosa.load(FILENAME, SAMPLING_RATE)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(FIGSIZE))

    ### パワースペクトログラムの表示
    plt.subplot(3, 1, 1)
    librosa.display.specshow(D,
                            y_axis="hz",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("power spectrogram")

    ### 対数パワースペクトログラムの表示
    plt.subplot(3, 1, 2)
    librosa.display.specshow(D,
                            y_axis="log",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("log-power spectrogram")

    ### FBANKのスペクトログラムの表示
    plt.subplot(3, 1, 3)
    mel = librosa.feature.melspectrogram(y=data,
                                        sr=sr,
                                        n_fft=1024,
                                        hop_length=512,
                                        power=1,
                                        n_mels=40)

    # 対数を取る
    log_mel=librosa.amplitude_to_db(mel, ref=np.max)

    # 対数パワースペクトルを表示
    librosa.display.specshow(log_mel,
                            y_axis="mel",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("mel spectrogram")

    plt.tight_layout()

    savepath = FILENAME[:-4] + ".jpg"
    plt.savefig(savepath)

    plt.show()
"""