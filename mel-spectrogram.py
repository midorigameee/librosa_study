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


# setting veriable
FILENAME = "data_1s.wav"
SAMPLING_RATE = 44100
FIGSIZE = (12, 10)


if __name__ == "__main__":
    data, sr = librosa.load(FILENAME, SAMPLING_RATE)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(FIGSIZE))

    ### パワースペクトログラムの表示
    plt.subplot(3,1,1)
    librosa.display.specshow(D,
                            y_axis="hz",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("power spectrogram")

    ### 対数パワースペクトログラムの表示
    plt.subplot(3,1,2)        
    librosa.display.specshow(D,
                            y_axis="log",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("log-power spectrogram")

    ### FBANKのスペクトログラムの表示
    plt.subplot(3,1,3)
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
    plt.show()