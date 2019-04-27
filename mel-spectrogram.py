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
FIGSIZE = (12, 8)


if __name__ == "__main__":
    data, sr = librosa.load(FILENAME, SAMPLING_RATE)

    stft = librosa.stft(data,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    plt.figure(figsize=(FIGSIZE))

    plt.subplot(2,1,1)    
    # 周波数領域をメル周波数領域に変換
    mel = librosa.feature.melspectrogram(y=data,
                                        sr=sr,
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
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("mel spectrogram")

    plt.subplot(2,1,2)
    # 周波数領域をメル周波数領域に変換
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