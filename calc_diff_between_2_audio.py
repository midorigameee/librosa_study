# 音源ココから
# https://www.mitsue.co.jp/service/audio_and_video/audio_production/high_resolution_narration.html
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys


# setting veriable
SAMPLING_RATE = 44100
FIGSIZE = (12, 10)


def create_spectrogram(filename_1, filename_2, sampling_rate):
    sr = sampling_rate
    data_1, _ = librosa.load(filename_1, sr)
    data_2, _ = librosa.load(filename_2, sr)

    stft_1 = librosa.stft(data_1,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")

    stft_2 = librosa.stft(data_2,
                        n_fft=1024,
                        hop_length=512,
                        window="hamming")

    sub_stft = spectrum_substraction(stft_1, stft_2)

    # D = librosa.amplitude_to_db(np.abs(stft_1), ref=np.max)
    D = librosa.amplitude_to_db(np.abs(sub_stft), ref=np.max)
    # D = librosa.power_to_db(np.abs(sub_stft), ref=np.max)

    plt.figure(figsize=(FIGSIZE))

    ### パワースペクトログラムの表示
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D,
                            y_axis="hz",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("power spectrogram")

    ### 対数パワースペクトログラムの表示
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D,
                            y_axis="log",
                            x_axis="time",
                            sr=sr,
                            fmax=sr/2,
                            hop_length=512)
    plt.colorbar(format="%+2.0f db")
    plt.title("log-power spectrogram")

    """
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
    """

    # レイアウトを整える
    plt.tight_layout()

    savepath = "spectrogram_subtraction.jpg"

    plt.savefig(savepath)


def spectrum_substraction(stft_1, stft_2):
    print("stft_1.shape : {}" .format(stft_1.shape))
    print("stft_2.shape : {}" .format(stft_2.shape))

    if stft_1.shape[1] == stft_2.shape[1]:
        diff = []

        for i in range(stft_1.shape[1]):
            D_1 = np.abs(stft_1)
            D_2 = np.abs(stft_2)

            diff_temp = D_1.T[i] - D_2.T[i]
            # diff_temp = stft_1.T[i] - stft_2.T[i]

            diff.append(diff_temp)

        diff = np.array(diff).T
        print("diff.shape : {}" .format(diff.shape))

        return diff
    else:
        print("The lengthes of them are not equal.")
        return None


def main():
    args = sys.argv

    if len(args) >= 3:
        audio_1 = args[1]
        audio_2 = args[2]

        print("audio_1 : {}" .format(audio_1))
        print("audio_2 : {}" .format(audio_2))

        create_spectrogram(audio_1, audio_2, SAMPLING_RATE)

    else:
        print("Arguments are too short.")


if __name__ == "__main__":
    main()
