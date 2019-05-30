import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def main():
    main_name = "chAngE.wav"
    inst_name = "chAngE_inst.wav"
    plt.figure(figsize=(10, 15))

    main_wav, sr = librosa.load(main_name)
    print("file_name:{}, sr:{}" .format(main_name, sr))
    print(main_wav.shape)

    inst_wav, sr = librosa.load(inst_name)
    print("file_name:{}, sr:{}" .format(inst_name, sr))
    print(inst_wav.shape)


    main_power_spec = np.abs(librosa.stft(main_wav))
    print("power_spec.shape:", main_power_spec.shape)

    inst_power_spec = np.abs(librosa.stft(inst_wav))
    print("power_spec.shape:", inst_power_spec.shape)

    plt.subplot(3, 1, 1)    # (row, colum, num)
    librosa.display.specshow(librosa.amplitude_to_db(main_power_spec, ref=np.max), y_axis='log', x_axis='time')
    wav_title = "main_Power_spectrogram"
    plt.title(wav_title)
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 2)    # (row, colum, num)
    librosa.display.specshow(librosa.amplitude_to_db(inst_power_spec, ref=np.max), y_axis='log', x_axis='time')
    wav_title = "inst_Power_spectrogram"
    plt.title(wav_title)
    plt.colorbar(format='%+2.0f dB')

    main_len = main_power_spec.shape[1]
    inst_len = inst_power_spec.shape[1]

    if main_len > inst_len:
        diff_len = inst_len
    else:
        diff_len = main_len

    print

    diff_power_spec = []
    for i in range(diff_len):
        diff = main_power_spec.T[i] - inst_power_spec.T[i]
        # print("diff.shape:", diff.shape)

        diff_power_spec.append(diff)

    diff_power_spec = np.array(diff_power_spec).T
    print("diff_power_spec.shape:", diff_power_spec.shape)


    plt.subplot(3, 1, 3)    # (row, colum, num)
    librosa.display.specshow(librosa.amplitude_to_db(diff_power_spec, ref=np.max), y_axis='log', x_axis='time')
    wav_title = "diff_Power_spectrogram"
    plt.title(wav_title)
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(wav_title + ".jpg")
    plt.clf()

    inv_wav = librosa.core.istft(main_power_spec)
    diff_wav = librosa.core.istft(diff_power_spec)
    librosa.output.write_wav('inv.wav', inv_wav, sr)
    librosa.output.write_wav('diff.wav', diff_wav, sr)


if __name__ == '__main__':
    main()
