import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from librosa import display

"""
np.fftの戻り値は
サンプル数が1024なら1024で返ってくる。
しかし後ろ半分（512以降）は意味のないデータ
→真ん中がナイキスト周波数（？）
"""

N = 1024            # サンプル数
dt = 1/(N+1)          # サンプリング周期 [s]
f1, f2, f3 = 32, 128, 256 # 周波数 [Hz]

# 時間 [s] 
t = np.arange(0, N*dt, dt)

# 周波数軸
hz = np.linspace(0, 1/dt, N)

# 信号の作成(周波数32[Hz], 振幅3の正弦波
#           周波数128[Hz], 振幅0.3の正弦波
#           周波数256[Hz], 振幅0.2の正弦波)
# sin(t) = Asin(2pi * f0 * t)
y_original = np.sin(2*np.pi*f1*t) \
    + np.sin(2*np.pi*f2*t) \
    + np.sin(2*np.pi*f3*t)

# フィルタ適用前の振幅スペクトル
F2 = np.fft.fft(y_original)
Amp2 = np.abs(F2/(N/2))

# バンドパスフィルタの設計
band_pass_filter = signal.firwin(numtaps=51, cutoff=[50, 150], 
                                fs=1/dt, pass_zero=False)

# フィルタを信号に適用する
y_filtered = signal.lfilter(band_pass_filter, 1, y_original)

# フィルタ適用後の振幅スペクトル
F3 = np.fft.fft(y_filtered)
Amp3 = np.abs(F3/(N/2))

plt.figure(figsize=(16, 9))

# 波形データの表示
plt.subplot(2, 1, 1)
plt.plot(t, y_original, color="red")
plt.plot(t, y_filtered, color="blue")

# 振幅スペクトルの表示
plt.subplot(2, 1, 2)
plt.plot(hz, Amp2, color="red")
plt.plot(hz, Amp3, color="blue")

plt.show()
plt.clf()

plt.figure(figsize=(16, 9))
plt.subplot(2, 1, 1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max)
librosa.display.specshow(D, sr=1/dt, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('original')

plt.subplot(2, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y_filtered)), ref=np.max)
librosa.display.specshow(D, sr=1/dt, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('filtered')

plt.show()
plt.clf()