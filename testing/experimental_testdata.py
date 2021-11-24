# author: SaKuRa Pop
# data: 2021/8/9 17:47
from Bi_directional_LSTM_model import Bi_directional_LSTM
from bp_kalman_filter import BpFilter, array_to_tensor, plain_kalman
from scipy.signal import savgol_filter
import numpy.fft as fft
import torch
import numpy as np
import matplotlib.pyplot as plt
from 模拟数据的测试结果 import compute_snr, fourier_transform, SG_filter, Bp_Kalman_filter, Multi_averaging_filter

data_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(1)训练数据\实验数据\投射信号与透射光谱\original_spectra.npy"
noisy_spectra = np.load(data_path)  # = [100, 1111]
print(noisy_spectra.shape)

path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(1)训练数据\模拟数据\投射信号与透射光谱\de_noised_spectra.npy"
denoised_spectra = np.load(path2)[::10]  # (100, 1111) 透射谱（吸收谱）
print(denoised_spectra.shape)

bi_direction_lstm = Bi_directional_LSTM()
Bp_kalman_filter = Bp_Kalman_filter()
S_G_filter = SG_filter()
multi_averaging_filter = Multi_averaging_filter()

index = 60
noisy_sample = noisy_spectra[index]  # 选择600ppm作为测试数据

sg_result = S_G_filter.predict(noisy_sample)
maf_result = multi_averaging_filter.predict(noisy_sample, 20, mode="same")
kf_result = plain_kalman(index, noisy_spectra, denoised_spectra)
bp_kf_result = Bp_kalman_filter.predict(index, noisy_spectra, denoised_spectra)
bi_LSTM_result = bi_direction_lstm.predict(noisy_sample)

save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\实验数据的测试"
np.savetxt(save_path + r"\sg_result.txt", sg_result)
np.savetxt(save_path + r"\maf_result.txt", maf_result)
np.savetxt(save_path + r"\kf_result.txt", kf_result)
np.savetxt(save_path + r"\bp_kf_result.txt", bp_kf_result)
np.savetxt(save_path + r"\Bi_lstm_result.txt", bi_LSTM_result)
np.savetxt(save_path + r"\noisy_sample.txt", noisy_sample)

plt.subplot(5, 1, 1)
plt.xlim(38, 1070)
plt.ylim(0.95, 1.025)
plt.plot(noisy_sample, label="experimental noisy spectra")
plt.plot(sg_result, linewidth=2, label="S-G")
plt.legend()

plt.subplot(5, 1, 2)
plt.xlim(38, 1070)
plt.ylim(0.95, 1.025)
plt.plot(noisy_sample, label="experimental noisy spectra")
plt.plot(maf_result, linewidth=2, label="MAF")
plt.legend()

plt.subplot(5, 1, 3)
plt.xlim(38, 1070)
plt.ylim(0.95, 1.025)
plt.plot(noisy_sample, label="experimental noisy spectra")
plt.plot(kf_result, linewidth=2, label="KF")
plt.legend()

plt.subplot(5, 1, 4)
plt.xlim(38, 1070)
plt.ylim(0.95, 1.025)
plt.plot(noisy_sample, label="experimental noisy spectra")
plt.plot(bp_kf_result, linewidth=2, label="BP-KF")
plt.legend()

plt.subplot(5, 1, 5)
plt.xlim(38, 1070)
plt.ylim(0.95, 1.025)
plt.plot(noisy_sample, label="experimental noisy spectra")
plt.plot(bi_LSTM_result, linewidth=2, label="Bi-LSTM")
plt.legend()

plt.show()

