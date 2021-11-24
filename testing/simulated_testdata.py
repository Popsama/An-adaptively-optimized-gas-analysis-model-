# author: SaKuRa Pop
# data: 2021/8/9 10:04
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from utils import Encoder, Attention, Decoder, Bi_lstm_NN_filter, Data_set, init_weights, training
from Bi_directional_LSTM_model import Bi_directional_LSTM
from bp_kalman_filter import BpFilter, array_to_tensor, plain_kalman
from scipy.signal import savgol_filter
import numpy.fft as fft


def compute_snr(pure_signal, noisy_signal):
    signal_to_noise_ratio = 10 * (np.log10(np.std(pure_signal)/np.std(noisy_signal-pure_signal)))
    return signal_to_noise_ratio


def fourier_transform(sample, mode="p"):
    if mode == "p":
        N = len(sample)  # 1111
        sample_ft = fft.fft(sample)  # 信号的傅里叶变换 (复数)
        frequencies = fft.fftfreq(sample_ft.size, 1/N)  # 傅里叶变换后的每种频率分量的频率值
        amplitude = np.abs(sample_ft)  # 傅里叶变换的振幅
        p = 20*np.log10(amplitude)
        frequency = frequencies[frequencies > 0]
        p = p[frequencies > 0]
        return p, frequency
    elif mode == "a":
        N = len(sample)  # 1111
        sample_ft = fft.fft(sample)  # 信号的傅里叶变换 (复数)
        frequencies = fft.fftfreq(sample_ft.size, 1 / N)  # 傅里叶变换后的每种频率分量的频率值
        amplitude = np.abs(sample_ft)  # 傅里叶变换的振幅
        frequency = frequencies[frequencies > 0]
        amplitude = amplitude[frequencies > 0]
        return amplitude, frequency


class SG_filter():

    def __init__(self):
        self.window_length = 11
        self.polynomial = 2

    def predict(self, data):
        prediction = savgol_filter(data, self.window_length, self.polynomial)
        return prediction


class Bp_Kalman_filter():

    def __init__(self):
        self.Gpu = torch.device("cuda")
        self.model = BpFilter().to(self.Gpu)
        model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\深度学习滤波器\滤波模型\透射谱的滤波模型\BP_KF.pt"
        self.model.load_state_dict(torch.load(model_save_path))

    def predict(self, index, CH4_noisy_spectral, CH4_no_noise_spectral):
        data = plain_kalman(index, CH4_noisy_spectral, CH4_no_noise_spectral)
        data = array_to_tensor(data)
        data = data.reshape(1111, 1)
        prediction = self.model(data)
        prediction = prediction.cpu().detach().numpy()
        return prediction


class Multi_averaging_filter():
    def __init__(self):
        None

    def predict(self, spectrum, kernel_size, mode="full"):
        result = np.convolve(spectrum, np.ones((kernel_size,))/kernel_size, mode=mode)
        return result


if __name__ == "__main__":
    """模拟数据 （透射谱）"""
    no_noise_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(1)训练数据\模拟数据\投射信号与透射光谱\de_noised_spectra.npy"
    denoised_spectra = np.load(no_noise_path)  # (1000, 1111) 透射谱（吸收谱）
    print(denoised_spectra.shape)
    noisy_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(1)训练数据\模拟数据\投射信号与透射光谱\noisy_spectra.npy"
    noisy_spectra = np.load(noisy_path)  # (1000, 1111) 透射谱（吸收谱）
    print(noisy_spectra.shape)

    index = 900
    noisy_sample = noisy_spectra[index]  # 选择600ppm作为测试数据
    denoised_ground_truth = denoised_spectra[index]  # 选择600ppm无噪声信号作为真值

    ####################################################################################################################
    # Bi-directional-LSTM
    bi_direction_lstm = Bi_directional_LSTM()
    filtering_result = bi_direction_lstm.predict(noisy_sample)

    plt.figure()
    plt.plot(noisy_sample, label="original data")
    snr = compute_snr(denoised_ground_truth, filtering_result)
    plt.plot(filtering_result, linewidth=2,
             label="filtered data, SNR={}".format(snr))
    original_snr = compute_snr(denoised_ground_truth, noisy_sample)
    plt.plot(denoised_spectra[index], linewidth=2,
             label="original signal, SNR={}".format(original_snr))
    plt.legend()

    ####################################################################################################################

    # BP-KF filtering
    Bp_kalman_filter = Bp_Kalman_filter()
    # index = 900
    bp_filtering_result = Bp_kalman_filter.predict(index, noisy_spectra, denoised_spectra)

    plt.figure()
    plt.plot(noisy_sample, label="original data")
    snr = compute_snr(denoised_ground_truth, bp_filtering_result)
    plt.plot(bp_filtering_result, linewidth=2,
             label="filtered data, SNR={}".format(snr))
    original_snr = compute_snr(denoised_ground_truth, noisy_sample)
    plt.plot(denoised_spectra[index], linewidth=2,
             label="original signal, SNR={}".format(original_snr))
    plt.legend()

    ####################################################################################################################

    # Plain Kalman filtering
    """plain KF滤波1次"""
    """对所有的噪声信号进行KF滤波，滤波后的信号可以作为BP神经网络的输入，而无噪声的信号即为label（ground truth）"""
    kalman_filtering_result = plain_kalman(index, noisy_spectra, denoised_spectra)
    print(kalman_filtering_result.shape)  # (1111,)

    plt.figure()
    plt.plot(noisy_sample, label="original data")
    snr = compute_snr(denoised_ground_truth, kalman_filtering_result)
    plt.plot(kalman_filtering_result, linewidth=2,
             label="filtered data, SNR={}".format(snr))
    original_snr = compute_snr(denoised_ground_truth, noisy_sample)
    plt.plot(denoised_spectra[index], linewidth=2,
             label="original signal, SNR={}".format(original_snr))
    plt.legend()

    ####################################################################################################################

    # S-G filtering
    S_G_filter = SG_filter()
    sg_result = S_G_filter.predict(noisy_sample)

    plt.figure()
    plt.plot(noisy_sample, label="original data")
    snr = compute_snr(denoised_ground_truth, sg_result)
    plt.plot(sg_result, linewidth=2,
             label="filtered data, SNR={}".format(snr))
    original_snr = compute_snr(denoised_ground_truth, noisy_sample)
    plt.plot(denoised_spectra[index], linewidth=2,
             label="original signal, SNR={}".format(original_snr))
    plt.legend()

    ####################################################################################################################

    # Moving average filtering
    multi_averaging_filter = Multi_averaging_filter()
    maf_result = multi_averaging_filter.predict(noisy_sample, 20, mode="same")
    # maf_result2 = multi_averaging_filter.predict(noisy_sample, 50, mode="full")
    # maf_result3 = multi_averaging_filter.predict(noisy_sample, 50, mode="valid")

    plt.figure()
    plt.ylim(0.93, 1.03)
    plt.plot(noisy_sample, label="original data")
    snr = compute_snr(denoised_ground_truth, maf_result)
    plt.plot(maf_result, linewidth=2,
             label="filtered data, SNR={}".format(snr))
    original_snr = compute_snr(denoised_ground_truth, noisy_sample)
    plt.plot(denoised_spectra[index], linewidth=2,
             label="original signal, SNR={}".format(original_snr))
    plt.legend()
    plt.show()

    ####################################################################################################################
    ####################################################################################################################

    plt.subplot(6, 1, 1)
    plt.xlim(38, 1070)
    plt.ylim(0.935, 1.025)
    plt.plot(noisy_sample, label="origin+noise")
    plt.plot(denoised_spectra[index], linewidth=2, label="original signal")
    plt.legend()
    path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\原始噪声谱.npy"
    np.save(path1, noisy_sample)
    path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\原始无噪声谱.npy"
    np.save(path2, denoised_spectra[index])
    path3 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\原始噪声谱.txt"
    np.savetxt(path3, noisy_sample)
    path4 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\原始无噪声谱.txt"
    np.savetxt(path4, denoised_spectra[index])

    plt.subplot(6, 1, 2)
    plt.xlim(38, 1070)
    plt.ylim(0.935, 1.025)
    plt.plot(noisy_sample, label="noisy signal")
    plt.plot(sg_result, linewidth=2, label="S-G denoised signal")
    plt.legend()
    path5 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\SG滤波结果.npy"
    np.save(path5, sg_result)
    path6 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\SG滤波结果.txt"
    np.savetxt(path6, sg_result)


    plt.subplot(6, 1, 3)
    plt.xlim(38, 1070)
    plt.ylim(0.935, 1.025)
    plt.plot(noisy_sample, label="noisy signal")
    plt.plot(maf_result, linewidth=2, label="MAF denoised signal")
    plt.legend()
    path7 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\MAF滤波结果.npy"
    np.save(path7, maf_result)
    path8 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\MAF滤波结果.txt"
    np.savetxt(path8, maf_result)

    plt.subplot(6, 1, 4)
    plt.xlim(38, 1070)
    plt.ylim(0.935, 1.025)
    plt.plot(noisy_sample, label="noisy signal")
    plt.plot(kalman_filtering_result, linewidth=2, label="Plain KF denoised signal")
    plt.legend()
    path9 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\KF滤波结果.npy"
    np.save(path9, kalman_filtering_result)
    path10 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\KF滤波结果.txt"
    np.savetxt(path10, kalman_filtering_result)

    plt.subplot(6, 1, 5)
    plt.xlim(38, 1070)
    plt.ylim(0.935, 1.025)
    plt.plot(noisy_sample, label="noisy signal")
    plt.plot(bp_filtering_result, linewidth=2, label="BP-KF denoised signal")
    plt.legend()
    path11 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\BP-KF滤波结果.npy"
    np.save(path11, bp_filtering_result)
    path12 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\BP-KF滤波结果.txt"
    np.savetxt(path12, bp_filtering_result)

    plt.subplot(6, 1, 6)
    plt.xlim(38, 1070)
    plt.ylim(0.935, 1.025)
    plt.plot(noisy_sample, label="noisy signal")
    plt.plot(filtering_result, linewidth=2, label="Bi-directional-LSTM denoised signal")
    plt.legend()
    path13 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\LSTM滤波结果.npy"
    np.save(path13, filtering_result)
    path14 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试\LSTM滤波结果.txt"
    np.savetxt(path14, filtering_result)

    plt.show()