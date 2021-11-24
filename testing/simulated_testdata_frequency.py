# author: SaKuRa Pop
# data: 2021/8/9 11:17
from Bi_directional_LSTM_model import Bi_directional_LSTM
from bp_kalman_filter import BpFilter, array_to_tensor, plain_kalman
from scipy.signal import savgol_filter
import numpy.fft as fft
import torch
import numpy as np
import matplotlib.pyplot as plt
from 模拟数据的测试结果 import compute_snr, fourier_transform, SG_filter, Bp_Kalman_filter, Multi_averaging_filter

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
    ####################################################################################################################
    # BP-KF filtering
    Bp_kalman_filter = Bp_Kalman_filter()
    # index = 900
    bp_filtering_result = Bp_kalman_filter.predict(index, noisy_spectra, denoised_spectra)
    ####################################################################################################################
    # Plain Kalman filtering
    """plain KF滤波1次"""
    """对所有的噪声信号进行KF滤波，滤波后的信号可以作为BP神经网络的输入，而无噪声的信号即为label（ground truth）"""
    kalman_filtering_result = plain_kalman(index, noisy_spectra, denoised_spectra)
    print(kalman_filtering_result.shape)  # (1111,)
    ####################################################################################################################
    # S-G filtering
    S_G_filter = SG_filter()
    sg_result = S_G_filter.predict(noisy_sample)
    ####################################################################################################################
    # Moving average filtering
    multi_averaging_filter = Multi_averaging_filter()
    maf_result = multi_averaging_filter.predict(noisy_sample, 20, mode="same")

    ####################################################################################################################
    p = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器"\
        r"\(3)Testing\模拟数据的测试\DOSC.csv"
    dosc_result = np.loadtxt(p)  #
    print(dosc_result.shape)
    plt.figure()
    plt.plot(dosc_result)
    plt.show()
    ####################################################################################################################

    sample_amplitude1, frequencies1 = fourier_transform(filtering_result, mode="a")
    sample_amplitude2, frequencies2 = fourier_transform(sg_result, mode="a")
    sample_amplitude3, frequencies3 = fourier_transform(maf_result, mode="a")
    sample_amplitude4, frequencies4 = fourier_transform(kalman_filtering_result, mode="a")
    sample_amplitude5, frequencies5 = fourier_transform(bp_filtering_result.squeeze(), mode="a")
    sample_amplitude6, frequencies6 = fourier_transform(denoised_ground_truth, mode="a")
    sample_amplitude7, frequencies7 = fourier_transform(noisy_sample, mode="a")
    sample_amplitude8, frequencies8 = fourier_transform(dosc_result, mode="a")


    path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(3)Testing\模拟数据的测试"
    np.save(path + r"\origin+noise_FFT.npy", sample_amplitude7)
    np.savetxt(path + r"\origin+noise_FFT.txt", sample_amplitude7)
    np.save(path + r"\origin_FFT.npy", sample_amplitude6)
    np.savetxt(path + r"\origin_FFT.txt", sample_amplitude6)
    np.save(path + r"\SG_FFT.npy", sample_amplitude2)
    np.savetxt(path + r"\SG_FFT.txt", sample_amplitude2)
    np.save(path + r"\MAF_FFT.npy", sample_amplitude3)
    np.savetxt(path + r"\MAF_FFT.txt", sample_amplitude3)
    np.save(path + r"\KF_FFT.npy", sample_amplitude4)
    np.savetxt(path + r"\KF_FFT.txt", sample_amplitude4)
    np.save(path + r"\BP_KF_FFT.npy", sample_amplitude5)
    np.savetxt(path + r"\BP_KF_FFT.txt", sample_amplitude5)
    np.save(path + r"\Bi_LSTM_FFT.npy", sample_amplitude1)
    np.savetxt(path + r"\Bi_LSTM_FFT.txt", sample_amplitude1)
    np.savetxt(path + r"\DOSC_FFT.txt", sample_amplitude8)

    residual1 = sample_amplitude7 - sample_amplitude6
    residual2 = sample_amplitude2 - sample_amplitude6
    residual3 = sample_amplitude3 - sample_amplitude6
    residual4 = sample_amplitude4 - sample_amplitude6
    residual5 = sample_amplitude5 - sample_amplitude6
    residual6 = sample_amplitude1 - sample_amplitude6
    residual7 = sample_amplitude8 - sample_amplitude6

    np.save(path + r"\noise_residual.npy", residual1)
    np.savetxt(path + r"\noise_residual.txt", residual1)
    np.save(path + r"\S-G_residual.npy", residual2)
    np.savetxt(path + r"\S-G_residual.txt", residual2)
    np.save(path + r"\MAF_residual.npy", residual3)
    np.savetxt(path + r"\MAF_residual.txt", residual3)
    np.save(path + r"\KF_residual.npy", residual4)
    np.savetxt(path + r"\KF_residual.txt", residual4)
    np.save(path + r"\BP-KF_residual.npy", residual5)
    np.savetxt(path + r"\BP-KF_residual.txt", residual5)
    np.save(path + r"\Bi_LSTM_residual.npy", residual6)
    np.savetxt(path + r"\Bi_LSTM_residual.txt", residual6)
    np.savetxt(path + r"\DOSC_residual.txt", residual7)

    plt.subplot(6, 1, 1)
    plt.xlim(0, 550)
    # plt.yscale("log")
    plt.plot(frequencies1, np.abs(residual1), label="noisy signal")
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.xlim(0, 550)
    # plt.yscale("log")
    plt.plot(frequencies1, np.abs(residual2), label="S-G")
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.xlim(0, 550)
    # plt.yscale("log")
    plt.plot(frequencies1, np.abs(residual7), label="DOSC")
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.xlim(0, 550)
    # plt.yscale("log")
    plt.plot(frequencies1, np.abs(residual4), label="KF")
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.xlim(0, 550)
    # plt.yscale("log")
    plt.plot(frequencies1, np.abs(residual5), label="BP-KF")
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.xlim(0, 550)
    # plt.yscale("log")
    plt.plot(frequencies1, np.abs(residual6), label="Bi-LSTM")
    plt.legend()

    plt.show()