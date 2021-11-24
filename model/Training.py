# author: SaKuRa Pop
# data: 2021/7/30 9:09
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

if __name__ == "__main__":

    # 一些超参数
    input_dim = 1
    output_dim = 1
    enc_hid_dim = 512
    dec_hid_dim = 256
    attention_dim = 128
    dropout = 0.3
    num_layers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 暂时用不到

    # 根据超参数定意好模型encoder attention layer decoder
    encoder = Encoder(input_dim, enc_hid_dim, num_layers, dec_hid_dim, dropout)
    attention_layer = Attention(enc_hid_dim, attention_dim, dec_hid_dim)
    decoder = Decoder(output_dim, enc_hid_dim, num_layers, dec_hid_dim, dropout, attention_layer)

    # 生成模型实例
    bi_direc_lstm = Bi_lstm_NN_filter(encoder, decoder, device)

    # 初始化weights
    # bi_direc_lstm.apply(init_weights)

    optimizer = optim.Adam(bi_direc_lstm.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # 导入数据

    path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(1)训练数据\模拟数据\投射信号与透射光谱\splited_noisy_spectra.npy"
    path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(1)训练数据\模拟数据\投射信号与透射光谱\splited_de_noised_spectra.npy"

    noisy_spectra = np.load(path1)
    de_noised_spectra = np.load(path2)

    print(noisy_spectra.shape)
    print(de_noised_spectra.shape)

    input_data = torch.from_numpy(noisy_spectra).type(torch.cuda.FloatTensor).unsqueeze(2)
    input_label = torch.from_numpy(de_noised_spectra).type(torch.cuda.FloatTensor).unsqueeze(2)

    print(input_data.shape)
    print(input_label.shape)

    simulated_dataset = Data_set(input_data, input_label, batch_size=111)

    # 可视化一下
    index = 3850
    plt.figure()
    plt.plot(noisy_spectra[index])
    plt.plot(de_noised_spectra[index])
    plt.show()

    # 开始训练
    begin_time = time.time()
    train_loss = training(bi_direc_lstm, simulated_dataset, optimizer, criterion, epochs=50, device=device)
    end_time = time.time()
    total_time_cost = (end_time - begin_time) / 60
    print("总训练用时：{} 分钟".format(total_time_cost))
    # 训练结束

    """保存模型"""
    model_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\带有注意力机制的LSTM甲烷浓度探测器\(2)Architecture\model\model2.pt"
    torch.save(bi_direc_lstm.state_dict(), model_save_path)


    # 可视化误差
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(train_loss, label="train loss")

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label="train loss")
    ax = plt.gca()
    ax.set_yscale("log")
    plt.show()


    # # 可视化结果
    # sample = noisy_spectra[999]
    # ground_truth = de_noised_spectra[999].squeeze()
    # sample = torch.from_numpy(sample).type(torch.float32).unsqueeze(0).unsqueeze(2)
    # result = bi_direc_lstm(sample)
    # result = result.squeeze().detach().numpy()  # array [1111]
    # sample = sample.squeeze().detach().numpy()  # array [1111]
    #
    # plt.figure()
    # plt.plot(sample, alpha=0.5, label="noisy")
    # plt.plot(result, label="denoised")
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(result, label="nnf prediction")
    # plt.plot(ground_truth, label="ground_truth")
    # plt.legend()
    # plt.show()
    #
    # sample2 = input_data[9]
    # ground_truth2 = input_label[9]
    # result2 = bi_direc_lstm(sample2.unsqueeze(0))
    # result2 = result2.squeeze().detach().numpy()  # array [1111]
    # sample2 = sample2.squeeze().detach().numpy()  # array [1111]
    # ground_truth2 = ground_truth2.squeeze().detach().numpy()
    #
    # plt.figure()
    # plt.plot(sample2, alpha=0.5, label="noisy")
    # plt.plot(result2, label="denoised")
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(result2, label="nnf prediction")
    # plt.plot(ground_truth2, label="ground_truth")
    # plt.legend()
    # plt.show()