# author: SaKuRa Pop
# data: 2021/7/6 20:52
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import time


class Encoder(nn.Module):

    def __init__(self, input_dim, enc_hid_dim, n_layers, dec_hid_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.n_layers = n_layers
        self.dec_hid_dim = dec_hid_dim
        self.lstm = nn.LSTM(input_dim, enc_hid_dim, n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2 * n_layers, dec_hid_dim * n_layers)
        self.dropout = nn.Dropout(dropout)
        self.n_direction = 2
        self.dec_hid_dim = dec_hid_dim

    def forward(self, input_sequence):
        # input_sequence = noisy_spectra = [batch_size, seq_len] = [1000, 1111]
        outputs, (last_hidden, last_cell) = self.lstm(input_sequence)
        # outputs is the outputs from top layer
        # last_hidden is the hidden state of the last time step from both two layers
        # last_cell is the cell state of the last time step from both two layer
        # hidden, cell = [batch, n_layers * n_directions, enc_hidden_dim] = [batch, 2*2, enc_hid_dim]
        # outputs = [batch, seq_len, enc_hid_dim * n_directions] = [1000, 1111, enc_hid_dim*2]

        last_hidden_reshape = torch.reshape(last_hidden, [-1, self.n_layers * self.enc_hid_dim * self.n_direction])
        last_cell_reshape = torch.reshape(last_cell, [-1, self.n_layers * self.enc_hid_dim * self.n_direction])
        # last_hidden_reshape = [batch, n_layer * n_direction * enc_hid_dim]
        # last_cell_reshape = [batch, n_layer * n_direction * enc_hid_dim]

        hidden = torch.tanh(self.fc(last_hidden_reshape))
        cell = torch.tanh(self.fc(last_cell_reshape))
        # hidden = [batch, dec_hid_dim * n_layers]
        # cell = [batch, dec_hid_dim * n_layers]
        hidden = hidden.permute(1, 0)
        hidden = hidden.reshape(self.n_layers, -1, self.dec_hid_dim)
        cell = cell.permute(1, 0)
        cell = cell.reshape(self.n_layers, -1, self.dec_hid_dim)
        # hidden, cell = [n_layers, batch, dec_hid_dim]
        # outputs = [batch, seq_len, enc_hid_dim * n_directions] = [batch, 1111, enc_hid_dim*2]
        return outputs, (hidden, cell)


class Attention(nn.Module):

    def __init__(self, enc_hid_dim, attention_dim, dec_hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.attention_dim = attention_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear((enc_hid_dim * 2) + (2 * dec_hid_dim), attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, hidden):
        # hidden, cell = [n_layers*n_directions, batch, dec_hid_dim]
        # encoder_outputs = [batch,seq_len,enc_hid_dim*n_directions] = [batch, 1111, enc_hid_dim*2]
        hidden_permute = hidden.permute(1, 0, 2)
        # hidden = [batch, n_layers, dec_hid_dim]
        hidden_reshape = hidden_permute.reshape(-1, 2 * self.dec_hid_dim)
        # hidden = [batch, n_layer * dec_hid_dim]
        seq_len = encoder_outputs.shape[1]
        hidden_reshape = hidden_reshape.unsqueeze(1)  # hidden = [batch, 1, n_layer * dec_hid_dim]
        hidden_reshape = hidden_reshape.repeat(1, seq_len, 1)  # hidden=[batch, seq_len, n_l * dec_h]

        energy = torch.tanh(self.attn(torch.cat((hidden_reshape, encoder_outputs), dim=2)))
        # energy = [batch, seq_len, attention_dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch, seq_len]
        weights = F.softmax(attention, dim=1)
        # weights = [batch, seq_len] = [batch, 1111]
        return weights


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, n_layers, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.lstm = nn.LSTM(2 * enc_hid_dim, dec_hid_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, encoder_outputs, hidden, cell):
        # hidden, cell = [n_layers*n_directions, batch, dec_hid_dim]= [2, batch, dec_hid_dim]
        # outputs = [batch, seq_len, enc_hid_dim * n_directions] = [batch, 1111, enc_hid_dim*2]
        weights = self.attention(encoder_outputs, hidden)
        # weights = [batch, seq_len] = [batch, 1111]
        weights = weights.unsqueeze(1)  # weights = [batch, 1, seq_len]

        context = torch.bmm(weights, encoder_outputs)
        # context = [batch, 1, enc_hid_dim *2]
        decoder_outputs, (hidden, cell) = self.lstm(context, (hidden, cell))
        # decoder_outputs = [1, batch, dec_hid_dim]
        # hidden, cell = [n_layers, batch, dec_hid_dim]
        prediction = F.relu(self.fc_out(decoder_outputs.squeeze(1)))
        # prediction = [batch, output_dim] = [batch, 1]
        return prediction, (hidden, cell)


class Bi_lstm_NN_filter(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, noisy_samples):
        # noisy_samples = [batch, seq_len] = [batch, 1111]
        # denoised_samples = [batch, seq_len] = [batch, 1111]
        # splited_noisy_spectra = [batch, 111]; batch = 1000*20
        seq_len = noisy_samples.shape[1]  # 111
        predictions = torch.zeros_like(noisy_samples)
        # predictions = [batch, seq_len] is a tensor to store decoder outputs

        encode_outputs, (hidden, cell) = self.encoder(noisy_samples)

        for time_step in range(seq_len):
            decoder_output, (hidden, cell) = self.decoder(encode_outputs, hidden, cell)
            predictions[:, time_step] = decoder_output
        # predictions = [batch, seq_len]
        return predictions


def Data_set(x, y, batch_size):
    """
    生成data_loader实例。可以定义batch_size
    :param input_data: 希望作为训练input的数据，tensor类型
    :param label_data: 希望作为训练label的数据，tensor类型
    :param batch_size: batch size
    :return: data_loader实例
    """
    data_set = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              shuffle=True,
                                              batch_size=batch_size,
                                              num_workers=0)
    return data_loader


def init_weights(model):
    for name, parameters in model.named_parameters():
        nn.init.xavier_uniform(parameters.data)


def training(model, data_loader, optimizer, criterion, epochs, device):
    model.train()
    model.to(device)
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for data_index, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            train_loss.append(float(loss))
            loss.backward()
            optimizer.step()
            # print(f"epoch:{epoch} [{data_index*len(batch_x)}/{len(data_loader.dataset)}"
            #       f"{100*data_index/len(data_loader):.2f}]\t"
            #       f"train loss: {loss.item()}")
            print("epoch: {} [{}/{} {:.2f}%] train loss: {}".format(epoch, data_index * len(batch_x),
                                                                    len(data_loader.dataset),
                                                                    100 * data_index / len(data_loader),
                                                                    loss.item())
                  )
    return train_loss


if __name__ == "__main__":
    input_dim = 1
    output_dim = 1
    enc_hid_dim = 512
    dec_hid_dim = 256
    attention_dim = 128
    dropout = 0.3
    num_layers = 2
    device = torch.device('cuda')  # 暂时用不到

    encoder = Encoder(input_dim, enc_hid_dim, num_layers, dec_hid_dim, dropout)
    attention_layer = Attention(enc_hid_dim, attention_dim, dec_hid_dim)
    decoder = Decoder(output_dim, enc_hid_dim, num_layers, dec_hid_dim, dropout, attention_layer)
    bi_direc_lstm = Bi_lstm_NN_filter(encoder, decoder, device)

    input_sample = torch.randn(10, 1111, 1)
    predictions = bi_direc_lstm(input_sample)
    print(predictions.shape)