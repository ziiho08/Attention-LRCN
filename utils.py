import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import torch.nn.functional as F

# Loss function for knowledge distillation
def WKD_loss(logits, labels, teacher_softmax, batch_W, temp):
    student_softmax = F.log_softmax(logits / temp, dim=1)
    loss_pointwise = teacher_softmax * (torch.log(teacher_softmax) - student_softmax) * batch_W
    distillation_loss = (loss_pointwise.sum() / student_softmax.shape[0]) * (2*temp*temp)
    student_loss = F.cross_entropy(input=logits, target=labels)
    total_loss = 0.1*student_loss + 0.9*distillation_loss
    return total_loss

# Dataset for Weight Knowledge Distillation (WKD)
class WKD_Dataset(Dataset):
    def __init__(self, idx_train, teacher, data, device, class_num):
        ppg_windows = data[idx_train, :7680]
        target = data[idx_train, 7680]
        # Teachers' predictions and calculated weights from teachers' uncertainty
        teacher_data = teacher[:, :class_num]
        weight = teacher[:, class_num + 1:]

        # Spectrogram
        WIN_LEN = 64 #Window length
        OVERLAP = 16
        CUT_OFF_FREQ = 8 #HZ
        spctrogram_list = []
        for i in range(ppg_windows.shape[0]):
            x = ppg_windows[i, :]
            x_stft = librosa.stft(x, n_fft=WIN_LEN, hop_length=OVERLAP, win_length=WIN_LEN)
            magnitude = np.abs(x_stft)
            log_spectrogram = librosa.amplitude_to_db(magnitude)
            spctrogram_list.append(log_spectrogram)
        spectrogram_np = np.array(spctrogram_list)

        X_train = torch.tensor(spectrogram_np[:, np.newaxis, :CUT_OFF_FREQ + 1, :]).float()
        ATT_train = torch.tensor(spectrogram_np[:, np.newaxis, :, :]).float()
        T_tensor = torch.tensor(teacher_data).float()
        weight_tensor = torch.tensor(weight).float()
        y_train = torch.tensor(target.astype(np.int64))

        self.ppg = ppg_windows
        self.x_data = X_train.to(device)
        self.att_data = ATT_train.to(device)
        self.y_data = y_train.to(device)
        self.teacher_data = T_tensor.to(device)
        self.weight_tensor = weight_tensor.to(device)

    def __getitem__(self, index):
        return self.x_data[index], self.att_data[index], self.y_data[index], self.teacher_data[index], \
               self.weight_tensor[index]
    def __len__(self):
        return self.x_data.shape[0]

# Dataset for Attention-LRCN (baseline)
class Dataset(Dataset):
    def __init__(self, idx_train, data, device):

        ppg_windows = data[idx_train, :7680]
        target = data[idx_train, 7680]

        # Spectrogram
        WIN_LEN = 64 # Window length
        OVERLAP = 16
        CUT_OFF_FREQ = 8 # HZ
        spctrogram_list = []
        for i in range(ppg_windows.shape[0]):
            x = ppg_windows[i, :]
            x_stft = librosa.stft(x, n_fft=WIN_LEN, hop_length=OVERLAP, win_length=WIN_LEN)
            magnitude = np.abs(x_stft)
            log_spectrogram = librosa.amplitude_to_db(magnitude)
            spctrogram_list.append(log_spectrogram)
        spectrogram_np = np.array(spctrogram_list)
        X_train = torch.tensor(spectrogram_np[:, np.newaxis, :CUT_OFF_FREQ + 1, :]).float()
        ATT_train = torch.tensor(spectrogram_np[:, np.newaxis, :, :]).float()
        y_train = torch.tensor(target.astype(np.int64))

        self.ppg = ppg_windows
        self.x_data = X_train.to(device)
        self.att_data = ATT_train.to(device)
        self.y_data = y_train.to(device)

    def __getitem__(self, index):
        return self.x_data[index], self.att_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]

class Net_Attention(nn.Module):
    def __init__(self):
        super(Net_Attention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (9, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 9), padding=(0, 4))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (9, 1), padding=(4, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (9, 1), padding=(4, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, (9, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (9, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 1, (9, 1)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((1, 2), (1, 2))

    def forward(self, x):
        out = self.conv1(x)  # [8, 16, 16, 240]
        out = self.max_pool(self.relu(out))

        out = out + self.conv2(out)  # [8, 32, 8, 120]
        out = self.max_pool(self.relu(out))

        out = out + self.conv3(out)  # [8, 32, 8, 120]
        out = self.max_pool(self.relu(out))

        out = self.conv4(out)  # [8, 1, 1, 60]
        out = torch.squeeze(out, 1)

        return out

class Net_LRCN(nn.Module):
    def __init__(self):
        super(Net_LRCN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (9, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 9), padding=(0, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 9), padding=(0, 4))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 9), padding=(0, 4)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 9), padding=(0, 4)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(64)
        )
        self.lstm1 = nn.LSTM(64, 64, 2, bidirectional=True)  # (Hin, Hout, num_layers)
        #self.lstm2 = nn.LSTM(64, 64, 2, bidirectional=True)

        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((1, 2), (1, 2))

    def forward(self, x):
        hidden1 = torch.zeros(4, x.shape[0], 64, requires_grad=True).to(x.device)
        cell1 = torch.zeros(4, x.shape[0], 64, requires_grad=True).to(x.device)

        out = self.conv1(x)  # [8, 32, 1, 240]
        out = self.max_pool(self.relu(out))

        out = out + self.conv2(out)  #
        out = self.max_pool(self.relu(out))

        out = out + self.conv3(out)  # [8, 64, 1, 60]
        out = self.max_pool(self.relu(out))

        out = torch.squeeze(out, 2)  # [8, 64, 60]
        out = torch.transpose(out, 0, 2)
        out = torch.transpose(out, 1, 2)  # [60, -1, 64]

        out, hidden1 = self.lstm1(out, (hidden1, cell1))

        out = torch.transpose(out, 0, 1)
        out = torch.transpose(out, 1, 2)  # [8, 64, 60]

        out = self.bn(out)

        return out

class Net_FC(nn.Module):
    def __init__(self, CLASS_NUM):
        super(Net_FC, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, CLASS_NUM)
        )

    def forward(self, feat, attention):
        feat_attn = torch.mul(feat, attention)  # [8, 64, 60]
        out = torch.sum(feat_attn, 2)  # [8, 64]
        out = self.fc(out)  # [8, 2]

        return out

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

