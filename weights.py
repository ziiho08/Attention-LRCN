from utils import *
import torch.nn.functional as F
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument("--class_num", type=int, default=3, help='class number 2 or 3')
parser.add_argument('--temp', type=int, default=2, help='temperature')
parser.add_argument('--window_size', type=int, default=200, help='Window size for WKD')
parser.add_argument('--exp_param', type=int, default=7, help='Parameter for exponential function')
opt = parser.parse_args()

DEVICE = torch.device(f'cuda:{opt.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE)
print('Current cuda device : ', torch.cuda.current_device())

# Load data
DATA = np.load('./subject_merge.npy')

subjects = DATA[:,7681] # Mapping class 2 to 0 for binary classification
if opt.class_num == 2:
    for i in range(DATA.shape[0]):
        if DATA[i, 7680] == 2:
            DATA[i, 7680] = 0

net_lrcn = Net_LRCN().to(DEVICE)
net_att = Net_Attention().to(DEVICE)
net_fc = Net_FC(opt.class_num).to(DEVICE)

# Subject number for WESAD datset
# Use [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17] list for the entire training
subject_list = [2]

pred_list = []
label_list =[]
for subject_num in subject_list:
    # Weight initialization
    net_lrcn.apply(weight_reset)
    net_att.apply(weight_reset)
    net_fc.apply(weight_reset)

    # Load models
    lrcn_PATH = f'models/teacher/S{subject_num}_net_lrcn.pt'
    net_att_PATH = f'models/teacher/S{subject_num}_net_att.pt'
    net_fc_PATH = f'models/teacher/S{subject_num}_net_fc.pt'

    net_lrcn.load_state_dict(torch.load(lrcn_PATH))
    net_att.load_state_dict(torch.load(net_att_PATH))
    net_fc.load_state_dict(torch.load(net_fc_PATH))

    idx_train = np.where(subjects != subject_num)[0]
    dataset = Dataset(idx_train, DATA, DEVICE)
    train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)

    net_lrcn.eval()
    net_att.eval()
    net_fc.eval()

    batch_list = []
    with torch.no_grad():
        for itr, batch_data in enumerate(train_loader):
            inputs, inputs_attn, labels = batch_data

            feat = net_lrcn(inputs)
            attention = net_att(inputs_attn)
            tr_output = net_fc(feat, attention)

            argmax_label = torch.argmax(tr_output, dim=1)
            tr_argmax_np = argmax_label.detach().cpu().numpy()
            tr_label_np = labels.detach().cpu().numpy()

            pred_list = pred_list + list(argmax_label.detach().cpu().numpy())
            label_list = label_list + list(labels.detach().cpu().numpy())

            tr_softmax = F.softmax(tr_output/opt.temp, dim=1)
            tr_softmax_np = tr_softmax.detach().cpu().numpy()

            merge_np = np.hstack([tr_softmax_np, tr_argmax_np[:, np.newaxis], tr_label_np[:, np.newaxis]])
            batch_list.append(merge_np)

    batch_np = np.vstack(batch_list)
    np.save(f'./teacher_npy/S{subject_num}_merge.npy', batch_np)

# Calculating weights from uncertainty
for subject_num in subject_list:
    data = np.load(f'./S{subject_num}_merge.npy')
    w_size = int((opt.window_size-1)/2)
    preds = data[:, :opt.class_num]
    std = []
    weight_list = []
    for i in range(len(preds)):
        window=[]
        if i<w_size+1:
            window.append(preds[:w_size*2+1])
        else:
            window.append(preds[i-w_size:i+w_size])
        window_mean = np.mean(np.array(window),axis=1)
        squared = (window-window_mean) * (window-window_mean)
        window_var = np.mean(squared, axis=1)
        window_std = np.sqrt(window_var)
        std.append(np.sum(window_std))
    std_np = np.array(std)
    weight = np.exp(-std_np * opt.exp_param) # Weights for WKD
    merge_np = np.hstack([preds, std_np[:, np.newaxis], weight[:, np.newaxis]])
    np.save(f'./teacher_npy/S{subject_num}.npy', merge_np)
