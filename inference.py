from utils import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as acc
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument("--class_num", type=int, default=3, help='class number 2 or 3')
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
    lrcn_PATH = f'./models/student/S{subject_num}_net_lrcn.pt'
    net_att_PATH = f'./models/student/S{subject_num}_net_att.pt'
    net_fc_PATH = f'./models/student/S{subject_num}_net_fc.pt'

    net_lrcn.load_state_dict(torch.load(lrcn_PATH)) # If you have less than 2 GPUs, please add -> map_location='cuda:0'
    net_att.load_state_dict(torch.load(net_att_PATH))
    net_fc.load_state_dict(torch.load(net_fc_PATH))

    idx_test = np.where(subjects == subject_num)[0]
    test_data = Dataset(idx_test, DATA, DEVICE)
    test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=False)

    net_lrcn.eval()
    net_att.eval()
    net_fc.eval()

    batch_list = []
    with torch.no_grad():
        for itr, batch_data in enumerate(test_loader):
            inputs, inputs_attn, labels = batch_data

            feat = net_lrcn(inputs)
            attention = net_att(inputs_attn)
            val_output = net_fc(feat, attention)

            pred = torch.argmax(val_output, dim=1)

            pred_list = pred_list + list(pred.detach().cpu().numpy())
            label_list = label_list + list(labels.detach().cpu().numpy())

        pred_np = np.array(pred_list)
        label_np = np.array(label_list)

        accuracy = acc(pred_np, label_np)
        F1 = f1_score(label_np, pred_np, average='macro')

        print("[S" + str(subject_num) + "]", "Acc: %.4f" % (accuracy * 100),"F1-score: %.4f" % (F1 * 100))

