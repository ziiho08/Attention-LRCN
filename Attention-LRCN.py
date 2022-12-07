import os
import copy
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as acc
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--class_num", type=int, default=3, help='class number 2 or 3')
parser.add_argument("--BATCH_SIZE", type=int, default=256)
parser.add_argument("--LEARNING_RATE", type=float, default=0.0001)
parser.add_argument("--EPOCH", type=int, default=20)
parser.add_argument('--GPU_NUM', type=int, default=0)
opt = parser.parse_args()
print(opt)

DEVICE = torch.device(f'cuda:{opt.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE)
print('Current cuda device : ', torch.cuda.current_device())

DATA_PATH = './subject_merge.npy'
DATA = np.load(DATA_PATH)
subjects = DATA[:,7681]

if opt.class_num == 2: # Mapping class 2 to 0 for binary classification
    for i in range(DATA.shape[0]):
        if DATA[i, 7680] == 2:
            DATA[i, 7680] = 0

# Network configuration
net_lrcn = Net_LRCN().to(DEVICE)
net_att = Net_Attention().to(DEVICE)
net_fc = Net_FC(opt.class_num).to(DEVICE)

best_model_wts_lrcn = copy.deepcopy(net_lrcn.state_dict())
best_model_wts_att = copy.deepcopy(net_att.state_dict())
best_model_wts_fc = copy.deepcopy(net_fc.state_dict())

loss_func = nn.CrossEntropyLoss()
params = list(net_lrcn.parameters())+list(net_att.parameters())+list(net_fc.parameters())

# Subject number for WESAD datset
# Use [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17] list for the entire training
subject_list = [2]
for subject_num in subject_list:
    idx_train = np.where(subjects != subject_num)[0]
    idx_test = np.where(subjects == subject_num)[0]

    train_data = Dataset(idx_train, DATA, DEVICE)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.BATCH_SIZE, shuffle=True)
    test_data = Dataset(idx_test, DATA, DEVICE)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(params, lr=opt.LEARNING_RATE)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    # Weight initialization
    net_lrcn.apply(weight_reset)
    net_att.apply(weight_reset)
    net_fc.apply(weight_reset)
    best_acc, best_fc = 0.0, 0.0
    for epoch in range(opt.EPOCH):
        # TRAIN
        net_lrcn.train()
        net_att.train()
        net_fc.train()

        tr_loss, tr_output = 0.0, 0.0
        for itr, batch_data in enumerate(train_loader):
            inputs, inputs_attn, labels = batch_data
            if inputs.shape[0] == 1: continue # for batch normalization
            feat = net_lrcn(inputs)
            attention = net_att(inputs_attn)
            tr_output = net_fc(feat, attention)
            loss = loss_func(tr_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        # TEST
        if (epoch+1) % 1 == 0:
            net_lrcn.eval()
            net_att.eval()
            net_fc.eval()
            pred_list, label_list = [],[]
            attn_list = []
            for itr, test_data in enumerate(test_loader):
                inputs, inputs_attn, labels = test_data
                feat = net_lrcn(inputs)
                attention = net_att(inputs_attn)
                val_output = net_fc(feat, attention)

                pred = torch.argmax(val_output, dim=1)

                pred_list = pred_list + list(pred.detach().cpu().numpy())
                label_list = label_list + list(labels.detach().cpu().numpy())

                attn_mean = torch.mean(attention.detach()).cpu().numpy()
                attn_list.append(attn_mean)

            pred_np = np.array(pred_list)
            label_np = np.array(label_list)

            accuracy = acc(pred_np, label_np)
            F1 = f1_score(label_np, pred_np, average='macro')

            print("[S"+str(subject_num)+"]", "Epoch:", (epoch + 1), "Acc: %.4f" %(accuracy*100), "F1-score: %.4f" %(F1*100))

            if accuracy > best_acc:
                best_acc = accuracy.copy()
                best_fc = F1.copy()
                best_model_wts_lrcn = copy.deepcopy(net_lrcn.state_dict())
                best_model_wts_att = copy.deepcopy(net_att.state_dict())
                best_model_wts_fc = copy.deepcopy(net_fc.state_dict())

del train_data, train_loader, test_data, test_loader
torch.cuda.empty_cache()

