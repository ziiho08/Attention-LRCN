import os
import pickle
import numpy as np

savePath = 'merged_PPG'
subject_list = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

# E4 (wrist) Sampling Frequencies
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}
label_dict = {'baseline': 1, 'stress': 2, 'amusement': 3}
int_to_label = {1: 'baseline', 2: 'stress', 3: 'amusement'}
label_to_class = {'baseline': 0, 'stress': 1, 'amusement': 2}

subject_idx = 1
for subject_id in subject_list:
    main_path = './data'
    subject_number = subject_id
    subject_name = f'S{subject_number}'
    with open(os.path.join(main_path, subject_name) + '/' + subject_name + '.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    ppg_signal = data['signal']['wrist']['BVP']  # 64 Hz
    label = data['label']  # 700 Hz

    idx_sync = [round(fs_dict['label'] / fs_dict['BVP'] * i) for i in range(len(ppg_signal))]  # 700/64xi
    label_sync = label[idx_sync]

    sample_num = []
    for label_value in label_dict.values():
        file_name = f'S{subject_number}_' + int_to_label[label_value]
        ppg_signal_ = ppg_signal[label_sync == label_value]
        np.save(os.path.join(savePath, file_name), ppg_signal_)
        sample_num.append(len(ppg_signal_))

    print('[S' + str(subject_id) + ' samples] Total:', len(ppg_signal), ',  baseline:', sample_num[0],
          ', stress:', sample_num[1], ',  amusement:', sample_num[2])

    subject_idx += 1

window_len = 120 * fs_dict['BVP']
overlap = 1 * fs_dict['BVP']  # [sec]x64Hz
ppg_window_list = []
label_subject_list = []
for subject_number in subject_list:
    subject_name = f'S{subject_number}'

    for label_key in label_dict.keys():
        ppg_signal_ = np.load(os.path.join(savePath, subject_name) + '_' + label_key + '.npy')
        idx_st = list(range(0, len(ppg_signal_) - window_len + 1, overlap))

        print('[' + subject_name + ']', label_key, 'num:', len(idx_st))

        for itr_window in range(len(idx_st)):
            ppg_window_list.append(ppg_signal_[idx_st[itr_window]:idx_st[itr_window] + window_len, 0])
            label_subject_list.append([label_to_class[label_key], subject_number])
ppg_window_np = np.array(ppg_window_list)
label_subject_np = np.array(label_subject_list)
data_np = np.concatenate((ppg_window_np, label_subject_np), axis=1)

file_name = 'subject_merge.npy'
np.save(os.path.join(savePath, file_name), data_np)
