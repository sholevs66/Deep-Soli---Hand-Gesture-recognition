# Demo code to extract data in python
import h5py
import os
import json
import argparse
import torch
import tools


def load_data(args):
    import numpy as np
    reshape_flag = int(args['reshape_flag'])
    zero_pad = int(args['zero_pad'])

    use_channel = 0
    f=open(os.path.join('config','file_half.json'))
    idx = json.load(f)

    data_train = []
    label_train = []
    data_vld = []
    label_vld = []

    for d in idx['train']:
        q = h5py.File(os.path.join('dsp', d + '.h5'), 'r')
        temp = torch.from_numpy(q['ch{}'.format(use_channel)][()])  # [seq_len, 1024]
        if zero_pad == 1:
            temp = torch.cat((temp, torch.zeros(130 - temp.shape[0], 1024)))        # max sequence length is 130
        elif zero_pad > 1:
            if temp.shape[0] >= zero_pad:
                temp = temp[0:40, :]
                label_train.append(torch.from_numpy(q['label'][0:zero_pad, 0]).long())
            else:
                temp = torch.cat((temp, torch.zeros(40 - temp.shape[0], 1024)))  # pad to length zero_pad
                label_train.append(torch.from_numpy(q['label'][:, 0]).long())
        if reshape_flag == 0:
            data_train.append(temp)
        else:
            data_train.append(temp.reshape(-1, 32, 32))

    # drop sequences with less than 40 samples - 33 in original training set
    idx_to_drop = [label_train[idx].shape[0] for idx in range(0, len(idx['train']))]
    idx_to_drop = np.asarray(idx_to_drop)
    idx_to_drop = np.where(idx_to_drop < 40)[0]
    label_train = [i for j, i in enumerate(label_train) if j not in idx_to_drop]
    data_train = [i for j, i in enumerate(data_train) if j not in idx_to_drop]

    labels_tr_l = torch.LongTensor(tools.get_last_element(label_train, len(label_train)))

    for d in idx['eval']:
        q = h5py.File(os.path.join('dsp', d + '.h5'), 'r')
        temp = torch.from_numpy(q['ch{}'.format(use_channel)][()])  # [seq_len, 1024]
        if zero_pad == 1:
            temp = torch.cat((temp, torch.zeros(145 - temp.shape[0], 1024)))  # max sequence length is 145
        elif zero_pad > 1:
            if temp.shape[0] >= zero_pad:
                temp = temp[0:40, :]
                label_vld.append(torch.from_numpy(q['label'][0:40, 0]).long())
            else:
                temp = torch.cat((temp, torch.zeros(40 - temp.shape[0], 1024)))  # pad to length zero_pad
                label_vld.append(torch.from_numpy(q['label'][:, 0]).long())
        if reshape_flag == 0:
            data_vld.append(temp)
        else:
            data_vld.append(temp.reshape(-1, 32, 32))

    # drop sequences with less than 40 samples - 23 sequences in original eval set
    idx_to_drop = [label_vld[idx].shape[0] for idx in range(0, len(idx['eval']))]
    idx_to_drop = np.asarray(idx_to_drop)
    idx_to_drop = np.where(idx_to_drop < 40)[0]
    label_vld = [i for j, i in enumerate(label_vld) if j not in idx_to_drop]
    data_vld = [i for j, i in enumerate(data_vld) if j not in idx_to_drop]

    labels_vl_l = torch.LongTensor(tools.get_last_element(label_vld, len(label_vld)))


    return data_train, label_train, labels_tr_l, data_vld, label_vld, labels_vl_l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='provide arguments')
    parser.add_argument('--reshape_flag', help='Reshape data to 32x32 matrices', default=1)
    parser.add_argument('--zero_pad', help='If true, zero pad train & validation sequences to max length, if Int than pad/subsample to Int', default=40)
    args = vars(parser.parse_args())
    load_data(args)