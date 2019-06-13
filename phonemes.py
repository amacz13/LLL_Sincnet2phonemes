#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Tuesday Dec 2 2015

@author: Anthony Larcher
"""


import numpy as np
import numpy
import torch
import sidekit
import os
import csv_read


class Net(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.lin1 = torch.nn.Linear(360, 650)
        self.lin2 = torch.nn.Linear(650, 650)
        self.lin3 = torch.nn.Linear(650, 80)
        self.lin4 = torch.nn.Linear(80, 650)
        self.lin5 = torch.nn.Linear(650, 650)
        self.lin6 = torch.nn.Linear(650, 2304)

    def forward(self, x):
        m = torch.nn.Sigmoid()
        x = m(self.lin1(x))
        x = m(self.lin2(x))
        x = self.lin3(x)
        x = m(self.lin4(x))
        x = m(self.lin5(x))
        x = self.lin6(x)
        return x


if __name__ == '__main__':
    # Directory of the files
    training_feature_dir = "D:\\Documents\\Universite\\4A\\LLL\\LLL_DB\\h5"
    # Label file
    label_file_name = "mini.ali"

    # Features parameters
    feature_context = (7, 7)
    feature_size = 360

    # DNN Configuration
    nb_epoch = 3
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_buffer_size=300
    batch_size=512
    output_file_name="dnn/Stat_extractor_{}.mdl"

    # Load the labels and get the number of output classes

    print("Load training feature labels")
    with open(label_file_name, 'r') as inputf:
        lines = [line.rstrip() for line in inputf]
        print(lines[1].split('_')[0] + '_' + lines[1].split('_')[1])
        seg_list = [(line.split('_')[0] + '_' + line.split('_')[1],
            int(line.split('_')[2].split('-')[0]),
            int(line.split(' ')[0].split('_')[2].split('-')[1]),
            np.array([int(x) for x in line.split()[1:]]))
            for line in lines[1:-1]]
        max_list = [seg[3].max() for seg in seg_list]
        nclasses = int(max(max_list) + 1)

    print("Number of output classes = {}".format(nclasses))

    # Create FeaturesServer to load features
    fs_params = {"feature_filename_structure":training_feature_dir+"{}.h5",
                "dataset_list": ["fb"],
                "context":feature_context,
                "feat_norm":"cmvn",
                "global_cmvn":True}

    np.random.seed(42)
    shuffle_idx = np.random.permutation(np.arange(len(seg_list)))
    seg_list = [seg_list[idx] for idx in shuffle_idx]

    net = Net()
    data = np.load("mean_std.npz")
    input_mean = data["mean"]
    input_std = data["std"]

    # split the list of files to process
    training_segment_sets = [seg_list[i:i + segment_buffer_size]
                             for i in range(0, len(seg_list), segment_buffer_size)]

    # Initialized cross validation error
    last_cv_error = -1 * numpy.inf

    for ep in range(nb_epoch):

        print("Start epoch {} / {}".format(ep + 1, nb_epoch))
        features_server = sidekit.FeaturesServer(**fs_params)
        running_loss = accuracy = n = nbatch = 0.0

        # Move model to requested device (GPU)
        net.to(device)

        # Set training parameters
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(net.parameters())

        for idx_mb, file_list in enumerate(training_segment_sets):
            l = []
            f = []
            for idx, val in enumerate(file_list):
                show, s, _, label = val
                #show = show.replace("/",os.path.sep)
                print("Show : ",show)
                e = s + len(label)
                l.append(label)
                # Load the segment of frames plus left and right context
                feat, _ = features_server.load(show,
                                               start=s - features_server.context[0],
                                               stop=e + features_server.context[1])
                # Get features in context
                f.append(features_server.get_context(feat=feat,
                                                     label=None,
                                                     start=features_server.context[0],
                                                     stop=feat.shape[0]-features_server.context[1])[0])
            lab = numpy.hstack(l)
            fea = numpy.vstack(f).astype(numpy.float32)
            assert numpy.all(lab != -1) and len(lab) == len(fea)  # make sure that all frames have defined label
            shuffle = numpy.random.permutation(len(lab))
            label = lab.take(shuffle, axis=0)
            data = fea.take(shuffle, axis=0)

            # normalize the input
            data = (data - input_mean) / input_std

            # Send data and label to the GPU
            data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
            label = torch.from_numpy(label).to(device)

            for jj, (X, t) in enumerate(zip(torch.split(data, batch_size), torch.split(label, batch_size))):

                optimizer.zero_grad()
                lab_pred = net.forward(X)
                loss = criterion(lab_pred, t)
                loss.backward()
                optimizer.step()

                accuracy += (torch.argmax(lab_pred.data, 1) == t).sum().item()
                nbatch += 1
                n += len(X)
                running_loss += loss.item() / (batch_size * nbatch)
                if nbatch % 200 == 199:
                    print("loss = {} | accuracy = {} ".format(running_loss,  accuracy / n) )

        optimizer.zero_grad()
        running_loss = accuracy = n = nbatch = 0.0

        # Save the current version of the network
        torch.save(net.to('cpu').state_dict(), output_file_name.format(ep))
