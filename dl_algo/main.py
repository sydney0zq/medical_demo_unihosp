import os
import pdb
import torch
from torch.utils import data
import numpy as np
import random
import torch.nn as nn



#### helper of plot ####
import matplotlib.pyplot as plt
def plot_dualy_sharex(x, ys, xlabel, ylabels, figname=None):
    params = {"figsize": (20, 10), "dpi": 60, }

    assert len(ys) == 2
    assert len(x) == len(ys[0]) and len(x) == len(ys[1])
    f, ax1 = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
    ax2 = ax1.twinx()
    
    # If you want scatter plot
    #ax1.scatter(x, ys[0], c='b', label=ylabels[0],)
    #ax2.scatter(x, ys[1], c='r', label=ylabels[1],)

    ax1.plot(x, ys[0], c='b', ls='-', label=ylabels[0],)
    ax2.plot(x, ys[1], c='r', ls='-', label=ylabels[1],)
    ax1.set_xticklabels(x, rotation=0)

    # If you want to put legend in the fig
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    # Else if you want to put lengend outside the fig
    #f.legend(loc='upper right')
    
    plt.grid(True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabels[0])
    ax2.set_ylabel(ylabels[1])
    
    if figname is not None:
        plt.savefig(figname)


#### dataset ####
class CTDataset(data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        
        train_list = []
        for s in os.listdir(self.data_root):
            if s.startswith('covid'):
                train_list.append([os.path.join(self.data_root, s), 1])
            elif s.startswith('normal'):
                train_list.append([os.path.join(self.data_root, s), 0])
            else:
                assert ValueError("sample should be covid19 or normal")
        self.train_list = train_list * 8
        random.shuffle(self.train_list)

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        ct_path, ct_label = self.train_list[index]
        ct_imgs = np.load(ct_path)  # ct_imgs.shape: 38x320x416
        
        # !! Do your augmentation here

        # Normalization to -1~1
        ct_imgs = np.array(ct_imgs, dtype=np.float32)
        ct_imgs = ct_imgs / 127.5 - 1.

        # Our network input is NCTHW
        ct_imgs = np.expand_dims(ct_imgs, 0)   # 38x320x416 -> 1x38x320x416

        # Convert to torch data structure
        ct_imgs_th = torch.from_numpy(ct_imgs).float()
        ct_label_th = torch.from_numpy(np.array(ct_label, dtype=np.uint8)).long()
        return ct_imgs_th, ct_label_th

#### deep learning network ####
from model.baseline_i3d import ENModel


if __name__ == "__main__":
    mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    os.makedirs("ckpt", exist_ok=True)

    # hyper-parameters
    model = ENModel()
    if mode == 'gpu':   
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4,)
    epoch = 10
    batch_size = 3
    # build dataset
    dataset = CTDataset(data_root="../train_data")
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # record
    loss_rec = []
    acc_rec = []

    # train network
    for e in range(epoch):
        print("epoch {} started...".format(e))
        epoch_loss = 0
        epoch_acc = 0
        for ct_imgs, ct_label in dataloader:
            print("current training images.shape: {}, label: {}".format(ct_imgs.shape, ct_label))
            if mode == 'gpu':
                ct_imgs, ct_label = ct_imgs.cuda(), ct_label.cuda()
            output = model(ct_imgs)
            
            loss = criterion(output, ct_label)
            acc = ((output[:, 1] > output[:, 0]).long() == ct_label).sum() / ct_label.size(0)
            print("===> loss is {}, acc is {}".format(loss.item(), acc.item()))

            # record
            epoch_loss += loss.item()
            epoch_acc += ((output[:, 1] > output[:, 0]).long() == ct_label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss is backward and the network is optimized...")
        epoch_loss = epoch_loss / len(dataset)
        epoch_acc = epoch_acc / len(dataset)
        loss_rec.append(epoch_loss); acc_rec.append(epoch_acc)
        print("===> epoch loss is {}, epoch acc is {}".format(epoch_loss, epoch_acc))
        torch.save(model.state_dict(), "ckpt/epoch-{:02d}.pth".format(e))

    # plot
    plot_dualy_sharex([*range(epoch)], ys=[loss_rec, acc_rec], 
                        xlabel="epoch", ylabels=["epoch_loss", "epoch_acc"],
                        figname="train_curve.png")
    
    # validate the network
    with torch.no_grad():
        model.eval()
        for ct_imgs, ct_label in dataloader:
            print("current validating images.shape: {}, label: {}".format(ct_imgs.shape, ct_label))
            if mode == 'gpu':
                ct_imgs, ct_label = ct_imgs.cuda(), ct_label.cuda()
            output = model(ct_imgs)
            
            acc = ((output[:, 1] > output[:, 0]).long() == ct_label).sum() / ct_label.size(0)
            print("===> on validation loss is {}, acc is {}".format(loss.item(), acc.item()))

