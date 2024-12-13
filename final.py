import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset, ConcatDataset

import time
import numpy as np
from src.hypermodel import CNN, CNN_ResNET, MacroRank
from src.utl import InversePairs
from src.hyperdataset import PlainClusterSet as dtb


class Settings:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Dataset settings
        self.data_path = "data/"
        self.train_ratio = 0.8
        self.train_design = ["mgc_des_perf_a", "mgc_fft_a", "mgc_fft_b"]
        self.test_design = ["mgc_fft_a", "mgc_fft_b"]
        # Model settings
        self.model = "GNN"
        self.nhid = 16
        self.layers = 2
        self.egnn_layers = 3
        self.egnn_nhid = 16
        self.skip_cnt = False
        self.pos_encode = True
        # Solver settings
        self.epoch = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.dropout_ratio = 0.25
        self.batch_step = 8
        self.batch_size = 32
        self.patience = 50
        self.save_path = "save_model/"
        # Test settings
        self.label = [0, 1, 4]


#######################################################
## Build Data Loader
def build_data_loader(settings):
    dataset = dtb(root=settings.data_path, settings=settings)

    settings.num_node_features = dataset.num_node_features
    settings.num_edge_features = dataset.num_edge_features
    settings.num_pin_features = dataset.num_pin_features

    train_designs = dataset.train_file_names
    test_designs = dataset.test_file_names
    train_sets = []
    test_sets = []
    test_loader = {}
    num_training = 0
    num_testing = 0
    for design in train_designs:
        train_sets.append(Subset(dataset, range(dataset.ptr[design], dataset.ptr[design] + dataset.file_num[design])))
        num_training += dataset.file_num[design]
    for design in test_designs:
        test_set = Subset(dataset, range(dataset.ptr[design], dataset.ptr[design] + dataset.file_num[design]))
        num_testing += dataset.file_num[design]
        test_loader[design] = DataLoader(test_set, batch_size=settings.batch_size, shuffle=True)
    train_set = ConcatDataset(train_sets)
    print("Total %d training data, %d testing data." % (num_training, num_testing), flush=True)
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True)
    return dataset, train_loader, test_loader


#######################################################
## Build Loss Function
def build_loss_function(settings):
    def BCELoss(out, data):
        if len(settings.label) > 1:
            label = torch.tensor(settings.label[0]).long().to(config.device)
            y = data.y[:, label]
            w = data.w[:, label]
            return F.mse_loss(out[0].view(-1) * w, y.view(-1) * w)
        else:
            label = settings.label[0]
            y = data.y[:, label]
            w = data.w[:, label]
            return F.binary_cross_entropy(target=y.view(-1), input=out[0].view(-1), weight=w)

    return BCELoss


def build_accuracy_function(settings):
    label = settings.label[0]  # TODO: check here wwhy it only use the 1 label

    def BEQAcc(out, data):
        y = data.y[:, label]
        mask1, mask5, mask0 = (out[0] > 0.5), (out[0] == 0.5), (out[0] < 0.5)
        mask = 1.0 * mask1 + 0.5 * mask5
        return torch.eq(mask.view(-1), y.view(-1)).float().mean()

    return BEQAcc


def build_model(settings):
    if settings.model == "CNN":
        return CNN(settings)
    elif settings.model == "CNN_ResNET":
        return CNN_ResNET(settings)
    elif settings.model == "GNN":
        return MacroRank(settings)
    else:
        raise ValueError("Invalid model")


#######################################################
## Test function
def test(model, dataloader, settings):
    with torch.no_grad():
        model.eval()
        lenth = len(dataloader)  # if epoch % 5 == 0 else int(len(loader)/5)
        maes = []
        accs = []
        ipes = []
        for i, label in enumerate(settings.label):
            correct = 0.0
            loss = 0.0
            reals = []
            preds = []
            for i, data in enumerate(dataloader):
                if i >= lenth:
                    break
                data = data.to(settings.device)
                out = model(data).view(-1)
                y = data.y[:, label].view(-1)

                preds.extend(out.detach().cpu().numpy().tolist())
                reals.extend(data.y[:, label].cpu().numpy().tolist())

                correct += torch.mean(torch.abs((y - out) / y)).item()
                loss += F.l1_loss(out, y).item()
            # rank loss
            Rp = np.argsort(preds)
            Rr = np.argsort(np.array(reals)[Rp])
            rankacc = InversePairs(Rr.tolist()) / (len(reals) ** 2 - len(reals)) * 2
            # print('[{}]MAE=\t{:4f}\tMRE={:4f}\tIPE={:4f}'.format(label,loss/len(loader),correct/len(loader),rankacc),end='\t')
            maes.append(loss / lenth)
            accs.append(correct / lenth)
            ipes.append(rankacc)
    return np.mean(maes), np.mean(accs), np.mean(ipes)


#######################################################
## Initial setting
config = Settings()
np.random.seed(212)
torch.manual_seed(212)
torch.cuda.manual_seed_all(212)
torch.cuda.manual_seed(212)

torch.set_num_threads(32)

# Load data
dataset, train_loader, test_loader = build_data_loader(config)

# Define model, loss function, optimizer, and learning rate scheduler
model = build_model(config)
criterion = build_loss_function(config)  # Mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=config.patience, verbose=True)
accuracy = build_accuracy_function(config)

test_designs = dataset.test_file_names
save_last_path = config.save_path + "last.pth"

# for param in model.features.parameters():  # 'features' contains convolutional layers
#     param.requires_grad = False

# Train model
for epoch in range(config.epoch):
    model.train()
    train_time = time.time()
    train_loss = 0.0
    train_acc = 0.0

    for i, data in enumerate(train_loader):  # Assume dataloader is defined
        data = data.to(config.device)
        outputs = [model(data)]  # Forward pass
        loss = criterion(outputs, data) / config.batch_step  # Compute loss
        loss.backward()  # Backward pass
        if (i + 1) % config.batch_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            train_loss += loss.mean().item()
            train_acc += accuracy(outputs, data).item()

    val_losses = []
    rank_errs = []
    print("[Epoch\t{}]\tTrain loss:\t{:.4f}\tTrain acc:\t{:.4f}".format(epoch, train_loss / len(train_loader) * config.batch_step, train_acc / len(train_loader)), flush=True, end="\t")

    for design in test_designs:
        _, val_loss, rank_err = test(model, test_loader[design], config)
        val_losses.append(val_loss)
        rank_errs.append(rank_err)

    mean_val_loss = np.mean(val_losses)
    mean_rank_err = np.mean(rank_errs)

    print("{} mre:\t{:.4f}\t{} ipe:\t{:.4f}\tTime:{:.2f}\tlr:{:.5f}".format("Test", mean_val_loss, "Test", mean_rank_err, time.time() - train_time, optimizer.param_groups[0]["lr"]))

    scheduler.step(mean_val_loss)

state = {"model": model.state_dict(), "val_loss": mean_val_loss, "rank_err": mean_rank_err}
torch.save(state, save_last_path)
