"""
Basic MLP training functionality, needs to be integrated with LGBM/ Type2 GNN training pipelines
"""
import os, sys

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.evaluate import compute_binary_metrics


class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layers, dropout=0.1):
        """
        Initialize basic MLP model.
        :param n_in: Number of input features
        :param n_out: Number of output features
        :param n_hidden: Number of hidden nodes in each layer
        :param n_layers: Number of hidden layers
        """
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.inp = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU())
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.BatchNorm1d(n_hidden))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.inp(x)
        x = self.hidden(x)
        x = self.out(x)
        
        return x


class AMLLoader(Dataset):
    def __init__(self, data):
        """
        Batch loader for AML data.
        :param data: torch_geometric.Data instance
        """
        super(AMLLoader, self).__init__()
        if isinstance(data, Data):
            self.x = torch.tensor(data.x).float()
            self.y = torch.tensor(data.y).long()
        elif isinstance(data, tuple):
            self.x = data[0].float()
            self.y = data[1].long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_mlp(tr_data, te_data, tr_inds, val_inds, params, args):
    """
    MLP training.

    :param tr_data: Training data
    :param te_data: Test data
    :param tr_inds: Train indices
    :param val_inds: Validation indices
    :param params: Model parameters
    :param args: Auxiliary command line arguments
    :return: Model and train/ validation/ test metrics
    """
    device = args.device
    n_hidden, ce_w1, ce_w2, lr, norm, n_layers = params
    fn_norm = args.fn_norms[norm]

    x_tr, y_tr = fn_norm(tr_data.x[tr_inds].detach().clone().to(device), device), tr_data.y[tr_inds]
    x_val, y_val = fn_norm(tr_data.x[val_inds].detach().clone().to(device), device), tr_data.y[val_inds]
    x_te, y_te = fn_norm(te_data.x.detach().clone().to(device), device), te_data.y

    loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([ce_w1, ce_w2]).to(device))
    model = MLP(n_in=x_tr.shape[1], n_out=2, n_hidden=n_hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(AMLLoader((x_tr, y_tr)), batch_size=1024, shuffle=True)
    val_loader = DataLoader(AMLLoader((x_val, y_val)), batch_size=1024, shuffle=True)
    test_loader = DataLoader(AMLLoader((x_te, y_te)), batch_size=1024, shuffle=True)

    for i in tqdm(range(args.n_epochs)):
        train_losses = []
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            model.train()
            model.zero_grad()
            tr_pred = model(x.to(device))

            l = loss(tr_pred, y.to(device).squeeze())
            train_losses.append(l.detach().item())

            l.backward()
            opt.step()

    train_metrics = evaluate_mlp(model, train_loader, loss, device)
    val_metrics = evaluate_mlp(model, val_loader, loss, device)
    test_metrics = evaluate_mlp(model, test_loader, loss, device)

    # tqdm.write(f"train loss: {np.mean(train_losses):.4f} || test loss: {np.mean(test_losses):.4f} || "
    #            f"train F1: {train_metrics[-1]} || val F1: {val_metrics[-1]} || test F1: {test_metrics[-1]}")

    return model, train_metrics, val_metrics, test_metrics


@torch.no_grad()
def evaluate_mlp(model, loader, loss_fn, device):
    """
    Evaluate MLP performance on a chosen dataset
    :param model: MLP model instance
    :param loader: Inherited DataSet instance with __getitem__ method implemented
    :param loss_fn: Loss function of choice
    :param device: GPU or CPU
    :return: Computed metrics
    """
    preds, target = [], []

    losses = []
    with torch.no_grad():
        model.eval()
        for (x, y) in loader:
            pred = model(x.to(device))
            preds.append(pred.argmax(-1))
            target.append(y)
            if y.shape[0] > 1:
                losses.append(loss_fn(pred, y.to(device).squeeze()).item())

    preds, target = torch.cat(preds).tolist(), torch.cat(target).squeeze().tolist()
    metrics = compute_binary_metrics(preds, target)

    return metrics
