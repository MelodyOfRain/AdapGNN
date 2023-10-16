import torch.nn.functional as F
from math import sqrt
import torch
from utils.dataset import prepare
from torch_geometric.loader import DataLoader
from utils.config import CONFIG as config


@torch.no_grad()
def test(loader, model, device):
    mse = []
    model.eval()
    for data in loader:
        data = data.to(device)
        if data.edge_attr is not None:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        else:
            out = model(data.x, data.edge_index, data.batch)
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
    return sqrt(float(torch.cat(mse, dim=0).mean()))


def train(epochs, model, task_name, path, device, lr, wd, fn, batch_size=512, shuffle=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -lr,
                                 weight_decay=10 ** -wd)
    
    train_dataset, val_dataset, test_dataset = prepare(task_name, path,
                                                       featurizer=config[task_name]['featurizer'],
                                                       descriptor=config[task_name]['descriptor'],
                                                       des_para=config[task_name]['para'],
                                                       shuffle=shuffle)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_param = {}
    best_param["train_epoch"] = 0
    best_param["val_epoch"] = 0
    best_param["train_RMSE"] = 9e8
    best_param["val_RMSE"] = 9e8

    iter = 0
    val_interval = int(10*512/batch_size)
    
    for epoch in range(1, 1 + epochs):
        for data in train_loader:
            model.train()
            data = data.to(device)
            optimizer.zero_grad()
            if data.edge_attr is not None:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            else:
                out = model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            if iter % val_interval == 0:
                val_RMSE = test(val_loader, model, device)
                if val_RMSE < best_param["val_RMSE"]:
                    best_param["val_epoch"] = iter
                    best_param["val_RMSE"] = val_RMSE
                    torch.save(model, fn)
                if (iter - best_param["val_epoch"]) > 10 * val_interval:
                    break
                print(f'Epoch: {iter // val_interval:04d}, Loss: {loss.item():.4f} Val RMSE: {val_RMSE:.4f}')
            optimizer.step()
            iter += 1
    model = torch.load(fn)
    test_RMSE = test(test_loader, model, device)
    return test_RMSE
