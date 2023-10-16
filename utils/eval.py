import sys
sys.path.append('..')
import torch
import numpy as np
from utils.dataset import prepare
from torch_geometric.loader import DataLoader
from utils.config import CONFIG as config
from math import sqrt
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def compute_mute_rmse(datadir,modeldir,task_name, modeltype, device):
    std = torch.load(datadir + '/%s/processed/std_mean.pt' % task_name)[0]
    score = []

    _, _, test_dataset = prepare(task_name, datadir, featurizer=config[task_name]['featurizer']
                                 , descriptor=config[task_name]['descriptor'],
                                 des_para=config[task_name]['para'], shuffle=False)
    loader = DataLoader(test_dataset, batch_size=3000)
    for i in range(1,4):
        fn = modeldir + '/%s/%s/%s%s.pkl' % (task_name, modeltype, task_name, i)
        model = torch.load(fn, map_location=device)
        model.eval()
        rmse = []
        for data in loader:
            data.to(device)
            target = data.y
            data.x = data.x * data.node_mask.view(-1, 1)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            rmse.append((target - out) ** 2)
        rmse = torch.cat(rmse)
        rmse = sqrt(torch.mean(rmse)) * std
        score.append(rmse)
    return [np.mean(score), np.std(score)]


def compute_roc_auc(datadir,modeldir,task_name, modeltype):
    score = []

    _, _, test_dataset = prepare(task_name, datadir, featurizer=config[task_name]['featurizer']
                                 , descriptor=config[task_name]['descriptor'],
                                 des_para=config[task_name]['para'], shuffle=False)
    loader = DataLoader(test_dataset, batch_size=30000)
    for data in loader:
        label = data.node_mask.cpu().view(-1).numpy()
    for i in range(1,4):
        fn = modeldir.replace('model', 'weight') + '/%s/%s/%s%s.pkl' % (task_name, modeltype, task_name, i)
        weight = torch.load(fn, map_location='cpu').view(-1)
        weight = weight.cpu().numpy()
        score.append(float(roc_auc_score(label, weight)))
    score_mean = np.mean(score) * 100
    if score_mean<50:
        score_mean = 100 - score_mean
    score_std = np.std(score) * 100
    return [score_mean, score_std]


@torch.no_grad()
def compute_pre_rmse(datadir,modeldir,task_name, modeltype, device):
    std = torch.load(datadir + '/%s/processed/std_mean.pt' % task_name)[0]
    score = []

    _, _, test_dataset = prepare(task_name, datadir, featurizer=config[task_name]['featurizer']
                                 , descriptor=config[task_name]['descriptor'],
                                 des_para=config[task_name]['para'], shuffle=False)
    loader = DataLoader(test_dataset, batch_size=3000)
    for i in range(1,4):
        fn = modeldir + '/%s/%s/%s%s.pkl' % (task_name, modeltype, task_name, i)
        model = torch.load(fn, map_location=device)
        model.eval()
        rmse = []
        for data in loader:
            data.to(device)
            target = data.y
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            rmse.append((target - out) ** 2)
        rmse = torch.cat(rmse)
        rmse = sqrt(torch.mean(rmse)) * std
        score.append(rmse)
    return [np.mean(score), np.std(score)]
