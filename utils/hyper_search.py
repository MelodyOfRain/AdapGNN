import sys
sys.path.append('..')
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
from model.gnn import*
from model.adaptive_gnn import*
from utils.train import train
import numpy as np
from utils.config import CONFIG as benchmark_config
import os.path as osp
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--task_name',type=str,help='name for specific prediction task')
parser.add_argument('--modeltype',type=str,help='name for specific model')
args = parser.parse_args()

def main_gcn(hc,nl,lr,wd,dropout):
    hc = int(hc) # hideen channels
    nl = int(nl) # number of layers
    fn = osp.join(root,'tmp/%s_%s.model'%(modeltype,task_name))
    model_class = eval(modeltype)
    model = model_class(in_channels=benchmark_config[task_name]['node_dim'],
                        edge_dim=benchmark_config[task_name]['edge_dim'],
                        hidden_channels=hc,num_layers=nl,dropout=dropout,out_channels=1).to(device)
    val = -train(300, model,task_name=task_name,path=path,device=device,lr=lr,wd=wd,fn=fn)
    return val

def main_afp(hc,nl,nt,lr,wd,dropout):
    hc = int(hc) # hideen channels
    nl = int(nl) # number of layers
    nt = int(nt) # number of timesteps
    fn = osp.join(root,'tmp/%s_%s.model'%(modeltype,task_name))
    model_class = eval(modeltype)
    model = model_class(in_channels=benchmark_config[task_name]['node_dim'],
                        edge_dim=benchmark_config[task_name]['edge_dim'],
                        hidden_channels=hc,num_layers=nl,num_timesteps=nt,dropout=dropout,out_channels=1).to(device)
    val = -train(300, model,task_name=task_name,path=path,device=device,lr=lr,wd=wd,fn=fn)
    return val

if __name__ == '__main__':
    device = 'cuda:0'
    task_name = args.task_name  # 'carbon_cnt', 'noncarbon_cnt', ...
    modeltype = args.modeltype # 'GCN' , 'AFP', 'AdapGCNL1', 'AdapGCNL2', 'AdapAFPL1', 'AdapAFPL2'
    root = os.path.abspath('../save')
    print(root)
    if not osp.exists(osp.join(root,'tmp')):
        os.makedirs(osp.join(root,'tmp'))

    if not osp.exists(osp.join(root,'optim')):
        os.makedirs(osp.join(root,'optim'))
    
    path = os.path.abspath('../datasets/hyperopt')
    if not osp.exists(osp.join(path,task_name,'raw')):
        os.makedirs(osp.join(path,task_name,'raw'))
    shutil.copy('../data/smiles_opt.pt',osp.join(path,task_name,'raw','smiles.pt'))
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='UCB')
    if 'GCN' in modeltype:
        param = {
                    'hc':('int',[10,300]),
                    'nl':('int',[2,6]),
                    'lr':('cont',[2,5]),
                    'wd':('cont',[2,5]),
                    'dropout':('cont',[0,0.5]),
        }
        np.random.seed(1)
        gpgo = GPGO(gp, acq, main_gcn, param)
        sys.stdout = open(root+'/optim/%s_%s.txt'%(modeltype,task_name),'w')
        gpgo.run(max_iter=30,init_evals=1)
    else:
        param = {
                    'hc':('int',[10,300]),
                    'nl':('int',[2,6]),
                    'nt':('int',[2,6]),
                    'lr':('cont',[2,5]),
                    'wd':('cont',[2,5]),
                    'dropout':('cont',[0,0.5]),
        }
        np.random.seed(1)
        gpgo = GPGO(gp, acq, main_afp, param)
        sys.stdout = open(root+'/optim/%s_%s.txt'%(modeltype,task_name),'w')
        gpgo.run(max_iter=30,init_evals=1)
    