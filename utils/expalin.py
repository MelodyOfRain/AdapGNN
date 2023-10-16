from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from utils.dataset import prepare
import torch
from torch_geometric.loader import DataLoader
from utils.config import CONFIG as config
import tqdm as tqdm


def bat_explain(modeldir, datadir, task_name, modeltype, device):
    for i in range(3):
        fn = modeldir + '/%s/%s/%s%s.pkl' % (task_name, modeltype, task_name, i+1)
        model = torch.load(fn,map_location=device)
        explain(fn, task_name, datadir, device, model)
    return

def explain(fn, task_name, datadir, device, model):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explainer_config=dict(
            explanation_type='phenomenon',
            node_mask_type='object',
            edge_mask_type=None),
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        )
    )
    weight = []
    _, _, test_dataset = prepare(task_name, datadir, featurizer=config[task_name]['featurizer']
                                 , descriptor=config[task_name]['descriptor'],
                                 des_para=config[task_name]['para'], shuffle=False)
    loader = DataLoader(test_dataset, batch_size=1000)
    for data in tqdm(loader):
        data.to(device)
        mask = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch, target=data.y)
        weight.append(mask['node_mask'])
    weight = torch.cat(weight)
    torch.save(weight, fn.replace('model', 'weight'))
    return
