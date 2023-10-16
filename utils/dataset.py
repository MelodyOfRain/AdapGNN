import os.path as osp
from typing import Callable, Optional

import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.smiles import from_smiles
from tqdm import tqdm


def prepare(datanme, path, featurizer, descriptor, des_para, shuffle=False):
    """
    Args:
        datanme (str): task name
        path (str): path to load data
        featurizer (Callable): featurizer for transform smiles to graph representation
        descriptor (Callable): fucntion for generating target descriptor
        des_para (dict): parameters for descriptor generation fucntion
        shuffle (bool, optional): Defaults to False.

    """    
    dataset = MolExplain(path, name=datanme, pre_transform=featurizer, descriptor=descriptor, des_para=des_para)
    if shuffle:
        dataset.shuffle()
    N = len(dataset) // 10
    test_dataset = dataset[:N]
    val_dataset = dataset[N:2 * N]
    train_dataset = dataset[2 * N:]
    return train_dataset, val_dataset, test_dataset


class MolExplain(InMemoryDataset):

    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            descriptor: Optional[Callable] = None,
            des_para: Optional[dict] = None
    ):
        """
            root (str): the root path for benchmark dataset
            name (str): the name of task
            transform (Optional[Callable], optional): Defaults to None.
            pre_transform (Optional[Callable], optional): Defaults to None.
            pre_filter (Optional[Callable], optional): Defaults to None.
            descriptor (Optional[Callable], optional): fucntion for generating target descriptor
            des_para (Optional[dict], optional): parameters for descriptor generation fucntion
        """  
        # assert name in 
        self.name = name
        self.descriptor = descriptor
        self.des_para = des_para
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'smiles.pt'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):

        data_list = []
        smiles = torch.load(self.raw_paths[0])
        with tqdm(total=len(smiles)) as pbar:
            for s in smiles:
                data = from_smiles(s)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                mask, contribs, y = self.descriptor(data.smiles, **self.des_para)
                data.y = y
                data.atom_contribs = contribs.view(-1)
                data.node_mask = mask.view(-1)
                data_list.append(data)
                pbar.update(1)

        data, indice = self.collate(data_list)
        data.node_mask = data.node_mask.long()
        max_contrib = torch.max(torch.abs(data.atom_contribs * data.node_mask))
        data.atom_contribs[(data.node_mask == 1) & (data.atom_contribs > 0)] += max_contrib
        data.atom_contribs[(data.node_mask == 1) & (data.atom_contribs < 0)] -= max_contrib
        for i in range(len(data.smiles)):
            data.y[i] = torch.sum(data.atom_contribs[indice['x'][i]:indice['x'][i + 1]])
        std_mean = torch.std_mean(data.y)
        data.y = (data.y - std_mean[1]) / std_mean[0]
        data.y = data.y.view(-1, 1)
        torch.save((data, indice), self.processed_paths[0])
        torch.save(std_mean, self.processed_dir + '/std_mean.pt')

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
