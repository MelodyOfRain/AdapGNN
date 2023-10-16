import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch import nn

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    global_add_pool
)
from torch_geometric.nn.models.attentive_fp import GATEConv

class AdapAFPL1(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.lin_node = Linear(in_channels,1)

        self.lin1 = Linear(in_channels, hidden_channels)
        if edge_dim!=None:
            self.gate_conv = GATEConv(hidden_channels+in_channels, hidden_channels, edge_dim,
                                  dropout)
        else:
            self.gate_conv = GATConv(hidden_channels+in_channels, hidden_channels,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels+in_channels,hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        self.predict = Linear(hidden_channels,out_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.predict.reset_parameters()

    def forward(self,x,edge_index,edge_attr,batch) -> Tensor:
        """"""
        # Atom Embedding:
        x0 = x
        x = F.leaky_relu_(self.lin1(x))
        node_weight = self.lin_node(x0)
        node_weight = torch.sigmoid(node_weight)
        if self.edge_dim!=None:
            h = F.elu_(self.gate_conv(torch.cat([x0*node_weight,x],1), edge_index, edge_attr))
        else:
            h = F.elu_(self.gate_conv(torch.cat([x0*node_weight,x],1), edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(torch.cat([x0*node_weight,x],1), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.predict(out)
        return out
    
class AdapAFPL2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin_node = Linear(hidden_channels,1)
        self.weight_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)

        if edge_dim!=None:
            self.gate_conv = GATEConv(hidden_channels+in_channels, hidden_channels, edge_dim,
                                  dropout)
        else:
            self.gate_conv = GATConv(hidden_channels+in_channels, hidden_channels,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels+in_channels,hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        self.predict = Linear(hidden_channels,out_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.predict.reset_parameters()

    def forward(self,x,edge_index,edge_attr,batch) -> Tensor:
        """"""
        # Atom Embedding:
        x0 = x
        x = F.leaky_relu_(self.lin1(x))
        node_weight = self.weight_conv(x,edge_index,edge_attr)
        node_weight = self.lin_node(node_weight)
        node_weight = torch.sigmoid(node_weight)
        if self.edge_dim!=None:
            h = F.elu_(self.gate_conv(torch.cat([x0*node_weight,x],1), edge_index, edge_attr))
        else:
            h = F.elu_(self.gate_conv(torch.cat([x0*node_weight,x],1), edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(torch.cat([x0*node_weight,x],1), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.predict(out)
        return out
    
class AdapGCNL1(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                num_layers: int,
                dropout=0):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.num_layers = num_layers
        self.atom_convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.lin_node  = mlp(hidden_channels,128,1)
        self.predict = Linear(hidden_channels, out_channels)
        for i in range(num_layers):
            conv = GCNConv(2*hidden_channels,hidden_channels,cached=False,
                                 normalize=True,add_self_loops=False)
            self.atom_convs.append(conv)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        for conv in self.atom_convs:
            conv.reset_parameters()
        self.predict.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch) -> Tensor:
       
        x = x0 = F.leaky_relu_(self.lin1(x))
        node_weight = self.lin_node(x0)
        node_weight = torch.sigmoid(node_weight)

        #message passing
        for conv in self.atom_convs:
            x = conv(torch.cat([x,x0*node_weight],1), edge_index, None).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
         
        out = global_add_pool(x, batch)
        out = self.predict(out)
        return out

class AdapGCNL2(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                num_layers: int,
                edge_dim : int,
                dropout=0,
                weight_conv_type = 'gat'):
        super().__init__()

        self.lin1 = Linear(in_channels, hidden_channels)
        self.num_layers = num_layers
        self.atom_convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.lin_node = mlp(hidden_channels,128,1)
        assert weight_conv_type in ['gat','gcn']
        self.weight_conv_type = weight_conv_type
        if weight_conv_type == 'gat':
            self.weight_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        else:
            self.weight_conv = GCNConv(hidden_channels, hidden_channels,cached=False,
                                 normalize=True,add_self_loops=False)
        self.predict = Linear(hidden_channels, out_channels)

        for i in range(num_layers):
            conv = GCNConv(2*hidden_channels,hidden_channels,cached=False,
                                 normalize=True,add_self_loops=False)
            self.atom_convs.append(conv)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        for conv in self.atom_convs:
            conv.reset_parameters()
        self.predict.reset_parameters()
        self.weight_conv.reset_parameters

    def forward(self, x, edge_index, edge_attr, batch) -> Tensor:
       
        x = x0 = F.leaky_relu_(self.lin1(x))
        if self.weight_conv_type == 'gat':
            node_weight = self.weight_conv(x, edge_index, edge_attr)
        else:
            node_weight = self.weight_conv(x, edge_index)
        node_weight = self.lin_node(x0)
        node_weight = torch.sigmoid(node_weight)

        #message passing
        for conv in self.atom_convs:
            x = conv(torch.cat([x,x0*node_weight],1), edge_index, None).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
         
        out = global_add_pool(x, batch)
        out = self.predict(out)
        return out
         
class mlp(nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel,num_layer=2) -> None:
        super().__init__()
        self.num_layer = num_layer
        for i in range(num_layer):
            if i==0:
                self.add_module('lin'+str(i),nn.Linear(in_channel,hidden_channel))
            elif i<num_layer-1:
                self.add_module('lin'+str(i),nn.Linear(hidden_channel,hidden_channel))
            else:
                self.add_module('lin'+str(i),nn.Linear(hidden_channel,out_channel))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.num_layer):
            getattr(self,'lin'+str(i)).reset_parameters()
        
    def forward(self,x):
        for i in range(self.num_layer-1):
            x = F.relu(getattr(self,'lin'+str(i))(x))
        return getattr(self,'lin'+str(self.num_layer-1))(x)
            