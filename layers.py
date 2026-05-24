import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
try:
    from torch_sparse import coalesce as _coalesce_impl

    def coalesce(edge_index, edge_attr, m, n):
        return _coalesce_impl(edge_index, edge_attr, m, n)
except ImportError:
    from torch_geometric.utils import coalesce as _coalesce_impl

    def coalesce(edge_index, edge_attr, m, n):
        num_nodes = max(int(m), int(n))
        return _coalesce_impl(edge_index, edge_attr, num_nodes=num_nodes, reduce='sum')
_norm_layer_factory = {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}
_act_layer_factory = {'relu': nn.ReLU, 'relu6': nn.ReLU6, 'sigmoid': nn.Sigmoid, 'prelu': nn.PReLU, 'silu': nn.SiLU}

def create_spectral_features(pos_edge_index: torch.LongTensor, neg_edge_index: torch.LongTensor, node_num: int, dim: int) -> torch.FloatTensor:
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_index = edge_index.to(torch.device('cpu'))
    pos_val = torch.full((pos_edge_index.size(1),), 2, dtype=torch.float)
    neg_val = torch.full((neg_edge_index.size(1),), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)
    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)
    edge_index, val = coalesce(edge_index, val, node_num, node_num)
    val = val - 1
    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    adjacency = sp.coo_matrix((val, edge_index), shape=(node_num, node_num))
    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(adjacency)
    features = svd.components_.T
    return torch.from_numpy(features).to(torch.float)

class MLP(nn.Module):

    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5, sigmoid=False, tanh=False):
        super().__init__()
        assert num_layer >= 2, 'The number of layers should be larger or equal to 2.'
        if norm_layer in _norm_layer_factory:
            self.norm_layer = _norm_layer_factory[norm_layer]
        if act_layer in _act_layer_factory:
            self.act_layer = _act_layer_factory[act_layer]
        if p_drop > 0:
            self.dropout = nn.Dropout
        layers = []
        layers.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            layers.append(self.norm_layer(dim_hidden))
        if act_layer:
            layers.append(self.act_layer(inplace=True))
        if p_drop > 0:
            layers.append(self.dropout(p_drop))
        for _ in range(num_layer - 2):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                layers.append(self.norm_layer(dim_hidden))
            if act_layer:
                layers.append(self.act_layer(inplace=True))
            if p_drop > 0:
                layers.append(self.dropout(p_drop))
        layers.append(nn.Linear(dim_hidden, dim_pred))
        if sigmoid:
            layers.append(nn.Sigmoid())
        if tanh:
            layers.append(nn.Tanh())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
