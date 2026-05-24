from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers import MLP, create_spectral_features

class NTOSFLSequenceMixer(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

    def reset_parameters(self):
        return None

    def forward(self, x: Tensor) -> Tensor:
        return x

class DWConv1dSequenceMixer(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = max(d_model, int(round(d_model * 4)))
        kernel_size = 3
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.05)
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.in_proj.reset_parameters()
        self.dwconv.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.in_proj(self.norm(x))
        if h.size(0) > 1:
            h = self.dwconv(h.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        h = self.dropout(self.out_proj(F.gelu(h)))
        return residual + h

def _build_sequence_mixer(d_model: int, use_nto_sfl: bool) -> nn.Module:
    if use_nto_sfl:
        return NTOSFLSequenceMixer(d_model=d_model)
    return DWConv1dSequenceMixer(d_model=d_model)

class DLGSFR(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

    def reset_parameters(self):
        return None

    def forward(self, h: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        return h

class SignedSequenceResidualConv(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_relations: int=2, self_loop: bool=True, dropout: float=0.2, use_nto_sfl: bool=True, use_dlg_sfr: bool=False):
        super().__init__()
        self.num_relations = num_relations
        self.self_loop = self_loop
        self.use_nto_sfl = bool(use_nto_sfl)
        self.use_dlg_sfr = bool(use_dlg_sfr)
        self.relation_layers = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.self_loop_layer = nn.Linear(in_dim, out_dim, bias=False) if self_loop else None
        self.and_pair_norm = nn.LayerNorm(out_dim)
        self.sequence_mixer = _build_sequence_mixer(d_model=out_dim, use_nto_sfl=self.use_nto_sfl)
        self.dlg_sfr = DLGSFR(out_dim) if self.use_dlg_sfr else None
        self.reset_parameters()

    def _reset_sequence_mixer(self):
        if hasattr(self.sequence_mixer, 'reset_parameters'):
            self.sequence_mixer.reset_parameters()
            return
        norm = getattr(self.sequence_mixer, 'norm', None)
        if norm is not None and hasattr(norm, 'reset_parameters'):
            norm.reset_parameters()
        out_proj = getattr(self.sequence_mixer, 'out_proj', None)
        if isinstance(out_proj, nn.Linear):
            out_proj.reset_parameters()

    def reset_parameters(self):
        for layer in self.relation_layers:
            nn.init.xavier_uniform_(layer.weight)
        if self.self_loop and self.self_loop_layer is not None:
            nn.init.xavier_uniform_(self.self_loop_layer.weight)
        self.and_pair_norm.reset_parameters()
        self._reset_sequence_mixer()
        if self.dlg_sfr is not None:
            self.dlg_sfr.reset_parameters()

    @staticmethod
    def _invert_permutation(order: Tensor) -> Tensor:
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return inverse

    def _apply_sequence_mixer(self, agg: Tensor, seq_order: Optional[Tensor]) -> Tensor:
        if agg.size(0) <= 1:
            return agg
        if seq_order is None:
            seq_order = torch.arange(agg.size(0), device=agg.device)
        else:
            seq_order = seq_order.to(device=agg.device, dtype=torch.long)
        if seq_order.numel() != agg.size(0):
            raise ValueError(f'seq_order length {seq_order.numel()} does not match node count {agg.size(0)}')
        inverse_order = self._invert_permutation(seq_order)
        seq_features = agg.index_select(0, seq_order)
        mixed = self.sequence_mixer(seq_features)
        return mixed.index_select(0, inverse_order)

    def _aggregate_relations(self, x: Tensor, edge_indices, and_pair_index: Optional[Tensor]=None) -> Tensor:
        out_dim = self.self_loop_layer.out_features if self.self_loop_layer is not None else self.relation_layers[0].out_features
        agg = x.new_zeros((x.size(0), out_dim))
        shared_layer = self.relation_layers[0]
        pos_edge_index = edge_indices[0]
        if pos_edge_index.numel() > 0:
            source, target = (pos_edge_index[0], pos_edge_index[1])
            messages = shared_layer(x.index_select(0, source))
            agg.scatter_add_(0, target.unsqueeze(-1).expand(-1, messages.size(1)), messages)
        neg_edge_index = edge_indices[1]
        if neg_edge_index.numel() > 0:
            source, target = (neg_edge_index[0], neg_edge_index[1])
            messages = -shared_layer(x.index_select(0, source))
            agg.scatter_add_(0, target.unsqueeze(-1).expand(-1, messages.size(1)), messages)
        if and_pair_index is not None and and_pair_index.numel() > 0:
            and_pair_index = and_pair_index.to(device=x.device)
            src0 = and_pair_index[0].long()
            src1 = and_pair_index[1].long()
            dst = and_pair_index[2].long()
            sign0 = and_pair_index[3].to(dtype=x.dtype).unsqueeze(-1)
            sign1 = and_pair_index[4].to(dtype=x.dtype).unsqueeze(-1)
            m0 = shared_layer(x.index_select(0, src0)) * sign0
            m1 = shared_layer(x.index_select(0, src1)) * sign1
            and_msg = self.and_pair_norm(m0 * m1)
            agg.index_add_(0, dst, and_msg)
        if self.self_loop and self.self_loop_layer is not None:
            agg = agg + self.self_loop_layer(x)
        return agg

    def forward(self, x: Tensor, edge_indices, seq_order: Optional[Tensor]=None, and_pair_index: Optional[Tensor]=None) -> Tensor:
        agg = self._aggregate_relations(x, edge_indices, and_pair_index=and_pair_index)
        if self.use_dlg_sfr and self.dlg_sfr is not None:
            agg = self.dlg_sfr(agg, edge_indices[0], edge_indices[1])
        seq_out = self._apply_sequence_mixer(agg, seq_order)
        return seq_out

class StandardLinearAdapter(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class GraphFeatureDropout(nn.Module):

    def __init__(self, p: float=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if self.p >= 1.0:
            return x.new_zeros(x.shape)
        keep_prob = 1.0 - self.p
        mask = x.new_empty((1, x.size(-1))).bernoulli_(keep_prob)
        mask = mask / keep_prob
        return x * mask

class _BaseBiSeqGate(nn.Module):

    def __init__(self, args, node_num: int, device: torch.device, in_dim: int=64, out_dim: int=64, layer_num: int=2, dropout: float=0.2, lamb: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.node_num = int(node_num)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.lamb = float(max(0.0, min(1.0, lamb)))
        self.device = device
        self.task_type = getattr(args, 'task_type', 'prob')
        hidden_dim = out_dim // 2
        self.pos_edge_index = None
        self.neg_edge_index = None
        self.x = None
        self.conv1 = SignedSequenceResidualConv(in_dim=in_dim, out_dim=hidden_dim, num_relations=2, dropout=dropout, use_nto_sfl=True, use_dlg_sfr=False)
        self.srus = StandardLinearAdapter(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(SignedSequenceResidualConv(in_dim=hidden_dim, out_dim=hidden_dim, num_relations=2, dropout=dropout, use_nto_sfl=True, use_dlg_sfr=False))
            self.hidden_norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layer = GraphFeatureDropout(dropout) if self.task_type == 'tt' else nn.Dropout(dropout)
        self.weight = nn.Linear(hidden_dim, out_dim)
        self.output_norm = nn.LayerNorm(out_dim)
        self.readout_prob = MLP(out_dim, out_dim, 1, num_layer=3, p_drop=dropout, norm_layer='layernorm', act_layer='relu')
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.srus.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()
        self.input_norm.reset_parameters()
        for norm in self.hidden_norms:
            norm.reset_parameters()
        self.output_norm.reset_parameters()

    def _infer_node_num(self, edge_index_s: Tensor) -> int:
        if self.node_num > 0:
            return self.node_num
        if edge_index_s.numel() == 0:
            return 0
        return int(edge_index_s[:, :2].max().item()) + 1

    def get_x_edge_index(self, init_emb: Optional[Tensor], edge_index_s: Tensor, pos_edge_index: Optional[Tensor]=None, neg_edge_index: Optional[Tensor]=None):
        if pos_edge_index is None or neg_edge_index is None:
            pos_mask = edge_index_s[:, 2] > 0
            neg_mask = edge_index_s[:, 2] < 0
            if torch.any(pos_mask):
                self.pos_edge_index = edge_index_s[pos_mask][:, :2].t().contiguous()
            else:
                self.pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            if torch.any(neg_mask):
                self.neg_edge_index = edge_index_s[neg_mask][:, :2].t().contiguous()
            else:
                self.neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            self.pos_edge_index = pos_edge_index.to(device=self.device, dtype=torch.long)
            self.neg_edge_index = neg_edge_index.to(device=self.device, dtype=torch.long)
        if init_emb is None:
            init_emb = create_spectral_features(pos_edge_index=self.pos_edge_index, neg_edge_index=self.neg_edge_index, node_num=self._infer_node_num(edge_index_s), dim=self.in_dim).to(self.device)
        else:
            init_emb = init_emb.to(self.device)
        self.x = init_emb

    def forward(self, init_emb: Optional[Tensor], edge_index_s: Tensor, seq_order: Optional[Tensor]=None, pos_edge_index: Optional[Tensor]=None, neg_edge_index: Optional[Tensor]=None, and_pair_index: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index)
        z = self.conv1(self.x, [self.pos_edge_index, self.neg_edge_index], seq_order=seq_order, and_pair_index=and_pair_index)
        z = F.elu(self.input_norm(z))
        z = self.srus(z)
        z = self.dropout_layer(z)
        for conv, norm in zip(self.convs, self.hidden_norms):
            h = F.elu(conv(z, [self.pos_edge_index, self.neg_edge_index], seq_order=seq_order, and_pair_index=and_pair_index))
            z = norm(h + self.lamb * z)
            z = self.dropout_layer(z)
        z = self.weight(z)
        z_clean = F.elu(self.output_norm(z))
        z_prob = self.dropout_layer(z_clean)
        prob = torch.sigmoid(self.readout_prob(z_prob))
        return (z_clean, prob)

class BiSeqGate(_BaseBiSeqGate):

    def __init__(self, args, node_num: int, device: torch.device, in_dim: int=64, out_dim: int=64, layer_num: int=2, dropout: float=0.2, lamb: float=0.5, **kwargs):
        if getattr(args, 'use_nto_sfl', None) is None:
            args.use_nto_sfl = True
        if getattr(args, 'use_dlg_sfr', None) is None:
            args.use_dlg_sfr = True
        self.use_nto_sfl = bool(args.use_nto_sfl)
        self.use_dlg_sfr = bool(args.use_dlg_sfr)
        self.dlg_sfr_last_k_layers = max(0, int(layer_num) - 1)
        super().__init__(args=args, node_num=node_num, device=device, in_dim=in_dim, out_dim=out_dim, layer_num=layer_num, dropout=dropout, lamb=lamb, **kwargs)
        if self.use_dlg_sfr:
            self._enable_dlg_sfr_on_hidden_layers(device=device)
        if not self.use_nto_sfl:
            self._replace_sequence_mixers(device=device)

    def _make_conv(self, in_dim: int, out_dim: int, use_nto_sfl: bool, use_dlg_sfr: bool, device: torch.device) -> SignedSequenceResidualConv:
        return SignedSequenceResidualConv(in_dim=in_dim, out_dim=out_dim, num_relations=2, dropout=self.dropout_layer.p, use_nto_sfl=use_nto_sfl, use_dlg_sfr=use_dlg_sfr).to(device)

    def _replace_conv(self, base_conv: SignedSequenceResidualConv, in_dim: int, out_dim: int, use_nto_sfl: bool, use_dlg_sfr: bool, device: torch.device) -> SignedSequenceResidualConv:
        new_conv = self._make_conv(in_dim=in_dim, out_dim=out_dim, use_nto_sfl=use_nto_sfl, use_dlg_sfr=use_dlg_sfr, device=device)
        new_conv.load_state_dict(base_conv.state_dict(), strict=False)
        return new_conv

    def _enable_dlg_sfr_on_hidden_layers(self, device: torch.device):
        if len(self.convs) == 0:
            return
        start_idx = max(0, len(self.convs) - self.dlg_sfr_last_k_layers)
        hidden_dim = self.weight.in_features
        for layer_idx in range(start_idx, len(self.convs)):
            base_conv = self.convs[layer_idx]
            self.convs[layer_idx] = self._replace_conv(base_conv=base_conv, in_dim=hidden_dim, out_dim=hidden_dim, use_nto_sfl=True, use_dlg_sfr=True, device=device)

    def _replace_sequence_mixers(self, device: torch.device):
        hidden_dim = self.weight.in_features
        self.conv1 = self._replace_conv(base_conv=self.conv1, in_dim=self.in_dim, out_dim=hidden_dim, use_nto_sfl=False, use_dlg_sfr=False, device=device)
        for layer_idx, base_conv in enumerate(self.convs):
            self.convs[layer_idx] = self._replace_conv(base_conv=base_conv, in_dim=hidden_dim, out_dim=hidden_dim, use_nto_sfl=False, use_dlg_sfr=bool(getattr(base_conv, 'use_dlg_sfr', False)), device=device)
