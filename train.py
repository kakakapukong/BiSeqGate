import argparse
import copy
import logging
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.seed import seed_everything
from tqdm import tqdm
from load_data import load_aig_data
from model import BiSeqGate

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PolarGate_processed')
    parser.add_argument('--task_type', type=str, default='prob', choices=['prob', 'tt'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--layer_num', type=int, default=6)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--split_file', type=str, default='0.05-0.05-0.9')
    parser.add_argument('--feature_type', type=str, default='one-hot', choices=['deepgate', 'spectral', 'one-hot'])
    parser.add_argument('--loss_type', type=str, default='mae', choices=['mae', 'mse'])
    parser.add_argument('--name_others', type=str, default='')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--seq_order', type=str, default='topo', choices=['topo', 'random', 'block_shuffled', 'interleaved_window', 'level_shuffled'])
    parser.add_argument('--seq_block_size', type=int, default=64)
    parser.add_argument('--seq_window_size', type=int, default=64)
    parser.add_argument('--use_nto_sfl', dest='use_nto_sfl', action='store_true')
    parser.add_argument('--disable_nto_sfl', dest='use_nto_sfl', action='store_false')
    parser.add_argument('--use_dlg_sfr', dest='use_dlg_sfr', action='store_true')
    parser.add_argument('--disable_dlg_sfr', dest='use_dlg_sfr', action='store_false')
    parser.set_defaults(use_nto_sfl=None, use_dlg_sfr=None)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.data_root is None:
        args.data_root = str(script_dir.parent.joinpath('AIGDataset'))
    args.data_root_path = Path(args.data_root).joinpath(args.dataset)
    args.pi_edges_path = args.data_root_path.joinpath('npz', 'pi_edges.npz')
    args.tt_pair_path = args.data_root_path.joinpath('npz', 'labels.npz')
    args.output_root = script_dir
    args.ft_saved_dir = args.output_root.joinpath('ft_saved')
    args.results_dir = args.output_root.joinpath('results')
    os.makedirs(args.ft_saved_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    if args.use_dlg_sfr is None:
        args.use_dlg_sfr = True
    if args.use_nto_sfl is None:
        args.use_nto_sfl = True
    args.residual_weight = 0.5
    args.dlg_sfr_last_k_layers = max(0, int(args.layer_num) - 1)
    args.ablation_tag = f'seq{args.seq_order}_nto_sfl{int(bool(args.use_nto_sfl))}_dlg_sfr{int(bool(args.use_dlg_sfr))}'
    if args.name_others:
        args.experiment_tag = f'{args.ablation_tag}_{args.name_others}'
    else:
        args.experiment_tag = args.ablation_tag
    artifact_stem = f'{args.task_type}_{args.dataset}_BiSeqGate_{args.layer_num}_{args.experiment_tag}'
    args.ft_model_path = args.ft_saved_dir.joinpath(f'{artifact_stem}_state_dict.pth')
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    return args

def get_logger(name, logfile=None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def remap_tensor(raw_data, data_dir):
    import json
    id_map_file = Path(data_dir).joinpath('processed', 'node_id_map.json')
    with open(id_map_file, 'r') as file:
        id_dict = json.load(file)
    new_idx = list(map(int, list(id_dict.keys())))
    return raw_data[new_idx]

def remap_edges(edge_index, data_dir):
    import json
    id_map_file = Path(data_dir).joinpath('processed', 'node_id_map.json')
    with open(id_map_file, 'r') as file:
        id_dict = json.load(file)
    new_edge_index = edge_index.clone()
    for old_pos, new_pos in id_dict.items():
        old_pos = int(old_pos)
        new_pos = int(new_pos)
        new_edge_index[edge_index == old_pos] = new_pos
    return new_edge_index

def one_hot(idx, length):
    if isinstance(idx, int):
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    return torch.zeros((len(idx), length)).scatter_(1, idx, 1)

def construct_node_feature(x, num_gate_types=3):
    gate_list = np.float32(x[:, 1])
    return one_hot(gate_list, num_gate_types)

def zero_normalization(x):
    if x.shape[0] <= 1:
        return x
    mean_x = torch.mean(x)
    std_x = torch.std(x) + 1e-08
    return (x - mean_x) / std_x

def compute_topological_order(edge_index_s: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes <= 1 or edge_index_s.numel() == 0:
        return torch.arange(num_nodes, dtype=torch.long)
    edges = edge_index_s[:, :2].detach().cpu()
    indegree = [0] * num_nodes
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in edges.tolist():
        adjacency[src].append(dst)
        indegree[dst] += 1
    queue = deque((node for node, degree in enumerate(indegree) if degree == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in adjacency[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != num_nodes:
        return torch.arange(num_nodes, dtype=torch.long)
    return torch.tensor(order, dtype=torch.long)

def _safe_positive_int(value, default_value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default_value
    return max(1, value)

def _make_seq_generator(args, edge_index_s: torch.Tensor, num_nodes: int, salt: int=0) -> torch.Generator:
    edge_cpu = edge_index_s[:, :3].detach().cpu().long() if edge_index_s.numel() > 0 else torch.zeros((0, 3), dtype=torch.long)
    if edge_cpu.numel() == 0:
        graph_hash = int(num_nodes) * 1000003
    else:
        src_sum = int(edge_cpu[:, 0].sum().item())
        dst_sum = int(edge_cpu[:, 1].sum().item())
        sign_sum = int(edge_cpu[:, 2].sum().item())
        graph_hash = int(num_nodes) * 1000003 + src_sum * 9176 + dst_sum * 131 + sign_sum * 17
    seed = (int(args.seed) * 1009 + graph_hash + int(salt) * 7919) % (2 ** 63 - 1)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return generator

def _split_into_chunks(order: torch.Tensor, chunk_size: int):
    return [order[start:start + chunk_size] for start in range(0, order.numel(), chunk_size)]

def _compute_graph_levels(edge_index_s: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes <= 1 or edge_index_s.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.long)
    edges = edge_index_s[:, :2].detach().cpu()
    indegree = [0] * num_nodes
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in edges.tolist():
        adjacency[src].append(dst)
        indegree[dst] += 1
    queue = deque((node for node, degree in enumerate(indegree) if degree == 0))
    levels = [0] * num_nodes
    visited_count = 0
    while queue:
        node = queue.popleft()
        visited_count += 1
        for nxt in adjacency[node]:
            levels[nxt] = max(levels[nxt], levels[node] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited_count != num_nodes:
        return torch.zeros(num_nodes, dtype=torch.long)
    return torch.tensor(levels, dtype=torch.long)

def build_sequence_order(args, edge_index_s: torch.Tensor, num_nodes: int) -> torch.Tensor:
    topo_order = compute_topological_order(edge_index_s, num_nodes)
    if args.seq_order == 'topo':
        return topo_order
    if args.seq_order == 'random':
        generator = _make_seq_generator(args, edge_index_s, num_nodes, salt=1)
        return torch.randperm(num_nodes, dtype=torch.long, generator=generator)
    if args.seq_order == 'level_shuffled':
        levels = _compute_graph_levels(edge_index_s, num_nodes)
        generator = _make_seq_generator(args, edge_index_s, num_nodes, salt=2)
        ordered = []
        for level in torch.unique(levels, sorted=True).tolist():
            nodes = torch.nonzero(levels == int(level), as_tuple=False).view(-1)
            if nodes.numel() > 1:
                nodes = nodes[torch.randperm(nodes.numel(), generator=generator)]
            ordered.append(nodes)
        return torch.cat(ordered, dim=0) if ordered else torch.arange(num_nodes, dtype=torch.long)
    if args.seq_order == 'block_shuffled':
        block_size = _safe_positive_int(getattr(args, 'seq_block_size', 64), 64)
        blocks = _split_into_chunks(topo_order, block_size)
        if len(blocks) <= 1:
            return topo_order
        generator = _make_seq_generator(args, edge_index_s, num_nodes, salt=3)
        perm = torch.randperm(len(blocks), generator=generator).tolist()
        return torch.cat([blocks[i] for i in perm], dim=0)
    if args.seq_order == 'interleaved_window':
        window_size = _safe_positive_int(getattr(args, 'seq_window_size', 64), 64)
        windows = _split_into_chunks(topo_order, window_size)
        if len(windows) <= 1:
            return topo_order
        interleaved = windows[0::2] + windows[1::2]
        return torch.cat(interleaved, dim=0)
    return topo_order

def build_signed_fanin_pair_index(edge_index_s: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index_s.device
    if num_nodes <= 0 or edge_index_s.numel() == 0:
        return torch.zeros((5, 0), dtype=torch.long, device=device)
    fanins = [[] for _ in range(num_nodes)]
    edges_cpu = edge_index_s[:, :3].detach().cpu().tolist()
    for src, dst, sign in edges_cpu:
        fanins[int(dst)].append((int(src), int(sign)))
    pairs = []
    for dst, items in enumerate(fanins):
        if len(items) == 2:
            items = sorted(items, key=lambda item: item[0])
            src0, sign0 = items[0]
            src1, sign1 = items[1]
            pairs.append([src0, src1, dst, sign0, sign1])
    if len(pairs) == 0:
        return torch.zeros((5, 0), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()

def load_data_signed_parallel(args, graph_dirs, pi_edges_dict, tt_pair_dict):
    total_data = []

    def process_data(data_dir):
        graph_name = os.path.basename(data_dir)
        data = load_aig_data(dataset=args.dataset, root=data_dir, train_size=0.8, val_size=0.1, test_size=0.1, data_split=1).to(args.device)
        data.to_unweighted()
        node_features_tensor = None
        if args.feature_type == 'one-hot':
            node_features = np.genfromtxt(os.path.join(data_dir, 'raw/node-feat.csv'), delimiter=',')
            node_features_tensor_d = torch.from_numpy(node_features).float()
            node_features_tensor_o = construct_node_feature(node_features_tensor_d).to(args.device)
            node_features_tensor = remap_tensor(node_features_tensor_o, data_dir)
            assert node_features_tensor.shape[0] == data.num_nodes
        edge_index = data.edge_index.t()
        edge_sign = data.edge_weight.long()
        edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)
        pos_mask = edge_sign > 0
        neg_mask = edge_sign < 0
        pos_edge_index = edge_index[pos_mask].t().contiguous() if torch.any(pos_mask) else torch.zeros((2, 0), dtype=torch.long, device=args.device)
        neg_edge_index = edge_index[neg_mask].t().contiguous() if torch.any(neg_mask) else torch.zeros((2, 0), dtype=torch.long, device=args.device)
        not_edges = edge_index_s[neg_mask][:, :2].contiguous()
        and_edges = edge_index_s[torch.argsort(edge_index_s[:, 1])]
        and_edges = and_edges[and_edges[:, 2] == 1][:, :2].contiguous()
        and_pair_index = build_signed_fanin_pair_index(edge_index_s, data.num_nodes)
        pi_edges_list = list(pi_edges_dict[graph_name].values())
        pi_edges_signed = np.concatenate(pi_edges_list, axis=0)
        pi_edges_signed_tensor = torch.from_numpy(pi_edges_signed).long()
        pi_edges_weight = pi_edges_signed_tensor[:, 2].unsqueeze(1)
        new_pi_edges = remap_edges(pi_edges_signed_tensor[:, :2], data_dir)
        pi_edges_signed_tensor = torch.cat([new_pi_edges, pi_edges_weight], dim=1).to(args.device)
        tt_pair_index = tt_pair_dict[graph_name]['tt_pair_index']
        tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long, device=args.device)
        tt_pair_index_tensor = remap_edges(tt_pair_index, data_dir).t().contiguous()
        tt_dis_tensor = torch.tensor(tt_pair_dict[graph_name]['tt_dis'], dtype=torch.float32, device=args.device)
        node_labels = np.genfromtxt(os.path.join(data_dir, 'raw/prob.csv'), delimiter=None)
        node_labels_tensor = torch.from_numpy(node_labels).float().to(args.device)
        node_labels_tensor = remap_tensor(node_labels_tensor, data_dir)
        node_labels_tensor_2d = node_labels_tensor.unsqueeze(1)
        pi_and_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=args.device)
        if neg_edge_index.numel() > 0:
            pi_and_mask[neg_edge_index[1]] = False
        seq_order = build_sequence_order(args, edge_index_s, data.num_nodes).to(args.device)
        total_data.append({'data_dir': data_dir, 'edge_index_s': edge_index_s, 'pos_edge_index': pos_edge_index, 'neg_edge_index': neg_edge_index, 'pi_edges_signed_tensor': pi_edges_signed_tensor, 'tt_pair_index_tensor': tt_pair_index_tensor, 'node_features_tensor': node_features_tensor, 'node_labels_tensor': node_labels_tensor, 'node_labels_tensor_2d': node_labels_tensor_2d, 'tt_dis_tensor': tt_dis_tensor, 'not_edges': not_edges, 'and_edges': and_edges, 'and_pair_index': and_pair_index, 'pi_and_mask': pi_and_mask, 'seq_order': seq_order})
    with ThreadPoolExecutor(max_workers=max(1, int(args.num_workers))) as executor:
        list(tqdm(executor.map(process_data, graph_dirs), total=len(graph_dirs), desc='Processing Graphs'))
    return total_data

def parse_data_parallel(args):
    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'train.txt')) as file:
        lines = file.readlines()
    train_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(train_file)
    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'valid.txt')) as file:
        lines = file.readlines()
    valid_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(valid_file)
    with open(os.path.join(args.data_root_path, 'split', args.split_file, 'test.txt')) as file:
        lines = file.readlines()
    test_file = [os.path.join(args.data_root_path, line.strip()) for line in lines]
    random.shuffle(test_file)
    pi_edges_dict = np.load(args.pi_edges_path, allow_pickle=True)['pi_edges'].item()
    tt_pair_dict = np.load(args.tt_pair_path, allow_pickle=True)['labels'].item()
    train_data = load_data_signed_parallel(args, train_file, pi_edges_dict, tt_pair_dict)
    valid_data = load_data_signed_parallel(args, valid_file, pi_edges_dict, tt_pair_dict)
    test_data = load_data_signed_parallel(args, test_file, pi_edges_dict, tt_pair_dict)
    return (train_data, valid_data, test_data)

def forward_sample(model, sample):
    return model(sample['node_features_tensor'], sample['edge_index_s'], seq_order=sample['seq_order'], pos_edge_index=sample.get('pos_edge_index'), neg_edge_index=sample.get('neg_edge_index'), and_pair_index=sample.get('and_pair_index'))

def compute_tt_distance(out_emb, tt_pair_index_tensor):
    node_a = out_emb[tt_pair_index_tensor[0]]
    node_b = out_emb[tt_pair_index_tensor[1]]
    return 1 - torch.cosine_similarity(node_a, node_b, eps=1e-08)

def compute_supervised_losses(args, out_emb, out, sample, need_prob=True, need_tt=True):
    prob_loss = None
    tt_loss = None
    if need_prob:
        node_labels_tensor = sample.get('node_labels_tensor_2d')
        if node_labels_tensor is None:
            node_labels_tensor = sample['node_labels_tensor'].unsqueeze(1)
        if args.loss_type == 'mae':
            prob_loss = F.l1_loss(out, node_labels_tensor)
        else:
            prob_loss = F.mse_loss(out, node_labels_tensor)
    if need_tt:
        emb_dis = compute_tt_distance(out_emb, sample['tt_pair_index_tensor'])
        tt_target = sample['tt_dis_tensor']
        emb_dis_z = zero_normalization(emb_dis)
        tt_dis_z = zero_normalization(tt_target)
        if args.loss_type == 'mae':
            tt_loss = F.l1_loss(emb_dis_z, tt_dis_z)
        else:
            tt_loss = F.mse_loss(emb_dis_z, tt_dis_z)
    return (prob_loss, tt_loss)

def test(args, model, test_data):
    model.eval()
    results = {'prob': 0.0, 'tt': 0.0}
    with torch.no_grad():
        for sample in test_data:
            out_emb, out = forward_sample(model, sample)
            prob_loss, tt_loss = compute_supervised_losses(args, out_emb, out, sample, need_prob=True, need_tt=True)
            results['prob'] += float(prob_loss.item())
            results['tt'] += float(tt_loss.item())
    num_samples = max(1, len(test_data))
    results['prob'] /= num_samples
    results['tt'] /= num_samples
    return results

def train(args, model, optimizer, train_data, valid_data, logger):
    logger.info('*********** Start End-to-End Train ***********')
    best_loss = float('inf')
    patience = args.patience
    total_time = 0.0
    cnt_epoch = 0
    best_model_info = None
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_prob_loss = 0.0
        total_tt_loss = 0.0
        random.shuffle(train_data)
        accum_count = 0
        for idx, sample in enumerate(train_data):
            out_emb, out = forward_sample(model, sample)
            prob_loss, tt_loss = compute_supervised_losses(args, out_emb, out, sample, need_prob=True, need_tt=True)
            if args.task_type == 'prob':
                loss = prob_loss
            else:
                loss = tt_loss
            total_prob_loss += float(prob_loss.item())
            total_tt_loss += float(tt_loss.item())
            loss.backward()
            accum_count += 1
            if (idx + 1) % args.batch_size == 0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accum_count)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
        if accum_count > 0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accum_count)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_time += time.time() - t
        cnt_epoch += 1
        valid_results = test(args, model, valid_data)
        total_prob_loss /= max(1, len(train_data))
        total_tt_loss /= max(1, len(train_data))
        logger.info('Epoch: {:02d} | [Train] Prob: {:.4f} Func: {:.4f} | [Valid] Prob: {:.4f} Func: {:.4f}'.format(epoch, total_prob_loss, total_tt_loss, valid_results['prob'], valid_results['tt']))
        valid_metric = valid_results['prob'] if args.task_type == 'prob' else valid_results['tt']
        if valid_metric < best_loss:
            best_loss = valid_metric
            best_model_info = {'model_state_dict': copy.deepcopy(model.state_dict()), 'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()), 'args': vars(args)}
            patience = args.patience
        patience -= 1
        if patience <= 0:
            break
    if best_model_info is not None:
        torch.save(best_model_info, args.ft_model_path)
    return total_time / max(1, cnt_epoch)

def load_model(args):
    model = BiSeqGate(args=args, node_num=0, device=args.device, in_dim=args.in_dim, out_dim=args.out_dim, layer_num=args.layer_num, dropout=args.dropout, lamb=args.residual_weight).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return (model, optimizer)
if __name__ == '__main__':
    args = parameter_parser()
    seed_everything(args.seed)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    logname = f'{args.task_type}_{args.dataset}_BiSeqGate_{args.layer_num}__{args.split_file}_{args.experiment_tag}_{timestamp}.log'
    logfile = str(args.results_dir.joinpath(logname))
    logger = get_logger(__name__, logfile=logfile)
    train_data, valid_data, test_data = parse_data_parallel(args)
    model, optimizer = load_model(args)
    avg_train_time = train(args, model, optimizer, train_data, valid_data, logger)
    del model
    model, optimizer = load_model(args)
    checkpoint = torch.load(args.ft_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = test(args, model, test_data)
    logger.info('*********** Test Result ***********')
    logger.info('[Test] Prob: {:.4f} Func: {:.4f} | Avg.Train_time: {:.4f}'.format(test_results['prob'], test_results['tt'], avg_train_time))
