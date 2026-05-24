# BiSeqGate

## 1. Abstract

As integrated-circuit technologies continue to evolve toward higher complexity and larger scale, modern Electronic Design Automation (EDA) workflows increasingly require fine-grained AND-Inverter Graph (AIG)-based representation and analysis techniques for logic synthesis, equivalence checking, and functional verification. However, practical AIGs often contain long logic chains, high-fanout propagation paths, and signed positive/negative connections, making it difficult for existing methods to capture path-sensitive ordered dependencies while preserving discriminative logical semantics during deep propagation.

To address these challenges, we propose BiSeqGate, a topology-aware spatial-sequential AIG representation learning framework that effectively captures long-range ordered dependencies along logic-topological signal propagation while preserving discriminative semantics associated with signed logical relations during deep aggregation. Specifically, BiSeqGate incorporates a bidirectional sequence mixing mechanism based on node topological order, enabling the model to characterize sequential signal propagation from primary inputs to outputs and to capture the inherent long-range path dependencies and topological ordering of circuit behavior.

In addition, BiSeqGate introduces an incremental graph semantic enhancement strategy that separately aggregates positive and negative relations and explicitly injects their semantic discrepancy into node representations. This design preserves the logical-polarity distinction between direct and inverted propagation during multi-layer feature propagation, thereby alleviating discriminative semantic degradation and feature homogenization while improving the model's ability to capture logic circuit semantics through bidirectional sequence-aware graph learning.

We conduct comprehensive experiments on 9,933 valid subcircuit samples collected from logic synthesis, circuit testing, and hardware design benchmarks. Experimental results demonstrate that the proposed method consistently outperforms state-of-the-art methods.

## 2. File Structure

```text
BiSeqGate/
|-- README.md
|-- requirements.txt
|-- layers.py
|-- load_data.py
|-- preprocess_data.py
|-- model.py
|-- train.py
`-- AIGDataset/
    |-- PolarGate_raw/
    |   |-- npz/
    |   |   |-- graphs.npz
    |   |   `-- labels.npz
    |   `-- bench/
    |       |-- <circuit_name>.bench
    |       `-- ...
    `-- PolarGate_processed/
        |-- npz/
        |   |-- labels.npz
        |   `-- pi_edges.npz
        |-- split/
        |   `-- 0.05-0.05-0.9/
        |       |-- train.txt
        |       |-- valid.txt
        |       `-- test.txt
        `-- <circuit_name>/
            |-- raw/
            |   |-- node-feat.csv
            |   |-- signed_edge.csv
            |   `-- prob.csv
            `-- processed/
                |-- data.pt
                `-- node_id_map.json
```

## 3. Environment Setup

- **GPU:** NVIDIA A800 * 1
- **CUDA Version:** 11.8.0
- **OS:** Ubuntu 22.04.5 LTS

### Conda Environment

```bash
conda create -n biseqgate python=3.8.20
conda activate biseqgate
pip install -r requirements.txt
```

## 4. Running Commands

Preprocess data:

```bash
python preprocess_data.py \
  --base_dir ./ \
  --split_ratio 0.05 0.05 0.9
```

Train on SPP:

```bash
python train.py \
  --dataset PolarGate_processed \
  --task_type prob \
  --data_root ./AIGDataset \
  --split_file 0.05-0.05-0.9 \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 6 \
  --dropout 0.05 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --seq_order topo \
  --device 0
```

Train on TTDP:

```bash
python train.py \
  --dataset PolarGate_processed \
  --task_type tt \
  --data_root ./AIGDataset \
  --split_file 0.05-0.05-0.9 \
  --feature_type one-hot \
  --in_dim 3 \
  --out_dim 256 \
  --layer_num 6 \
  --dropout 0.05 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --seq_order topo \
  --device 0
```

If CUDA is not available, set `--device -1`.
