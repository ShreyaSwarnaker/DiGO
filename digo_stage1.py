import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import Dataset
from torch_scatter import scatter_mean
from collections import Counter
from tqdm import tqdm
import multiprocessing
import gc
from time import time

torch.backends.cudnn.benchmark = True

TRAIN_CSV = "path_to_train_data.csv"
VAL_CSV   = "path_to_val_data.csv"
TEST_CSV  = "path_to_test_data.csv"
IC_CSV    = "path_to_ic.txt"

SEQ_COL = 'sequence'
ID      = 'ID'

OUT_DIR = "path_to_output_directory"
os.makedirs(OUT_DIR, exist_ok=True)

AMINO_ACIDS = ['A','C','D','E','F','G','H','I','K','L',
               'M','N','P','Q','R','S','T','V','W','Y']
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

FEATURE_MODE      = 'statistics'
MAX_POSITION_BINS = 100
MAX_RES_LENGTH    = 2000

EMBEDDING_DIM = 64
HIDDEN_DIM    = 256
ESM_LAYERS    = [20, 27, 33]   # late-stage layers
LABEL_SMOOTH  = 0.05
DROPOUT       = 0.4
USE_GCN_IN_CLASSIFIER = True   

ALPHA      = 0.5
BETA       = 0.5
SELF_LOOPS = True
EPOCHS     = 200
LR         = 0.0005
MC_SAMPLES = 5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED       = 456

BATCH_SIZE    = 32
GRAPH_VERSION = "graph_version"    

#Uncomment below whichever branch is being evaluated
MF_ROOT = "GO:0003674"
"""
CC_ROOT = "GO:0005575"
BP_ROOT = "GO:0008150"
"""

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===========================================================================
# SUPERMAGO evaluation  (IDENTICAL to esm7 / esm15)
# ===========================================================================

def get_ancestors(ontology, term):
    list_of_terms = [term]
    data = []
    while list_of_terms:
        new_term = list_of_terms.pop(0)
        if new_term not in ontology:
            break
        data.append(new_term)
        for parent_term in ontology[new_term]['parents']:
            if parent_term in ontology:
                list_of_terms.append(parent_term)
    return data


def generate_ontology(file, specific_space=False, name_specific_space=''):
    ontology = {}
    gene, flag = {}, False
    with open(file) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if line == '[Term]':
                if 'id' in gene:
                    ontology[gene['id']] = gene
                gene = {}
                gene['parents'], gene['alt_ids'] = [], []
                flag = True
            elif line == '[Typedef]':
                flag = False
            else:
                if not flag:
                    continue
                items = line.split(': ')
                if items[0] == 'id':
                    gene['id'] = items[1]
                elif items[0] == 'alt_id':
                    gene['alt_ids'].append(items[1])
                elif items[0] == 'namespace':
                    if specific_space:
                        if name_specific_space == items[1]:
                            gene['namespace'] = items[1]
                        else:
                            gene = {}; flag = False
                    else:
                        gene['namespace'] = items[1]
                elif items[0] == 'is_a':
                    gene['parents'].append(items[1].split(' ! ')[0])
                elif items[0] == 'name':
                    gene['name'] = items[1]
                elif items[0] == 'is_obsolete':
                    gene = {}; flag = False
    key_list = list(ontology.keys())
    for key in key_list:
        ontology[key]['ancestors'] = get_ancestors(ontology, key)
        for alt_id in ontology[key]['alt_ids']:
            ontology[alt_id] = ontology[key]
    for key, value in ontology.items():
        if 'children' not in value:
            value['children'] = []
        for p_id in value['parents']:
            if p_id in ontology:
                if 'children' not in ontology[p_id]:
                    ontology[p_id]['children'] = []
                ontology[p_id]['children'].append(key)
    return ontology


def propagate_preds(predictions, ontologies_names, ontology):
    ont_n = ontologies_names.tolist()
    list_of_parents = []
    for idx_term in range(len(ont_n)):
        this_list = []
        term = ont_n[idx_term]
        if term in ontology:
            for parent in ontology[term]['ancestors']:
                if parent in ont_n:
                    this_list.append(ont_n.index(parent))
        list_of_parents.append(list(set(this_list)))
    for idx_protein in tqdm(range(len(predictions)), desc="Propagating predictions"):
        for idx_term in range(len(ont_n)):
            for idx_parent in list_of_parents[idx_term]:
                predictions[idx_protein, idx_parent] = max(
                    predictions[idx_protein, idx_parent],
                    predictions[idx_protein, idx_term])
    return predictions


def compute_metrics(y_true, y_prob, ic_dict, ontologies_names,
                    ontology, root=BP_ROOT, propagate=True):
    y_prob = y_prob.copy().astype(np.float32)
    y_true = y_true.astype(np.int32)
    if propagate:
        y_prob = propagate_preds(y_prob, ontologies_names, ontology)
    ont_n = ontologies_names.tolist()
    N = len(y_prob)
    wfmax = fmax = fmax_s = 0.0
    smin = 1e100
    pr_arr, rc_arr = [], []
    for tau in tqdm(np.linspace(0, 1, 101), desc="Evaluating thresholds"):
        wpr, wrc, num_prot_w = 0.0, 0.0, 0
        pr_s, rc_s, num_prot_s = 0.0, 0.0, 0
        pr_n, rc_n, num_prot_n = 0.0, 0.0, 0
        ru, mi = 0.0, 0.0
        for i in range(N):
            protein_pred = set(np.array(ont_n)[y_prob[i] >= tau].tolist())
            protein_gt   = set(np.array(ont_n)[y_true[i] == 1].tolist())
            if len(protein_gt) == 0:
                continue
            ic_pred      = sum(ic_dict.get(q, 0.0) for q in protein_pred)
            ic_gt        = sum(ic_dict.get(q, 0.0) for q in protein_gt)
            ic_intersect = sum(ic_dict.get(q, 0.0) for q in protein_pred & protein_gt)
            if ic_pred > 0:
                num_prot_w += 1; wpr += ic_intersect / ic_pred
            if ic_gt > 0:
                wrc += ic_intersect / ic_gt
            if len(protein_pred) > 0:
                num_prot_n += 1
                pr_n += len(protein_pred & protein_gt) / len(protein_pred)
            rc_n += len(protein_pred & protein_gt) / len(protein_gt)
            tp_set = protein_pred & protein_gt
            for go_id in protein_pred - tp_set: mi += ic_dict.get(go_id, 0.0)
            for go_id in protein_gt   - tp_set: ru += ic_dict.get(go_id, 0.0)
            pred_nr = protein_pred - {root}; gt_nr = protein_gt - {root}
            if len(pred_nr) > 0:
                num_prot_s += 1; pr_s += len(pred_nr & gt_nr) / len(pred_nr)
            if len(gt_nr) > 0:
                rc_s += len(pred_nr & gt_nr) / len(gt_nr)
        tau_wpr = wpr / num_prot_w if num_prot_w > 0 else 0.0
        tau_wrc = wrc / N
        if tau_wpr + tau_wrc > 0:
            wfmax = max(wfmax, 2*tau_wpr*tau_wrc/(tau_wpr+tau_wrc))
        tau_pr_n = pr_n / num_prot_n if num_prot_n > 0 else 0.0
        tau_rc_n = rc_n / N
        if tau_pr_n + tau_rc_n > 0:
            fmax = max(fmax, 2*tau_pr_n*tau_rc_n/(tau_pr_n+tau_rc_n))
        pr_arr.append(tau_pr_n); rc_arr.append(tau_rc_n)
        ru = ru/N; mi = mi/N
        smin = min(smin, math.sqrt(ru**2 + mi**2))
        tau_pr_s = pr_s / num_prot_s if num_prot_s > 0 else 0.0
        tau_rc_s = rc_s / N
        if tau_pr_s + tau_rc_s > 0:
            fmax_s = max(fmax_s, 2*tau_pr_s*tau_rc_s/(tau_pr_s+tau_rc_s))
    pr_arr = np.array(pr_arr); rc_arr = np.array(rc_arr)
    si = np.argsort(rc_arr)
    rc_s2 = rc_arr[si]; pr_s2 = pr_arr[si]
    auprc = float(np.trapz(pr_s2, rc_s2))
    ipr_arr, irc_arr = [], []
    for tau in np.linspace(0, 1, 101):
        idx = np.where(rc_s2 >= tau)[0]
        if len(idx) > 0:
            irc_arr.append(tau); ipr_arr.append(float(np.max(pr_s2[idx])))
    iauprc = float(np.trapz(ipr_arr, irc_arr)) if len(irc_arr) > 1 else 0.0
    return {"Fmax": fmax, "Fmax_star": fmax_s, "wFmax": wfmax, "Smin": smin,
            "AUPRC": auprc, "IAuPRC": iauprc, "AUPRC_micro": auprc}


# ===========================================================================
# Data loading  (IDENTICAL to esm15)
# ===========================================================================

def load_csv_multilabel_data(file_path, seq_col=SEQ_COL, id_col=ID):
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}  shape={df.shape}")
    sequences = [s.strip().upper() for s in df[seq_col].astype(str).tolist()]
    label_cols = [c for c in df.columns if c != seq_col and c != id_col]
    labels = df[label_cols].values.astype(np.float32)
    valid_idx = [i for i, seq in enumerate(sequences) if len(seq) > 0]
    sequences = [sequences[i] for i in valid_idx]
    labels    = labels[valid_idx]
    print(f"  Valid: {len(sequences)}  Labels: {labels.shape[1]}  Sparsity: {labels.mean():.4f}")
    return sequences, labels, df


# ===========================================================================
# Physicochemical features  (IDENTICAL to esm15)
# ===========================================================================

PHYSCHEM = {
    'A': [0.0,  1.8, 0.0,  88.6, 0.36, 0.0, 0.0],
    'C': [0.0,  2.5, 0.0, 108.5, 0.35, 0.0, 1.0],
    'D': [-1.0,-3.5, 1.0, 111.1, 0.51, 0.0, 1.0],
    'E': [-1.0,-3.5, 1.0, 138.4, 0.50, 0.0, 1.0],
    'F': [0.0,  2.8, 0.0, 189.9, 0.31, 1.0, 0.0],
    'G': [0.0, -0.4, 0.0,  60.1, 0.54, 0.0, 0.0],
    'H': [0.5, -3.2, 1.0, 153.2, 0.32, 1.0, 1.0],
    'I': [0.0,  4.5, 0.0, 166.7, 0.46, 0.0, 0.0],
    'K': [1.0, -3.9, 1.0, 168.6, 0.47, 0.0, 1.0],
    'L': [0.0,  3.8, 0.0, 166.7, 0.45, 0.0, 0.0],
    'M': [0.0,  1.9, 0.0, 162.9, 0.36, 0.0, 0.0],
    'N': [0.0, -3.5, 1.0, 114.1, 0.46, 0.0, 1.0],
    'P': [0.0, -1.6, 0.0, 112.7, 0.51, 0.0, 0.0],
    'Q': [0.0, -3.5, 1.0, 143.8, 0.49, 0.0, 1.0],
    'R': [1.0, -4.5, 1.0, 173.4, 0.53, 0.0, 1.0],
    'S': [0.0, -0.8, 1.0,  89.0, 0.51, 0.0, 1.0],
    'T': [0.0, -0.7, 1.0, 116.1, 0.44, 0.0, 1.0],
    'V': [0.0,  4.2, 0.0, 140.0, 0.40, 0.0, 0.0],
    'W': [0.0, -0.9, 0.0, 227.8, 0.31, 1.0, 1.0],
    'Y': [0.0, -1.3, 1.0, 193.6, 0.42, 1.0, 1.0],
}
_raw  = torch.tensor([PHYSCHEM[aa] for aa in AMINO_ACIDS], dtype=torch.float32)
_mean = _raw.mean(dim=0)
_std  = _raw.std(dim=0); _std[_std == 0] = 1.0
PHYSCHEM_MATRIX = (_raw - _mean) / _std

print("Using device:", DEVICE)
print("GPU available:", torch.cuda.is_available())

esm_model      = None
batch_converter = None


# ===========================================================================
# ESM multi-layer precomputation  (IDENTICAL to esm15)
# ===========================================================================

def precompute_esm_embeddings_multilayer(seqs, save_path, layers=ESM_LAYERS, batch_size=8):
    if os.path.exists(save_path):
        print(f"  Found cached: {save_path}")
        return
    import threading
    num_gpus = torch.cuda.device_count()
    usable_gpus = [i for i in range(num_gpus)
                   if (torch.cuda.get_device_properties(i).total_memory
                       - torch.cuda.memory_reserved(i)) / 1e9 > 8.0]
    if not usable_gpus:
        usable_gpus = [0]
    print(f"  Usable GPUs: {usable_gpus}  Layers: {layers}")
    chunk_size = (len(seqs) + len(usable_gpus) - 1) // len(usable_gpus)
    chunks  = [seqs[i:i+chunk_size] for i in range(0, len(seqs), chunk_size)]
    results = [None] * len(chunks)
    errors  = [None] * len(chunks)

    def process_chunk(gpu_id, chunk, idx):
        import esm
        device = torch.device(f'cuda:{gpu_id}')
        try:
            m, alpha = esm.pretrained.esm2_t33_650M_UR50D()
            m = m.to(device).eval()
            bc = alpha.get_batch_converter()
            embs = []
            with torch.inference_mode():
                for i in range(0, len(chunk), batch_size):
                    batch = [("protein", s[:1022]) for s in chunk[i:i+batch_size]]
                    _, _, tokens = bc(batch)
                    tokens = tokens.to(device)
                    out  = m(tokens, repr_layers=layers, return_contacts=False)
                    layer_vecs = [out["representations"][l][:, 1:-1, :].mean(dim=1)
                                  for l in layers]
                    embs.append(torch.cat(layer_vecs, dim=1).cpu())
                    if i % 200 == 0:
                        print(f"  GPU {gpu_id}: {i}/{len(chunk)}")
            results[idx] = torch.cat(embs, dim=0)
            torch.cuda.empty_cache()
        except Exception as e:
            errors[idx] = e
            print(f"  GPU {gpu_id} failed: {e}")

    threads = []
    for t_idx, (gpu_id, chunk) in enumerate(zip(usable_gpus, chunks)):
        t = threading.Thread(target=process_chunk, args=(gpu_id, chunk, t_idx))
        threads.append(t); t.start()
    for t in threads:
        t.join()
    for idx, err in enumerate(errors):
        if err is not None:
            process_chunk(usable_gpus[0], chunks[idx], idx)
    embeddings = torch.cat(results, dim=0)
    torch.save(embeddings, save_path)
    print(f"  Saved: {save_path}  shape={embeddings.shape}")


# ===========================================================================
# GO adjacency  (IDENTICAL to esm15; unused in flat mode)
# ===========================================================================

def build_go_adjacency(label_names, go_obo_path):
    go_parents = {}
    current_id = None
    with open(go_obo_path) as f:
        for line in f:
            line = line.strip()
            if line == '[Term]': current_id = None
            elif line.startswith('id: GO:'): current_id = line.split('id: ')[1]
            elif line.startswith('is_a:') and current_id:
                parent = line.split('is_a: ')[1].split(' ')[0]
                go_parents.setdefault(current_id, []).append(parent)
    n = len(label_names)
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    adj = torch.zeros(n, n)
    for i, label in enumerate(label_names):
        for parent in go_parents.get(label, []):
            if parent in label_to_idx:
                adj[i, label_to_idx[parent]] = 1.0
    return adj


# ===========================================================================
# Model components  (IDENTICAL to esm15)
# ===========================================================================

class GOHierarchyLayer(torch.nn.Module):
    """IDENTICAL to esm7/esm15."""
    def __init__(self, num_classes, go_adj_matrix):
        super().__init__()
        self.register_buffer('adj', go_adj_matrix)
        self.gcn = torch.nn.Linear(num_classes, num_classes, bias=False)

    def forward(self, logits):
        return logits + self.gcn(torch.sigmoid(logits) @ self.adj)


class AATransformerLayer(torch.nn.Module):
    """IDENTICAL to esm7/esm15."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn  = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm  = torch.nn.LayerNorm(dim)
        self.ff    = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2), torch.nn.ReLU(), torch.nn.Linear(dim * 2, dim))
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ===========================================================================
# Feature extraction  (IDENTICAL to esm15)
# ===========================================================================

def max_run_length(sequence, aa):
    max_run = current = 0
    for c in sequence:
        if c == aa: current += 1; max_run = max(max_run, current)
        else: current = 0
    return max_run

def mean_inter_gap(positions):
    if len(positions) < 2: return 0.0
    return float(np.mean(np.diff(positions)))

def get_high_freq_tripeptides(sequence, percentile=90):
    tripeptides = [sequence[i:i+3] for i in range(len(sequence) - 2)]
    counts = Counter(tripeptides)
    if not counts: return set()
    threshold = np.percentile(list(counts.values()), percentile)
    return {tp for tp, c in counts.items() if c >= threshold}

def neighborhood_aa_frequencies(sequence, positions, k=2):
    if len(positions) == 0: return np.zeros(20, dtype=np.float32)
    counts = np.zeros(20, dtype=np.float32); total = 0
    L = len(sequence)
    for pos in positions:
        for i in range(max(0, pos - k), min(L, pos + k + 1)):
            if i == pos: continue
            aa = sequence[i]
            if aa in AA_TO_INDEX: counts[AA_TO_INDEX[aa]] += 1; total += 1
    if total > 0: counts /= total
    return counts

def sequence_to_amino_acid_nodes_statistics(sequence, k=2):
    L = len(sequence)
    node_features = np.zeros((20, 30), dtype=np.float32)
    positions_dict = {aa: [] for aa in AMINO_ACIDS}
    for i, aa in enumerate(sequence):
        if aa in positions_dict: positions_dict[aa].append(i)
    dipeptides = [sequence[i:i+2] for i in range(L - 1)]
    hft = get_high_freq_tripeptides(sequence)
    for aa in AMINO_ACIDS:
        idx = AA_TO_INDEX[aa]; positions = positions_dict[aa]; count = len(positions)
        if count == 0: continue
        positions = np.array(positions, dtype=np.float32)
        norm_pos  = positions / max(L - 1, 1)
        node_features[idx, 0]  = count / L
        node_features[idx, 1]  = norm_pos.mean()
        node_features[idx, 2]  = norm_pos.std()
        node_features[idx, 3]  = mean_inter_gap(positions) / L
        node_features[idx, 4]  = max_run_length(sequence, aa) / L
        node_features[idx, 5]  = np.sum(norm_pos <= 0.2) / count
        node_features[idx, 6]  = np.sum(norm_pos >= 0.8) / count
        first  = sum(dp[0] == aa for dp in dipeptides)
        second = sum(dp[1] == aa for dp in dipeptides)
        total_dp = max(len(dipeptides), 1)
        node_features[idx, 7]  = first  / total_dp
        node_features[idx, 8]  = second / total_dp
        hf_hits = 0
        for pos in positions.astype(int):
            for start in (pos - 2, pos - 1, pos):
                if 0 <= start <= L - 3 and sequence[start:start+3] in hft:
                    hf_hits += 1; break
        node_features[idx, 9]     = hf_hits / count
        node_features[idx, 10:30] = neighborhood_aa_frequencies(sequence, positions.astype(int), k=k)
    return node_features

def sequence_to_amino_acid_nodes(sequence, mode=FEATURE_MODE):
    if mode == 'statistics': return sequence_to_amino_acid_nodes_statistics(sequence)
    raise ValueError(f"Unknown mode: {mode}")

def get_feature_dim(mode=FEATURE_MODE):
    if mode == 'statistics': return 37
    raise ValueError(f"Unknown mode: {mode}")


# ===========================================================================
# Residue graph  (IDENTICAL to esm15)
# ===========================================================================

def build_residue_graph(sequence, pe_dim=EMBEDDING_DIM):
    sequence = sequence[:MAX_RES_LENGTH]; L = len(sequence)
    position = torch.arange(L, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / pe_dim))
    pe = torch.zeros(L, pe_dim)
    pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
    pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
    aa_onehot = torch.zeros(L, 20, dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INDEX: aa_onehot[i, AA_TO_INDEX[aa]] = 1.0
    return torch.cat([pe, aa_onehot], dim=1), None


class ResidueProteinEncoder(torch.nn.Module):
    def __init__(self, dim, in_dim):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.LayerNorm(dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.LayerNorm(dim))

    def forward(self, res_x, res_batch):
        res_x = self.input_proj(res_x)
        return scatter_mean(self.mlp(res_x), res_batch, dim=0)


# ===========================================================================
# Graph building  (IDENTICAL to esm15)
# ===========================================================================

from torch_geometric.data import Data

def build_protein_graph(sequence, label):
    x = torch.tensor(sequence_to_amino_acid_nodes(sequence), dtype=torch.float)
    x = torch.cat([x, PHYSCHEM_MATRIX], dim=1)
    edges = []
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i+1]
        if a in AA_TO_INDEX and b in AA_TO_INDEX:
            edges.append([AA_TO_INDEX[a], AA_TO_INDEX[b]])
    if not edges: edges = [[i, i] for i in range(20)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor(label, dtype=torch.float).unsqueeze(0)
    res_x, _ = build_residue_graph(sequence)
    res_batch = torch.zeros(res_x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, res_x=res_x, res_batch=res_batch, y=y)

def build_graph_worker(args):
    seq, label = args
    g = build_protein_graph(seq, label)
    return {'x': g.x.numpy(), 'edge_index': g.edge_index.numpy(),
            'res_x': g.res_x.numpy(), 'res_batch': g.res_batch.numpy(), 'y': g.y.numpy()}

def build_graph_dataset_parallel(seqs, labels, num_workers=32):
    print(f"  Building graphs using {num_workers} CPU cores...")
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(num_workers, maxtasksperchild=200) as pool:
        raw = pool.map(build_graph_worker, list(zip(seqs, labels)))
    return [Data(x=torch.from_numpy(r['x']), edge_index=torch.from_numpy(r['edge_index']),
                 res_x=torch.from_numpy(r['res_x']), res_batch=torch.from_numpy(r['res_batch']),
                 y=torch.from_numpy(r['y'])) for r in raw]


class ProteinDataset(Dataset):
    def __init__(self, graphs, esm_embeddings):
        self.graphs         = graphs
        self.esm_embeddings = esm_embeddings

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.esm_emb = self.esm_embeddings[idx].unsqueeze(0)
        return data


from torch_geometric.loader import DataLoader


# ===========================================================================
# GCN  (IDENTICAL to esm15)
# ===========================================================================

class DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0,
                 self_loops=True, adaptive=False):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.alpha = alpha; self.beta = beta
        self.self_loops = self_loops; self.adaptive = adaptive
        if adaptive:
            self.alpha_param = torch.nn.Parameter(torch.tensor(alpha))
            self.beta_param  = torch.nn.Parameter(torch.tensor(beta))

    def forward(self, x, edge_index):
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        alpha = self.alpha_param if self.adaptive else self.alpha
        beta  = self.beta_param  if self.adaptive else self.beta
        norm  = alpha * deg_inv_sqrt[row] + beta * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, alpha, beta, self_loops):
        super().__init__()
        self.conv1 = DirectedGCNConv(in_ch,  hid_ch, alpha, beta, self_loops)
        self.conv2 = DirectedGCNConv(hid_ch, hid_ch, alpha, beta, self_loops)
        self.conv3 = DirectedGCNConv(hid_ch, out_ch,  alpha, beta, self_loops)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)


class DirectedGCNConvEncoder(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, alpha, beta, self_loops, adaptive=False):
        super().__init__()
        self.source_conv = GCNEncoder(in_ch, hid_ch, out_ch, alpha, beta, self_loops)
        self.target_conv = GCNEncoder(in_ch, hid_ch, out_ch, alpha, beta, self_loops)

    def forward(self, x, edge_index):
        return self.source_conv(x, edge_index), self.target_conv(x, edge_index)


# ===========================================================================
# DiGAEClassifier
#
# CHANGE 1: USE_GCN_IN_CLASSIFIER = True  ->  clf_in uses
#   [z_aa_mod | z_res_mod | z_esm_raw]  (gcn_flat + embedding_dim + esm_raw_dim)
#   instead of esm15's [z_esm_raw | z_res_mod] only.
#
# CHANGE 2: esm_direct branch REMOVED.
#   esm15:  logits = self.classifier(clf_in) + self.esm_direct(z_esm_raw)
#   here:   logits = self.classifier(clf_in)          <-  l_hat_main only
# ===========================================================================

class DiGAEClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim, num_classes,
                 alpha=1.0, beta=0.0, self_loops=True, adaptive=False,
                 num_aa=20, go_adj=None, esm_raw_dim=3840):
        super().__init__()

        self.res_protein_encoder = ResidueProteinEncoder(
            dim=embedding_dim, in_dim=embedding_dim + 20)
        self.encoder = DirectedGCNConvEncoder(
            in_channels, hidden_channels, embedding_dim, alpha, beta, self_loops, adaptive)
        self.go_hierarchy = GOHierarchyLayer(num_classes, go_adj) if go_adj is not None else None
        self.num_aa       = num_aa
        self.res_norm     = torch.nn.LayerNorm(embedding_dim)
        self.current_epoch = 0
        self.aa_transformer = AATransformerLayer(dim=embedding_dim, num_heads=4)

        # GCN fusion gate (kept, feeds the residual encoder gating)
        gcn_flat = embedding_dim * num_aa * 2
        self.fusion_gate = torch.nn.Linear(gcn_flat, embedding_dim)

        # No esm_proj - raw ESM goes straight to classifier
        self.esm_raw_dim = esm_raw_dim

        # CHANGE 1: USE_GCN_IN_CLASSIFIER=True -> full concatenation
        if USE_GCN_IN_CLASSIFIER:
            clf_in = gcn_flat + embedding_dim + esm_raw_dim
        else:
            clf_in = esm_raw_dim + embedding_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(clf_in, hidden_channels * 2),
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden_channels, num_classes),
        )

        # CHANGE 2: esm_direct branch removed entirely (no shortcut term)

    def pool_gcn(self, z_source, z_target, batch):
        num_graphs = batch.max().item() + 1
        pooled = []
        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            pooled.append(torch.cat([z_source[idx].flatten(), z_target[idx].flatten()]))
        return torch.stack(pooled)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z_source, z_target   = self.encoder(x, edge_index)
        num_graphs  = batch.max().item() + 1
        total_nodes = z_source.shape[0]
        assert total_nodes == num_graphs * self.num_aa

        z_src_3d = self.aa_transformer(z_source.view(num_graphs, self.num_aa, -1))
        z_tgt_3d = self.aa_transformer(z_target.view(num_graphs, self.num_aa, -1))
        z_source  = z_src_3d.view(total_nodes, -1)
        z_target  = z_tgt_3d.view(total_nodes, -1)
        z_aa = self.pool_gcn(z_source, z_target, batch)   # (B, gcn_flat)

        # Residue encoder with GCN-gated fusion (unchanged logic from esm7/esm15)
        res_x     = data.res_x.to(x.device)
        res_batch = data.res_batch.to(x.device)
        z_res     = self.res_norm(self.res_protein_encoder(res_x, res_batch))
        warmup    = max(0.1, min(1.0, self.current_epoch / 10))
        z_res     = z_res * warmup
        fusion_alpha = torch.sigmoid(self.fusion_gate(z_aa))
        z_res_mod    = fusion_alpha * z_res   # (B, embedding_dim)

        # Raw ESM, no projection
        z_esm_raw = data.esm_emb.squeeze(1).to(x.device)   # (B, esm_raw_dim)

        # CHANGE 1: assemble classifier input with USE_GCN_IN_CLASSIFIER=True
        if USE_GCN_IN_CLASSIFIER:
            z_aa_mod = z_aa + torch.cat(
                [z_res_mod] * (z_aa.shape[1] // z_res_mod.shape[1]), dim=1)
            clf_in = torch.cat([z_aa_mod, z_res_mod, z_esm_raw], dim=1)
        else:
            clf_in = torch.cat([z_esm_raw, z_res_mod], dim=1)

        # CHANGE 2: main classifier ONLY - no esm_direct shortcut added
        logits = self.classifier(clf_in)

        if self.go_hierarchy is not None:
            logits = self.go_hierarchy(logits)
        return logits


# ===========================================================================
# Loss  (IDENTICAL to esm15 - label smoothing BCE)
# ===========================================================================

def bce_label_smooth(logits, targets, eps=LABEL_SMOOTH):
    logits  = logits.float()
    targets = targets.float()
    smooth  = targets * (1.0 - eps) + (1.0 - targets) * (eps / 2.0)
    return F.binary_cross_entropy_with_logits(logits, smooth, reduction='mean')


# ===========================================================================
# Training helpers  (IDENTICAL to esm15)
# ===========================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss   = bce_label_smooth(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, mc_samples=1):
    if mc_samples <= 1:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for data in loader:
                data  = data.to(device)
                probs = torch.sigmoid(model(data))
                all_probs.append(probs.cpu())
        return torch.cat(all_probs).numpy()
    else:
        model.train()
        accum = None
        with torch.no_grad():
            for _ in range(mc_samples):
                batch_probs = []
                for data in loader:
                    data  = data.to(device)
                    probs = torch.sigmoid(model(data))
                    batch_probs.append(probs.cpu())
                sample = torch.cat(batch_probs).numpy()
                accum  = sample if accum is None else accum + sample
        model.eval()
        return accum / mc_samples


def save_checkpoint(model, optimizer, scheduler, epoch, best_val, path):
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch, "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val": best_val}, path)


# ===========================================================================
# Main  (IDENTICAL to esm15 - flat multi-label mode, no ontology file)
# ===========================================================================

def main():
    print("=" * 60)
    print("DiGAE v22  -  GCN-in-classifier, no ESM-direct shortcut")
    print(f"  ESM layers: {ESM_LAYERS}  |  use_gcn_in_clf: {USE_GCN_IN_CLASSIFIER}")
    print(f"  label_smooth: {LABEL_SMOOTH}  |  dropout: {DROPOUT}")
    print("=" * 60)

    train_seqs, train_labels, df_train = load_csv_multilabel_data(TRAIN_CSV)
    val_seqs,   val_labels,   df_val   = load_csv_multilabel_data(VAL_CSV)
    test_seqs,  test_labels,  df_test  = load_csv_multilabel_data(TEST_CSV)

    meta_cols        = ["ID", "sequence"]
    label_names      = [c for c in df_test.columns if c not in meta_cols]
    ontologies_names = np.array(label_names)
    NUM_CLASSES      = train_labels.shape[1]
    print(f"  Classes: {NUM_CLASSES}")

    print("Skipping OBO ontology loading (Flat Multi-Label Mode)...")
    ontology = {}

    ic_dict = {}
    with open(IC_CSV, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ic_dict[parts[0]] = float(parts[1])
    print(f"  IC values loaded: {len(ic_dict)} terms")

    layer_tag = "_".join(str(l) for l in ESM_LAYERS)
    train_esm_path = f"train_L{layer_tag}.pt"
    val_esm_path   = f"val_L{layer_tag}.pt"
    test_esm_path  = f"test_L{layer_tag}.pt"

    print(f"Precomputing ESM embeddings (layers {ESM_LAYERS})...")
    precompute_esm_embeddings_multilayer(train_seqs, train_esm_path, ESM_LAYERS)
    precompute_esm_embeddings_multilayer(val_seqs,   val_esm_path,   ESM_LAYERS)
    precompute_esm_embeddings_multilayer(test_seqs,  test_esm_path,  ESM_LAYERS)

    global esm_model
    del esm_model
    gc.collect()
    torch.cuda.empty_cache()

    _tmp = torch.load(train_esm_path, map_location="cpu")
    esm_raw_dim = _tmp.shape[1]
    del _tmp
    print(f"  ESM raw dim: {esm_raw_dim}  ({len(ESM_LAYERS)} * 1280, no projection)")

    train_graph_path = f"train_graphs_{GRAPH_VERSION}.pt"
    val_graph_path   = f"val_graphs_{GRAPH_VERSION}.pt"
    test_graph_path  = f"test_graphs_{GRAPH_VERSION}.pt"

    if os.path.exists(train_graph_path):
        print("Loading cached graphs...")
        train_graphs = torch.load(train_graph_path, weights_only=False)
        val_graphs   = torch.load(val_graph_path,   weights_only=False)
        test_graphs  = torch.load(test_graph_path,  weights_only=False)
    else:
        print("Building graphs...")
        train_graphs = build_graph_dataset_parallel(train_seqs, train_labels)
        val_graphs   = build_graph_dataset_parallel(val_seqs,   val_labels)
        test_graphs  = build_graph_dataset_parallel(test_seqs,  test_labels)
        torch.save(train_graphs, train_graph_path)
        torch.save(val_graphs,   val_graph_path)
        torch.save(test_graphs,  test_graph_path)

    train_esm = torch.load(train_esm_path, map_location="cpu")
    val_esm   = torch.load(val_esm_path,   map_location="cpu")
    test_esm  = torch.load(test_esm_path,  map_location="cpu")

    train_dataset = ProteinDataset(train_graphs, train_esm)
    val_dataset   = ProteinDataset(val_graphs,   val_esm)
    test_dataset  = ProteinDataset(test_graphs,  test_esm)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_seqs)}  Val: {len(val_seqs)}  Test: {len(test_seqs)}")

    go_adj = None
    """
    go_adj = build_go_adjacency(label_names, GO_OBO_PATH).to(DEVICE) \
             if os.path.exists(GO_OBO_PATH) else None
    if go_adj is not None:
        print(f"  GO adjacency: {go_adj.sum().item():.0f} edges")
    """
    fdim  = get_feature_dim()
    model = DiGAEClassifier(
        in_channels=fdim, hidden_channels=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES,
        alpha=ALPHA, beta=BETA, self_loops=SELF_LOOPS,
        adaptive=False, go_adj=go_adj, esm_raw_dim=esm_raw_dim,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    CHECKPOINT_PATH = os.path.join(OUT_DIR, "training_checkpoint.pt")
    start_epoch = 0; best_val_acc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        ckpt      = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val"]
        print(f"  Resuming from epoch {start_epoch}")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Classifier input dim: {HIDDEN_DIM*2} hidden, ESM raw={esm_raw_dim}")

    print("\n" + "=" * 60 + "\nTraining...\n" + "=" * 60)
    best_epoch = 0

    for epoch in range(start_epoch, EPOCHS):
        tstart = time()
        model.current_epoch = epoch
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()

        # Only evaluate heavy metrics every 5 epochs (plus first/last)
        is_eval_epoch = (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == EPOCHS

        if is_eval_epoch:
            val_probs   = evaluate(model, val_loader, DEVICE, mc_samples=1)
            val_metrics = compute_metrics(
                y_true=val_labels, y_prob=val_probs,
                ic_dict=ic_dict, ontologies_names=ontologies_names,
                ontology=ontology, root=BP_ROOT, propagate=False)
            val_fmax = val_metrics["Fmax"]

            if val_fmax > best_val_acc:
                best_val_acc = val_fmax; best_epoch = epoch + 1
                state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state, os.path.join(OUT_DIR, "best_digae_model.pth"))

            tend = time()
            print(f"Epoch {epoch+1}/{EPOCHS} ({tend-tstart:.1f}s) [EVAL]  Loss: {train_loss:.4f}  "
                  f"Val Fmax: {val_fmax:.4f}  Fmax*: {val_metrics['Fmax_star']:.4f}  "
                  f"wFmax: {val_metrics['wFmax']:.4f}  Smin: {val_metrics['Smin']:.4f}  "
                  f"Best: {best_val_acc:.4f}")
        else:
            tend = time()
            print(f"Epoch {epoch+1}/{EPOCHS} ({tend-tstart:.1f}s)  Loss: {train_loss:.4f}  "
                  f"Best Val Fmax so far: {best_val_acc:.4f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, CHECKPOINT_PATH)
    print(f"\nBest val Fmax: {best_val_acc:.4f} at epoch {best_epoch}")

    print("\n" + "=" * 60 + "\nTesting...\n" + "=" * 60)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    raw_model.load_state_dict(
        torch.load(os.path.join(OUT_DIR, "best_digae_model.pth"), map_location=DEVICE))

    print(f"  MC-dropout ({MC_SAMPLES} samples)...")
    test_probs   = evaluate(model, test_loader, DEVICE, mc_samples=MC_SAMPLES)
    test_metrics = compute_metrics(
        y_true=test_labels, y_prob=test_probs,
        ic_dict=ic_dict, ontologies_names=ontologies_names,
        ontology=ontology, root=BP_ROOT, propagate=False)

    print("\nTest Results:")
    for k in ["Fmax", "Fmax_star", "wFmax", "Smin", "AUPRC", "IAuPRC"]:
        print(f"  {k:<12} {test_metrics[k]:.4f}")

    test_preds = (test_probs >= 0.5).astype(int)
    np.save(os.path.join(OUT_DIR, 'test_predictions.npy'),   test_preds)
    np.save(os.path.join(OUT_DIR, 'test_probabilities.npy'), test_probs)
    pd.DataFrame(test_probs, columns=label_names).to_csv(
        os.path.join(OUT_DIR, "test_predictions_probs.csv"), index=False)

    predicted_go_terms = []
    for i in range(test_probs.shape[0]):
        idx = np.where(test_probs[i] >= 0.5)[0]
        predicted_go_terms.append(";".join([label_names[j] for j in idx]))
    results_df = df_test.copy()
    results_df["Predicted_GO_Terms"] = predicted_go_terms
    results_df.to_excel(os.path.join(OUT_DIR, 'test_results_detailed.xlsx'), index=False)

    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write("DiGAE v22\n" + "=" * 60 + "\n\n")
        f.write(f"ESM layers: {ESM_LAYERS}  raw dim: {esm_raw_dim} (no projection)\n")
        f.write(f"USE_GCN_IN_CLASSIFIER: {USE_GCN_IN_CLASSIFIER}\n")
        f.write("esm_direct shortcut branch: REMOVED (logits = classifier(clf_in) only)\n")
        f.write(f"label_smooth: {LABEL_SMOOTH}  dropout: {DROPOUT}\n")
        f.write(f"Epochs: {EPOCHS}  LR: {LR}  Seed: {SEED}\n\n")
        f.write("Test Performance:\n")
        for k, v in test_metrics.items():
            if k != "AUPRC_micro":
                f.write(f"  {k}: {v:.4f}\n")

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    import esm as esm_lib
    esm_model, esm_alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    esm_model       = esm_model.to(DEVICE).eval()
    batch_converter = esm_alphabet.get_batch_converter()
    main()
