import os
import math
import multiprocessing
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_scatter import scatter_mean
from tqdm import tqdm
from sklearn.metrics import average_precision_score

_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz

# ===========================================================================
# CONFIG
# ===========================================================================

# ---- Which ontology branch ----
NAMESPACE = 'molecular_function'      # molecular_function | biological_process | cellular_component

ROOT_BY_NAMESPACE = {
    'molecular_function': 'GO:0003674',
    'biological_process': 'GO:0008150',
    'cellular_component': 'GO:0005575',
}
GO_ROOT = ROOT_BY_NAMESPACE[NAMESPACE]

# ---- Data paths ----
VAL_CSV     = "path_to_val.csv"
TEST_CSV    = "path_to_test.csv"
IC_CSV      = "path_to_ic.txt"

# Set to the go.obo path
GO_OBO_PATH = "path_to_go.obo"

ANNOTATION_FILES = [
    "path_to_train_annotation.txt",
    "path_to_valid_annotationgo.txt",
    "path_to_test_annotation.txt",
]

OUT_DIR = "output_directory"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_COL = 'sequence'
ID_COL  = 'ID'

RELATION_POLICY = {
    'is_a':                   True,    # always keep
    'part_of':                True,    # keep - required for BP / mf
    'regulates':              False,   # CAFA excludes regulation edges
    'positively_regulates':   False,
    'negatively_regulates':   False,
    'omfurs_in':              False,
    'has_part':               False,   # inverse of part_of - would invert the DAG
}

# ---- Evaluation clauses ----
PROPAGATE_PREDICTIONS = True   
REPORT_PROP_GT        = True   
REPORT_RAW_GT         = True   

# ---- Model configs ----
MODEL_CONFIGS = [
    dict(
        name           = "model_name",
        pth_path       = "model.pth",
        esm_val_pt     = "esm_val.pt",
        esm_test_pt    = "esm_test.pt",
        graph_val_pt   = "graph_val.pt",
        graph_test_pt  = "graph_test.pt",
        esm_layers     = [20, 27, 33],
        hidden_dim     = 256,
        use_gcn_in_clf = True,
        dropout        = 0.4,
    ),
]

# ---- Architecture constants (must match training) ----
EMBEDDING_DIM  = 64
ALPHA          = 0.5
BETA           = 0.5
SELF_LOOPS     = True
MAX_RES_LENGTH = 2000
BATCH_SIZE     = 64
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Calibration search ----
T_MIN, T_MAX, T_STEPS          = 0.3, 5.0, 100
N_THRESH_CANDIDATES            = 200
MIN_POSITIVES_FOR_CLASS_THRESH = 3


# ===========================================================================
# Amino acid + physicochemical constants
# ===========================================================================

AMINO_ACIDS = ['A','C','D','E','F','G','H','I','K','L',
               'M','N','P','Q','R','S','T','V','W','Y']
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

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


# ===========================================================================
# If OBO parsing with is_a AND part_of
# ===========================================================================

def _enabled_relations():
    return {r for r, on in RELATION_POLICY.items() if on and r != 'is_a'}


def parse_obo(obo_path, namespace):

    keep_rels = _enabled_relations()
    ontology, gene, in_term = {}, {}, False
    rel_counts = defaultdict(int)

    with open(obo_path) as f:
        for line in f:
            line = line.rstrip('\n')

            if line == '[Term]':
                if 'id' in gene:
                    ontology[gene['id']] = gene
                # 'parents'      -> used for PROPAGATION (is_a + part_of)
                # 'parents_isa'  -> used for the GOHierarchyLayer ADJACENCY,
                #                   which must reproduce training exactly.
                gene = {'parents': [], 'parents_isa': [], 'alt_ids': []}
                in_term = True
                continue

            if line == '[Typedef]':
                if 'id' in gene:
                    ontology[gene['id']] = gene
                gene, in_term = {}, False
                continue

            if not in_term or ': ' not in line:
                continue

            tag, _, val = line.partition(': ')

            if tag == 'id':
                gene['id'] = val
            elif tag == 'alt_id':
                gene['alt_ids'].append(val)
            elif tag == 'name':
                gene['name'] = val
            elif tag == 'is_obsolete':
                gene, in_term = {}, False
            elif tag == 'namespace':
                if val != namespace:
                    gene, in_term = {}, False
                else:
                    gene['namespace'] = val
            elif tag == 'is_a':
                _p = val.split(' ! ')[0].strip()
                gene['parents'].append(_p)
                gene['parents_isa'].append(_p)
                rel_counts['is_a'] += 1
            elif tag == 'relationship':
                # val looks like: "part_of GO:0005634 ! nucleus"
                bits = val.split()
                if len(bits) >= 2:
                    rel, target = bits[0], bits[1]
                    if rel in keep_rels and target.startswith('GO:'):
                        gene['parents'].append(target)
                        rel_counts[rel] += 1
                    else:
                        rel_counts[f'{rel} (skipped)'] += 1

    if 'id' in gene:
        ontology[gene['id']] = gene

    for t in ontology.values():
        t['parents']     = [p for p in t['parents']     if p in ontology]
        t['parents_isa'] = [p for p in t.get('parents_isa', []) if p in ontology]

    # Transitive closure -> ancestors
    for key in list(ontology.keys()):
        ontology[key]['ancestors'] = _ancestors_bfs(ontology, key)
    for key in list(ontology.keys()):
        for alt in ontology[key].get('alt_ids', []):
            ontology[alt] = ontology[key]

    print(f"  OBO parsed: {len(ontology)} terms in '{namespace}'")
    for r, c in sorted(rel_counts.items()):
        print(f"    {r:<28} {c}")
    return ontology


def _ancestors_bfs(ontology, term):
    
    seen, queue, out = {term}, [term], []
    while queue:
        t = queue.pop(0)
        if t not in ontology:
            continue
        out.append(t)
        for p in ontology[t]['parents']:
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return out


def edges_from_obo(label_names, ontology):
    """
    Build (child_idx, parent_idx) ancestor-closure edge arrays restricted to
    the label set. Returns int32 arrays.
    """
    idx = {t: i for i, t in enumerate(label_names)}
    child, parent = [], []
    for t, i in idx.items():
        if t not in ontology:
            continue
        for anc in ontology[t]['ancestors']:
            if anc != t and anc in idx:
                child.append(i)
                parent.append(idx[anc])
    return np.array(child, dtype=np.int32), np.array(parent, dtype=np.int32)


# ===========================================================================
# If reconstructing DAG from annotation subsumption
# ===========================================================================

def edges_from_annotations(label_names, annotation_files):
   
    label_set = set(label_names)
    idx = {t: i for i, t in enumerate(label_names)}

    term_prot = defaultdict(set)
    for fp in annotation_files:
        if not os.path.exists(fp):
            print(f"    WARNING: annotation file not found: {fp}")
            continue
        with open(fp) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] in label_set:
                    term_prot[parts[1]].add(parts[0])
        print(f"    Loaded: {fp}")

    ann_terms = [t for t in label_names if term_prot.get(t)]
    sets      = [term_prot[t] for t in ann_terms]
    n         = len(ann_terms)
    print(f"  Terms with annotations: {n} / {len(label_names)}")

    child, parent = [], []
    for i in range(n):
        si = sets[i]
        for j in range(n):
            if i != j and si <= sets[j]:
                child.append(idx[ann_terms[i]])
                parent.append(idx[ann_terms[j]])

    return np.array(child, dtype=np.int32), np.array(parent, dtype=np.int32)


# ===========================================================================
# Fast propagation via ancestor-closure grouping
# ===========================================================================

def build_propagation_groups(child_arr, parent_arr):
    """
    Group ancestor edges by PARENT.

    Because child_arr/parent_arr encode the FULL ancestor closure (not just
    direct parents), propagation reduces to:

        out[:, p] = max over {p and all descendants of p} of preds[:, .]

    which is one vectorized column-max per parent term. That is at most L
    iterations rather than E (number of edges), and because it reads from the
    unmodified input it needs no topological ordering.
    """
    if len(child_arr) == 0:
        return []
    by_parent = defaultdict(list)
    for c, p in zip(child_arr.tolist(), parent_arr.tolist()):
        by_parent[p].append(c)
    groups = []
    for p, cs in by_parent.items():
        cs.append(p)                                  # include self
        groups.append((p, np.array(sorted(set(cs)), dtype=np.int64)))
    return groups


def propagate(scores, groups):
   
    if not groups:
        return scores.copy()
    out = scores.copy()
    for p, desc in groups:
        out[:, p] = scores[:, desc].max(axis=1)
    return out


def propagate_labels(labels, groups):
    if not groups:
        return labels.astype(np.int32)
    prop = propagate(labels.astype(np.float32), groups)
    return (prop > 0.5).astype(np.int32)


# ===========================================================================
# Ontology resolution 
# ===========================================================================

def resolve_ontology(label_names):
    
    if_obo = bool(GO_OBO_PATH) and os.path.exists(GO_OBO_PATH)

    if if_obo:
        enabled = ['is_a'] + sorted(_enabled_relations())
        print(f"  Relations used: {', '.join(enabled)}")
        ontology = parse_obo(GO_OBO_PATH, NAMESPACE)
        child, parent = edges_from_obo(label_names, ontology)
        regime = "A_obo"
    else:
        print("If reconstructing DAG from annotations")
        ontology = None
        child, parent = edges_from_annotations(label_names, ANNOTATION_FILES)
        regime = "B_annotations"

    groups = build_propagation_groups(child, parent)

    covered = len({int(c) for c in child.tolist()}) if len(child) else 0
    pct = 100.0 * covered / max(len(label_names), 1)
    print(f"  Ancestor edges: {len(child)}")
    print(f"  Coverage: {covered}/{len(label_names)} terms ({pct:.1f}%) have >=1 ancestor")
    if len(child) == 0:
        print("  WARNING: no edges. Propagation is a no-op; only rawGT is meaningful.")
    elif pct < 50:
        print("  WARNING: coverage < 50%. Propagated metrics may understate "
              "performance on sparsely-connected terms.")

    with open(os.path.join(OUT_DIR, "ontology_source.txt"), 'w') as f:
        f.write(f"regime: {regime}\n")
        f.write(f"namespace: {NAMESPACE}\n")
        f.write(f"root: {GO_ROOT}\n")
        f.write(f"relations: {RELATION_POLICY}\n")
        f.write(f"label_terms: {len(label_names)}\n")
        f.write(f"ancestor_edges: {len(child)}\n")
        f.write(f"terms_with_ancestors: {covered} ({pct:.2f}%)\n")

    return groups, ontology, regime


# ===========================================================================
# Metrics
# ===========================================================================

def _stats_at_tau(pred_mask, gt_mask, has_gt, ic_vec, root_idx, N):
    
    tp = (pred_mask & gt_mask).sum(axis=1).astype(np.float32)
    pp = pred_mask.sum(axis=1).astype(np.float32)
    gt = gt_mask.sum(axis=1).astype(np.float32)

    has_pp = pp > 0
    pr_n   = np.where(has_pp, tp / np.maximum(pp, 1e-12), 0.0)
    rc_n   = np.where(has_gt, tp / np.maximum(gt, 1e-12), 0.0)
    n_pp   = (has_pp & has_gt).sum()
    tau_pr = pr_n[has_pp & has_gt].sum() / n_pp if n_pp > 0 else 0.0
    tau_rc = rc_n[has_gt].sum() / N

    ic_pred = pred_mask.astype(np.float32) @ ic_vec
    ic_gt   = gt_mask.astype(np.float32)   @ ic_vec
    ic_tp   = (pred_mask & gt_mask).astype(np.float32) @ ic_vec
    has_icp = ic_pred > 0
    wpr = np.where(has_icp & has_gt, ic_tp / np.maximum(ic_pred, 1e-12), 0.0)
    wrc = np.where(ic_gt > 0,        ic_tp / np.maximum(ic_gt, 1e-12),   0.0)
    n_icp   = (has_icp & has_gt).sum()
    tau_wpr = wpr[has_icp & has_gt].sum() / n_icp if n_icp > 0 else 0.0
    tau_wrc = wrc[has_gt].sum() / N

    mi = ((pred_mask & ~gt_mask).astype(np.float32) @ ic_vec).sum() / N
    ru = ((gt_mask & ~pred_mask).astype(np.float32) @ ic_vec).sum() / N

    if root_idx is not None:
        pm = pred_mask.copy(); pm[:, root_idx] = False
        gm = gt_mask.copy();   gm[:, root_idx] = False
        tp2 = (pm & gm).sum(axis=1).astype(np.float32)
        pp2 = pm.sum(axis=1).astype(np.float32)
        gt2 = gm.sum(axis=1).astype(np.float32)
        hp2, hg2 = pp2 > 0, gt2 > 0
        pr2 = np.where(hp2, tp2 / np.maximum(pp2, 1e-12), 0.0)
        rc2 = np.where(hg2, tp2 / np.maximum(gt2, 1e-12), 0.0)
        n2  = (hp2 & hg2).sum()
        tau_pr_s = pr2[hp2 & hg2].sum() / n2 if n2 > 0 else 0.0
        tau_rc_s = rc2[hg2].sum() / N
    else:
        tau_pr_s, tau_rc_s = tau_pr, tau_rc

    return dict(tau_pr=tau_pr, tau_rc=tau_rc, tau_wpr=tau_wpr,
                tau_wrc=tau_wrc, ru=ru, mi=mi,
                tau_pr_s=tau_pr_s, tau_rc_s=tau_rc_s)


def tawfn_fmax(y_true, y_prob):
    best = 0.0
    tc = y_true.sum(axis=1)
    for tau in np.arange(0.01, 1.00, 0.01):
        pred = y_prob >= tau
        tp   = np.logical_and(pred, y_true).sum(axis=1)
        pc   = pred.sum(axis=1)
        vp   = pc > 0
        prec = (tp[vp] / pc[vp]).mean() if vp.any() else 0.0
        rec  = (tp / np.maximum(tc, 1)).mean()
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def tawfn_smin(y_true, y_prob, ic_vec):
    best = np.inf
    for tau in np.arange(0.01, 1.00, 0.01):
        pred = y_prob >= tau
        fn = np.logical_and(y_true == 1, pred == 0).astype(np.float32) @ ic_vec
        fp = np.logical_and(y_true == 0, pred == 1).astype(np.float32) @ ic_vec
        un = np.logical_or(y_true == 1, pred == 1).astype(np.float32) @ ic_vec
        un = np.where(un == 0, 1e-12, un)
        best = min(best, np.sqrt((fn / un) ** 2 + (fp / un) ** 2).mean())
    return best


def compute_metrics(y_true, y_prob, ic_vec, root_idx):
 
    y_prob = y_prob.astype(np.float32)
    y_true = y_true.astype(np.int32)

    gt_mask = y_true.astype(bool)
    has_gt  = gt_mask.any(axis=1)
    # Recall / Smin denominator: proteins carrying at least one annotation.
    n_gt    = int(has_gt.sum())
    if n_gt == 0:
        return {k: 0.0 for k in
                ("Fmax", "Fmax_star", "wFmax", "Smin", "AUPRC", "IAuPRC",
                 "TAWFN_Fmax", "TAWFN_Smin", "TAWFN_MicroAUPR", "TAWFN_MacroAUPR")}

    fmax = wfmax = fmax_s = 0.0
    smin = 1e100
    pr_arr, rc_arr = [], []

    for tau in np.linspace(0, 1, 101):
        v = _stats_at_tau(y_prob >= tau, gt_mask, has_gt, ic_vec, root_idx, n_gt)
        a, b = v['tau_pr'], v['tau_rc']
        if a + b > 0: fmax = max(fmax, 2 * a * b / (a + b))
        pr_arr.append(a); rc_arr.append(b)
        c, d = v['tau_wpr'], v['tau_wrc']
        if c + d > 0: wfmax = max(wfmax, 2 * c * d / (c + d))
        smin = min(smin, math.sqrt(v['ru'] ** 2 + v['mi'] ** 2))
        e, g = v['tau_pr_s'], v['tau_rc_s']
        if e + g > 0: fmax_s = max(fmax_s, 2 * e * g / (e + g))

    pr_arr, rc_arr = np.array(pr_arr), np.array(rc_arr)
    si = np.argsort(rc_arr)
    rc2, pr2 = rc_arr[si], pr_arr[si]
    auprc = float(_TRAPZ(pr2, rc2))

    ipr, irc = [], []
    for tau in np.linspace(0, 1, 101):
        k = np.where(rc2 >= tau)[0]
        if len(k):
            irc.append(tau); ipr.append(float(pr2[k].max()))
    iauprc = float(_TRAPZ(ipr, irc)) if len(irc) > 1 else 0.0

    def _safe_ap(average):
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                v = average_precision_score(y_true, y_prob, average=average)
            return 0.0 if v is None or not np.isfinite(v) else float(v)
        except Exception:
            return 0.0

    micro = _safe_ap('micro')
    macro = _safe_ap('macro')

    return {
        "Fmax": fmax, "Fmax_star": fmax_s, "wFmax": wfmax, "Smin": smin,
        "AUPRC": auprc, "IAuPRC": iauprc,
        "TAWFN_Fmax": tawfn_fmax(y_true, y_prob),
        "TAWFN_Smin": tawfn_smin(y_true, y_prob, ic_vec),
        "TAWFN_MicroAUPR": micro, "TAWFN_MacroAUPR": macro,
    }


def fmax_only(y_true_mask, y_prob, groups):
    
    p = propagate(y_prob.astype(np.float32), groups) if PROPAGATE_PREDICTIONS else y_prob
    has_gt = y_true_mask.any(axis=1)
    gt_c   = y_true_mask.sum(axis=1).astype(np.float32)
    n_gt   = int(has_gt.sum())
    if n_gt == 0:
        return 0.0
    best   = 0.0
    for tau in np.linspace(0, 1, 51):
        pm = p >= tau
        tp = (pm & y_true_mask).sum(axis=1).astype(np.float32)
        pp = pm.sum(axis=1).astype(np.float32)
        hp = pp > 0
        n  = (hp & has_gt).sum()
        a  = np.where(hp, tp / np.maximum(pp, 1e-12), 0.0)[hp & has_gt].sum() / n if n > 0 else 0.0
        b  = np.where(has_gt, tp / np.maximum(gt_c, 1e-12), 0.0)[has_gt].sum() / n_gt
        if a + b > 0:
            best = max(best, 2 * a * b / (a + b))
    return best


# ===========================================================================
# Data loading
# ===========================================================================

def load_csv(path):
   
    df = pd.read_csv(path)

    raw   = df[SEQ_COL].where(df[SEQ_COL].notna(), "").astype(str).tolist()
    seqs  = [s.strip().upper() for s in raw]
    lcols = [c for c in df.columns if c not in (SEQ_COL, ID_COL)]
    labs  = df[lcols].values.astype(np.float32)

    bad   = {"", "NAN", "NONE", "NULL"}
    keep  = [i for i, s in enumerate(seqs) if s not in bad]

    if len(keep) != len(seqs):
        print(f"  NOTE: dropped {len(seqs) - len(keep)} rows with empty/NaN sequences")

    seqs = [seqs[i] for i in keep]
    labs = labs[keep]
    df   = df.iloc[keep].reset_index(drop=True)     # keep df in register

    print(f"  {path}: {len(seqs)} seqs, {labs.shape[1]} labels")
    return seqs, labs, lcols, df


def load_ic(path, label_names):
   
    ic = {}
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            ic = dict(zip(df.iloc[:, 0].astype(str).str.strip(),
                          pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0.0)))
    except Exception:
        pass

    if sum(1 for t in label_names if t in ic) < 0.5 * len(label_names):
        ic = {}
        with open(path) as f:
            for line in f:
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        ic[parts[0].strip()] = float(parts[1])
                    except ValueError:
                        continue

    hit = sum(1 for t in label_names if t in ic)
    print(f"  IC loaded: {len(ic)} entries, {hit}/{len(label_names)} label terms matched")
    if hit < 0.5 * len(label_names):
        print("  WARNING: IC covers < 50% of label terms. Check the file format. "
              "Missing terms get IC = 0, which deflates wFmax and Smin.")
    return ic


# ===========================================================================
# Model architecture 
# ===========================================================================

class GOHierarchyLayer(torch.nn.Module):
    def __init__(self, nc, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.gcn = torch.nn.Linear(nc, nc, bias=False)
    def forward(self, logits):
        return logits + self.gcn(torch.sigmoid(logits) @ self.adj)


class AATransformerLayer(torch.nn.Module):
    def __init__(self, dim, nh=4):
        super().__init__()
        self.attn  = torch.nn.MultiheadAttention(dim, nh, batch_first=True)
        self.norm  = torch.nn.LayerNorm(dim)
        self.ff    = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2), torch.nn.ReLU(),
            torch.nn.Linear(dim * 2, dim))
        self.norm2 = torch.nn.LayerNorm(dim)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm(x + a)
        return self.norm2(x + self.ff(x))


class ResidueProteinEncoder(torch.nn.Module):
    def __init__(self, dim, in_dim):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_dim, dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.LayerNorm(dim), torch.nn.ReLU(),
            torch.nn.Linear(dim, dim), torch.nn.LayerNorm(dim))
    def forward(self, rx, rb):
        return scatter_mean(self.mlp(self.input_proj(rx)), rb, dim=0)


class DirectedGCNConv(MessagePassing):
    def __init__(self, ic, oc, alpha=1., beta=0., sl=True):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(ic, oc)
        self.alpha, self.beta, self.sl = alpha, beta, sl
    def forward(self, x, ei):
        if self.sl:
            ei, _ = add_self_loops(ei, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = ei
        d = degree(col, x.size(0), dtype=x.dtype).pow(-0.5)
        d[d == float('inf')] = 0.
        return self.propagate(ei, x=x, norm=self.alpha * d[row] + self.beta * d[col])
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCNEncoder(torch.nn.Module):
    def __init__(self, ic, hc, oc, alpha, beta, sl):
        super().__init__()
        self.conv1 = DirectedGCNConv(ic, hc, alpha, beta, sl)
        self.conv2 = DirectedGCNConv(hc, hc, alpha, beta, sl)
        self.conv3 = DirectedGCNConv(hc, oc, alpha, beta, sl)
    def forward(self, x, ei):
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(x, ei)), ei)), ei)


class DirectedGCNConvEncoder(torch.nn.Module):
    def __init__(self, ic, hc, oc, alpha, beta, sl):
        super().__init__()
        self.source_conv = GCNEncoder(ic, hc, oc, alpha, beta, sl)
        self.target_conv = GCNEncoder(ic, hc, oc, alpha, beta, sl)
    def forward(self, x, ei):
        return self.source_conv(x, ei), self.target_conv(x, ei)


class DiGAEClassifier(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, emb_dim, num_classes,
                 alpha, beta, sl, go_adj=None,
                 esm_raw_dim=3840, use_gcn=True, dropout=0.4, num_aa=20):
        super().__init__()
        self.num_aa, self.use_gcn = num_aa, use_gcn
        self.current_epoch = 200
        self.res_protein_encoder = ResidueProteinEncoder(dim=emb_dim, in_dim=emb_dim + 20)
        self.encoder  = DirectedGCNConvEncoder(in_ch, hidden_ch, emb_dim, alpha, beta, sl)
        self.go_hier  = GOHierarchyLayer(num_classes, go_adj) if go_adj is not None else None
        self.res_norm = torch.nn.LayerNorm(emb_dim)
        self.aa_transformer = AATransformerLayer(dim=emb_dim, nh=4)

        gcn_flat = emb_dim * num_aa * 2
        self.fusion_gate = torch.nn.Linear(gcn_flat, emb_dim)
        clf_in = (gcn_flat + emb_dim + esm_raw_dim) if use_gcn else (esm_raw_dim + emb_dim)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(clf_in, hidden_ch * 2),
            torch.nn.LayerNorm(hidden_ch * 2), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_ch * 2, hidden_ch),
            torch.nn.LayerNorm(hidden_ch), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_ch, num_classes))
        #self.esm_direct = torch.nn.Sequential(
        #    torch.nn.Linear(esm_raw_dim, hidden_ch), torch.nn.GELU(),
        #    torch.nn.Dropout(dropout), torch.nn.Linear(hidden_ch, num_classes))

    def _pool(self, zs, zt, batch):
        ng = batch.max().item() + 1
        return torch.stack([
            torch.cat([zs[(batch == g).nonzero(as_tuple=True)[0]].flatten(),
                       zt[(batch == g).nonzero(as_tuple=True)[0]].flatten()])
            for g in range(ng)])

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        zs, zt = self.encoder(x, ei)
        ng = batch.max().item() + 1
      
        assert zs.shape[0] == ng * self.num_aa, (
            f"node count {zs.shape[0]} != batch {ng} x {self.num_aa} nodes; "
            f"the cached graph .pt file does not match this architecture")
        zs = self.aa_transformer(zs.view(ng, self.num_aa, -1)).view(zs.shape)
        zt = self.aa_transformer(zt.view(ng, self.num_aa, -1)).view(zt.shape)
        z_aa = self._pool(zs, zt, batch)
        zr = self.res_norm(self.res_protein_encoder(
            data.res_x.to(x.device), data.res_batch.to(x.device)))
        zrm = torch.sigmoid(self.fusion_gate(z_aa)) * zr
        ze  = data.esm_emb.squeeze(1).to(x.device)
        if self.use_gcn:
            zm = z_aa + torch.cat([zrm] * (z_aa.shape[1] // zrm.shape[1]), dim=1)
            ci = torch.cat([zm, zrm, ze], dim=1)
        else:
            ci = torch.cat([ze, zrm], dim=1)
        logits = self.classifier(ci) #+ self.esm_direct(ze)
        if self.go_hier is not None:
            logits = self.go_hier(logits)
        return logits


# ===========================================================================
# Graph / feature construction
# ===========================================================================

def max_run_length(seq, aa):
    mx = cur = 0
    for c in seq:
        if c == aa:
            cur += 1; mx = max(mx, cur)
        else:
            cur = 0
    return mx


def mean_inter_gap(pos):
    return float(np.mean(np.diff(pos))) if len(pos) >= 2 else 0.


def get_hft(seq, pct=90):
    tri = [seq[i:i + 3] for i in range(len(seq) - 2)]
    cnt = Counter(tri)
    if not cnt:
        return set()
    return {t for t, c in cnt.items() if c >= np.percentile(list(cnt.values()), pct)}


def nbr_freq(seq, positions, k=2):
    if not len(positions):
        return np.zeros(20, dtype=np.float32)
    counts = np.zeros(20, dtype=np.float32); total = 0; L = len(seq)
    for pos in positions:
        for i in range(max(0, pos - k), min(L, pos + k + 1)):
            if i == pos:
                continue
            if seq[i] in AA_TO_INDEX:
                counts[AA_TO_INDEX[seq[i]]] += 1; total += 1
    if total:
        counts /= total
    return counts


def seq_to_nodes(seq):
    L = len(seq); nf = np.zeros((20, 30), dtype=np.float32)
    pd_ = {aa: [] for aa in AMINO_ACIDS}
    for i, aa in enumerate(seq):
        if aa in pd_:
            pd_[aa].append(i)
    dip = [seq[i:i + 2] for i in range(L - 1)]
    hft = get_hft(seq)
    for aa in AMINO_ACIDS:
        idx = AA_TO_INDEX[aa]; pos = pd_[aa]; cnt = len(pos)
        if cnt == 0:
            continue
        pos = np.array(pos, dtype=np.float32); np_ = pos / max(L - 1, 1)
        nf[idx, 0] = cnt / L; nf[idx, 1] = np_.mean(); nf[idx, 2] = np_.std()
        nf[idx, 3] = mean_inter_gap(pos) / L
        nf[idx, 4] = max_run_length(seq, aa) / L
        nf[idx, 5] = np.sum(np_ <= 0.2) / cnt
        nf[idx, 6] = np.sum(np_ >= 0.8) / cnt
        f = sum(d[0] == aa for d in dip); s = sum(d[1] == aa for d in dip)
        td = max(len(dip), 1)
        nf[idx, 7] = f / td; nf[idx, 8] = s / td
        hh = 0
        for p in pos.astype(int):
            for st in (p - 2, p - 1, p):
                if 0 <= st <= L - 3 and seq[st:st + 3] in hft:
                    hh += 1; break
        nf[idx, 9] = hh / cnt
        nf[idx, 10:30] = nbr_freq(seq, pos.astype(int))
    return nf


def build_res_graph(seq, pe_dim=EMBEDDING_DIM):
    seq = seq[:MAX_RES_LENGTH]; L = len(seq)
    pos = torch.arange(L, dtype=torch.float32)
    dt  = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32)
                    * (-math.log(10000.) / pe_dim))
    pe = torch.zeros(L, pe_dim)
    pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * dt)
    pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * dt)
    oh = torch.zeros(L, 20, dtype=torch.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_INDEX:
            oh[i, AA_TO_INDEX[aa]] = 1.
    return torch.cat([pe, oh], dim=1)


def build_graph(seq, label):
    x = torch.tensor(seq_to_nodes(seq), dtype=torch.float)
    x = torch.cat([x, PHYSCHEM_MATRIX], dim=1)
    edges = [[AA_TO_INDEX[seq[i]], AA_TO_INDEX[seq[i + 1]]]
             for i in range(len(seq) - 1)
             if seq[i] in AA_TO_INDEX and seq[i + 1] in AA_TO_INDEX]
    if not edges:
        edges = [[i, i] for i in range(20)]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y  = torch.tensor(label, dtype=torch.float).unsqueeze(0)
    rx = build_res_graph(seq)
    rb = torch.zeros(rx.size(0), dtype=torch.long)
    return Data(x=x, edge_index=ei, res_x=rx, res_batch=rb, y=y)


class ProteinDataset(Dataset):
    def __init__(self, graphs, esm):
        self.g, self.e = graphs, esm
    def __len__(self):
        return len(self.g)
    def __getitem__(self, i):
        d = self.g[i]; d.esm_emb = self.e[i].unsqueeze(0); return d


# ===========================================================================
# Model loading + inference
# ===========================================================================

def detect_go_adj_in_checkpoint(pth_path):
    
    try:
        state = torch.load(pth_path, map_location='cpu')
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        return any(k.startswith('go_hier.') for k in state)
    except Exception as e:
        print(f"  Could not inspect checkpoint ({e}); assuming no GOHierarchyLayer.")
        return False


def build_go_adj_matrix(label_names, ontology, relations='isa_only'):
  
    key = 'parents_isa' if relations == 'isa_only' else 'parents'
    idx = {t: i for i, t in enumerate(label_names)}
    adj = torch.zeros(len(label_names), len(label_names))
    n = 0
    for t, i in idx.items():
        if t in ontology:
            for p in ontology[t].get(key, []):
                if p in idx:
                    adj[i, idx[p]] = 1.0
                    n += 1
    print(f"    adjacency built from '{key}': {n} edges")
    return adj


def load_model(cfg, num_classes, go_adj, device):
    esm_tmp = torch.load(cfg['esm_val_pt'], map_location='cpu')
    esm_raw_dim = (esm_tmp.shape[1] * esm_tmp.shape[2]
                   if esm_tmp.dim() == 3 else esm_tmp.shape[1])
    del esm_tmp

    model = DiGAEClassifier(
        in_ch=37, hidden_ch=cfg['hidden_dim'], emb_dim=EMBEDDING_DIM,
        num_classes=num_classes, alpha=ALPHA, beta=BETA, sl=SELF_LOOPS,
        go_adj=go_adj, esm_raw_dim=esm_raw_dim,
        use_gcn=cfg['use_gcn_in_clf'], dropout=cfg['dropout']).to(device)

    state = torch.load(cfg['pth_path'], map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"  Weights: {cfg['pth_path']}  ESM dim: {esm_raw_dim}")
    return model


def preflight(cfg, n_val, n_test, num_classes):
    
    ok = True
    for split, n_rows, gkey, ekey in (("val", n_val, 'graph_val_pt', 'esm_val_pt'),
                                      ("test", n_test, 'graph_test_pt', 'esm_test_pt')):
        try:
            graphs = torch.load(cfg[gkey], weights_only=False)
            esm    = torch.load(cfg[ekey], map_location='cpu')
        except Exception as e:
            print(f"    {split}: could not load cached tensors ({type(e).__name__}: {e})")
            ok = False
            continue

        n_g, n_e = len(graphs), esm.shape[0]
        if not (n_g == n_e == n_rows):
            print(f"    {split}: ROW COUNT MISMATCH -> csv={n_rows}, "
                  f"graphs={n_g}, esm={n_e}")
            print(("      The cached .pt files were built from a different CSV. "
                  "Rebuild them, or proteins will be paired with the wrong "
                  "embeddings and every metric will be meaningless."))
            ok = False
        else:
            print(f"    {split}: {n_rows} rows, graphs and ESM aligned")
        del graphs, esm
    return ok


def make_loader(graph_pt, esm_pt):
    graphs = torch.load(graph_pt, weights_only=False)
    esm    = torch.load(esm_pt, map_location='cpu')
    if esm.dim() == 3:
        esm = esm.view(esm.shape[0], -1)
    return DataLoader(ProteinDataset(graphs, esm),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def get_logits(model, loader, device):
    model.eval(); out = []
    with torch.no_grad():
        for data in loader:
            out.append(model(data.to(device)).cpu())
    return torch.cat(out).numpy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# ===========================================================================
# Calibration techniques
# ===========================================================================

def find_temperature(val_logits, val_gt_mask, groups):
    print(f"  Temp search: {T_STEPS} coarse steps in [{T_MIN}, {T_MAX}]")
    best_T, best_f = 1.0, -1.
    for T in tqdm(np.linspace(T_MIN, T_MAX, T_STEPS), desc="  Grid"):
        f = fmax_only(val_gt_mask, sigmoid(val_logits / T), groups)
        if f > best_f:
            best_f, best_T = f, T
    for T in np.linspace(max(T_MIN, best_T * 0.9), min(T_MAX, best_T * 1.1), 30):
        f = fmax_only(val_gt_mask, sigmoid(val_logits / T), groups)
        if f > best_f:
            best_f, best_T = f, T
    print(f"  Best T = {best_T:.4f}   val Fmax = {best_f:.4f}")
    return best_T


def per_class_thresholds(val_probs, val_labels):
    N, L  = val_probs.shape
    cands = np.linspace(0.01, 0.99, N_THRESH_CANDIDATES).astype(np.float32)
    gt    = val_labels.astype(np.float32)
    n_pos = gt.sum(axis=0).astype(np.int32)

    best_gt_thr, best_gf = 0.5, -1.
    for t in cands:
        pred  = (val_probs >= t).astype(np.float32)
        tp    = np.sum(pred * gt)
        denom = 2 * tp + np.sum(pred * (1 - gt)) + np.sum((1 - pred) * gt)
        f1    = 2 * tp / denom if denom > 0 else 0.
        if f1 > best_gf:
            best_gf, best_gt_thr = f1, t
    print(f"  Global threshold = {best_gt_thr:.4f} (micro F1 = {best_gf:.4f})")

    best_f1 = np.zeros(L, dtype=np.float32)
    thr     = np.full(L, best_gt_thr, dtype=np.float32)
    opt     = n_pos >= MIN_POSITIVES_FOR_CLASS_THRESH
    print(f"  Optimizing {opt.sum()} classes; {(~opt).sum()} rare classes use fallback")

    for t in tqdm(cands, desc="  Per-class thresh"):
        pred  = (val_probs >= t).astype(np.float32)
        tp    = (pred * gt).sum(axis=0)
        denom = 2 * tp + (pred * (1 - gt)).sum(axis=0) + ((1 - pred) * gt).sum(axis=0)
        f1    = np.where(denom > 0, 2 * tp / denom, 0.)
        imp   = (f1 > best_f1) & opt
        thr[imp]     = t
        best_f1[imp] = f1[imp]
    return thr, best_gt_thr


def normalize_thresholds(probs, thr, groups=None):
    """
    Remap probabilities so every class decides at 0.5
    Passing `groups` re-propagates after the remap to restore the invariant.
    This is the correct order: threshold first (so each class's own decision
    point is honoured), then re-propagate (so the hierarchy stays consistent).
    """
    eps = 1e-6
    t = np.clip(thr, eps, 1 - eps).astype(np.float32)[np.newaxis, :]
    p = probs.astype(np.float32)
    out = np.where(p < t, p / (2.0 * t),
                   0.5 + (p - t) / (2.0 * (1.0 - t))).astype(np.float32)
    if groups:
        out = propagate(out, groups)
    return out


# ===========================================================================
# Dual-clause reporting
# ===========================================================================

def report_dual(probs_raw, gt_raw, gt_prop, label_names, df,
                ic_vec, root_idx, groups, tag, results,
                pre_propagated=False):
    """
    Evaluates one calibration method under BOTH ground-truth clauses.

    probs_raw      : model probabilities (before propagation unless
                     pre_propagated=True)
    gt_raw         : labels exactly as stored in the CSV
    gt_prop        : labels with ancestors filled in
    pre_propagated : set True when the caller already propagated, so the
                     per-class-threshold path can reuse this function
                     instead of duplicating the reporting logic.

    Predictions are propagated once and the SAME array is scored against both
    ground-truth variants, so the only thing differing between clauses is the
    ground truth.
    """
    print(f"\n{'=' * 62}\n{tag}\n{'=' * 62}")

    if pre_propagated or not PROPAGATE_PREDICTIONS:
        probs_eval = probs_raw.astype(np.float32)
    else:
        probs_eval = propagate(probs_raw.astype(np.float32), groups)

    keys = ["Fmax", "Fmax_star", "wFmax", "Smin", "AUPRC", "IAuPRC",
            "TAWFN_Fmax", "TAWFN_Smin", "TAWFN_MicroAUPR", "TAWFN_MacroAUPR"]

    clauses = []
    if REPORT_PROP_GT:
        clauses.append(("propGT", gt_prop))
    if REPORT_RAW_GT:
        clauses.append(("rawGT", gt_raw))

    out = {}
    for cname, gt in clauses:
        m = compute_metrics(gt, probs_eval, ic_vec, root_idx)
        out[cname] = m
        results[f"{tag}::{cname}"] = m

    header = "  {:<18}".format("metric") + "".join(f"{c:>12}" for c, _ in clauses)
    print(header)
    print("  " + "-" * (18 + 12 * len(clauses)))
    for k in keys:
        row = "  {:<18}".format(k) + "".join(f"{out[c][k]:>12.4f}" for c, _ in clauses)
        print(row)

    # Diagnostic: do the two clauses agree?
    if len(clauses) == 2:
        d = abs(out["propGT"]["Fmax"] - out["rawGT"]["Fmax"])
        if d < 1e-6:
            print("  -> Identical. CSV labels were already propagated by the dataset.")
        else:
            print(f"  -> Fmax differs by {d:.4f}. CSV stores direct annotations only; "
                  f"propGT is the CAFA-correct number.")

    np.save(os.path.join(OUT_DIR, f"{tag}_probs.npy"), probs_eval)
    np.save(os.path.join(OUT_DIR, f"{tag}_preds.npy"), (probs_eval >= 0.5).astype(np.int8))

    gt_col = [";".join(label_names[j] for j in np.where(probs_eval[i] >= 0.5)[0])
              for i in range(probs_eval.shape[0])]
    od = df.copy()
    if len(od) == len(gt_col):
        od["Predicted_GO_Terms"] = gt_col
        # Never let a missing optional dependency throw away a completed run:
        # inference and calibration are the expensive parts and are already
        # persisted as .npy above. Fall back to CSV if openpyxl is absent.
        try:
            od.to_excel(os.path.join(OUT_DIR, f"{tag}_detailed.xlsx"), index=False)
        except Exception as e:
            od.to_csv(os.path.join(OUT_DIR, f"{tag}_detailed.csv"), index=False)
            print(f"  NOTE: xlsx write failed ({type(e).__name__}); wrote CSV instead.")
    else:
        print(f"  WARNING: df rows ({len(od)}) != prediction rows ({len(gt_col)}); "
              f"skipping the detailed table.")

    with open(os.path.join(OUT_DIR, f"{tag}_metrics.txt"), 'w') as f:
        f.write(f"{tag}\n{'=' * 62}\n")
        for cname, _ in clauses:
            f.write(f"\n[{cname}]\n")
            for k, v in out[cname].items():
                f.write(f"  {k}: {v:.4f}\n")
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 62)
    print("M_calib_universal  -  OBO-aware / OBO-free calibration")
    print("=" * 62)

    val_seqs,  val_labels,  label_names, df_val  = load_csv(VAL_CSV)
    test_seqs, test_labels, _,           df_test = load_csv(TEST_CSV)
    NUM_CLASSES = len(label_names)
    print(f"  GO terms: {NUM_CLASSES}   namespace: {NAMESPACE}   root: {GO_ROOT}")

    ic_dict  = load_ic(IC_CSV, label_names)
    ic_vec   = np.array([ic_dict.get(t, 0.0) for t in label_names], dtype=np.float32)
    root_idx = label_names.index(GO_ROOT) if GO_ROOT in label_names else None

    print()
    groups, ontology, regime = resolve_ontology(label_names)

    # Ground truth in both states, computed once
    val_gt_raw   = val_labels.astype(np.int32)
    test_gt_raw  = test_labels.astype(np.int32)
    val_gt_prop  = propagate_labels(val_labels,  groups)
    test_gt_prop = propagate_labels(test_labels, groups)

    added = int(test_gt_prop.sum() - test_gt_raw.sum())
    print(f"\n  GT propagation added {added} positive labels on test "
          f"({test_gt_raw.sum()} -> {test_gt_prop.sum()})")
    if added == 0 and len(groups) > 0:
        print("  -> Dataset labels were already ancestor-closed.")

    # Proteins with zero annotations are excluded from the recall denominator
    # (CAFA convention). Report the count so the exclusion is never silent.
    n_unann_test = int((test_gt_prop.sum(axis=1) == 0).sum())
    n_unann_val  = int((val_gt_prop.sum(axis=1) == 0).sum())
    print(f"  Proteins with NO annotation: test {n_unann_test}/{len(test_gt_prop)}, "
          f"val {n_unann_val}/{len(val_gt_prop)}")
    if n_unann_test:
        print("  -> excluded from the recall denominator, per CAFA. If you expect "
              "every test protein to be annotated, check the label columns.")

    # Temperature search targets the CAFA-correct clause when available
    val_search_mask = (val_gt_prop if REPORT_PROP_GT else val_gt_raw).astype(bool)

    valid = []
    for cfg in MODEL_CONFIGS:
        missing = [k for k in ('pth_path', 'esm_val_pt', 'esm_test_pt',
                               'graph_val_pt', 'graph_test_pt')
                   if not os.path.exists(cfg[k])]
        if missing:
            print(f"\n  SKIPPING {cfg['name']}: missing {missing}")
            continue
        print(f"\n  Preflight {cfg['name']}:")
        if preflight(cfg, len(val_labels), len(test_labels), NUM_CLASSES):
            valid.append(cfg)
        else:
            print(f"  SKIPPING {cfg['name']}: preflight failed")
    if not valid:
        print("ERROR: no usable model configs."); return

    results = {}
    ens_val, ens_tst = [], []   # temperature-scaled logits pooled for ensembling
    val_gt_fit_global = val_gt_prop if REPORT_PROP_GT else val_gt_raw

    for cfg in valid:
        name = cfg['name']
        print(f"\n{'=' * 62}\nModel: {name}\n{'=' * 62}")

        # go_adj must mirror training, detected from the checkpoint itself
        needs_adj = detect_go_adj_in_checkpoint(cfg['pth_path'])
        if needs_adj and ontology is not None:
            go_adj = build_go_adj_matrix(label_names, ontology).to(DEVICE)
            print(f"  GOHierarchyLayer: ENABLED ({int(go_adj.sum())} edges)")
        elif needs_adj:
            print("  ERROR: checkpoint expects GOHierarchyLayer but no OBO is "
                  "available to build the adjacency. Skipping this model.")
            continue
        else:
            go_adj = None
            print("  GOHierarchyLayer: disabled (not present in checkpoint)")

        model = load_model(cfg, NUM_CLASSES, go_adj, DEVICE)

        print("  Val inference...")
        val_logits = get_logits(model, make_loader(cfg['graph_val_pt'], cfg['esm_val_pt']), DEVICE)
        print("  Test inference...")
        tst_logits = get_logits(model, make_loader(cfg['graph_test_pt'], cfg['esm_test_pt']), DEVICE)

        # --- 1. Baseline ---
        report_dual(sigmoid(tst_logits), test_gt_raw, test_gt_prop,
                    label_names, df_test, ic_vec, root_idx, groups,
                    f"{name}_baseline", results)

        # --- 2. Temperature scaling ---
        print(f"\n  [{name}] Temperature scaling...")
        best_T = find_temperature(val_logits, val_search_mask, groups)
        np.save(os.path.join(OUT_DIR, f"{name}_best_T.npy"), np.array([best_T]))

        vp_T = sigmoid(val_logits / best_T)
        tp_T = sigmoid(tst_logits / best_T)
        report_dual(tp_T, test_gt_raw, test_gt_prop, label_names, df_test,
                    ic_vec, root_idx, groups, f"{name}_temp", results)

        ens_val.append(val_logits / best_T)
        ens_tst.append(tst_logits / best_T)

        # --- 3. Per-class thresholds ---
        print(f"\n  [{name}] Per-class thresholds...")
        vp_T_eval = propagate(vp_T, groups) if PROPAGATE_PREDICTIONS else vp_T
        thr, _ = per_class_thresholds(vp_T_eval, val_gt_fit_global)
        np.save(os.path.join(OUT_DIR, f"{name}_class_thresholds.npy"), thr)

        tp_T_eval = propagate(tp_T, groups) if PROPAGATE_PREDICTIONS else tp_T
        # groups passed so the True Path Rule is restored after the remap
        tp_pc = normalize_thresholds(tp_T_eval, thr,
                                     groups if PROPAGATE_PREDICTIONS else None)
        # pre_propagated=True: reuse the same reporting path (and get the
        # xlsx / metrics.txt written) without propagating a second time
        report_dual(tp_pc, test_gt_raw, test_gt_prop, label_names, df_test,
                    ic_vec, root_idx, groups, f"{name}_temp_perclass",
                    results, pre_propagated=True)

        del model
        torch.cuda.empty_cache()

    # --- Ensemble ---
    if len(ens_tst) > 1:
        print(f"\n{'=' * 62}\nEnsemble ({len(ens_tst)} models)\n{'=' * 62}")
        evp = sigmoid(np.mean(ens_val, axis=0))
        etp = sigmoid(np.mean(ens_tst, axis=0))

        report_dual(etp, test_gt_raw, test_gt_prop, label_names, df_test,
                    ic_vec, root_idx, groups, "ensemble_temp", results)

        print("\n  Ensemble per-class thresholds...")
        evp_eval = propagate(evp, groups) if PROPAGATE_PREDICTIONS else evp
        etp_eval = propagate(etp, groups) if PROPAGATE_PREDICTIONS else etp
        e_thr, _ = per_class_thresholds(evp_eval, val_gt_fit_global)
        np.save(os.path.join(OUT_DIR, "ensemble_class_thresholds.npy"), e_thr)
        etp_pc = normalize_thresholds(etp_eval, e_thr,
                                      groups if PROPAGATE_PREDICTIONS else None)
        report_dual(etp_pc, test_gt_raw, test_gt_prop, label_names, df_test,
                    ic_vec, root_idx, groups, "ensemble_temp_perclass",
                    results, pre_propagated=True)

    # --- Summary ---
    print("\n" + "=" * 78)
    print(f"  {'Method::Clause':<46}{'Fmax':>8}{'wFmax':>8}{'Smin':>8}{'AUPRC':>8}")
    print("  " + "-" * 76)
    for tag, m in results.items():
        print(f"  {tag:<46}{m['Fmax']:>8.4f}{m['wFmax']:>8.4f}"
              f"{m['Smin']:>8.4f}{m['AUPRC']:>8.4f}")

    prop_only = {k: v for k, v in results.items() if k.endswith("::propGT")}
    pool = prop_only if prop_only else results
    best = max(pool, key=lambda t: pool[t]["Fmax"])
    print(f"\n  Best: {best}   Fmax = {pool[best]['Fmax']:.4f}")

    rows = []
    for tag, m in results.items():
        method, _, clause = tag.partition("::")
        rows.append({"method": method, "gt_clause": clause, "regime": regime, **m})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "calibration_summary.csv"), index=False)
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
