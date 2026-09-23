import os, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 更稳地限制CPU线程
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, math, random, gc, time, platform, argparse
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
torch.set_num_threads(1)

# ================= 全局默认参数 =================

DATA_ROOT = Path("/path/to/US_EQ_EX")

DOMAIN_DATA_DIRS = {
    "base": DATA_ROOT / "BASE" / "processing_BASE_outputs" / "legacy_dataset",
    "enam": DATA_ROOT / "ENAM" / "processing_ENAM_outputs" / "legacy_dataset",
    "msh": DATA_ROOT / "MSH" / "processing_MSH_outputs" / "legacy_dataset",
    "hlp": DATA_ROOT / "HLP" / "processing_HLP_outputs" / "legacy_dataset",
}

OUT_ROOT = Path("./outputs/DANN")

DOMAINS   = ["base", "enam", "msh", "hlp"]
MODALITY  = "STFT"
SEEDS     = [0, 1, 2, 3, 4]


BATCH_SRC       = 64
BATCH_TGT       = 64
EPOCHS          = 50

LR              = 5e-5
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 2
PREFETCH_FACTOR = 2
AMP             = True

# 域对抗相关
LAMBDA_DOMAIN   = 0.05
LAMBDA_ADAPT    = True
LAMBDA_MIN      = 0.02
LAMBDA_MAX      = 0.40
LAMBDA_HARD_MAX = 0.12

DOM_TARGET_CENTER   = 0.50
DOM_TARGET_BAND     = 0.05
LAMBDA_KP           = 0.70
LAMBDA_MIN_STEP     = 0.01
LAMBDA_MAX_STEP     = 0.10

# GRL / alpha 调度
ALPHA_SLOPE         = 0.6
ALPHA_MAX_INIT      = 0.10
ALPHA_MIN_CAP       = 0.04
ALPHA_MAX_CAP       = 0.30
ALPHA_HARD_MAX      = 0.12
ALPHA_ADAPT         = True
ALPHA_STEP          = 0.02

DANN_WARMUP_EPOCHS  = 5
PATIENCE            = 20
GRAD_CLIP_NORM      = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

USE_DATAPARALLEL    = False

CLASS_NAMES = ["earthquake", "explosion"]
FORCE_X_KEY = None
FORCE_Y_KEY = None

CROSS_TRAIN_ALL = True
MANUAL_PAIRS = [("base", "enam")]

THRESH_SCAN_STEP = 0.002

# 健康监测 & SAFE MODE
DOMACC_EMA_BETA       = 0.9
TRIG_A_LOW            = 0.45
TRIG_B_HIGH           = 0.58
TRIG_K_CONSEC         = 3
TRIG_C_M_UP           = 3
TRIG_C_F1_DROP_RATIO  = 0.02
A_LAMBDA_DEC          = 0.02
A_ALPHA_DEC           = 0.02
A_DOMLR_MUL           = 0.5
B_LAMBDA_INC          = 0.02
B_ALPHA_INC           = 0.02
SAFE_A_ACCUM_S        = 8
SAFE_RECOVER_BAND_L   = 0.48
SAFE_RECOVER_BAND_U   = 0.52
SAFE_RECOVER_NEED_R   = 3
SAFE_RESUME_STEP      = 0.02

USE_EMA              = True
EMA_BETA             = 0.999
USE_GROUPNORM        = False
GN_GROUPS            = 32
USE_SWA              = False
SWA_START_EPOCH      = 40

# ===== 半监督默认 =====
SEMI                 = 1          # 1=半监督，0=纯 UDA
TGT_LABEL_RATIO      = 0.10       # 目标域有标签比例
W_TGT_SUP            = 1.0        # 目标域监督权重

# ===== 域对抗是否包含目标有标签 =====
DOM_INCLUDE_LABELED  = 0          # 0=只用无标签目标做对抗；1=有标签也参加
DOM_LABELED_WEIGHT   = 0.5        # 有标签目标在域对抗中的权重

# =================== 小工具 ===================
def log(*a): print(*a, flush=True)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False

def _wif(worker_id: int):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed); random.seed(base_seed)

def device_info():
    if torch.cuda.is_available():
        ng = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(ng)]
        return {"device":"cuda","num_gpus":ng,"gpu_names":names,"cuda_version":torch.version.cuda}
    return {"device":"cpu","num_gpus":0,"gpu_names":[],"cuda_version":None}

def data_pt_path(data_root: str, ds: str, split: str):
    ds = ds.lower()
    if ds not in DOMAIN_DATA_DIRS:
        raise ValueError(
            f"Unknown domain {ds!r}; expected one of {sorted(DOMAIN_DATA_DIRS)}"
        )
    base = f"{ds}_dataset_zrt_waveforms_ZRT_DanFenLiang_40HZ_90S_EventFiltered"
    if split == "train":
        fn = f"{base}_train_{MODALITY}_raw_ALL13.pt"
    elif split == "valid":
        fn = f"{base}_valid_{MODALITY}_raw.pt"
    elif split == "test":
        fn = f"{base}_test_{MODALITY}_raw.pt"
    else:
        raise ValueError(f"Unknown split {split!r}; expected train, valid, or test")

    pt_path = os.path.join(DOMAIN_DATA_DIRS[ds], fn)
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(
            f"Missing {ds.upper()} {split} dataset: {pt_path}. "
            f"Generate that domain's legacy_dataset artifacts first."
        )
    return pt_path

def compute_metrics(y_true, y_pred, n_classes=2):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class_acc = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        per_class_acc.append(float(tp / (tp + fn + 1e-12)))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm.tolist()
    }

# =================== 数据集 ===================
def _labels_to_float_tensor(y):
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    elif isinstance(y, (list,tuple)):
        buf=[]
        for v in y:
            if isinstance(v, torch.Tensor):
                buf.append(float(v.view(-1)[0].item()))
            else:
                vv = np.array(v).reshape(-1); buf.append(float(vv[0]))
        y_np = np.array(buf, dtype=np.float32)
    else:
        y_np = np.array(y).reshape(-1).astype(np.float32)
    return torch.from_numpy(y_np.reshape(-1)).float().view(-1,1)

def _pick_xy_from_dict(d: dict):
    if "data" in d and ("labels_num" in d or "label" in d or "labels" in d):
        xraw = d["data"]
        if isinstance(xraw, list):
            x = torch.stack([torch.as_tensor(xx) for xx in xraw], dim=0)
        else:
            x = torch.as_tensor(xraw)
        if "labels_num" in d:
            y = d["labels_num"]
        elif "label" in d:
            y = d["label"]
        else:
            y = d["labels"]
        return x, y

    if FORCE_X_KEY and FORCE_Y_KEY and FORCE_X_KEY in d and FORCE_Y_KEY in d:
        return d[FORCE_X_KEY], d[FORCE_Y_KEY]

    for kx,ky in [("data","labels"),("x","y"),("inputs","targets"),
                  ("images","labels"),("waveforms","labels"),("samples","targets")]:
        if kx in d and ky in d:
            return d[kx], d[ky]
    return None

class PTDataset(Dataset):
    def __init__(self, pt_path, mean: float, std: float):
        obj = torch.load(pt_path, map_location="cpu")
        if isinstance(obj, dict):
            xy = _pick_xy_from_dict(obj)
            if xy is None:
                raise ValueError(f"Unrecognized dict keys in {pt_path}: {list(obj.keys())}")
            x,y = xy
        elif isinstance(obj, (list,tuple)):
            xs,ys=[],[]
            for it in obj:
                if isinstance(it,(list,tuple)) and len(it)==2:
                    xi,yi = it
                elif isinstance(it,dict) and ("data" in it and ("label" in it or "labels_num" in it or "labels" in it)):
                    xi,yi = it["data"], it.get("labels_num", it.get("label", it.get("labels")))
                else:
                    raise ValueError(f"Unrecognized element in {pt_path}: {type(it)}")
                xs.append(torch.as_tensor(xi))
                yi = np.array(yi).reshape(-1); ys.append(float(yi[0]))
            x = torch.stack(xs, dim=0); y = np.array(ys, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported .pt format: {type(obj)}")

        x = torch.as_tensor(x, dtype=torch.float32)
        y = _labels_to_float_tensor(y)

        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim !=4 or x.shape[1]!=1:
            raise ValueError(f"x shape should be [N,1,H,W], got {tuple(x.shape)}")

        x = torch.log1p(torch.clamp(x, min=0.0))
        x = (x - float(mean)) / (float(std) + 1e-6)

        self.x = x.contiguous()
        self.y = y.contiguous()

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

def compute_mean_std_for_source_train(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "data" in obj:
        xraw = obj["data"]
        if isinstance(xraw, list):
            x = torch.stack([torch.as_tensor(xx) for xx in xraw], dim=0)
        else:
            x = torch.as_tensor(xraw)
    elif isinstance(obj,(list,tuple)):
        xs=[]
        for it in obj:
            if isinstance(it,(list,tuple)) and len(it)==2:
                xi,_ = it
            elif isinstance(it,dict) and ("data" in it):
                xi = it["data"]
            else:
                continue
            xs.append(torch.as_tensor(xi))
        x = torch.stack(xs, dim=0)
    else:
        raise ValueError("Unsupported training .pt format for mean/std")
    if x.ndim==3: x=x.unsqueeze(1)
    x = torch.log1p(torch.clamp(torch.as_tensor(x, dtype=torch.float32), min=0.0))
    mean = x.mean().item(); std = x.std().item()
    return mean, std

# =================== EMA（修复版） ===================
class EMAHelper:
    def __init__(self, model, beta=EMA_BETA):
        self.beta = beta
        self.model = model
        dev = next((p.device for p in model.parameters() if p.requires_grad), torch.device("cpu"))
        self.shadow = {k: v.detach().clone().to(dev) for k, v in model.state_dict().items()
                       if isinstance(v, torch.Tensor) and v.dtype.is_floating_point}
        self.backup = None

    def to(self, device: torch.device):
        for k in list(self.shadow.keys()):
            self.shadow[k] = self.shadow[k].to(device)

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if (k in self.shadow) and isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                if self.shadow[k].device != v.device:
                    self.shadow[k] = self.shadow[k].to(v.device)
                self.shadow[k].mul_(self.beta).add_(v.detach(), alpha=1.0 - self.beta)

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                       for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            if (k in self.shadow) and isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                if self.shadow[k].device != v.device:
                    self.shadow[k] = self.shadow[k].to(v.device)
                v.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self):
        if self.backup is not None:
            for k, v in self.model.state_dict().items():
                if k in self.backup:
                    vv = self.backup[k]
                    if isinstance(v, torch.Tensor) and isinstance(vv, torch.Tensor) and vv.device != v.device:
                        vv = vv.to(v.device)
                    if isinstance(v, torch.Tensor) and isinstance(vv, torch.Tensor):
                        v.copy_(vv)
                    else:
                        setattr(self.model, k, vv)
            self.backup = None

# =================== 模型 ===================
def make_norm(channels, use_gn=USE_GROUPNORM, groups=GN_GROUPS):
    if use_gn:
        g = min(groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, channels)
    else:
        return nn.BatchNorm2d(channels)

class CNN3(nn.Module):
    def __init__(self, in_channels=1, use_gn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1   = make_norm(32, use_gn)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = make_norm(64, use_gn)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = make_norm(128, use_gn)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANN(nn.Module):
    def __init__(self, in_channels=1, input_size=(256,256), feature_dim_reduced=128, use_gn=False):
        super().__init__()
        self.feature = CNN3(in_channels=in_channels, use_gn=use_gn)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            feat = self.feature(dummy)
            self.feature_dim = feat.shape[1]
        self.reduce = nn.Linear(self.feature_dim, feature_dim_reduced)
        self.cls_head = nn.Linear(feature_dim_reduced, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim_reduced, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x, alpha=1.0):
        h = self.feature(x)
        h = self.reduce(h)
        cls_logit = self.cls_head(h)
        rev = GradientReversalFunction.apply(h, alpha)
        dom_logits = self.domain_head(rev)
        return cls_logit, dom_logits

def freeze_bn(m: nn.Module):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# =================== 评估/标定/阈值 ===================
@torch.no_grad()
def eval_binary_fixed_thresh(model, loader, device, thresh=0.5):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    tot_loss, n = 0.0, 0
    y_true, y_pred, y_prob = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits, _ = model(xb, alpha=0.0)
        loss = bce(logits, yb).item()
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        preds = (probs >= float(thresh)).astype(int)
        tot_loss += loss; n += yb.numel()
        y_true.extend(yb.detach().cpu().numpy().ravel().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())
    metrics = compute_metrics(y_true, y_pred, n_classes=2)
    try:
        metrics["auc"] = float(roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_prob)))
    except Exception:
        metrics["auc"] = 0.0
    metrics["loss"] = tot_loss / max(1, n)
    return metrics

@torch.no_grad()
def best_thresh_on_valid(model, loader, device, step=THRESH_SCAN_STEP):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits, _ = model(xb, alpha=0.0)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        ys.append(yb.numpy().ravel()); ps.append(prob)
    y_true = np.concatenate(ys).astype(int)
    p = np.concatenate(ps)
    grid = list(np.arange(0.0, 1.0+1e-12, step))
    qs = np.quantile(p, [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
    cand = np.unique(np.clip(np.concatenate([grid, qs]), 0, 1))
    best_f1, best_t = -1.0, 0.5
    for t in cand:
        y_pred = (p >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

class TempScale(nn.Module):
    def __init__(self): super().__init__(); self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, z): return z / self.logT.exp()

@torch.no_grad()
def _collect_logits_and_labels(model, loader, device):
    Z, Y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        z, _ = model(xb, alpha=0.0)
        Z.append(z.detach()); Y.append(yb.to(device))
    return torch.cat(Z, dim=0), torch.cat(Y, dim=0)

def fit_temperature(model, src_val_loader, device, max_iter=200):
    ts = TempScale().to(device)
    bce = nn.BCEWithLogitsLoss()
    Z, Y = _collect_logits_and_labels(model, src_val_loader, device)
    opt = optim.LBFGS(ts.parameters(), lr=1e-2, max_iter=max_iter)
    def closure():
        opt.zero_grad(); loss = bce(ts(Z), Y); loss.backward(); return loss
    opt.step(closure)
    return ts

@torch.no_grad()
def _sweep_thresh_from_probs(y_true, probs, step=0.01):
    y_true = np.asarray(y_true).astype(int); p = np.asarray(probs).astype(float)
    grid = list(np.arange(0.0, 1.0 + 1e-12, step))
    qs = np.quantile(p, [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99])
    cand = np.unique(np.clip(np.concatenate([grid, qs]), 0, 1))
    best_f1, best_t = -1.0, 0.5
    for t in cand:
        y_pred = (p >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, float(t)
    return best_t, best_f1

@torch.no_grad()
def find_best_t_with_ts(model, loader, device, ts, step=0.01):
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device); z, _ = model(xb, alpha=0.0)
        p = torch.sigmoid(ts(z)).squeeze(1).cpu().numpy()
        ys.append(yb.numpy().ravel()); ps.append(p)
    y_true = np.concatenate(ys).astype(int); probs = np.concatenate(ps)
    return _sweep_thresh_from_probs(y_true, probs, step=step)

@torch.no_grad()
def eval_with_ts(model, loader, device, ts, thresh):
    ys, ps = [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    tot_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        z, _ = model(xb, alpha=0.0)
        loss = bce(z, yb).item()
        p = torch.sigmoid(ts(z)).squeeze(1).cpu().numpy()
        ys.append(yb.cpu().numpy().ravel()); ps.append(p)
        tot_loss += loss; n += yb.numel()
    y_true = np.concatenate(ys).astype(int); probs  = np.concatenate(ps)
    y_pred = (probs >= float(thresh)).astype(int)
    metrics = compute_metrics(y_true, y_pred, n_classes=2)
    metrics["loss"] = tot_loss / max(1, n)
    pr = float((probs >= float(thresh)).mean())
    return metrics, pr

@torch.no_grad()
def rate_matched_threshold(model, src_val_loader, tgt_loader, device, ts, t_src, step=0.001):
    ps_src = []
    for xb, _ in src_val_loader:
        xb = xb.to(device); z,_ = model(xb, alpha=0.0)
        ps_src.append(torch.sigmoid(ts(z)).squeeze(1).cpu().numpy())
    ps_src = np.concatenate(ps_src)
    pr_src = float((ps_src >= float(t_src)).mean())

    ps_tgt = []
    for xb, _ in tgt_loader:
        xb = xb.to(device); z,_ = model(xb, alpha=0.0)
        ps_tgt.append(torch.sigmoid(ts(z)).squeeze(1).cpu().numpy())
    ps_tgt = np.concatenate(ps_tgt)

    qs = np.quantile(ps_tgt, np.clip(np.arange(0,1+1e-9,step),0,1))
    cand = np.unique(np.concatenate([qs, [t_src]]))
    best_t, best_gap = float(t_src), 1.0
    for t in cand:
        gap = abs(float((ps_tgt >= float(t)).mean()) - pr_src)
        if gap < best_gap:
            best_gap, best_t = gap, float(t)
    return best_t, pr_src

# =================== 训练单对 ===================
def _ensure_hist_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_cls_loss,train_tgt_sup_loss,train_dom_loss,train_dom_acc,dom_acc_ema,lambda_domain,alpha_max_curr,src_valid_loss,src_valid_macro_f1_at_t,t_star,lr_main,lr_dom,triggerA,triggerB,triggerC,safe_mode\n")

def split_target_labeled_unlabeled(dataset: Dataset, ratio: float, seed: int):
    n = len(dataset)
    if n == 0 or ratio <= 0.0:
        return None, dataset
    if ratio >= 1.0:
        return dataset, None
    idx = list(range(n))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(idx)
    k = int(round(n * ratio))
    k = max(1, k)  # 至少拿 1 条出来
    lab_idx = idx[:k]; unlab_idx = idx[k:]
    lab_ds = Subset(dataset, lab_idx) if k > 0 else None
    unlab_ds = Subset(dataset, unlab_idx) if len(unlab_idx) > 0 else None
    return lab_ds, unlab_ds

def train_one_pair(
    data_root, out_root,
    source, target, seed, device,
    batch_src=BATCH_SRC, batch_tgt=BATCH_TGT,
    epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY,
    lambda_domain=LAMBDA_DOMAIN, patience=PATIENCE,
    num_workers=NUM_WORKERS,
    # 控制器
    lambda_adapt=LAMBDA_ADAPT,
    lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
    dom_target_center=DOM_TARGET_CENTER, dom_target_band=DOM_TARGET_BAND,
    lambda_kp=LAMBDA_KP, lambda_min_step=LAMBDA_MIN_STEP, lambda_max_step=LAMBDA_MAX_STEP,
    alpha_slope=ALPHA_SLOPE, alpha_max_init=ALPHA_MAX_INIT,
    alpha_min_cap=ALPHA_MIN_CAP, alpha_max_cap=ALPHA_MAX_CAP,
    alpha_adapt=ALPHA_ADAPT, alpha_step=ALPHA_STEP,
    lambda_hard_max=LAMBDA_HARD_MAX, alpha_hard_max=ALPHA_HARD_MAX,
    use_gn=USE_GROUPNORM,
    use_ema=USE_EMA, ema_beta=EMA_BETA,
    # 半监督相关
    semi=SEMI, tgt_label_ratio=TGT_LABEL_RATIO, w_tgt_sup=W_TGT_SUP,
    dom_include_labeled=DOM_INCLUDE_LABELED, dom_labeled_weight=DOM_LABELED_WEIGHT
):
    set_seed(seed)
    mode_tag = "ssda" if semi and tgt_label_ratio>0 else "dann"
    run_suffix = f"stft_{mode_tag}_seed{seed}" + (f"_r{tgt_label_ratio:.2f}" if semi and tgt_label_ratio>0 else "")
    run_dir = os.path.join(out_root, f"{source}_to_{target}", run_suffix)

    os.makedirs(run_dir, exist_ok=True)
    best_path   = os.path.join(run_dir, "best.pth")
    latest_path = os.path.join(run_dir, "latest.pth")
    hist_csv    = os.path.join(run_dir, "history.csv")
    result_js   = os.path.join(run_dir, "results.json")

    if os.path.isfile(result_js) and os.path.isfile(best_path):
        log(f"[SKIP] Found results & best for {source}->{target} seed={seed}, skipping training.")
        with open(result_js, "r", encoding="utf-8") as f:
            return json.load(f)

    devinfo = device_info()
    envinfo = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cuda": devinfo.get("cuda_version"),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES","")
    }

    # 源域 mean/std
    src_train_path = data_pt_path(data_root, source, "train")
    src_mean, src_std = compute_mean_std_for_source_train(src_train_path)

    # 数据集
    src_train = PTDataset(src_train_path, mean=src_mean, std=src_std)
    src_valid = PTDataset(data_pt_path(data_root, source,"valid"), mean=src_mean, std=src_std)
    src_test  = PTDataset(data_pt_path(data_root, source,"test"),  mean=src_mean, std=src_std)
    tgt_train_full = PTDataset(data_pt_path(data_root, target,"train"), mean=src_mean, std=src_std)
    tgt_test  = PTDataset(data_pt_path(data_root, target,"test"),  mean=src_mean, std=src_std)

    # 半监督：拆分目标域训练集
    tgt_lab_ds, tgt_unlab_ds = split_target_labeled_unlabeled(tgt_train_full, float(tgt_label_ratio), seed=seed)

    # dataloader 公共配置
    dl_ctx = mp.get_context("spawn")
    dl_common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_wif,
        multiprocessing_context=dl_ctx,
    )
    if num_workers > 0:
        dl_common["prefetch_factor"] = PREFETCH_FACTOR

    src_tr_loader = DataLoader(src_train, batch_size=batch_src, shuffle=True, drop_last=True, **dl_common)
    src_va_loader = DataLoader(src_valid, batch_size=batch_src, shuffle=False, **dl_common)
    src_te_loader = DataLoader(src_test,  batch_size=batch_src, shuffle=False, **dl_common)

    # ====== 关键修复 1：目标有标签 loader 自动缩小 batch、且 drop_last=False ======
    tgt_lab_loader = None
    if tgt_lab_ds is not None and len(tgt_lab_ds) > 0:
        lab_bs = min(batch_tgt, len(tgt_lab_ds))  # 很少时用小 batch
        tgt_lab_loader = DataLoader(
            tgt_lab_ds,
            batch_size=lab_bs,
            shuffle=True,
            drop_last=False,   # 不能丢最后一批，否则会变成 0 个 batch
            **dl_common
        )

    # 无标签目标可以继续用大 batch、drop_last=True
    tgt_unlab_loader = None
    if tgt_unlab_ds is not None and len(tgt_unlab_ds) > 0:
        tgt_unlab_loader = DataLoader(
            tgt_unlab_ds,
            batch_size=batch_tgt,
            shuffle=True,
            drop_last=True,
            **dl_common
        )

    # 对抗用的目标域 loader
    # 这里要保证：只有真的能产 batch 的 loader 才参与
    if tgt_unlab_loader is None and tgt_lab_loader is not None:
        tgt_dom_loader = tgt_lab_loader
    elif tgt_unlab_loader is not None and tgt_lab_loader is None:
        tgt_dom_loader = tgt_unlab_loader
    else:
        if int(dom_include_labeled) == 1:
            # 都有，而且要并联
            tgt_dom_loader = (tgt_unlab_loader, tgt_lab_loader)
        else:
            tgt_dom_loader = tgt_unlab_loader if tgt_unlab_loader is not None else tgt_lab_loader

    tgt_te_loader = DataLoader(tgt_test, batch_size=batch_tgt, shuffle=False, **dl_common)

    # 模型
    model = DANN(in_channels=1, input_size=(256,256), feature_dim_reduced=128, use_gn=use_gn)
    if USE_DATAPARALLEL and torch.cuda.device_count()>1 and devinfo["device"]=="cuda":
        model = nn.DataParallel(model)
    model = model.to(device)

    ema = EMAHelper(model, beta=ema_beta) if use_ema else None

    base = (model.module if isinstance(model, nn.DataParallel) else model)
    feat_params   = list(base.feature.parameters()) + list(base.reduce.parameters()) + list(base.cls_head.parameters())
    dom_params    = list(base.domain_head.parameters())
    dom_lr_factor = 0.5
    optimizer = optim.Adam([
        {"params": feat_params, "lr": lr},
        {"params": dom_params, "lr": lr * dom_lr_factor},
    ], weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(enabled=(AMP and torch.cuda.is_available()))

    bce = nn.BCEWithLogitsLoss()
    ce_none = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")

    best_src_valid_f1_at_t = -1.0
    best_t_star = 0.5
    wait = 0
    t0 = time.time()

    current_lambda = float(lambda_domain)
    alpha_max_curr = float(alpha_max_init)
    current_lambda = min(current_lambda, lambda_hard_max)
    alpha_max_curr = min(alpha_max_curr, alpha_hard_max)

    start_epoch = 1
    if os.path.isfile(latest_path):
        ck = torch.load(latest_path, map_location="cpu")
        base.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 1)) + 1
        wait = int(ck.get("wait", 0))
        best_src_valid_f1_at_t = float(ck.get("best_src_valid_f1_at_t", -1.0))
        best_t_star = float(ck.get("best_t_star", 0.5))
        current_lambda = float(ck.get("current_lambda", current_lambda))
        alpha_max_curr = float(ck.get("alpha_max_curr", alpha_max_curr))
        dom_lr_factor  = float(ck.get("dom_lr_factor", dom_lr_factor))
        if ema is not None and "ema_shadow" in ck:
            model_dev = next((p.device for p in base.parameters() if p.requires_grad), torch.device("cpu"))
            for k in ema.shadow.keys():
                if k in ck["ema_shadow"]:
                    ema.shadow[k] = ck["ema_shadow"][k].to(model_dev).clone()
        log(f"[RESUME] {source}->{target}|seed={seed} from epoch {start_epoch} (wait={wait}, best_f1={best_src_valid_f1_at_t:.4f}, t*={best_t_star:.3f}, λ={current_lambda:.3f}, αmax={alpha_max_curr:.2f}, domLRx={dom_lr_factor:.2f})")

    _ensure_hist_header(hist_csv)

    devinfo_s = device_info()
    log(f"\n===== [DANN{'-SSDA' if (semi and tgt_label_ratio>0) else ''}] {source} -> {target} | seed={seed} | modality={MODALITY} | "
        f"S_tr={len(src_train)} S_va={len(src_valid)} S_te={len(src_test)} "
        f"T_tr_full={len(tgt_train_full)} T_tr_lab={len(tgt_lab_ds) if tgt_lab_ds else 0} T_tr_unlab={len(tgt_unlab_ds) if tgt_unlab_ds else 0} T_te={len(tgt_test)} | device={devinfo_s['device']} =====")
    log(f"[CTRL] dom_acc target={dom_target_center:.2f}±{dom_target_band:.2f} | "
        f"λ∈[{lambda_min:.2f},{min(lambda_max,lambda_hard_max):.2f}] init={current_lambda:.2f} | "
        f"alpha_max∈[{alpha_min_cap:.2f},{min(alpha_max_cap,alpha_hard_max):.2f}] init={alpha_max_curr:.2f} | "
        f"EMA={'on' if ema is not None else 'off'} GN={'on' if use_gn else 'off'} | "
        f"SEMI={'on' if (semi and tgt_label_ratio>0) else 'off'} ratio={tgt_label_ratio:.3f} w_tgt={w_tgt_sup:.2f} | "
        f"DOM(include_labeled)={bool(dom_include_labeled)} w_lab={dom_labeled_weight:.2f}")

    dom_acc_ema = 0.5
    trigA_consec = trigB_consec = trigC_consec = 0
    trigA_accum  = 0
    safe_mode = False
    safe_recover_consec = 0
    last_src_va_loss_list = []
    best_src_va_f1_seen = -1.0

    # 迭代次数
    if isinstance(tgt_dom_loader, tuple):
        iters = min(len(src_tr_loader), max(len(tgt_dom_loader[0]) if tgt_dom_loader[0] is not None else 0,
                                            len(tgt_dom_loader[1]) if tgt_dom_loader[1] is not None else 0))
    else:
        if tgt_dom_loader is not None:
            iters = min(len(src_tr_loader), len(tgt_dom_loader))
        else:
            iters = len(src_tr_loader)

    for epoch in range(start_epoch, epochs+1):
        model.train(); base.apply(freeze_bn)
        running = {"loss":[], "cls":[], "tgt_sup":[], "dom":[], "dom_acc":[]}

        src_iter = iter(src_tr_loader)

        # 对抗迭代器
        if isinstance(tgt_dom_loader, tuple):
            dom_unlab_iter = iter(tgt_dom_loader[0]) if tgt_dom_loader[0] is not None else None
            dom_lab_iter   = iter(tgt_dom_loader[1]) if tgt_dom_loader[1] is not None else None
            dom_iter_tuple = True
        else:
            dom_iter = iter(tgt_dom_loader) if tgt_dom_loader is not None else None
            dom_iter_tuple = False

        # 有标签监督迭代器
        lab_iter = iter(tgt_lab_loader) if tgt_lab_loader is not None else None

        pbar = tqdm(range(iters), dynamic_ncols=True,
                    desc=f"[{source}->{target}|seed{seed}] Train {epoch}/{epochs}")

        for i in pbar:
            # 源域：循环取样
            try:
                xs, ys = next(src_iter)
            except StopIteration:
                src_iter = iter(src_tr_loader); xs, ys = next(src_iter)

            # 对抗的目标域 batch
            if dom_iter_tuple:
                # 无标签对抗
                if dom_unlab_iter is not None:
                    try:
                        xt_unlab, _ = next(dom_unlab_iter)
                    except StopIteration:
                        dom_unlab_iter = iter(tgt_dom_loader[0]); xt_unlab, _ = next(dom_unlab_iter)
                else:
                    xt_unlab = None
                # 有标签对抗（可能很少，也循环）
                if dom_lab_iter is not None:
                    try:
                        xt_lab_adv, _ = next(dom_lab_iter)
                    except StopIteration:
                        dom_lab_iter = iter(tgt_dom_loader[1]); xt_lab_adv, _ = next(dom_lab_iter)
                else:
                    xt_lab_adv = None
            else:
                if dom_iter is not None:
                    try:
                        xt_any, _ = next(dom_iter)
                    except StopIteration:
                        dom_iter = iter(tgt_dom_loader); xt_any, _ = next(dom_iter)
                else:
                    xt_any = None
                xt_unlab, xt_lab_adv = xt_any, None

            # 有标签监督 batch（半监督才取）
            xt_lab_sup = yt_lab_sup = None
            if semi and tgt_lab_loader is not None and tgt_label_ratio>0:
                try:
                    xt_lab_sup, yt_lab_sup = next(lab_iter)
                except StopIteration:
                    # 关键修复 2：有标签的也循环
                    lab_iter = iter(tgt_lab_loader)
                    xt_lab_sup, yt_lab_sup = next(lab_iter)

            # 上设备
            xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
            if xt_unlab is not None:   xt_unlab = xt_unlab.to(device, non_blocking=True)
            if xt_lab_adv is not None: xt_lab_adv = xt_lab_adv.to(device, non_blocking=True)
            if xt_lab_sup is not None:
                xt_lab_sup = xt_lab_sup.to(device, non_blocking=True)
                yt_lab_sup = yt_lab_sup.to(device, non_blocking=True)

            # 调度
            p = float(i + (epoch-1)*iters) / (epochs*iters + 1e-12)
            alpha_sched = min(alpha_max_curr, alpha_slope * p)
            alpha_use = 0.0 if safe_mode else alpha_sched

            # ===== 组对抗输入 =====
            x_dom_parts = [xs]
            d_dom_parts = [torch.zeros(xs.size(0), dtype=torch.long, device=device)]
            w_dom_parts = [torch.ones(xs.size(0), dtype=torch.float32, device=device)]  # 源域=1

            if xt_unlab is not None:
                x_dom_parts.append(xt_unlab)
                d_dom_parts.append(torch.ones(xt_unlab.size(0), dtype=torch.long, device=device))
                w_dom_parts.append(torch.ones(xt_unlab.size(0), dtype=torch.float32, device=device))

            if dom_iter_tuple and (xt_lab_adv is not None) and (int(dom_include_labeled) == 1):
                x_dom_parts.append(xt_lab_adv)
                d_dom_parts.append(torch.ones(xt_lab_adv.size(0), dtype=torch.long, device=device))
                w_dom_parts.append(torch.full((xt_lab_adv.size(0),), float(dom_labeled_weight), dtype=torch.float32, device=device))

            x_for_dom = torch.cat(x_dom_parts, dim=0)
            d_for_dom = torch.cat(d_dom_parts, dim=0)
            w_for_dom = torch.cat(w_dom_parts, dim=0)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled() and torch.cuda.is_available()):
                # 统一前向
                cls_logit, dom_logits = model(x_for_dom, alpha=alpha_use)
                n_src = xs.size(0)
                src_cls_loss = bce(cls_logit[:n_src], ys)

                # 半监督监督项
                if semi and (xt_lab_sup is not None) and (yt_lab_sup is not None) and (tgt_label_ratio>0):
                    tgt_cls_logit, _ = model(xt_lab_sup, alpha=0.0)
                    tgt_sup_loss = bce(tgt_cls_logit, yt_lab_sup)
                else:
                    tgt_sup_loss = dom_logits.new_zeros(())

                # 域对抗损失
                if safe_mode:
                    dom_loss = dom_logits.new_zeros(())
                    effective_lambda = 0.0
                    dom_acc = 0.5
                else:
                    dom_losses_vec = ce_none(dom_logits, d_for_dom)
                    dom_loss = (dom_losses_vec * w_for_dom).sum() / (w_for_dom.sum() + 1e-8)
                    effective_lambda = 0.0 if epoch <= DANN_WARMUP_EPOCHS else current_lambda
                    with torch.no_grad():
                        d_pred = dom_logits.argmax(dim=1)
                        dom_acc = (d_pred == d_for_dom).float().mean().item()

                loss = src_cls_loss + effective_lambda * dom_loss + float(w_tgt_sup) * tgt_sup_loss

            # 反传
            if scaler.is_enabled() and torch.cuda.is_available():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((model.parameters()), GRAD_CLIP_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_((model.parameters()), GRAD_CLIP_NORM)
                optimizer.step()

            if ema is not None:
                ema.update()

            running["loss"].append(loss.item())
            running["cls"].append(src_cls_loss.item())
            running["tgt_sup"].append(float(tgt_sup_loss.item()) if (tgt_sup_loss is not None) else 0.0)
            running["dom"].append(float(dom_loss.item()) if not safe_mode else 0.0)
            running["dom_acc"].append(dom_acc)

            pbar.set_postfix(loss=f"{np.mean(running['loss']):.4f}",
                             cls=f"{np.mean(running['cls']):.4f}",
                             tgt_sup=f"{np.mean(running['tgt_sup']):.4f}",
                             dom=f"{np.mean(running['dom']):.4f}",
                             dom_acc=f"{np.mean(running['dom_acc']):.3f}",
                             alpha=f"{alpha_use:.2f}",
                             lam=f"{current_lambda:.3f}",
                             eff_lam=f"{effective_lambda:.3f}",
                             amax=f"{alpha_max_curr:.2f}",
                             SAFE=("Y" if safe_mode else "N"))

        # 源域 valid
        src_va_metrics_loss = eval_binary_fixed_thresh(model, src_va_loader, device, thresh=0.5)
        scheduler.step(src_va_metrics_loss["loss"])

        if ema is not None:
            ema.apply_shadow()
        try:
            t_star, f1_at_t = best_thresh_on_valid(model, src_va_loader, device, step=THRESH_SCAN_STEP)
        finally:
            if ema is not None:
                ema.restore()

        # ===== 稳态控制 =====
        dom_acc_mean = float(np.mean(running["dom_acc"])) if running["dom_acc"] else 0.5
        dom_acc_ema = DOMACC_EMA_BETA * dom_acc_ema + (1 - DOMACC_EMA_BETA) * dom_acc_mean

        if lambda_adapt and (epoch > DANN_WARMUP_EPOCHS) and (not safe_mode):
            err = dom_acc_mean - dom_target_center
            if abs(err) > dom_target_band:
                step = max(lambda_min_step, min(lambda_max_step, lambda_kp * abs(err)))
                if err > 0:
                    new_lambda = min(current_lambda + step, lambda_max)
                else:
                    new_lambda = max(current_lambda - step, lambda_min)
                new_lambda = min(new_lambda, lambda_hard_max)
                if abs(new_lambda - current_lambda) > 1e-8:
                    log(f"[CTRL|λ] epoch={epoch} dom_acc_mean={dom_acc_mean:.3f} "
                        f"λ {current_lambda:.3f}->{new_lambda:.3f}")
                    current_lambda = new_lambda
                if alpha_adapt:
                    need_more = (dom_acc_mean > dom_target_center + dom_target_band) and (current_lambda >= min(lambda_max, lambda_hard_max) - 1e-9)
                    need_less = (dom_acc_mean < dom_target_center - dom_target_band) and (current_lambda <= lambda_min + 1e-9)
                    if need_more:
                        new_amax = min(alpha_max_curr + alpha_step, min(alpha_max_cap, alpha_hard_max))
                        if new_amax > alpha_max_curr + 1e-8:
                            log(f"[CTRL|α] αmax {alpha_max_curr:.2f}->{new_amax:.2f} (+)")
                            alpha_max_curr = new_amax
                    if need_less:
                        new_amax = max(alpha_max_curr - alpha_step, alpha_min_cap)
                        if new_amax < alpha_max_curr - 1e-8:
                            log(f"[CTRL|α] αmax {alpha_max_curr:.2f}->{new_amax:.2f} (-)")
                            alpha_max_curr = new_amax

        # ===== 健康监测 =====
        trigA = trigB = trigC = 0
        last_src_va_loss_list.append(src_va_metrics_loss["loss"])
        best_src_va_f1_seen = max(best_src_va_f1_seen, f1_at_t)

        # A: dom_acc 一直低
        if dom_acc_ema < TRIG_A_LOW and not safe_mode:
            trigA_consec += 1
        else:
            trigA_consec = 0
        if trigA_consec >= TRIG_K_CONSEC and not safe_mode:
            trigA = 1
            old_lambda, old_amax = current_lambda, alpha_max_curr
            current_lambda = max(lambda_min, current_lambda - A_LAMBDA_DEC)
            alpha_max_curr = max(alpha_min_cap, alpha_max_curr - A_ALPHA_DEC)
            dom_lr_factor *= A_DOMLR_MUL
            optimizer.param_groups[1]["lr"] = lr * dom_lr_factor
            trigA_consec = 0; trigA_accum += 1
            log(f"[TRIG-A] dom_acc_ema={dom_acc_ema:.3f} ⇒ λ {old_lambda:.3f}->{current_lambda:.3f}, αmax {old_amax:.2f}->{alpha_max_curr:.2f}, domLRx->{dom_lr_factor:.3f}")

        # B: dom_acc 一直高
        if dom_acc_ema > TRIG_B_HIGH and not safe_mode:
            trigB_consec += 1
        else:
            trigB_consec = 0
        if trigB_consec >= TRIG_K_CONSEC and not safe_mode:
            trigB = 1
            old_lambda, old_amax = current_lambda, alpha_max_curr
            current_lambda = min(min(lambda_max, lambda_hard_max), current_lambda + B_LAMBDA_INC)
            alpha_max_curr = min(min(alpha_max_cap, alpha_hard_max), alpha_max_curr + B_ALPHA_INC)
            trigB_consec = 0
            log(f"[TRIG-B] dom_acc_ema={dom_acc_ema:.3f} ⇒ λ {old_lambda:.3f}->{current_lambda:.3f}, αmax {old_amax:.2f}->{alpha_max_curr:.2f}")

        # C: 源域 valid 连续升高 + F1 掉
        condC_loss = (len(last_src_va_loss_list) >= TRIG_C_M_UP and
                      all(last_src_va_loss_list[-k-1] < last_src_va_loss_list[-k] for k in range(1, TRIG_C_M_UP)))
        condC_f1   = (best_src_va_f1_seen > 0 and (best_src_va_f1_seen - f1_at_t) >= TRIG_C_F1_DROP_RATIO)
        if condC_loss and condC_f1 and not safe_mode:
            trigC_consec += 1
        else:
            trigC_consec = 0
        if trigC_consec >= 1 and not safe_mode:
            trigC = 1
            log(f"[TRIG-C] src_valid loss up & F1 drop ⇒ pause adversarial for next epoch")
            current_lambda = 0.0
            trigC_consec = 0

        # ===== SAFE MODE =====
        if (trigA_accum >= SAFE_A_ACCUM_S) and (not safe_mode):
            safe_mode = True
            for p in base.domain_head.parameters(): p.requires_grad = False
            current_lambda = 0.0; alpha_max_curr = 0.0
            log(f"[SAFE-MODE] Entered.")

        if safe_mode:
            if SAFE_RECOVER_BAND_L <= dom_acc_ema <= SAFE_RECOVER_BAND_U:
                safe_recover_consec += 1
            else:
                safe_recover_consec = 0
            if safe_recover_consec >= SAFE_RECOVER_NEED_R:
                safe_mode = False
                for p in base.domain_head.parameters(): p.requires_grad = True
                current_lambda = min(min(lambda_max, lambda_hard_max), current_lambda + SAFE_RESUME_STEP)
                alpha_max_curr = min(min(alpha_max_cap, alpha_hard_max), alpha_max_curr + SAFE_RESUME_STEP)
                safe_recover_consec = 0; trigA_accum = 0
                log(f"[SAFE-MODE] Recovered: resume adversarial (λ={current_lambda:.3f}, αmax={alpha_max_curr:.2f}).")

        # 写 history
        with open(hist_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{float(np.mean(running['loss'])):.6f},{float(np.mean(running['cls'])):.6f},"
                    f"{float(np.mean(running['tgt_sup'])):.6f},{float(np.mean(running['dom'])):.6f},"
                    f"{dom_acc_mean:.6f},{dom_acc_ema:.6f},"
                    f"{current_lambda:.6f},{alpha_max_curr:.6f},"
                    f"{src_va_metrics_loss['loss']:.6f},{f1_at_t:.6f},{t_star:.6f},"
                    f"{float(optimizer.param_groups[0]['lr']):.8f},{float(optimizer.param_groups[1]['lr']):.8f},"
                    f"{trigA},{trigB},{trigC},{int(safe_mode)}\n")

        # 保存最好
        if ema is not None:
            ema.apply_shadow()
        try:
            improved = f1_at_t > best_src_valid_f1_at_t
            if improved:
                best_src_valid_f1_at_t = f1_at_t; best_t_star = t_star; wait = 0
                torch.save(
                    {"model": base.state_dict(),
                     "source": source, "target": target, "seed": seed, "modality": MODALITY,
                     "class_names": CLASS_NAMES,
                     "src_mean": src_mean, "src_std": src_std,
                     "epoch": epoch, "uda": not (semi and tgt_label_ratio>0),
                     "lambda_domain": current_lambda,
                     "alpha_max_curr": alpha_max_curr,
                     "best_thresh": best_t_star,
                     "ema_used": bool(ema is not None),
                     "semi": bool(semi and tgt_label_ratio>0),
                     "tgt_label_ratio": float(tgt_label_ratio),
                     "w_tgt_sup": float(w_tgt_sup)},
                    best_path
                )
                log(f"  -> Save BEST to {best_path} (src_valid_macro-F1@t*={best_src_valid_f1_at_t:.4f}, t*={best_t_star:.3f})")
            else:
                wait += 1
        finally:
            if ema is not None:
                ema.restore()

        # 保存 latest
        ck = {
            "model": base.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "wait": wait,
            "best_src_valid_f1_at_t": best_src_valid_f1_at_t,
            "best_t_star": best_t_star,
            "current_lambda": current_lambda,
            "alpha_max_curr": alpha_max_curr,
            "dom_lr_factor": dom_lr_factor,
            "source": source, "target": target, "seed": seed,
            "src_mean": src_mean, "src_std": src_std,
        }
        if ema is not None:
            ck["ema_shadow"] = {k: v.cpu() for k,v in ema.shadow.items()}
        torch.save(ck, latest_path)

        if wait >= patience:
            log(f"EarlyStop: source-valid macro-F1@t* not improved for {patience} epochs.")
            break

    train_time = time.time() - t0

    # ===== 最终评估 =====
    ckpt = torch.load(best_path, map_location="cpu")
    base.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    t_star_ckpt = float(ckpt.get("best_thresh", 0.5))

    if ema is not None:
        ema.apply_shadow()
    try:
        src_te_metrics_plain = eval_binary_fixed_thresh(model, src_te_loader, device, thresh=t_star_ckpt)
        tgt_te_metrics_plain = eval_binary_fixed_thresh(model, tgt_te_loader, device, thresh=t_star_ckpt)

        log("\n==== Final (best by Src-Valid macro-F1@t* | Plain, no calib) ====")
        log(f"t*_ckpt = {t_star_ckpt:.3f}")
        log(f"Source({source}) Test: acc={src_te_metrics_plain['accuracy']:.4f} macroF1={src_te_metrics_plain['macro_f1']:.4f}")
        log(f"Target({target}) Test: acc={tgt_te_metrics_plain['accuracy']:.4f} macroF1={tgt_te_metrics_plain['macro_f1']:.4f}")

        ts = fit_temperature(model, src_va_loader, device)
        T_value = float(ts.logT.exp().item())
        log(f"[Calib] learned temperature T = {T_value:.3f}")

        t_star_cal, f1_src_va_cal = find_best_t_with_ts(model, src_va_loader, device, ts, step=THRESH_SCAN_STEP)
        log(f"[Src-Valid|Calib] macroF1@t*={f1_src_va_cal:.4f} (t*={t_star_cal:.3f}) | t*_ckpt={t_star_ckpt:.3f}")

        src_te_metrics_cal, pr_src_te = eval_with_ts(model, src_te_loader, device, ts, t_star_cal)
        tgt_te_metrics_cal, pr_tgt_te = eval_with_ts(model, tgt_te_loader, device, ts, t_star_cal)
        log(f"[Eval|Calib] PR_src_test@t*={pr_src_te:.3f}  PR_tgt_test@t*={pr_tgt_te:.3f}")

        t_rate, pr_src_val = rate_matched_threshold(model, src_va_loader, tgt_te_loader, device, ts, t_star_cal)
        tgt_te_metrics_rm, pr_tgt_rm = eval_with_ts(model, tgt_te_loader, device, ts, t_rate)
        log(f"[RateMatch] src_valid PR@t*={pr_src_val:.3f} -> choose t_rate={t_rate:.3f}  => tgt_PR={pr_tgt_rm:.3f}")
    finally:
        if ema is not None:
            ema.restore()

    results = {
        "mode": ("DANN_SSDA" if (semi and tgt_label_ratio>0) else "DANN_UDA"),
        "source":source, "target":target, "seed":seed, "modality":MODALITY,
        "class_names": CLASS_NAMES,
        "sizes": {
            "src_train": len(src_train), "src_valid": len(src_valid), "src_test": len(src_test),
            "tgt_train_full": len(tgt_train_full),
            "tgt_train_labeled": (len(tgt_lab_ds) if tgt_lab_ds else 0),
            "tgt_train_unlabeled": (len(tgt_unlab_ds) if tgt_unlab_ds else 0),
            "tgt_test": len(tgt_test)
        },
        "train_time_sec": train_time,
        "device_info": devinfo, "env_info": envinfo,
        "hyper_params": {
            "batch_src": batch_src, "batch_tgt": batch_tgt,
            "epochs": epochs, "lr": lr, "weight_decay": wd,
            "num_workers": num_workers, "prefetch_factor": (PREFETCH_FACTOR if num_workers>0 else 0), "amp": AMP,
            "lambda_domain_init": LAMBDA_DOMAIN,
            "lambda_domain_final": current_lambda,
            "lambda_adapt": lambda_adapt,
            "lambda_min": lambda_min, "lambda_max": min(lambda_max, lambda_hard_max),
            "dom_target_center": dom_target_center, "dom_target_band": dom_target_band,
            "lambda_kp": lambda_kp, "lambda_min_step": lambda_min_step, "lambda_max_step": lambda_max_step,
            "alpha_slope": alpha_slope,
            "alpha_max_init": ALPHA_MAX_INIT, "alpha_max_final": alpha_max_curr,
            "alpha_min_cap": ALPHA_MIN_CAP, "alpha_max_cap": min(alpha_max_cap, alpha_hard_max),
            "alpha_adapt": alpha_adapt, "alpha_step": alpha_step,
            "dann_warmup_epochs": DANN_WARMUP_EPOCHS,
            #"gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", GPU_ID),
            "use_dataparallel": USE_DATAPARALLEL,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "thresh_scan_step": THRESH_SCAN_STEP,
            "ema_used": bool(use_ema), "ema_beta": ema_beta,
            "use_gn": bool(use_gn), "gn_groups": GN_GROUPS,
            "hard_caps": {"lambda_hard_max": lambda_hard_max, "alpha_hard_max": alpha_hard_max},
            "health_triggers": {
                "domacc_ema_beta": DOMACC_EMA_BETA,
                "trigA_low": TRIG_A_LOW, "trigB_high": TRIG_B_HIGH,
                "K_consec": TRIG_K_CONSEC, "C_m_up": TRIG_C_M_UP, "C_f1_drop_ratio": TRIG_C_F1_DROP_RATIO,
                "actions": {"A":{"dLambda":-A_LAMBDA_DEC,"dAlphaMax":-A_ALPHA_DEC,"dom_lr_mul":A_DOMLR_MUL},
                            "B":{"dLambda":B_LAMBDA_INC,"dAlphaMax":B_ALPHA_INC},
                            "C":{"pause_one_epoch":True}}
            },
            "safe_mode": {
                "entered_when_A_accum_ge": SAFE_A_ACCUM_S,
                "recover_band":[SAFE_RECOVER_BAND_L, SAFE_RECOVER_BAND_U],
                "recover_need_R": SAFE_RECOVER_NEED_R,
                "resume_step": SAFE_RESUME_STEP
            },
            "semi": bool(semi and tgt_label_ratio>0),
            "tgt_label_ratio": float(tgt_label_ratio),
            "w_tgt_sup": float(w_tgt_sup),
            "dom_include_labeled": bool(dom_include_labeled),
            "dom_labeled_weight": float(dom_labeled_weight),
        },
        "final": {
            "src_test": src_te_metrics_plain,
            "tgt_test": tgt_te_metrics_plain,
            "best_thresh": t_star_ckpt,
            "best_thresh_ckpt": t_star_ckpt,
            "best_thresh_calib": t_star_cal,
            "rate_matched_thresh": t_rate,
            "calibration_T": T_value,
            "src_test_at_tcal": src_te_metrics_cal,
            "tgt_test_at_tcal": tgt_te_metrics_cal,
            "tgt_test_at_trate": tgt_te_metrics_rm
        },
        "paths": {"best_ckpt": best_path, "latest_ckpt": latest_path, "history_csv": hist_csv, "result_json": result_js},
        "timestamp": datetime.now().isoformat()
    }
    with open(result_js, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

# =================== 主流程 ===================
def parse_pairs_str_list(pairs_tokens):
    out = []
    if not pairs_tokens: return out
    for tok in pairs_tokens:
        if isinstance(tok, (list, tuple)) and len(tok) == 2:
            a, b = str(tok[0]).strip(), str(tok[1]).strip()
            if not a or not b: raise ValueError(f"Bad pair tuple: {tok}")
            out.append((a, b)); continue
        tok = str(tok).strip()
        if not tok: continue
        if "->" in tok: a,b = tok.split("->",1)
        elif ":" in tok: a,b = tok.split(":",1)
        elif "," in tok: a,b = tok.split(",",1)
        else: raise ValueError(f"Unrecognized pair token: {tok}")
        a,b = a.strip(), b.strip()
        if not a or not b: raise ValueError(f"Bad pair token: {tok}")
        out.append((a,b))
    return out

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--gpu", type=str, default=GPU_ID)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-src", type=int, default=BATCH_SRC)
    parser.add_argument("--batch-tgt", type=int, default=BATCH_TGT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-domain", type=float, default=LAMBDA_DOMAIN)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--domains", type=str, nargs="+", default=DOMAINS)
    parser.add_argument("--data-root", type=str, default=DOMAIN_DATA_DIRS)
    parser.add_argument("--out-root", type=str, default=OUT_ROOT)

    # 控制器参数
    parser.add_argument("--lambda-adapt", type=int, default=int(LAMBDA_ADAPT))
    parser.add_argument("--lambda-min", type=float, default=LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=LAMBDA_MAX)
    parser.add_argument("--lambda-hard-max", type=float, default=LAMBDA_HARD_MAX)
    parser.add_argument("--dom-target-center", type=float, default=DOM_TARGET_CENTER)
    parser.add_argument("--dom-target-band", type=float, default=DOM_TARGET_BAND)
    parser.add_argument("--lambda-kp", type=float, default=LAMBDA_KP)
    parser.add_argument("--lambda-min-step", type=float, default=LAMBDA_MIN_STEP)
    parser.add_argument("--lambda-max-step", type=float, default=LAMBDA_MAX_STEP)

    parser.add_argument("--alpha-slope", type=float, default=ALPHA_SLOPE)
    parser.add_argument("--alpha-max-init", type=float, default=ALPHA_MAX_INIT)
    parser.add_argument("--alpha-min-cap", type=float, default=ALPHA_MIN_CAP)
    parser.add_argument("--alpha-max-cap", type=float, default=ALPHA_MAX_CAP)
    parser.add_argument("--alpha-hard-max", type=float, default=ALPHA_HARD_MAX)
    parser.add_argument("--alpha-adapt", type=int, default=int(ALPHA_ADAPT))
    parser.add_argument("--alpha-step", type=float, default=ALPHA_STEP)

    # EMA/GN/SWA
    parser.add_argument("--ema", type=int, default=int(USE_EMA))
    parser.add_argument("--ema-beta", type=float, default=EMA_BETA)
    parser.add_argument("--use-gn", type=int, default=int(USE_GROUPNORM))
    parser.add_argument("--swa", type=int, default=int(USE_SWA))

    # 计划
    parser.add_argument("--cross-train-all", type=int, default=int(CROSS_TRAIN_ALL))
    parser.add_argument("--pairs", type=str, nargs="*", default=[f"{a}->{b}" for a,b in MANUAL_PAIRS])

    # 半监督
    parser.add_argument("--semi", type=int, default=SEMI)
    parser.add_argument("--tgt-label-ratio", type=float, default=TGT_LABEL_RATIO)
    parser.add_argument("--w-tgt-sup", type=float, default=W_TGT_SUP)

    # 域对抗额外开关
    parser.add_argument("--dom-include-labeled", type=int, default=DOM_INCLUDE_LABELED)
    parser.add_argument("--dom-labeled-weight", type=float, default=DOM_LABELED_WEIGHT)

    args = parser.parse_args()

    #if args.gpu is not None:
    #    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_root = args.data_root
    out_root  = args.out_root
    domains   = args.domains
    seeds     = args.seeds

    os.makedirs(out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] device={device} | cuda_available={torch.cuda.is_available()} | gpu={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    log(f"[INFO] MODALITY={MODALITY}")
    log(f"[INFO] cross_train_all={bool(args.cross_train_all)} | SEMI={bool(args.semi)} ratio={args.tgt_label_ratio:.3f} w_tgt={args.w_tgt_sup:.2f} | "
        f"DOM(include_labeled)={bool(args.dom_include_labeled)} w_lab={args.dom_labeled_weight:.2f}")

    if int(args.cross_train_all) == 1:
        plan_pairs = [(s,t) for s in domains for t in domains if s!=t]
    else:
        plan_pairs = parse_pairs_str_list(args.pairs)
        if not plan_pairs:
            raise ValueError("cross-train-all=0 但未提供 --pairs")

    log("[INFO] Training schedule (pairs × seeds):")
    for (s,t) in plan_pairs:
        log(f"  - {s}->{t} x {len(seeds)} seeds")

    all_runs = []
    for (src, tgt) in plan_pairs:
        for seed in seeds:
            set_seed(seed)
            res = train_one_pair(
                data_root, out_root,
                src, tgt, seed, device,
                batch_src=args.batch_src, batch_tgt=args.batch_tgt,
                epochs=args.epochs, lr=args.lr, wd=args.wd,
                lambda_domain=args.lambda_domain, patience=args.patience,
                num_workers=args.num_workers,
                lambda_adapt=bool(args.lambda_adapt),
                lambda_min=args.lambda_min, lambda_max=args.lambda_max,
                dom_target_center=args.dom_target_center, dom_target_band=args.dom_target_band,
                lambda_kp=args.lambda_kp, lambda_min_step=args.lambda_min_step, lambda_max_step=args.lambda_max_step,
                alpha_slope=args.alpha_slope, alpha_max_init=args.alpha_max_init,
                alpha_min_cap=args.alpha_min_cap, alpha_max_cap=args.alpha_max_cap,
                alpha_adapt=bool(args.alpha_adapt), alpha_step=args.alpha_step,
                lambda_hard_max=args.lambda_hard_max, alpha_hard_max=args.alpha_hard_max,
                use_gn=bool(args.use_gn),
                use_ema=bool(args.ema), ema_beta=args.ema_beta,
                semi=int(args.semi), tgt_label_ratio=float(args.tgt_label_ratio), w_tgt_sup=float(args.w_tgt_sup),
                dom_include_labeled=int(args.dom_include_labeled), dom_labeled_weight=float(args.dom_labeled_weight)
            )
            all_runs.append(res)
            torch.cuda.empty_cache(); gc.collect()

    # 汇总
    summary = {}
    seen_order = []
    for r in all_runs:
        key = f"{r['source']}->{r['target']}"
        if key not in seen_order:
            seen_order.append(key)

    for key in seen_order:
        src, tgt = key.split("->", 1)
        rows = [r for r in all_runs if r["source"]==src and r["target"]==tgt]
        tgt_acc  = [r["final"]["tgt_test"]["accuracy"] for r in rows]
        tgt_mf1  = [r["final"]["tgt_test"]["macro_f1"] for r in rows]
        src_acc  = [r["final"]["src_test"]["accuracy"] for r in rows]
        src_mf1  = [r["final"]["src_test"]["macro_f1"] for r in rows]
        def mean_ci(x):
            if not x: return 0.0, 0.0
            mu = float(np.mean(x)); sd = float(np.std(x, ddof=1)) if len(x)>1 else 0.0
            ci = 1.96*sd/ math.sqrt(len(x)) if len(x)>1 else 0.0
            return round(mu,4), round(ci,4)
        summary[key] = {
            "tgt_acc_mean": mean_ci(tgt_acc)[0], "tgt_acc_CI95": mean_ci(tgt_acc)[1],
            "tgt_macroF1_mean": mean_ci(tgt_mf1)[0], "tgt_macroF1_CI95": mean_ci(tgt_mf1)[1],
            "src_acc_mean": mean_ci(src_acc)[0], "src_acc_CI95": mean_ci(src_acc)[1],
            "src_macroF1_mean": mean_ci(src_mf1)[0], "src_macroF1_CI95": mean_ci(src_mf1)[1],
            "num_runs": len(rows)
        }

    out_sum = os.path.join(out_root, f"summary_{'ssda' if args.semi and args.tgt_label_ratio>0 else 'dann'}_{MODALITY.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "seeds": seeds,
            "domains_used": domains,
            "pairs_run": seen_order,
            "modality": MODALITY,
            "class_names": CLASS_NAMES,
            "data_root": data_root, "out_root": out_root,
            "note": ("SSDA with tgt-label-ratio=%.3f; Best ckpt chosen by SOURCE-VALID macro-F1 at best threshold. "
                     "Source mean/std used for all splits. Eval also includes TempScale & rate-matched threshold. "
                     "Lambda/alpha homeostatic controller + Health triggers + SAFE MODE + EMA weights.") % (args.tgt_label_ratio),
        }, f, indent=2, ensure_ascii=False)
    log(f"[DONE] summary saved to {out_sum}")

if __name__ == "__main__":
    main()
