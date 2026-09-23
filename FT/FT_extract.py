# -*- coding: utf-8 -*-

"""
三合一微调（全量微调：所有层都参与训练）
- 读取你此前的 best 基线模型（不覆盖）
- 在目标域 train/valid 上全量微调（支持按比例抽样目标域 train 子集，每个训练 seed 使用不同的随机子集）
- 微调后在四个域上测试
- 输出目录独立：{OUT_ROOT}/{source}_to_{target}/{modal_folder}_ftfull(_r{ratio})_seed{seed}/...
"""

# ========================
# 全局安静
# ========================
import os, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ========================
# 标准库 & 第三方
# ========================
import json, math, random, gc, time, platform
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

# ========================
# 1) 可调参数（集中管理）
# ========================

DATA_ROOT = Path("/path/to/US_EQ_EX")

DATA_DIRS = {
    "base": DATA_ROOT / "BASE" / "processing_BASE_outputs" / "legacy_dataset",
    "enam": DATA_ROOT / "ENAM" / "processing_ENAM_outputs" / "legacy_dataset",
    "msh": DATA_ROOT / "MSH" / "processing_MSH_outputs" / "legacy_dataset",
    "hlp": DATA_ROOT / "HLP" / "processing_HLP_outputs" / "legacy_dataset",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "FT"
)
BASELINE_ROOT = OUT_ROOT

DOMAINS       = ["base", "enam", "msh", "hlp"]   # 源域/目标域集合
MODALITY      = "STFT"                    
SEEDS         = [0, 1, 2, 3, 4]               
RUN_ALL_PAIRS = True                              # True: 跑所有 (source!=target) 组合；False: 只跑 FIXED_SRC→FIXED_TGT
FIXED_SRC     = "hlp"                            # RUN_ALL_PAIRS=False 时生效
FIXED_TGT     = "msh"                            # RUN_ALL_PAIRS=False 时生效

# ========= 新增：目标域训练集抽样控制（与训练 seed 绑定）=========
FT_TGT_RATIO = 0.10   # ∈(0,1]，例如 0.05 / 0.10；=1.0 表示用全量
# 说明：抽样随机性直接使用当前训练 seed，这样不同训练种子→不同子集，但比例一致

# 微调超参（全量微调）
FT_BATCH_SIZE    = 64
FT_EPOCHS        = 50             # 全量一般需要更久；可按valid曲线提前停止
FT_LR_FULL       = 1e-6           # ★全量微调学习率更小（从预训练点出发，防止破坏已有特征）
FT_WEIGHT_DECAY  = 1e-4           # 略小于你之前的1e-3，更保守；如需保持一致可改回1e-3
FT_PATIENCE      = 7              # 比只训头略大
NUM_WORKERS      = 4
AMP              = True

# GPU（用卡1）
GPU_ID            = "0"           # ★使用卡0
USE_DATAPARALLEL  = False

# 类名
CLASS_NAMES = ["earthquake", "explosion"]

# .pt 键名（保持自动探测）
FORCE_X_KEY = None
FORCE_Y_KEY = None

if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# ========================
# 打印
# ========================
def log(*args): print(*args, flush=True)

# ========================
# 2) 模型
# ========================
class CNN3(nn.Module):
    def __init__(self, in_channels=1, feature_dim_reduced=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flat_dim = 128 * 32 * 32
        self.reduce_fc = nn.Linear(self.flat_dim, 128)
        self.out_fc    = nn.Linear(128, 1)  # BCEWithLogits

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.reduce_fc(x))
        x = self.out_fc(x)
        return x  # [B,1]

# ========================
# 3) 数据解析与数据集
# ========================
def _labels_to_float_tensor(y):
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    elif isinstance(y, (list, tuple)):
        buf = []
        for v in y:
            if isinstance(v, torch.Tensor):
                buf.append(float(v.view(-1)[0].item()))
            else:
                vv = np.array(v).reshape(-1)
                buf.append(float(vv[0]))
        y_np = np.array(buf, dtype=np.float32)
    else:
        y_np = np.array(y).reshape(-1).astype(np.float32)
    y_np = y_np.reshape(-1)
    return torch.from_numpy(y_np).float().view(-1, 1)

def _pick_xy_from_dict(d, indices=None):
    if "data" in d and "labels_num" in d:
        xraw = d["data"]
        yraw = d["labels_num"]
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long)
            if isinstance(xraw, list):
                x = torch.stack([torch.as_tensor(xraw[i]) for i in indices], dim=0)
            else:
                x = torch.as_tensor(xraw).index_select(0, idx)
            y = torch.as_tensor(yraw).index_select(0, idx)
        else:
            x = (torch.stack([torch.as_tensor(v) for v in xraw], dim=0)
                 if isinstance(xraw, list) else torch.as_tensor(xraw))
            y = yraw
        return x, y
    if FORCE_X_KEY and FORCE_Y_KEY and FORCE_X_KEY in d and FORCE_Y_KEY in d:
        return d[FORCE_X_KEY], d[FORCE_Y_KEY]
    candidates = [
        ("data", "labels"), ("x", "y"), ("inputs", "targets"),
        ("images", "labels"), ("waveforms", "labels"), ("samples", "targets"),
    ]
    for kx, ky in candidates:
        if kx in d and ky in d:
            return d[kx], d[ky]
    return None

class PTDataset(Dataset):
    """
    - 读 .pt；支持训练集统计 mean/std，并在内部完成归一化；
    - 新增 indices：若提供，将只保留该子集，并在 is_train=True 时对“子集”重算 mean/std。
    """
    def __init__(self, pt_path, mean=None, std=None, is_train=False, indices=None):
        obj = torch.load(pt_path, map_location="cpu")
        if isinstance(obj, dict):
            picked = _pick_xy_from_dict(obj, indices=indices)
            if picked is None:
                raise ValueError(f"Unrecognized dict keys in {pt_path} -> keys={list(obj.keys())}")
            x, y = picked
            if "data" in obj and "labels_num" in obj:
                indices = None  # 已在 stack 之前完成子集筛选
        elif isinstance(obj, (list, tuple)):
            xs, ys = [], []
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    xi, yi = it
                elif isinstance(it, dict) and ("data" in it and ("label" in it or "labels_num" in it)):
                    xi, yi = it["data"], it.get("label", it.get("labels_num"))
                else:
                    raise ValueError(f"Unrecognized list element in {pt_path}: type={type(it)}")
                xs.append(torch.as_tensor(xi))
                yi = np.array(yi).reshape(-1)
                ys.append(float(yi[0]))
            x = torch.stack(xs, dim=0)
            y = np.array(ys, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported .pt format: {type(obj)} in {pt_path}")

        x = torch.as_tensor(x, dtype=torch.float32)
        y = _labels_to_float_tensor(y)

        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"x shape should be [N,1,H,W], got {tuple(x.shape)} from {pt_path}")

        # ---- 可选子集裁剪 ----
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long)
            x = x.index_select(0, idx)
            y = y.index_select(0, idx)

        # manuscript/legacy_dataset 中的 STFT 已经完成 abs + log1p。
        # 保留中心化后的负值，只在训练集上重新计算全局 mean/std。
        self.x_raw = x
        self.is_train = is_train

        if mean is None or std is None:
            if is_train:
                self.mean = self.x_raw.mean().item()
                self.std  = self.x_raw.std().item()
            else:
                raise ValueError("mean/std must be provided for non-train split")
        else:
            self.mean = float(mean); self.std = float(std)

        self.x = (self.x_raw - self.mean) / (self.std + 1e-6)
        self.y = y

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

# ========================
# 4) 通用工具
# ========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True; cudnn.benchmark = False

def device_info():
    if torch.cuda.is_available():
        ng = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(ng)]
        return {"device": "cuda", "num_gpus": ng, "gpu_names": names, "cuda_version": torch.version.cuda}
    else:
        return {"device": "cpu", "num_gpus": 0, "gpu_names": [], "cuda_version": None}

def data_pt_path(modality: str, ds: str, split: str):
    ds = ds.lower()
    if ds not in DATA_DIRS:
        raise ValueError(f"Unknown domain {ds!r}; expected one of {sorted(DATA_DIRS)}")
    base = f"{ds}_dataset_zrt_waveforms_ZRT_DanFenLiang_40HZ_90S_EventFiltered"
    if split == "train":
        fname = f"{base}_train_{modality}_raw_ALL13.pt"
    elif split == "valid":
        fname = f"{base}_valid_{modality}_raw.pt"
    elif split == "test":
        fname = f"{base}_test_{modality}_raw.pt"
    else:
        raise ValueError(f"Unknown split {split!r}; expected train, valid, or test")
    path = os.path.join(DATA_DIRS[ds], fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path

def compute_metrics(y_true, y_pred, n_classes=2):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
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

def strip_module_prefix(state_dict):
    """兼容 DataParallel 保存的权重（'module.' 前缀）"""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

# 旧基线 best 路径解析（不覆盖）
def baseline_ckpt_path(modality: str, source: str, seed: int):
    folder_map = {"STFT": "stft"}
    fname_map  = {"STFT": f"{source}_best.pth"}
    if modality not in folder_map:
        raise ValueError(f"Unsupported modality: {modality}")
    run_dir = os.path.join(BASELINE_ROOT, source, f"{folder_map[modality]}_seed{seed}")
    ckpt = os.path.join(run_dir, fname_map[modality])
    return ckpt, run_dir

# 本次微调的输出目录（不覆盖）；ratio<1时带上 _r{ratio}
def finetune_run_dir(modality: str, source: str, target: str, seed: int, ratio: float):
    folder_map = {"STFT": "stft", "CWT": "cwt", "LogMel40rel": "logmel"}
    rtag = f"_r{ratio:.2f}" if ratio < 0.9999 else ""
    out_dir = os.path.join(OUT_ROOT, f"{source}_to_{target}", f"{folder_map[modality]}_ftfull{rtag}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def choose_subset_indices(n: int, ratio: float, seed: int):
    """返回按比例采样的下标列表；保证至少1个样本；随机性绑定在训练 seed 上"""
    if ratio >= 1.0:
        return list(range(n))
    if ratio <= 0.0:
        raise ValueError("FT_TGT_RATIO 必须 > 0")
    k = max(1, int(round(n * ratio)))
    rng = np.random.RandomState(seed=seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:k].tolist()

# ========================
# 5) 微调（全量微调）
# ========================
def finetune_full_one_pair(modality: str, source: str, target: str, seed: int, device: torch.device):
    """
    加载 source 的 best 预训练 → 在 target train/valid 上全量微调（train可抽样；不同训练seed用不同子集） → 测试四个域
    """
    # ---- 路径/设备/环境 ----
    devinfo = device_info()
    envinfo = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cuda": devinfo.get("cuda_version"),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")
    }
    out_dir = finetune_run_dir(modality, source, target, seed, FT_TGT_RATIO)
    best_path   = os.path.join(out_dir, f"{source}_to_{target}_best_ftfull.pth")
    history_csv = os.path.join(out_dir, f"{source}_to_{target}_history_ftfull.csv")
    result_json = os.path.join(out_dir, f"{source}_to_{target}_results_ftfull.json")

    # 先检查 baseline，避免权重缺失时先加载数 GB 的目标数据。
    ckpt_path, base_run_dir = baseline_ckpt_path(modality, source, seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"预训练权重未找到：{ckpt_path}\n"
            f"请先运行 Baseline_extract 生成 source={source}, seed={seed} 的基线权重。"
        )

    # ---- 统计目标域 train 总样本数并抽样 ----
    full_train_pt = data_pt_path(modality, target, "train")
    tmp = torch.load(full_train_pt, map_location="cpu")
    if isinstance(tmp, dict):
        if "data" in tmp:
            n_train_total = len(tmp["data"])
        else:
            xraw, _ = _pick_xy_from_dict(tmp)
            n_train_total = int(xraw.shape[0]) if isinstance(xraw, torch.Tensor) else len(xraw)
    elif isinstance(tmp, (list, tuple)):
        n_train_total = len(tmp)
    else:
        raise ValueError(f"Unsupported .pt for counting: {type(tmp)}")
    del tmp
    gc.collect()

    subset_idx = choose_subset_indices(n_train_total, FT_TGT_RATIO, seed=seed)  # ★关键：用训练seed
    # 存档子集索引，便于复现
    np.save(os.path.join(out_dir, "subset_indices.npy"), np.array(subset_idx, dtype=np.int64))

    # ---- 数据集（训练子集上重算 mean/std；valid 用该 mean/std）----
    train_set = PTDataset(full_train_pt, mean=None, std=None, is_train=True, indices=subset_idx)
    mean, std = train_set.mean, train_set.std
    valid_set = PTDataset(data_pt_path(modality, target, "valid"), mean=mean, std=std, is_train=False)

    train_loader = DataLoader(train_set, batch_size=FT_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=FT_BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ---- 模型：加载基线 best ----
    log(f"[LOAD] baseline ckpt: {ckpt_path}")

    model = CNN3(in_channels=1).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = strip_module_prefix(state.get("model", state))
    model.load_state_dict(state_dict, strict=True)

    if USE_DATAPARALLEL and torch.cuda.device_count() > 1 and devinfo["device"] == "cuda":
        model = nn.DataParallel(model)

    # ---- 优化器/损失/调度/AMP ----
    # 全量微调：所有参数都参与
    params = [p for p in model.parameters()]
    optimizer = optim.Adam(params, lr=FT_LR_FULL, weight_decay=FT_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(enabled=(AMP and torch.cuda.is_available()))

    # ---- 训练 ----
    best_val_f1 = -1.0
    wait = 0
    history = []
    t0 = time.time()

    log(f"\n===== [FT-FULL] {source} → {target} | modality={modality} | seed={seed} "
        f"| Train(total)={n_train_total} | Train(used)={len(train_set)} | Valid {len(valid_set)} "
        f"| ratio={FT_TGT_RATIO:.4f} | device={devinfo['device']} =====")

    for epoch in range(1, FT_EPOCHS + 1):
        ep_start = time.time()

        # Train
        model.train()
        tr_loss, tr_true, tr_pred = 0.0, [], []
        pbar = tqdm(train_loader, desc=f"[FT-FULL {source}->{target}|seed{seed}] Train {epoch}/{FT_EPOCHS}",
                    dynamic_ncols=True, leave=False, mininterval=0.3)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled() and torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            tr_loss += loss.item() * xb.size(0)
            tr_true.extend(yb.detach().cpu().numpy().ravel().tolist())
            tr_pred.extend(preds.detach().cpu().numpy().ravel().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss /= len(train_set)
        tr_metrics = compute_metrics(tr_true, tr_pred, n_classes=2)

        # Valid
        model.eval()
        va_loss, va_true, va_pred = 0.0, [], []
        pbar_v = tqdm(valid_loader, desc=f"[FT-FULL {source}->{target}|seed{seed}] Valid {epoch}/{FT_EPOCHS}",
                      dynamic_ncols=True, leave=False, mininterval=0.3)
        with torch.no_grad():
            for xb, yb in pbar_v:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(enabled=scaler.is_enabled() and torch.cuda.is_available()):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                va_loss += loss.item() * xb.size(0)
                va_true.extend(yb.detach().cpu().numpy().ravel().tolist())
                va_pred.extend(preds.detach().cpu().numpy().ravel().tolist())
                pbar_v.set_postfix(loss=f"{loss.item():.4f}")

        va_loss /= len(valid_set)
        va_metrics = compute_metrics(va_true, va_pred, n_classes=2)
        scheduler.step(va_loss)

        cur_lr = float(optimizer.param_groups[0]["lr"])
        ep_time = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "valid_loss": va_loss,
            "lr": cur_lr,
            "epoch_time_sec": ep_time,
            "train_metrics": tr_metrics,
            "valid_metrics": va_metrics
        })

        if va_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = va_metrics["macro_f1"]
            # 保存：不覆盖基线；保存到新的 out_dir
            torch.save({
                "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "mean": mean, "std": std,
                "epoch": epoch,
                "train_set_size_total": n_train_total,
                "train_set_size_used": len(train_set),
                "valid_set_size": len(valid_set),
                "class_names": CLASS_NAMES,
                "source": source, "target": target, "modality": modality,
                "seed": seed, "finetune": "full",
                "tgt_ratio": float(FT_TGT_RATIO),
                "tgt_subset_seed": int(seed)  # ★记录：本次子集随机数种子=训练seed
            }, best_path)
            wait = 0; tag = "*"
        else:
            wait += 1; tag = " "

        log(f"[FT-FULL {source}->{target}|seed{seed}] ep{epoch:03d}{tag} "
            f"tr_loss={tr_loss:.4f} tr_f1={tr_metrics['macro_f1']:.4f} "
            f"va_loss={va_loss:.4f} va_f1={va_metrics['macro_f1']:.4f} "
            f"acc={va_metrics['accuracy']:.4f} lr={cur_lr:.2e} time={ep_time:.1f}s")

        if wait >= FT_PATIENCE:
            log(f"[FT-FULL {source}->{target}|seed{seed}] EarlyStop: valid macro-F1 no improve for {FT_PATIENCE} epochs.")
            break

    train_time = time.time() - t0

    # ---- 测试：四个域（统一用“训练子集”统计到的 mean/std 做归一化）----
    ckpt = torch.load(best_path, map_location="cpu")
    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt["model"])
    model.eval()

    def eval_on(domain):
        test_set = PTDataset(data_pt_path(modality, domain, "test"), mean=mean, std=std, is_train=False)
        test_loader = DataLoader(test_set, batch_size=FT_BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
        y_true, y_pred = [], []
        pbar_t = tqdm(test_loader, desc=f"[FT-FULL {source}->{target}|seed{seed}] Test @{domain}",
                      dynamic_ncols=True, leave=False, mininterval=0.3)
        with torch.no_grad():
            for xb, yb in pbar_t:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(enabled=scaler.is_enabled() and torch.cuda.is_available()):
                    logits = model(xb)
                preds = (torch.sigmoid(logits) > 0.5).float()
                y_true.extend(yb.detach().cpu().numpy().ravel().tolist())
                y_pred.extend(preds.detach().cpu().numpy().ravel().tolist())
        metrics = compute_metrics(y_true, y_pred, n_classes=2)
        return metrics, len(test_set)

    tests = {}
    for tgt_eval in DOMAINS:
        m, ntest = eval_on(tgt_eval)
        tests[tgt_eval] = {"num_samples": ntest, **m}
        log(f"[FT-FULL {source}->{target}|seed{seed}] Test@{tgt_eval}: "
            f"acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} weightedF1={m['weighted_f1']:.4f}")

    # 结果保存（新目录，不覆盖基线）
    results = {
        "mode": "finetune_full",
        "source": source, "target": target,
        "seed": seed, "modality": modality,
        "class_names": CLASS_NAMES,
        "train_samples_total": n_train_total,
        "train_samples_used": len(train_set),
        "valid_samples": len(valid_set),
        "tgt_ratio": float(FT_TGT_RATIO),
        "tgt_subset_seed": int(seed),
        "best_valid_macro_f1": best_val_f1,
        "train_time_sec": train_time,
        "timestamp": datetime.now().isoformat(),
        "device_info": devinfo, "env_info": envinfo,
        "hyper_params": {
            "batch_size": FT_BATCH_SIZE, "epochs": FT_EPOCHS, "lr_full": FT_LR_FULL,
            "weight_decay": FT_WEIGHT_DECAY, "patience": FT_PATIENCE,
            "num_workers": NUM_WORKERS, "amp": AMP,
            "gpu_id": GPU_ID, "use_dataparallel": USE_DATAPARALLEL
        },
        "history": history,
        "tests": tests,
        "paths": {
            "baseline_ckpt": ckpt_path,
            "baseline_run_dir": base_run_dir,
            "finetune_out_dir": out_dir,
            "finetune_best_ckpt": best_path,
            "finetune_history_csv": history_csv,
            "finetune_result_json": result_json,
            "subset_indices_npy": os.path.join(out_dir, "subset_indices.npy")
        }
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(history_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,valid_loss,train_macro_f1,valid_macro_f1,lr,epoch_time_sec\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']:.6f},{h['valid_loss']:.6f},"
                    f"{h['train_metrics']['macro_f1']:.6f},{h['valid_metrics']['macro_f1']:.6f},"
                    f"{h['lr']:.8f},{h['epoch_time_sec']:.3f}\n")

    return results


# ========================
# 6) 主流程
# ========================
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] device={device}; cuda_available={torch.cuda.is_available()} | modality={MODALITY} | tgt_ratio={FT_TGT_RATIO}")

    all_runs = []
    pairs = []
    if RUN_ALL_PAIRS:
        for s in DOMAINS:
            for t in DOMAINS:
                if s != t:
                    pairs.append((s, t))
    else:
        pairs.append((FIXED_SRC, FIXED_TGT))

    for (src, tgt) in pairs:
        for seed in SEEDS:
            set_seed(seed)
            cudnn.benchmark = True
            res = finetune_full_one_pair(MODALITY, src, tgt, seed, device)
            all_runs.append(res)
            torch.cuda.empty_cache(); gc.collect()

    # 汇总：每个 (src->tgt, eval_domain) 的均值 & 95%CI
    summary = {}
    for (src, tgt) in pairs:
        key = f"{src}->{tgt}"
        summary[key] = {}
        runs = [r for r in all_runs if r["source"] == src and r["target"] == tgt]
        for eval_ds in DOMAINS:
            vals_mf1 = [r["tests"][eval_ds]["macro_f1"] for r in runs]
            vals_wf1 = [r["tests"][eval_ds]["weighted_f1"] for r in runs]
            vals_acc = [r["tests"][eval_ds]["accuracy"] for r in runs]

            def mean_ci(x):
                mu = float(np.mean(x)) if len(x) else 0.0
                sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                ci = 1.96 * sd / math.sqrt(len(x)) if len(x) > 1 else 0.0
                return round(mu, 4), round(ci, 4)

            mu_mf1, ci_mf1 = mean_ci(vals_mf1)
            mu_wf1, ci_wf1 = mean_ci(vals_wf1)
            mu_acc, ci_acc = mean_ci(vals_acc)
            summary[key][eval_ds] = {
                "macroF1_mean": mu_mf1, "macroF1_CI95": ci_mf1,
                "weightedF1_mean": mu_wf1, "weightedF1_CI95": ci_wf1,
                "acc_mean": mu_acc, "acc_CI95": ci_acc
            }

    rtag = f"_r{FT_TGT_RATIO:.2f}" if FT_TGT_RATIO < 0.9999 else ""
    out_sum = os.path.join(
    OUT_ROOT,
    "finetune_full_stft_r0.10_summary.json"
    )
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "pairs": pairs,
            "seeds": SEEDS,
            "modality": MODALITY,
            "class_names": CLASS_NAMES,
            "data_root": DATA_DIRS,
            "out_root": OUT_ROOT,
            "gpu_id": GPU_ID,
            "tgt_ratio": float(FT_TGT_RATIO)
        }, f, indent=2, ensure_ascii=False)

    log(f"[DONE] finetune-full summary saved to {out_sum}")

if __name__ == "__main__":
    main()
