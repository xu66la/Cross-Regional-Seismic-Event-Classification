import argparse
import gc
import json
import math
import os
import platform
import random
import time
from datetime import datetime

# Set visibility before importing torch/CUDA.
DEFAULT_GPU = "0"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


MODALITY = "STFT"
CLASS_NAMES = ["earthquake", "explosion"]

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 10
NUM_WORKERS = 2
AMP = True
GRAD_CLIP_NORM = 1.0


def log(*args):
    print(*args, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def data_pt_path(domain: str, split: str) -> str:
    domain = domain.lower()
    if domain not in DATA_DIRS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {sorted(DATA_DIRS)}")
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split {split!r}; expected train, valid, or test")

    stem = f"{domain}_dataset_zrt_waveforms_ZRT_DanFenLiang_40HZ_90S_EventFiltered"
    suffix = (
        f"train_{MODALITY}_raw_ALL13.pt"
        if split == "train"
        else f"{split}_{MODALITY}_raw.pt"
    )
    path = os.path.join(DATA_DIRS[domain], f"{stem}_{suffix}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def run_paths(domain: str, seed: int):
    run_dir = os.path.join(OUT_ROOT, domain, f"stft_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "best": os.path.join(run_dir, f"{domain}_best.pth"),
        "latest": os.path.join(run_dir, f"{domain}_latest.pth"),
        "history": os.path.join(run_dir, f"{domain}_seed{seed}_history.csv"),
        "result": os.path.join(run_dir, f"{domain}_seed{seed}_results.json"),
    }


def labels_to_tensor(values):
    if isinstance(values, torch.Tensor):
        return values.detach().float().view(-1, 1)
    return torch.as_tensor(np.asarray(values), dtype=torch.float32).view(-1, 1)


class PTDataset(Dataset):
    def __init__(self, path: str, mean=None, std=None, is_train=False):
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or "data" not in obj or "labels_num" not in obj:
            raise ValueError(
                f"Expected dict with data and labels_num in {path}; "
                f"got {type(obj)} with keys={list(obj) if isinstance(obj, dict) else None}"
            )

        prep = obj.get("stft_prep", {})
        if not isinstance(prep, dict) or prep.get("log1p") is not True:
            raise ValueError(
                f"{path} is not marked as pre-log1p STFT data; refusing to guess preprocessing"
            )

        raw = obj["data"]
        x = (
            torch.stack([torch.as_tensor(item) for item in raw], dim=0)
            if isinstance(raw, list)
            else torch.as_tensor(raw)
        ).float()
        y = labels_to_tensor(obj["labels_num"])

        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 256, 256):
            raise ValueError(f"Expected x shape [N,1,256,256], got {tuple(x.shape)} from {path}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Feature/label count mismatch in {path}: {x.shape[0]} != {y.shape[0]}")
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite input values found in {path}")
        unique_labels = set(torch.unique(y).cpu().tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(f"Expected binary labels 0/1 in {path}, got {sorted(unique_labels)}")

        # Do not repeat abs/log1p. Preserve signed, already-prepared STFT values.
        if is_train:
            mean = float(x.mean().item())
            std = float(x.std().item())
        elif mean is None or std is None:
            raise ValueError("mean/std must be supplied for non-training splits")
        if not math.isfinite(float(std)) or float(std) <= 0:
            raise ValueError(f"Invalid training std: {std}")

        self.mean = float(mean)
        self.std = float(std)
        self.x = ((x - self.mean) / (self.std + 1e-6)).contiguous()
        self.y = y.contiguous()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CNN3(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.flat_dim = 128 * 32 * 32
        self.reduce_fc = nn.Linear(self.flat_dim, 128)
        self.out_fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x.flatten(1))
        x = torch.relu(self.reduce_fc(x))
        return self.out_fc(x)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast(enabled=(AMP and device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        pred = (torch.sigmoid(logits) >= 0.5).long()
        y_true.extend(yb.cpu().numpy().ravel().tolist())
        y_pred.extend(pred.cpu().numpy().ravel().tolist())
    return total_loss / max(1, len(loader.dataset)), compute_metrics(y_true, y_pred)


def train_one(domain: str, seed: int, device: torch.device, args):
    set_seed(seed)
    paths = run_paths(domain, seed)
    if os.path.isfile(paths["result"]) and os.path.isfile(paths["best"]) and not args.force:
        log(f"[SKIP] Completed baseline exists: {paths['best']}")
        with open(paths["result"], "r", encoding="utf-8") as handle:
            return json.load(handle)

    train_set = PTDataset(data_pt_path(domain, "train"), is_train=True)
    valid_set = PTDataset(
        data_pt_path(domain, "valid"), mean=train_set.mean, std=train_set.std
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"Training set ({len(train_set)}) is smaller than batch size ({args.batch_size})"
        )

    model = CNN3().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = GradScaler(enabled=(AMP and device.type == "cuda"))

    best_f1 = -1.0
    best_epoch = 0
    wait = 0
    start_epoch = 1
    if args.resume and os.path.isfile(paths["latest"]) and not args.force:
        checkpoint = torch.load(paths["latest"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_valid_macro_f1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        wait = int(checkpoint.get("wait", 0))
        log(f"[RESUME] {domain} seed={seed} from epoch {start_epoch}")

    if start_epoch == 1 or args.force:
        with open(paths["history"], "w", encoding="utf-8") as handle:
            handle.write("epoch,train_loss,valid_loss,train_macro_f1,valid_macro_f1,lr,epoch_time_sec\n")

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []
        progress = tqdm(
            train_loader,
            desc=f"[BASELINE {domain}|seed{seed}] {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )
        for xb, yb in progress:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * xb.size(0)
            pred = (torch.sigmoid(logits) >= 0.5).long()
            y_true.extend(yb.detach().cpu().numpy().ravel().tolist())
            y_pred.extend(pred.detach().cpu().numpy().ravel().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_set)
        train_metrics = compute_metrics(y_true, y_pred)
        valid_loss, valid_metrics = evaluate(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_started

        improved = valid_metrics["macro_f1"] > best_f1
        if improved:
            best_f1 = valid_metrics["macro_f1"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": train_set.mean,
                    "std": train_set.std,
                    "epoch": epoch,
                    "best_valid_macro_f1": best_f1,
                    "best_thresh": 0.5,
                    "train_set_size": len(train_set),
                    "valid_set_size": len(valid_set),
                    "class_names": CLASS_NAMES,
                    "source": domain,
                    "seed": seed,
                    "modality": MODALITY,
                    "preprocess": {
                        "input_artifact": "abs+log1p",
                        "repeat_log1p": False,
                        "normalization": "source_train_global_zscore",
                    },
                },
                paths["best"],
            )
        else:
            wait += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "wait": wait,
                "best_epoch": best_epoch,
                "best_valid_macro_f1": best_f1,
                "mean": train_set.mean,
                "std": train_set.std,
                "source": domain,
                "seed": seed,
            },
            paths["latest"],
        )
        with open(paths["history"], "a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{train_loss:.6f},{valid_loss:.6f},"
                f"{train_metrics['macro_f1']:.6f},{valid_metrics['macro_f1']:.6f},"
                f"{lr_now:.8f},{epoch_time:.3f}\n"
            )
        log(
            f"[BASELINE {domain}|seed{seed}] ep={epoch:03d}{'*' if improved else ' '} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"valid_f1={valid_metrics['macro_f1']:.4f} lr={lr_now:.2e}"
        )
        if wait >= args.patience:
            log(f"[EARLY STOP] no validation macro-F1 improvement for {args.patience} epochs")
            break

    if not os.path.isfile(paths["best"]):
        raise RuntimeError(f"No best checkpoint was produced: {paths['best']}")
    best = torch.load(paths["best"], map_location="cpu")
    model.load_state_dict(best["model"])

    tests = {}
    for eval_domain in DATA_DIRS:
        test_set = PTDataset(
            data_pt_path(eval_domain, "test"), mean=best["mean"], std=best["std"]
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs
        )
        test_loss, metrics = evaluate(model, test_loader, criterion, device)
        tests[eval_domain] = {"num_samples": len(test_set), "loss": test_loss, **metrics}
        log(
            f"[TEST {domain}|seed{seed}] @{eval_domain}: "
            f"acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f}"
        )
        del test_loader, test_set
        gc.collect()

    result = {
        "source": domain,
        "seed": seed,
        "modality": MODALITY,
        "best_epoch": int(best["epoch"]),
        "best_valid_macro_f1": float(best["best_valid_macro_f1"]),
        "train_samples": len(train_set),
        "valid_samples": len(valid_set),
        "mean": float(best["mean"]),
        "std": float(best["std"]),
        "tests": tests,
        "train_time_sec": time.time() - started,
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "paths": paths,
        "timestamp": datetime.now().isoformat(),
    }
    with open(paths["result"], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", choices=sorted(DATA_DIRS), default=["msh"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(
        f"[INFO] device={device} visible_gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"domains={args.domains} seeds={args.seeds}"
    )
    for domain in args.domains:
        for seed in args.seeds:
            train_one(domain, seed, device, args)
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
