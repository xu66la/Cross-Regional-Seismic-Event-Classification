import argparse, gc, json, math, os, platform, random, sys, time
from datetime import datetime

GPU_DEFAULT = "1"
if "--gpu" in sys.argv:
    try: GPU_DEFAULT = sys.argv[sys.argv.index("--gpu") + 1]
    except IndexError: pass
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEFAULT
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path

torch.set_num_threads(1)

DATA_ROOT = Path("/path/to/US_EQ_EX")

DATA_DIRS = {
    "base": DATA_ROOT / "BASE" / "processing_BASE_outputs" / "legacy_dataset",
    "enam": DATA_ROOT / "ENAM" / "processing_ENAM_outputs" / "legacy_dataset",
    "msh": DATA_ROOT / "MSH" / "processing_MSH_outputs" / "legacy_dataset",
    "hlp": DATA_ROOT / "HLP" / "processing_HLP_outputs" / "legacy_dataset",
}

OUT_ROOT = Path("./outputs/DAN")

DOMAINS, SEEDS = list(DATA_DIRS), [0, 1, 2, 3, 4]
PAIRS = [(source, target) for source in DOMAINS for target in DOMAINS if source != target]
MODALITY, CLASS_NAMES = "STFT", ["earthquake", "explosion"]
BATCH_SRC, BATCH_TGT, EPOCHS = 32, 32, 50
LR, WEIGHT_DECAY, PATIENCE, NUM_WORKERS = 5e-5, 1e-4, 20, 0
LAMBDA_MMD_MAX, WARMUP_EPOCHS, MMD_GAMMA = 0.10, 5, 10.0
KERNEL_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
SEMI, TGT_LABEL_RATIO, W_TGT_SUP = 1, 0.10, 1.0
EMA_BETA, USE_GN, THRESH_STEP = 0.999, True, 0.002


def log(*x): print(*x, flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic, cudnn.benchmark = True, False


def data_path(domain, split):
    domain = domain.lower()
    if domain not in DATA_DIRS or split not in {"train", "valid", "test"}:
        raise ValueError(f"Bad domain/split: {domain}/{split}")
    stem = f"{domain}_dataset_zrt_waveforms_ZRT_DanFenLiang_40HZ_90S_EventFiltered"
    tail = f"train_{MODALITY}_raw_ALL13.pt" if split == "train" else f"{split}_{MODALITY}_raw.pt"
    path = os.path.join(DATA_DIRS[domain], f"{stem}_{tail}")
    if not os.path.isfile(path): raise FileNotFoundError(path)
    return path


class PTDataset(Dataset):
    def __init__(self, path, mean=None, std=None, train=False):
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict) or "data" not in obj or "labels_num" not in obj:
            raise ValueError(f"Expected data/labels_num dict: {path}")
        prep = obj.get("stft_prep", {})
        if not isinstance(prep, dict) or prep.get("log1p") is not True:
            raise ValueError(f"Dataset is not marked as pre-log1p: {path}")
        raw = obj["data"]
        x = (torch.stack([torch.as_tensor(v) for v in raw]) if isinstance(raw, list)
             else torch.as_tensor(raw)).float()
        y = torch.as_tensor(obj["labels_num"]).float().view(-1, 1)
        if x.ndim == 3: x = x.unsqueeze(1)
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 256, 256) or len(x) != len(y):
            raise ValueError(f"Invalid shapes x={tuple(x.shape)}, y={tuple(y.shape)} in {path}")
        if not torch.isfinite(x).all(): raise ValueError(f"Non-finite data: {path}")
        if not set(torch.unique(y).tolist()).issubset({0.0, 1.0}): raise ValueError("Labels must be 0/1")
        # Artifacts already contain abs+log1p. Preserve signed centered values.
        if train: mean, std = x.mean().item(), x.std().item()
        elif mean is None or std is None: raise ValueError("mean/std required")
        if float(std) <= 0 or not math.isfinite(float(std)): raise ValueError(f"Invalid std={std}")
        self.mean, self.std = float(mean), float(std)
        self.x = ((x - self.mean) / (self.std + 1e-6)).contiguous()
        self.y = y.contiguous()

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def labeled_subset(dataset, ratio, seed):
    if ratio <= 0: return None
    if ratio > 1: raise ValueError("tgt-label-ratio must be in [0,1]")
    labels, rng, chosen = dataset.y.view(-1).numpy().astype(int), np.random.RandomState(seed), []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label); rng.shuffle(idx)
        if not len(idx): raise ValueError(f"Target train has no class {label}")
        n = len(idx) if ratio == 1 else max(1, round(len(idx) * ratio))
        chosen.extend(idx[:n].tolist())
    rng.shuffle(chosen)
    return Subset(dataset, chosen)


def dl(dataset, batch, shuffle, workers, device, drop=False):
    kw = dict(num_workers=workers, pin_memory=device.type == "cuda")
    if workers > 0: kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, batch_size=min(batch, len(dataset)), shuffle=shuffle,
                      drop_last=drop and len(dataset) >= batch, **kw)


def norm(channels, use_gn):
    if not use_gn: return nn.BatchNorm2d(channels)
    groups = min(32, channels)
    while channels % groups and groups > 1: groups -= 1
    return nn.GroupNorm(groups, channels)


class Features(nn.Module):
    def __init__(self, use_gn=True):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(1, 32, 3, padding=1), norm(32, use_gn)
        self.c2, self.n2 = nn.Conv2d(32, 64, 3, padding=1), norm(64, use_gn)
        self.c3, self.n3 = nn.Conv2d(64, 128, 3, padding=1), norm(128, use_gn)
        self.pool, self.drop = nn.MaxPool2d(2), nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.n1(self.c1(x))))
        x = self.pool(torch.relu(self.n2(self.c2(x))))
        x = self.pool(torch.relu(self.n3(self.c3(x))))
        return self.drop(x.flatten(1))


class DAN(nn.Module):
    def __init__(self, use_gn=True):
        super().__init__(); self.feature = Features(use_gn)
        self.reduce = nn.Linear(128 * 32 * 32, 128); self.cls = nn.Linear(128, 1)

    def forward(self, x):
        h = self.reduce(self.feature(x))
        return self.cls(h), h


def dist2(x, y):
    x, y = x.float(), y.float()
    return (x.square().sum(1, keepdim=True) + y.square().sum(1)[None] - 2 * x @ y.T).clamp_min(0)


def mk_mmd(source, target):
    n = min(len(source), len(target))
    if n < 2: return source.new_zeros(())
    source, target = source[:n].float(), target[:n].float()
    total = torch.cat([source, target]); distances = dist2(total, total)
    with torch.no_grad():
        mask = ~torch.eye(2*n, dtype=torch.bool, device=source.device)
        values = distances[mask]; values = values[values > 0]
        bandwidth = (values.median() if len(values) else distances.new_tensor(1.)).clamp_min(1e-6)
    def kernel(a, b):
        d = dist2(a, b)
        return sum(torch.exp(-d / (2 * bandwidth * s * s)) for s in KERNEL_SCALES)
    off = ~torch.eye(n, dtype=torch.bool, device=source.device)
    return kernel(source, source)[off].mean() + kernel(target, target)[off].mean() - 2*kernel(source, target).mean()


class EMA:
    def __init__(self, model, beta):
        self.model, self.beta, self.backup = model, beta, None
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.beta).add_(v.detach(), alpha=1-self.beta)
    @torch.no_grad()
    def apply(self):
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for k, v in self.shadow.items(): self.model.state_dict()[k].copy_(v)
    @torch.no_grad()
    def restore(self):
        for k, v in self.backup.items(): self.model.state_dict()[k].copy_(v)
        self.backup = None


@torch.no_grad()
def predictions(model, loader, device):
    model.eval(); zz, yy = [], []
    for x, y in loader:
        z, _ = model(x.to(device, non_blocking=True)); zz.append(z.float().cpu()); yy.append(y.float())
    if not zz: raise ValueError("Empty evaluation loader")
    return torch.cat(zz).view(-1), torch.cat(yy).view(-1)


def threshold_search(labels, probs):
    labels, probs = np.asarray(labels, int), np.asarray(probs, float)
    candidates = np.unique(np.r_[np.arange(0, 1.000001, THRESH_STEP), np.quantile(probs, [.01,.05,.1,.5,.9,.95,.99])])
    scores = [f1_score(labels, probs >= t, labels=[0,1], average="macro", zero_division=0) for t in candidates]
    i = int(np.argmax(scores)); return float(candidates[i]), float(scores[i])


def metrics(labels, probs, threshold):
    y = np.asarray(labels, int); p = np.asarray(probs); pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    out = {"accuracy": float(accuracy_score(y,pred)),
           "macro_f1": float(f1_score(y,pred,labels=[0,1],average="macro",zero_division=0)),
           "weighted_f1": float(f1_score(y,pred,labels=[0,1],average="weighted",zero_division=0)),
           "per_class_f1": f1_score(y,pred,labels=[0,1],average=None,zero_division=0).tolist(),
           "confusion_matrix": cm.tolist()}
    try: out["auc"] = float(roc_auc_score(y,p))
    except ValueError: out["auc"] = None
    return out


def evaluate(model, loader, device, threshold):
    z, y = predictions(model, loader, device); p = torch.sigmoid(z).numpy()
    out = metrics(y.numpy(), p, threshold)
    out["loss"] = float(nn.functional.binary_cross_entropy_with_logits(z,y).item())
    return out


def pair_list(tokens):
    out=[]
    for token in tokens:
        sep = "->" if "->" in token else ":" if ":" in token else None
        if not sep: raise ValueError(f"Bad pair: {token}")
        s,t = [v.strip().lower() for v in token.split(sep,1)]
        if s not in DATA_DIRS or t not in DATA_DIRS or s==t: raise ValueError(f"Bad pair: {s}->{t}")
        out.append((s,t))
    return out


def train_pair(source, target, seed, device, a):
    seed_all(seed); mode = "ssda" if a.semi and a.tgt_label_ratio > 0 else "uda"
    suffix = f"stft_dan_{mode}_seed{seed}" + (f"_r{a.tgt_label_ratio:.3f}" if mode=="ssda" else "")
    run = os.path.join(a.out_root, f"{source}_to_{target}", suffix); os.makedirs(run, exist_ok=True)
    path = {k: os.path.join(run,v) for k,v in {"best":"best.pth","latest":"latest.pth","history":"history.csv","result":"results.json"}.items()}
    if os.path.isfile(path["result"]) and os.path.isfile(path["best"]) and not a.force:
        with open(path["result"],encoding="utf-8") as f: return json.load(f)

    s_tr=PTDataset(data_path(source,"train"),train=True); mean,std=s_tr.mean,s_tr.std
    s_va=PTDataset(data_path(source,"valid"),mean,std); s_te=PTDataset(data_path(source,"test"),mean,std)
    t_tr=PTDataset(data_path(target,"train"),mean,std); t_te=PTDataset(data_path(target,"test"),mean,std)
    t_lab=labeled_subset(t_tr,a.tgt_label_ratio,seed) if a.semi and a.tgt_label_ratio>0 else None
    sl=dl(s_tr,a.batch_src,True,a.num_workers,device,True); tl=dl(t_tr,a.batch_tgt,True,a.num_workers,device,True)
    ll=dl(t_lab,a.batch_tgt,True,a.num_workers,device,False) if t_lab else None
    sv=dl(s_va,a.batch_src,False,a.num_workers,device); st=dl(s_te,a.batch_src,False,a.num_workers,device)
    tt=dl(t_te,a.batch_tgt,False,a.num_workers,device); iters=min(len(sl),len(tl))
    if iters < 1: raise ValueError("No training batches")

    model=DAN(bool(a.use_gn)).to(device); opt=optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=3)
    scaler=GradScaler(enabled=device.type=="cuda"); ema=EMA(model,a.ema_beta) if a.ema else None; bce=nn.BCEWithLogitsLoss()
    config={"method":"DAN","source":source,"target":target,"seed":seed,"mode":mode,"ratio":a.tgt_label_ratio,
            "lambda_mmd_max":a.lambda_mmd_max,"warmup_epochs":a.warmup_epochs,"w_tgt_sup":a.w_tgt_sup,
            "batch_src":a.batch_src,"batch_tgt":a.batch_tgt,"lr":a.lr,"weight_decay":a.weight_decay,
            "ema":bool(a.ema),"ema_beta":a.ema_beta,"use_gn":bool(a.use_gn),
            "kernel_scales":list(KERNEL_SCALES),"preprocess":"pre_log1p+source_zscore"}
    start,wait,best,best_epoch=1,0,-1.,0
    if a.resume and os.path.isfile(path["latest"]) and not a.force:
        ck=torch.load(path["latest"],map_location="cpu")
        if ck.get("config")!=config: raise ValueError("latest.pth configuration does not match this run")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"]); start=ck["epoch"]+1; wait=ck["wait"]; best=ck["best_f1"]; best_epoch=ck["best_epoch"]
        if ema and "ema" in ck: ema.shadow={k:v.to(device) for k,v in ck["ema"].items()}
    if start==1 or a.force:
        with open(path["history"],"w") as f: f.write("epoch,total_loss,src_loss,tgt_loss,mmd_loss,lambda_mmd,valid_loss,valid_f1,threshold,lr\n")

    begun=time.time(); log(f"[DAN-{mode.upper()}] {source}->{target} seed={seed} S={len(s_tr)} T={len(t_tr)} Tlab={len(t_lab) if t_lab else 0}")
    for epoch in range(start,a.epochs+1):
        model.train(); si,ti,li=iter(sl),iter(tl),iter(ll) if ll else None; runloss={k:[] for k in ("all","src","tgt","mmd","lam")}
        bar=tqdm(range(iters),desc=f"[{source}->{target}|{seed}] {epoch}/{a.epochs}",dynamic_ncols=True)
        for i in bar:
            try: xs,ys=next(si)
            except StopIteration: si=iter(sl); xs,ys=next(si)
            try: xt,_=next(ti)
            except StopIteration: ti=iter(tl); xt,_=next(ti)
            xl=yl=None
            if ll:
                try: xl,yl=next(li)
                except StopIteration: li=iter(ll); xl,yl=next(li)
                xl,yl=xl.to(device),yl.to(device)
            xs,ys,xt=xs.to(device),ys.to(device),xt.to(device); n=min(len(xs),len(xt))
            progress=(i+(epoch-1)*iters)/max(1,a.epochs*iters)
            lam=0. if epoch<=a.warmup_epochs else a.lambda_mmd_max*(2/(1+math.exp(-MMD_GAMMA*progress))-1)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                z,h=model(torch.cat([xs[:n],xt[:n]])); src=bce(z[:n],ys[:n]); mmd=mk_mmd(h[:n],h[n:])
                if xl is not None: zt,_=model(xl); tgt=bce(zt,yl)
                else: tgt=src.new_zeros(())
                loss=src+lam*mmd+a.w_tgt_sup*tgt
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite loss epoch={epoch} iter={i}")
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt); scaler.update()
            if ema: ema.update()
            for k,v in zip(runloss,(loss,src,tgt,mmd,lam)): runloss[k].append(float(v.item() if torch.is_tensor(v) else v))
            bar.set_postfix(loss=f"{np.mean(runloss['all']):.4f}",mmd=f"{np.mean(runloss['mmd']):.4f}",lam=f"{lam:.3f}")

        if ema: ema.apply()
        try:
            vz,vy=predictions(model,sv,device); vloss=nn.functional.binary_cross_entropy_with_logits(vz,vy).item()
            threshold,vf1=threshold_search(vy.numpy(),torch.sigmoid(vz).numpy()); improved=vf1>best
            if improved:
                best,best_epoch,wait=vf1,epoch,0
                torch.save({"model":model.state_dict(),"epoch":epoch,"best_valid_macro_f1":best,"best_thresh":threshold,
                            "src_mean":mean,"src_std":std,"class_names":CLASS_NAMES,"config":config},path["best"])
            else: wait+=1
        finally:
            if ema: ema.restore()
        sched.step(vloss)
        with open(path["history"],"a") as f:
            f.write(f"{epoch},{np.mean(runloss['all']):.6f},{np.mean(runloss['src']):.6f},{np.mean(runloss['tgt']):.6f},"
                    f"{np.mean(runloss['mmd']):.6f},{np.mean(runloss['lam']):.6f},{vloss:.6f},{vf1:.6f},{threshold:.6f},{opt.param_groups[0]['lr']:.8f}\n")
        latest={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),
                "epoch":epoch,"wait":wait,"best_f1":best,"best_epoch":best_epoch,"config":config}
        if ema: latest["ema"]={k:v.cpu() for k,v in ema.shadow.items()}
        torch.save(latest,path["latest"]); log(f"ep={epoch:03d}{'*' if improved else ' '} valid_f1={vf1:.4f} mmd={np.mean(runloss['mmd']):.4f}")
        if wait>=a.patience: break

    ck=torch.load(path["best"],map_location="cpu"); model.load_state_dict(ck["model"]); model.to(device).eval()
    # best.pth already holds the EMA weights from the best epoch; do not apply final EMA.
    threshold=float(ck["best_thresh"]); sm=evaluate(model,st,device,threshold); tm=evaluate(model,tt,device,threshold)
    result={"mode":f"DAN_{mode.upper()}","source":source,"target":target,"seed":seed,"best_epoch":ck["epoch"],
            "best_valid_macro_f1":ck["best_valid_macro_f1"],"best_thresh":threshold,"source_test":sm,"target_test":tm,
            "sizes":{"source_train":len(s_tr),"target_train":len(t_tr),"target_labeled":len(t_lab) if t_lab else 0},
            "config":config,"paths":path,"train_time_sec":time.time()-begun,"timestamp":datetime.now().isoformat(),
            "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":str(device)}}
    with open(path["result"],"w",encoding="utf-8") as f: json.dump(result,f,indent=2,ensure_ascii=False)
    log(f"[FINAL] source F1={sm['macro_f1']:.4f}, target F1={tm['macro_f1']:.4f}, t*={threshold:.3f}")
    return result


def parser():
    p=argparse.ArgumentParser(); p.add_argument("--gpu",default=GPU_DEFAULT); p.add_argument("--out-root",default=OUT_ROOT)
    p.add_argument("--epochs",type=int,default=EPOCHS); p.add_argument("--batch-src",type=int,default=BATCH_SRC); p.add_argument("--batch-tgt",type=int,default=BATCH_TGT)
    p.add_argument("--lr",type=float,default=LR); p.add_argument("--weight-decay",type=float,default=WEIGHT_DECAY); p.add_argument("--patience",type=int,default=PATIENCE)
    p.add_argument("--num-workers",type=int,default=NUM_WORKERS); p.add_argument("--lambda-mmd-max",type=float,default=LAMBDA_MMD_MAX); p.add_argument("--warmup-epochs",type=int,default=WARMUP_EPOCHS)
    p.add_argument("--semi",type=int,choices=[0,1],default=SEMI); p.add_argument("--tgt-label-ratio",type=float,default=TGT_LABEL_RATIO); p.add_argument("--w-tgt-sup",type=float,default=W_TGT_SUP)
    p.add_argument("--ema",type=int,choices=[0,1],default=1); p.add_argument("--ema-beta",type=float,default=EMA_BETA); p.add_argument("--use-gn",type=int,choices=[0,1],default=int(USE_GN))
    p.add_argument("--seeds",type=int,nargs="+",default=SEEDS); p.add_argument("--domains",nargs="+",choices=DOMAINS,default=DOMAINS)
    p.add_argument("--pairs",nargs="*",default=[f"{s}->{t}" for s,t in PAIRS]); p.add_argument("--cross-train-all",action="store_true")
    p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--force",action="store_true"); return p


def main():
    a=parser().parse_args()
    if a.epochs<1 or min(a.batch_src,a.batch_tgt)<2 or not 0<=a.tgt_label_ratio<=1 or a.lambda_mmd_max<0: raise ValueError("Invalid arguments")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(a.out_root,exist_ok=True)
    pairs=[(s,t) for s in a.domains for t in a.domains if s!=t] if a.cross_train_all else pair_list(a.pairs)
    runs=[]; log(f"[INFO] device={device}, pairs={pairs}, seeds={a.seeds}")
    for s,t in pairs:
        for seed in a.seeds:
            runs.append(train_pair(s,t,seed,device,a)); torch.cuda.empty_cache(); gc.collect()
    summary={}
    for s,t in pairs:
        vals=[r["target_test"]["macro_f1"] for r in runs if r["source"]==s and r["target"]==t]
        summary[f"{s}->{t}"]={"target_macro_f1_mean":float(np.mean(vals)),"target_macro_f1_std":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,"num_runs":len(vals)}
    out=os.path.join(a.out_root,f"summary_dan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    log(f"[DONE] {out}")


if __name__ == "__main__": main()
