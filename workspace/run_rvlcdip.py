# ============================================================
# RVL-CDIP × ColPali × 自己教師あり学習
# - 訓練: 正解ラベル不使用（MaxSim / InfoNCE / MSE / Diversity）
# - 検証: 正解ラベルで F1・精度を観察（選択基準は val 自己損失）
# - 推論時はタグ名定義と画像のみで任意データに適用可能
# ============================================================

import os
import sys
import gc
import re
import json
import math
import hashlib
import subprocess
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import psutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, f1_score, average_precision_score, accuracy_score
from transformers import ColPaliForRetrieval, ColPaliProcessor


# =========================
# 1. HF token setup
# =========================
def setup_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
            return token
    except Exception:
        pass

    return None


hf_token = setup_hf_token()
print("HF_TOKEN set:", bool(hf_token))

try:
    from huggingface_hub import login
    if hf_token:
        login(token=hf_token)
except Exception as e:
    print("huggingface_hub.login skipped:", repr(e))


# =========================
# 2. config
# =========================
MODEL_NAME = "vidore/colpali-v1.3-hf"
RVLCDIP_NAME = "aharley/rvl_cdip"

# データセットサイズ
MAX_TRAIN_ROWS = 12000  # フルは 320000
MAX_VAL_ROWS = 3000     # フルは 40000

TRAIN_IMAGE_ENCODE_BATCH = 2
VAL_IMAGE_ENCODE_BATCH = 2

MAX_PATCHES = 192
N_QUERIES = 6
N_HEADS = 8
EPOCHS = 12
LR = 3e-4
WEIGHT_DECAY = 1e-2

TRAIN_LOADER_BATCH = 16  # 訓練バッチサイズ増加
VAL_LOADER_BATCH = 16
SHARD_SIZE = 100        # シャードを小さくしてRAMスパイクを抑制
OVERWRITE_CACHE = False
IMAGE_SAVE_FORMAT = "PNG"
SEED = 42

# RVL-CDIPの16クラス
TAGS = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
TAG_TO_ID = {t: i for i, t in enumerate(TAGS)}

# 損失重み（訓練は自己教師あり損失のみ）
W_MAXSIM = 1.0
W_INFNCE = 0.5
W_MSE = 0.1
W_DIV = 0.1

# 保存先
CACHE_DIR = Path("rvlcdip_cache")
IMAGE_DIR = CACHE_DIR / "page_images"
SHARD_DIR = CACHE_DIR / "mv_shards"
OUT_DIR = Path("outputs_rvlcdip")
CKPT_DIR = OUT_DIR / "checkpoints"

for p in [CACHE_DIR, IMAGE_DIR, SHARD_DIR, OUT_DIR, CKPT_DIR]:
    p.mkdir(exist_ok=True, parents=True)


# =========================
# 3. helpers
# =========================
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def show_mem(tag=""):
    ram = psutil.virtual_memory()
    used = (ram.total - ram.available) / 1e9
    total = ram.total / 1e9
    print(f"[{tag}] RAM used: {used:.2f} GB / {total:.2f} GB")
    if torch.cuda.is_available():
        print(f"[{tag}] GPU alloc   : {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"[{tag}] GPU reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")


def normalize_text(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def to_pil(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
        if image_field.get("path") is not None:
            return Image.open(image_field["path"]).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_field)}")


def image_to_bytes(image_field: Any) -> bytes:
    if isinstance(image_field, dict) and image_field.get("bytes") is not None:
        return image_field["bytes"]
    pil = to_pil(image_field)
    buf = BytesIO()
    pil.save(buf, format=IMAGE_SAVE_FORMAT)
    return buf.getvalue()


def image_key(image_field: Any) -> str:
    return hashlib.md5(image_to_bytes(image_field)).hexdigest()


def tags_to_multi_hot(tags: List[str], tag_to_id: Dict[str, int]) -> torch.Tensor:
    y = torch.zeros(len(tag_to_id), dtype=torch.float32)
    for t in tags:
        if t in tag_to_id:
            y[tag_to_id[t]] = 1.0
    return y


def print_tag_stats(name: str, labels: torch.Tensor):
    counts = labels.sum(dim=0).tolist()
    print(f"\n[{name}] page count = {labels.size(0)}")
    for t, c in zip(TAGS, counts):
        print(f"  {t:20s}: {int(c)}")


# =========================
# 4. device / model
# =========================
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if torch.backends.mps.is_available():
        return torch.float32
    return torch.float32


if torch.cuda.is_available():
    _device = torch.device("cuda:0")
    _mps_mode = False
elif torch.backends.mps.is_available():
    _device = torch.device("mps")
    _mps_mode = True
else:
    _device = torch.device("cpu")
    _mps_mode = False
dtype = pick_dtype()

print("device:", _device, "mps_mode:", _mps_mode, "dtype:", dtype)

if torch.cuda.is_available():
    free_mem = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU free memory: {free_mem:.2f} GB")

processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
model = ColPaliForRetrieval.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).eval().to(_device)

print("model.device:", model.device)
show_mem("after model load")


# =========================
# 5. RVL-CDIP page index build
# =========================
def build_rvlcdip_page_index(split_name: str, max_rows: int, prefix: str) -> Path:
    """
    RVL-CDIPを読み、画像はdiskに退避しつつ、
    page単位のindex jsonを作る（ラベルは既存のものを使用）
    """
    index_path = CACHE_DIR / f"{prefix}_page_index.json"
    if index_path.exists() and not OVERWRITE_CACHE:
        print("Reuse page index:", index_path)
        return index_path

    print(f"Building {prefix} index from RVL-CDIP {split_name} split...")
    ds = load_dataset(RVLCDIP_NAME, split=split_name)
    ds = ds.select(range(min(max_rows, len(ds))))

    page_records = []

    for i, ex in enumerate(ds):
        # 画像をディスクに保存
        img_key = image_key(ex["image"])
        img_path = IMAGE_DIR / f"{img_key}.png"
        
        if not img_path.exists():
            img = to_pil(ex["image"])
            img.save(img_path, format="PNG")
        
        # RVL-CDIPのラベルは整数インデックス（0-15）
        label_idx = ex["label"]
        tag = TAGS[label_idx]
        
        page_records.append({
            "page_key": img_key,
            "image_path": str(img_path),
            "tags": [tag],  # シングルラベルだがリスト形式で統一
            "label_idx": label_idx,
        })

        if (i + 1) % 1000 == 0:
            print(f"[{prefix}] processed images: {i+1}/{len(ds)}")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(page_records, f, ensure_ascii=False, indent=2)

    print(f"[{prefix}] unique pages:", len(page_records))
    return index_path


# RVL-CDIPは "train" と "test" split（"validation" は存在しない）
train_index_path = build_rvlcdip_page_index("train", MAX_TRAIN_ROWS, "train")
val_index_path = build_rvlcdip_page_index("test", MAX_VAL_ROWS, "val")


def load_page_records(index_path: Path) -> List[dict]:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_records = load_page_records(train_index_path)
val_records = load_page_records(val_index_path)

train_labels = torch.stack([tags_to_multi_hot(r["tags"], TAG_TO_ID) for r in train_records], dim=0)
val_labels = torch.stack([tags_to_multi_hot(r["tags"], TAG_TO_ID) for r in val_records], dim=0)

print("train unique pages:", len(train_records))
print("val unique pages:", len(val_records))
print_tag_stats("train", train_labels)
print_tag_stats("val", val_labels)

del train_labels, val_labels
cleanup()


# =========================
# 6. encode images -> shard files
# =========================
@torch.no_grad()
def _encode_single(img: Image.Image) -> torch.Tensor:
    inputs = processor(images=[img], return_tensors="pt").to(model.device)
    out = model(**inputs)
    mv = out.embeddings.detach().to("cpu")
    del inputs, out
    if _mps_mode:
        torch.mps.synchronize()
    cleanup()
    return mv  # [1, P, D]


def encode_images_mv(pil_images: List[Image.Image], batch_size: int = 2) -> torch.Tensor:
    embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]

        try:
            inputs = processor(images=batch, return_tensors="pt").to(model.device)
            out = model(**inputs)
            mv = out.embeddings  # [B, P, D]
            embs.append(mv.detach().to("cpu"))
            del inputs, out, mv
            if _mps_mode:
                torch.mps.synchronize()
            cleanup()
        except Exception as e:
            print(f"  batch encode failed: {e}  → retrying one-by-one")
            try:
                del inputs, out, mv
            except Exception:
                pass
            if _mps_mode:
                torch.mps.synchronize()
            cleanup()
            # 1枚ずつ処理してメモリ圧力を下げる
            for img in batch:
                embs.append(_encode_single(img))

        if (i // batch_size) % 20 == 0:
            print(f"encoded {min(i + len(batch), len(pil_images))}/{len(pil_images)}")

    return torch.cat(embs, dim=0)


def save_mv_shards(records: List[dict], prefix: str, batch_size: int):
    shard_paths = []

    for s in range(0, len(records), SHARD_SIZE):
        shard_id = s // SHARD_SIZE
        shard_path = SHARD_DIR / f"{prefix}_{shard_id:03d}.pt"

        if shard_path.exists() and not OVERWRITE_CACHE:
            shard_paths.append(str(shard_path))
            print("Reuse shard:", shard_path)
            continue

        chunk = records[s : s + SHARD_SIZE]
        images = [Image.open(r["image_path"]).convert("RGB") for r in chunk]
        labels = torch.stack([tags_to_multi_hot(r["tags"], TAG_TO_ID) for r in chunk], dim=0)

        show_mem(f"{prefix} shard {shard_id} before encode")
        mv = encode_images_mv(images, batch_size=batch_size)
        show_mem(f"{prefix} shard {shard_id} after encode")

        torch.save(
            {
                "mv": mv.half(),   # CPU保持は float16
                "labels": labels,
                "meta": chunk,
            },
            shard_path,
        )
        shard_paths.append(str(shard_path))
        print("saved shard:", shard_path, "shape=", tuple(mv.shape))

        for im in images:
            im.close()
        del images, labels, mv, chunk
        cleanup()

    return shard_paths


train_shards = save_mv_shards(train_records, "train", TRAIN_IMAGE_ENCODE_BATCH)
val_shards = save_mv_shards(val_records, "val", VAL_IMAGE_ENCODE_BATCH)

print("num train shards:", len(train_shards))
print("num val shards:", len(val_shards))

del train_records, val_records
cleanup()
show_mem("after shard encode")


# =========================
# 7. encode tag texts
# =========================
@torch.no_grad()
def encode_tags_to_vec(tags: List[str], batch_size: int = 32) -> torch.Tensor:
    vecs = []
    for i in range(0, len(tags), batch_size):
        batch = tags[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model(**inputs)
        mv = out.embeddings  # [B, T, D]
        attn = inputs.get("attention_mask", None)

        if attn is None:
            pooled = mv.mean(dim=1)
        else:
            m = attn.unsqueeze(-1).to(mv.dtype)
            pooled = (mv * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        pooled = F.normalize(pooled.float(), dim=-1)
        vecs.append(pooled.detach().to("cpu"))

        del inputs, out, mv, pooled
        cleanup()

    return torch.cat(vecs, dim=0)


tag_matrix = encode_tags_to_vec(TAGS)
V, D = tag_matrix.shape
print("tag_matrix:", (V, D))


# =========================
# 8. model / losses
# =========================
class CrossAttnTagger(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, n_queries: int = 6, n_heads: int = 8):
        super().__init__()
        self.n_queries = n_queries
        self.query_embed = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, img_mv: torch.Tensor) -> torch.Tensor:
        B = img_mv.size(0)
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.attn(q, img_mv, img_mv, need_weights=False)
        attn_out = self.norm(attn_out)
        return self.fc(attn_out)


def subsample_patches(img_mv: torch.Tensor, max_patches: int = 256, deterministic: bool = False) -> torch.Tensor:
    B, P, D = img_mv.shape
    if P <= max_patches:
        return img_mv

    if deterministic:
        torch.manual_seed(SEED)

    indices = torch.randperm(P, device=img_mv.device)[:max_patches]
    indices = indices.sort()[0]
    return img_mv[:, indices, :]


device = _device

tagger = CrossAttnTagger(d_model=D, vocab_size=V, n_queries=N_QUERIES, n_heads=N_HEADS).to(device)
opt = torch.optim.AdamW(tagger.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

tag_matrix_dev = tag_matrix.to(device)


def page_logits_from_slot_logits(slot_logits: torch.Tensor) -> torch.Tensor:
    return slot_logits.max(dim=1)[0]


def logits_to_pred_vec(slot_logits: torch.Tensor, tag_mat: torch.Tensor) -> torch.Tensor:
    B, Q, V = slot_logits.shape
    probs = torch.sigmoid(slot_logits)
    pred_tag_vec = torch.matmul(probs, tag_mat)
    pred_tag_vec = F.normalize(pred_tag_vec, dim=-1)
    return pred_tag_vec


def maxsim_loss(pred_vecs: torch.Tensor, img_mv: torch.Tensor) -> torch.Tensor:
    B, Q, D = pred_vecs.shape
    _, P, _ = img_mv.shape

    pred_vecs_norm = F.normalize(pred_vecs, dim=-1)
    img_mv_norm = F.normalize(img_mv, dim=-1)

    sim = torch.matmul(pred_vecs_norm.view(B * Q, D), img_mv_norm.view(B * P, D).T)
    sim = sim.view(B, Q, B, P)

    maxsim_per_query = sim.diagonal(dim1=0, dim2=2).max(dim=-1)[0]
    return -maxsim_per_query.mean()


def infonce_loss(pred_vecs: torch.Tensor, img_mv: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    pred_pooled = F.normalize(pred_vecs.mean(dim=1), dim=-1)   # [B, D]
    img_pooled = F.normalize(img_mv.mean(dim=1), dim=-1)        # [B, D]
    sim = pred_pooled @ img_pooled.T / temperature              # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def diversity_loss(pred_vecs: torch.Tensor) -> torch.Tensor:
    B, Q, D = pred_vecs.shape
    pred_vecs_norm = F.normalize(pred_vecs, dim=-1)
    
    sim = torch.matmul(pred_vecs_norm.view(B * Q, D), pred_vecs_norm.view(B * Q, D).T)
    sim = sim.view(B, Q, B, Q)
    
    mask = torch.eye(B * Q, device=sim.device, dtype=torch.bool).view(B, Q, B, Q)
    sim_masked = sim.masked_fill(mask, 0.0)
    
    return sim_masked.abs().mean()


# =========================
# 9. self-supervised val loss
# =========================
@torch.no_grad()
def compute_val_self_loss(shard_paths: List[str]) -> float:
    tagger.eval()
    total = 0.0
    n = 0
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        mv = shard["mv"]
        dataset = TensorDataset(mv)
        loader = DataLoader(
            dataset,
            batch_size=min(VAL_LOADER_BATCH, len(dataset)),
            shuffle=False,
            drop_last=True,
        )
        for (batch_mv_cpu,) in loader:
            img_mv_b = batch_mv_cpu.to(device=device, dtype=torch.float32)
            img_mv_b = subsample_patches(img_mv_b, max_patches=MAX_PATCHES, deterministic=True)
            slot_logits = tagger(img_mv_b)
            pred_tag_vec = logits_to_pred_vec(slot_logits, tag_matrix_dev)
            pooled_img = img_mv_b.mean(dim=1)
            pooled_pred = pred_tag_vec.mean(dim=1)
            loss = (
                W_MAXSIM * maxsim_loss(pred_tag_vec, img_mv_b)
                + W_INFNCE * infonce_loss(pred_tag_vec, img_mv_b)
                + W_MSE * F.mse_loss(pooled_pred, pooled_img)
                + W_DIV * diversity_loss(pred_tag_vec)
            )
            total += float(loss.item())
            n += 1
            del img_mv_b, slot_logits, pred_tag_vec, pooled_img, pooled_pred, loss
            cleanup()
        del shard, mv
        cleanup()
    return total / max(n, 1)


# =========================
# 10. predict from shards
# =========================
@torch.no_grad()
def predict_from_shards(shard_paths: List[str], threshold: float = 0.5):
    tagger.eval()
    probs_all = []
    preds_all = []
    labels_all = []
    metas_all = []

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        mv = shard["mv"]
        labels = shard["labels"].float()
        meta = shard["meta"]

        dataset = TensorDataset(mv, labels)
        batch_size = min(VAL_LOADER_BATCH, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch_mv_cpu, batch_y_cpu in loader:
            img_mv_b = batch_mv_cpu.to(device=device, dtype=torch.float32)
            y_b = batch_y_cpu.to(device=device, dtype=torch.float32)

            img_mv_b = subsample_patches(img_mv_b, max_patches=MAX_PATCHES, deterministic=True)

            slot_logits = tagger(img_mv_b)
            page_logits = page_logits_from_slot_logits(slot_logits)

            probs = torch.sigmoid(page_logits).cpu()
            preds = (probs >= threshold).float()

            probs_all.append(probs)
            preds_all.append(preds)
            labels_all.append(y_b.cpu())

            del img_mv_b, y_b, slot_logits, page_logits, probs, preds
            cleanup()

        metas_all.extend(meta)
        del shard, mv, labels, meta, dataset, loader
        cleanup()

    probs_all = torch.cat(probs_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    return probs_all, preds_all, labels_all, metas_all


def compute_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, y_pred: torch.Tensor):
    yt = y_true.numpy().astype(np.int32)
    yp = y_pred.numpy().astype(np.int32)
    ys = y_prob.numpy().astype(np.float32)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        yt, yp, average=None, zero_division=0
    )

    subset_acc = (yt == yp).all(axis=1).mean()
    
    # シングルラベル精度を追加
    y_true_idx = yt.argmax(axis=1)
    y_pred_idx = yp.argmax(axis=1)
    single_label_acc = accuracy_score(y_true_idx, y_pred_idx)

    ap_list = []
    for j in range(yt.shape[1]):
        if yt[:, j].sum() > 0:
            ap_list.append(average_precision_score(yt[:, j], ys[:, j]))
    map_macro = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")

    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "subset_accuracy": float(subset_acc),
        "single_label_accuracy": float(single_label_acc),
        "mAP_macro": float(map_macro),
        "per_tag": {
            TAGS[j]: {
                "precision": float(per_p[j]),
                "recall": float(per_r[j]),
                "f1": float(per_f1[j]),
                "support": int(per_sup[j]),
            }
            for j in range(len(TAGS))
        }
    }


def find_best_threshold(shard_paths: List[str]):
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.10, 0.90, 17):
        probs, preds, labels, _ = predict_from_shards(shard_paths, threshold=float(thr))
        score = f1_score(labels.numpy(), preds.numpy(), average="micro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return best_thr, best_f1


def decode_multi_hot(y: torch.Tensor, tag_names: List[str]) -> List[List[str]]:
    out = []
    for i in range(y.size(0)):
        tags = [tag_names[j] for j in range(y.size(1)) if y[i, j].item() > 0.5]
        out.append(tags)
    return out


# =========================
# 11. training loop
# =========================
best_val_self_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    tagger.train()
    total = 0.0
    n = 0

    for shard_path in train_shards:
        shard = torch.load(shard_path, map_location="cpu")
        train_mv = shard["mv"]

        dataset = TensorDataset(train_mv)
        batch_size = min(TRAIN_LOADER_BATCH, len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=(len(dataset) > 1),
        )

        for (batch_mv_cpu,) in loader:
            img_mv_b = batch_mv_cpu.to(device=device, dtype=torch.float32)
            img_mv_b = subsample_patches(img_mv_b, max_patches=MAX_PATCHES, deterministic=False)

            slot_logits = tagger(img_mv_b)
            pred_tag_vec = logits_to_pred_vec(slot_logits, tag_matrix_dev)
            pooled_img = img_mv_b.mean(dim=1)
            pooled_pred = pred_tag_vec.mean(dim=1)

            loss_mse = F.mse_loss(pooled_pred, pooled_img)
            loss_max = maxsim_loss(pred_tag_vec, img_mv_b)
            loss_div = diversity_loss(pred_tag_vec)
            loss_nce = infonce_loss(pred_tag_vec, img_mv_b)

            loss = (
                W_MAXSIM * loss_max
                + W_INFNCE * loss_nce
                + W_MSE * loss_mse
                + W_DIV * loss_div
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            n += 1

            del img_mv_b, slot_logits, pred_tag_vec
            del pooled_img, pooled_pred, loss_mse, loss_max, loss_div, loss_nce, loss
            cleanup()

        del shard, train_mv, dataset, loader
        cleanup()

    # val 自己損失（モデル選択基準）
    val_self_loss = compute_val_self_loss(val_shards)

    # val メトリクス（観察用のみ、学習には不使用）
    best_thr, _ = find_best_threshold(val_shards)
    val_probs, val_preds, val_true, _ = predict_from_shards(val_shards, threshold=best_thr)
    val_metrics = compute_metrics(val_true, val_probs, val_preds)

    val_micro_f1 = val_metrics['micro_f1']
    val_acc = val_metrics['single_label_accuracy']
    print(
        f"epoch={epoch:02d}  "
        f"train_loss={total/max(n,1):.4f}  "
        f"val_self_loss={val_self_loss:.4f}  "
        f"[観察] val_acc={val_acc:.4f}  "
        f"val_micro_f1={val_micro_f1:.4f}  "
        f"best_thr={best_thr:.2f}"
    )

    ckpt = {
        "epoch": epoch,
        "seed": SEED,
        "tagger_state_dict": tagger.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "best_thr": best_thr,
        "val_self_loss": val_self_loss,
        "val_metrics": val_metrics,
        "config": {
            "MODEL_NAME": MODEL_NAME, "RVLCDIP_NAME": RVLCDIP_NAME,
            "MAX_TRAIN_ROWS": MAX_TRAIN_ROWS, "MAX_VAL_ROWS": MAX_VAL_ROWS,
            "MAX_PATCHES": MAX_PATCHES, "N_QUERIES": N_QUERIES, "N_HEADS": N_HEADS,
            "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
            "TRAIN_LOADER_BATCH": TRAIN_LOADER_BATCH, "SHARD_SIZE": SHARD_SIZE,
            "W_MAXSIM": W_MAXSIM, "W_INFNCE": W_INFNCE,
            "W_MSE": W_MSE, "W_DIV": W_DIV,
        },
    }
    torch.save(ckpt, CKPT_DIR / f"epoch_{epoch:02d}.pt")

    if val_self_loss < best_val_self_loss:
        best_val_self_loss = val_self_loss
        torch.save(ckpt, CKPT_DIR / "best.pt")
        print(f"  -> best model updated (val_self_loss={val_self_loss:.4f}, [観察] acc={val_acc:.4f})")

    del val_probs, val_preds, val_true, val_metrics
    cleanup()


# =========================
# 12. final evaluation
# =========================
best_thr, best_val_micro_f1 = find_best_threshold(val_shards)
val_probs, val_preds, val_true, val_meta = predict_from_shards(val_shards, threshold=best_thr)
final_metrics = compute_metrics(val_true, val_probs, val_preds)

print("\n========== FINAL VALIDATION ==========")
print(json.dumps({
    "threshold": best_thr,
    "metrics": final_metrics,
}, indent=2, ensure_ascii=False))

print("\n========== PER-TAG ==========")
for tag in TAGS:
    d = final_metrics["per_tag"][tag]
    print(
        f"{tag:20s} | "
        f"P={d['precision']:.4f}  "
        f"R={d['recall']:.4f}  "
        f"F1={d['f1']:.4f}  "
        f"support={d['support']}"
    )

print("\n========== SAMPLE PREDICTIONS ==========")
gt_tags = decode_multi_hot(val_true, TAGS)
pd_tags = decode_multi_hot(val_preds, TAGS)

for i in range(min(10, len(val_meta))):
    print(f"[val page {i:03d}]")
    print("GT  :", gt_tags[i])
    print("Pred:", pd_tags[i])
    print("Label idx:", val_meta[i].get("label_idx", "N/A"))
    print("-" * 80)


# =========================
# 13. save outputs
# =========================
_config = {
    "MODEL_NAME": MODEL_NAME, "RVLCDIP_NAME": RVLCDIP_NAME,
    "MAX_TRAIN_ROWS": MAX_TRAIN_ROWS, "MAX_VAL_ROWS": MAX_VAL_ROWS,
    "MAX_PATCHES": MAX_PATCHES, "N_QUERIES": N_QUERIES, "N_HEADS": N_HEADS,
    "EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
    "TRAIN_LOADER_BATCH": TRAIN_LOADER_BATCH, "SHARD_SIZE": SHARD_SIZE,
    "W_MAXSIM": W_MAXSIM, "W_INFNCE": W_INFNCE,
    "W_MSE": W_MSE, "W_DIV": W_DIV,
    "SEED": SEED,
}

torch.save(
    {
        "config": _config,
        "tags": TAGS,
        "tag_matrix": tag_matrix,
        "tagger_state_dict": tagger.state_dict(),
        "threshold": best_thr,
        "metrics": final_metrics,
        "train_shards": [str(p) for p in train_shards],
        "val_shards": [str(p) for p in val_shards],
    },
    OUT_DIR / "rvlcdip_16class_eval.pt",
)

torch.save(
    tagger.state_dict(),
    OUT_DIR / "rvlcdip_16class_tagger.pt",
)

with open(OUT_DIR / "rvlcdip_16class_metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "config": _config,
            "threshold": best_thr,
            "metrics": final_metrics,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\nsaved:")
print(str(OUT_DIR / "rvlcdip_16class_eval.pt"))
print(str(OUT_DIR / "rvlcdip_16class_tagger.pt"))
print(str(OUT_DIR / "rvlcdip_16class_metrics.json"))
print(f"checkpoints: {CKPT_DIR}/ (best.pt + epoch_XX.pt)")
