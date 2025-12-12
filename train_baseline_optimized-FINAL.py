# train_full_workflow_improved_22_labels.py
"""
Improved end-to-end workflow for multi-label emotion classification (GoEmotions) with:
- Asymmetric loss (ASL) or focal BCE + per-label balancing
- Aggressive oversampling for rare labels
- Per-label threshold optimization
- Robust label parsing and batch tokenization
- Rare-label recommendations and hooks
- Logit normalization, neutral downsampling, and medium data augmentation for rare labels
- Gradient accumulation, mixed precision, gradient clipping, capped pos_weight
- Automatic label-merging pipeline (28 -> 22 labels)
"""

import os
import re
import copy
import random
import argparse
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch import nn
from torch.nn.utils import clip_grad_norm_

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, precision_recall_fscore_support

# ---------------------------
# Config / Defaults
# ---------------------------
DEFAULTS = {
    "seed": 42,
    # DeBERTa v3 base (free, reasonably fast / accurate)
    "model_name": "microsoft/deberta-v3-base",
    "max_len": 128,
    "batch_size": 16,
    # training schedule
    "epochs": 15,
    "lr": 2e-5,
    "patience": 4,
    # NEW: final number of labels after merging
    "num_labels": 22,
    # original (source) number of labels in the dataset before merging
    "original_num_labels": 28,
    # denser sweep for per-label tuning
    "threshold_sweep": np.arange(0.01, 0.81, 0.01),
    "data_dir": "data",
    "train_file": "data/train.tsv",
    "dev_file": "data/dev.tsv",
    "labels_file": "data/emotion_labels.txt",
    "checkpoint": "best_deberta_goemotions_22labels.pt",
    "per_label_threshold": True,
    "use_asl": True,
    "use_focal_loss": False,
    "use_oversample": True,
    "rare_label_factor": 5.0,
    # ASL params
    "asl_gamma_pos": 0.0,
    "asl_gamma_neg": 1.0,
    "asl_clip": 0.05,
    "min_support_for_rare": 50,
    # augmentation level: 1=light,2=medium,3=heavy
    "augmentation_level": 2,
    "augmentation_multiplier_by_level": {1: 1, 2: 2, 3: 5},
    # neutral downsample fraction
    "neutral_keep_frac": 0.35,
    # grad accumulation default
    "grad_accum_steps": 2,
    # gradient clipping
    "max_grad_norm": 1.0,
    # cap pos_weight to avoid extreme values
    "pos_weight_cap": 40.0,
    # number of workers for DataLoader
    "num_workers": 2,
}

# ---------------------------
# Label merge plan (explicit)
# ---------------------------
# This mapping merges 28 GoEmotions labels into 22 final labels.
# Based on the user's approved merge plan:
# Merges:
# - disgust (11) + embarrassment (12) -> disgust_group
# - excitement (13) + joy (17)       -> joy_group
# - fear (14) + nervousness (19)     -> fear_group
# - grief (16) + sadness (25)        -> sadness_group
# - realization (22) + surprise (26) -> realization_group
# - relief (23) + remorse (24)       -> remorse_group

MERGE_PAIRS = {
    # new_name: [source_indices]
    "disgust_group": [11, 12],
    "joy_group": [13, 17],
    "fear_group": [14, 19],
    "sadness_group": [16, 25],
    "realization_group": [22, 26],
    "remorse_group": [23, 24],
}

# If labels_file exists and contains 28 names, we'll use it as the source order.
# Otherwise use a sensible default (order inferred from standard GoEmotions / earlier run output).
DEFAULT_ORIGINAL_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_extract_integers(s: str) -> List[int]:
    if not isinstance(s, str):
        s = str(s)
    matches = re.findall(r"\d+", s)
    if len(matches) == 0:
        return []
    return [int(m) for m in matches]

def clean_text_keep_emotion(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def to_multi_hot_from_indices(indices: List[int], num_labels: int):
    vec = np.zeros(num_labels, dtype=np.float32)
    for idx in indices:
        if 0 <= idx < num_labels:
            vec[idx] = 1.0
    return vec

# ---------------------------
# Column detection + load
# ---------------------------
def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    header_candidates_text = {"text", "comment", "sentence", "content"}
    header_candidates_labels = {"labels", "label", "emotion", "emotions", "tags"}

    colnames = [str(c).lower() for c in df.columns]
    for i, cname in enumerate(colnames):
        if cname in header_candidates_text:
            text_col = df.columns[i]
            break
    else:
        text_col = None

    for i, cname in enumerate(colnames):
        if cname in header_candidates_labels:
            labels_col = df.columns[i]
            break
    else:
        labels_col = None

    if text_col is not None and labels_col is not None:
        return text_col, labels_col

    n = len(df)
    best_text_col = None
    best_labels_col = None
    for c in df.columns:
        series = df[c].astype(str).fillna("")
        str_count = series.apply(lambda x: len(x) > 0).sum()
        label_like = series.apply(lambda x: bool(re.search(r"\d", x)) and all(part.strip().isdigit() for part in re.findall(r"\d+", x)))
        label_count = label_like.sum()
        if label_count > 0.5 * n:
            best_labels_col = c
        if str_count > 0.6 * n and label_count < 0.3 * n:
            best_text_col = c

    if best_text_col is None or best_labels_col is None:
        raise ValueError(f"Could not auto-detect text/labels columns. Columns: {list(df.columns)}. Try passing files with headers 'text' and 'labels'.")
    return best_text_col, best_labels_col

def load_and_process(path: str, source_num_labels: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    print(f"[load] {path} shape: {df.shape}")
    text_col, labels_col = detect_columns(df)
    print(f"[load] using text column '{text_col}' and labels column '{labels_col}'")
    df = df[[text_col, labels_col]].copy()
    df.columns = ["text", "labels"]
    df = df.dropna(subset=["text", "labels"]).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text_keep_emotion)
    df["label_list"] = df["labels"].apply(safe_extract_integers)
    # Filter rows with no labels (optional for GoEmotions)
    df = df[df["label_list"].apply(lambda L: len(L) > 0)].reset_index(drop=True)
    df["label_vector"] = df["label_list"].apply(lambda L: to_multi_hot_from_indices(L, source_num_labels).tolist())
    print(f"[load] processed shape: {df.shape}")
    return df

# ---------------------------
# Merge utilities
# ---------------------------
def build_merge_mapping(original_labels: List[str], merge_pairs: Dict[str, List[int]]):
    """Return (new_label_names, old2new_index_map)
    old2new_index_map: maps old index (0..original_num_labels-1) -> new index (0..new_num_labels-1)
    """
    original_num = len(original_labels)
    merged_to_newname = {}
    for newname, src_idxs in merge_pairs.items():
        for i in src_idxs:
            if i < 0 or i >= original_num:
                raise ValueError(f"merge pair index {i} out of range for original labels")
        merged_to_newname[tuple(sorted(src_idxs))] = newname

    new_labels = []
    old2new = {}
    seen_old = set()

    for old_idx, old_name in enumerate(original_labels):
        if old_idx in seen_old:
            continue
        # check if this old_idx is part of a merge pair
        merged_group = None
        for src_idxs, newname in merged_to_newname.items():
            if old_idx in src_idxs:
                merged_group = (src_idxs, newname)
                break
        if merged_group is None:
            # keep original label
            new_labels.append(old_name)
            new_idx = len(new_labels) - 1
            old2new[old_idx] = new_idx
            seen_old.add(old_idx)
        else:
            src_idxs, newname = merged_group
            new_labels.append(newname)
            new_idx = len(new_labels) - 1
            for s in src_idxs:
                old2new[s] = new_idx
                seen_old.add(s)
    return new_labels, old2new

def apply_label_merge_vector(old_vec: np.ndarray, old2new_map: Dict[int, int], new_num_labels: int) -> np.ndarray:
    new_vec = np.zeros(new_num_labels, dtype=np.float32)
    for old_i, val in enumerate(old_vec):
        if val:
            new_idx = old2new_map.get(old_i, None)
            if new_idx is None:
                # ignore out-of-range
                continue
            new_vec[new_idx] = 1.0 if val > 0 else new_vec[new_idx]
    return new_vec

def apply_label_merge_df(df: pd.DataFrame, old2new_map: Dict[int, int], new_num_labels: int):
    # expects df["label_vector"] as list-like of length original_num_labels
    merged_vectors = []
    for vec in df["label_vector"]:
        old_vec = np.array(vec, dtype=float)
        new_vec = apply_label_merge_vector(old_vec, old2new_map, new_num_labels)
        merged_vectors.append(new_vec.tolist())
    df = df.copy()
    df["label_vector"] = merged_vectors
    return df

# ---------------------------
# Dataset + collate
# ---------------------------
class GoEmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[float]]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


def collate_batch(batch: List[Dict[str,Any]], tokenizer, max_len:int):
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    enc = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels
    }

# ---------------------------
# Asymmetric Loss (ASL) for multi-label
# ---------------------------
class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8, reduction="mean", pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        if self.pos_weight is not None:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        else:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = prob * targets + (1 - prob) * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        focal_factor = (1 - pt) ** gamma

        if self.clip and self.clip > 0:
            prob_neg = (1 - prob).clamp(min=self.clip, max=1.0)
            pt = prob * targets + prob_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focal_factor = (1 - pt) ** gamma

        loss = focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# ---------------------------
# Focal BCE fallback
# ---------------------------
class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean", pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        if self.pos_weight is not None:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        else:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ---------------------------
# Small augmentation utils
# ---------------------------
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    _WN_AVAILABLE = True
except Exception:
    _WN_AVAILABLE = False


def get_synonyms(word: str) -> List[str]:
    if not _WN_AVAILABLE:
        return []
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if w.lower() != word.lower():
                syns.add(w)
    return list(syns)


def synonym_replacement(text: str, n_replacements: int = 1) -> str:
    words = text.split()
    if len(words) == 0:
        return text
    indices = list(range(len(words)))
    random.shuffle(indices)
    replaced = 0
    for idx in indices:
        w = re.sub(r"[^\w']", "", words[idx])
        if len(w) < 3:
            continue
        syns = get_synonyms(w)
        if syns:
            words[idx] = words[idx].replace(w, random.choice(syns))
            replaced += 1
        if replaced >= n_replacements:
            break
    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    keep = [w for w in words if random.random() > p]
    if len(keep) == 0:
        keep = [words[random.randrange(len(words))]]
    return " ".join(keep)


def random_swap(text: str, n_swaps: int = 1) -> str:
    words = text.split()
    L = len(words)
    if L < 2:
        return text
    for _ in range(n_swaps):
        i, j = random.sample(range(L), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def augment_text_medium(text: str) -> List[str]:
    aug1 = synonym_replacement(text, n_replacements=1)
    aug1 = random_swap(aug1, n_swaps=1)
    aug2 = random_deletion(text, p=0.12)
    aug2 = random_swap(aug2, n_swaps=1)
    return [aug1, aug2]

# ---------------------------
# Sampler
# ---------------------------
def create_sampler(df: pd.DataFrame, source_num_labels: int, rare_label_factor: float = 5.0):
    label_matrix = np.stack(df["label_vector"].values)  # (N, L_source)
    label_counts = label_matrix.sum(axis=0) + 1e-6
    inv_freq = 1.0 / label_counts
    rare_threshold = np.percentile(label_counts, 25)
    rare_mask = label_counts <= rare_threshold
    inv_freq = np.log1p(inv_freq * np.max(label_counts))
    inv_freq = inv_freq / inv_freq.mean()
    inv_freq[rare_mask] *= rare_label_factor
    sample_weights = (label_matrix * inv_freq).sum(axis=1)
    sample_weights = sample_weights + 1e-12
    sample_weights = sample_weights / sample_weights.sum()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# ---------------------------
# Eval / threshold utilities
# ---------------------------
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def normalize_logits(logits: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = logits.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return logits / std

def evaluate_preds(all_logits: np.ndarray, all_labels: np.ndarray, threshold: float):
    probs = sigmoid_np(all_logits)
    preds = (probs >= threshold).astype(int)
    micro = f1_score(all_labels, preds, average="micro", zero_division=0)
    macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    return micro, macro, preds, probs

def threshold_search_per_label(all_logits: np.ndarray, all_labels: np.ndarray, sweep: np.ndarray):
    norm_logits = normalize_logits(all_logits)
    L = all_logits.shape[1]
    best_ts = np.zeros(L, dtype=float)
    for i in range(L):
        best_t_i = 0.5
        best_f1_i = -1.0
        logits_i = norm_logits[:, i]
        labels_i = all_labels[:, i]
        for t in sweep:
            preds_i = (sigmoid_np(logits_i) >= t).astype(int)
            f1_i = f1_score(labels_i, preds_i, zero_division=0)
            if f1_i > best_f1_i:
                best_f1_i = f1_i
                best_t_i = t
        best_ts[i] = best_t_i
    probs = sigmoid_np(norm_logits)
    preds = (probs >= best_ts.reshape(1, -1)).astype(int)
    macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    return best_ts, macro

# ---------------------------
# Training / eval loops
# ---------------------------
def train_epoch_focal(model, loader, optimizer, scheduler, device, scaler=None, criterion=None, debug_labels=False, grad_accum_steps=1, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(loader, desc="Train")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if debug_labels and batch_idx == 0:
            label_sums = labels.sum(dim=0).cpu().numpy()
            print("[debug] label counts in first batch:", label_sums.astype(int))

        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels) / float(grad_accum_steps)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                if max_grad_norm is not None and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item()) * grad_accum_steps
        steps += 1
    return total_loss / max(1, steps)


def eval_full(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    logits_list = []
    labels_list = []
    steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if criterion is not None:
                loss = criterion(logits, labels)
            else:
                loss = torch.tensor(0.0)
            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            logits_list.append(logits_np)
            labels_list.append(labels_np)
            total_loss += float(loss.item())
            steps += 1
    if len(logits_list) == 0:
        return 0.0, np.zeros((0, cfg["num_labels"])), np.zeros((0, cfg["num_labels"]))
    all_logits = np.vstack(logits_list)
    all_labels = np.vstack(labels_list)
    return total_loss / max(1, steps), all_logits, all_labels

# ---------------------------
# Rare-label recommendations
# ---------------------------
def rare_label_recommendations(train_df, num_labels, min_support=50):
    label_matrix = np.stack(train_df["label_vector"].values)
    label_counts = label_matrix.sum(axis=0)
    print("\n[rare label analysis] label counts:", label_counts.astype(int))

    rare_labels = [i for i, c in enumerate(label_counts) if c < min_support]
    if rare_labels:
        print("[rare label analysis] rare labels (count < {}): {}".format(min_support, rare_labels))
        print("Recommendations:")
        print("1. Aggressive oversampling: ensured via `create_sampler()`.")
        print("2. Data augmentation: medium augmentation (2 variants per rare sample) is applied.")
        print("3. Consider lower batch size + gradient accumulation (configured).")
        print("4. Longer training and patience increased.")
        print("5. If too rare, consider merging or marking as 'other'.")
    return rare_labels

# ---------------------------
# Main
# ---------------------------
def main(args):
    global cfg
    cfg = {**DEFAULTS}
    cfg.update(vars(args))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device] Using:", device)

    # Load original label names (28) if available
    if os.path.exists(cfg["labels_file"]):
        with open(cfg["labels_file"], "r", encoding="utf-8") as f:
            original_label_names = [l.strip() for l in f.readlines() if l.strip()]
        if len(original_label_names) != cfg["original_num_labels"]:
            print(f"[warning] labels_file contains {len(original_label_names)} entries; expected {cfg['original_num_labels']}.")
            # fallback: pad/trim
            if len(original_label_names) < cfg["original_num_labels"]:
                original_label_names += DEFAULT_ORIGINAL_LABELS[len(original_label_names):cfg["original_num_labels"]]
            else:
                original_label_names = original_label_names[:cfg["original_num_labels"]]
    else:
        original_label_names = DEFAULT_ORIGINAL_LABELS.copy()
        print("[warning] labels_file not found; using default original label order.")

    # Build merge mapping
    new_label_names, old2new_map = build_merge_mapping(original_label_names, MERGE_PAIRS)
    new_num_labels = len(new_label_names)
    print(f"[labels] original={len(original_label_names)} -> merged={new_num_labels}")
    print("[labels] new label names:")
    for i, n in enumerate(new_label_names):
        print(f"{i:02d} {n}")

    # Load data using source label count
    train_df = load_and_process(cfg["train_file"], cfg["original_num_labels"])  # source vectors length 28
    dev_df = load_and_process(cfg["dev_file"], cfg["original_num_labels"])      # source vectors length 28
    print(f"[data] Train: {len(train_df)}, Dev: {len(dev_df)}")

    # Apply label merging to both train and dev
    train_df = apply_label_merge_df(train_df, old2new_map, new_num_labels)
    dev_df = apply_label_merge_df(dev_df, old2new_map, new_num_labels)

    # Neutral downsampling
    neutral_idx = None
    for i, nm in enumerate(new_label_names):
        if nm.lower() == "neutral":
            neutral_idx = i
            break
    if neutral_idx is None:
        neutral_idx = new_num_labels - 1

    def is_pure_neutral(row):
        vec = np.array(row["label_vector"], dtype=int)
        return (vec.sum() == 1) and (vec[neutral_idx] == 1)

    pure_neutral_mask = train_df.apply(is_pure_neutral, axis=1)
    pure_neutral_count = pure_neutral_mask.sum()
    print(f"[neutral] pure-neutral count before downsampling: {int(pure_neutral_count)}")
    if pure_neutral_count > 0:
        keep_frac = cfg.get("neutral_keep_frac", 0.35)
        keep_n = int(max(1, keep_frac * int(pure_neutral_count)))
        pure_neutral_indices = train_df[pure_neutral_mask].index.tolist()
        random.shuffle(pure_neutral_indices)
        keep_indices = set(pure_neutral_indices[:keep_n])
        keep_mask = train_df.index.isin(keep_indices) | (~pure_neutral_mask)
        train_df = train_df[keep_mask].reset_index(drop=True)
        print(f"[neutral] kept {keep_n} pure-neutral rows ({keep_frac*100:.0f}%) -> new train size: {len(train_df)}")
    else:
        print("[neutral] no pure-neutral rows found or already small; skipping downsample")

    # Rare-label augmentation (medium augmentation)
    train_matrix = np.stack(train_df["label_vector"].values)
    label_counts = train_matrix.sum(axis=0)
    rare_mask = label_counts < cfg.get("min_support_for_rare", 50)
    rare_labels = [i for i, m in enumerate(rare_mask) if m]
    print("[data] label counts (train) after neutral downsampling:", label_counts.astype(int))
    if rare_labels:
        print("[augment] rare labels detected:", rare_labels)
        multiplier = cfg.get("augmentation_multiplier_by_level", {1:1,2:2,3:5}).get(cfg.get("augmentation_level", 2), 2)
        new_rows = []
        for _, row in train_df.iterrows():
            vec = np.array(row["label_vector"], dtype=int)
            if any(vec[idx] == 1 for idx in rare_labels):
                try:
                    variants = augment_text_medium(row["text"])
                except Exception:
                    variants = []
                for i, aug_text in enumerate(variants[:multiplier]):
                    new_row = {
                        "text": aug_text,
                        "labels": row.get("labels", ""),
                        "label_list": row.get("label_list", []),
                        "label_vector": row["label_vector"],
                    }
                    new_rows.append(new_row)
        if len(new_rows) > 0:
            aug_df = pd.DataFrame(new_rows)
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
            print(f"[augment] added {len(new_rows)} augmented rows for rare labels (multiplier={multiplier})")
        else:
            print("[augment] no augmented rows created (augmentation functions may be unavailable).")
    else:
        print("[augment] no rare labels found (min_support threshold may be low).")

    # Recompute matrices and pos_weight after augmentation/downsampling
    train_matrix = np.stack(train_df["label_vector"].values)
    dev_matrix = np.stack(dev_df["label_vector"].values)
    print("[data] label counts (train) final:", train_matrix.sum(axis=0).astype(int))

    pos_counts = train_matrix.sum(axis=0)
    neg_counts = train_matrix.shape[0] - pos_counts
    pos_counts_clamped = np.maximum(pos_counts, 1.0)
    raw_pos_weight = np.array([(n / p if p > 0 else 1.0) for n, p in zip(neg_counts, pos_counts_clamped)], dtype=float)
    pos_weight_cap = cfg.get("pos_weight_cap", 40.0)
    raw_pos_weight = np.minimum(raw_pos_weight, pos_weight_cap)
    pos_weight = torch.tensor(raw_pos_weight, dtype=torch.float).to(device)
    print("[info] pos_weight per label (capped):", pos_weight)

    # Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds = GoEmotionDataset(train_df["text"].tolist(), train_df["label_vector"].tolist())
    dev_ds = GoEmotionDataset(dev_df["text"].tolist(), dev_df["label_vector"].tolist())
    collate = lambda batch: collate_batch(batch, tokenizer=tokenizer, max_len=cfg["max_len"])

    # Sampler / DataLoader
    if cfg["use_oversample"]:
        sampler = create_sampler(train_df, cfg["original_num_labels"], rare_label_factor=cfg.get("rare_label_factor", 5.0))
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=True)

    dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=new_num_labels, problem_type="multi_label_classification"
    )
    model.to(device)

    # Criterion selection
    if cfg["use_asl"]:
        criterion = AsymmetricLossMultiLabel(
            gamma_pos=cfg.get("asl_gamma_pos", 0.0),
            gamma_neg=cfg.get("asl_gamma_neg", 1.0),
            clip=cfg.get("asl_clip", 0.05),
            pos_weight=pos_weight
        )
    elif cfg["use_focal_loss"]:
        criterion = FocalBCEWithLogits(gamma=2.0, reduction="mean", pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"] // max(1, cfg.get("grad_accum_steps", 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=max(1, total_steps))
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    # Rare-label analysis & recommendations (prints)
    rare_label_recommendations(train_df, new_num_labels, min_support=cfg.get("min_support_for_rare", 50))

    best_macro = -1.0
    best_thresholds_per_label = np.array([0.5] * new_num_labels)
    patience = 0

    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    for epoch in range(cfg["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{cfg['epochs']} ===")
        train_loss = train_epoch_focal(model, train_loader, optimizer, scheduler, device, scaler=scaler, criterion=criterion, debug_labels=True, grad_accum_steps=grad_accum_steps, max_grad_norm=max_grad_norm)
        print("[train] loss:", round(train_loss, 4))

        val_loss, val_logits, val_labels = eval_full(model, dev_loader, device, criterion=criterion)
        print("[val] loss:", round(val_loss, 4))
        if val_logits.size == 0:
            print("[eval] no validation logits collected; skipping threshold search")
            continue

        print("[val] logits min/max:", float(val_logits.min()), float(val_logits.max()))
        print("[val] sigmoid mean:", float(sigmoid_np(val_logits).mean()))

        if cfg["per_label_threshold"]:
            thresholds, macro_local = threshold_search_per_label(val_logits, val_labels, sweep=cfg["threshold_sweep"])
            print(f"[threshold search] per-label thresholds computed (macro-F1={macro_local:.4f})")
            micro = f1_score(val_labels, (sigmoid_np(normalize_logits(val_logits)) >= thresholds.reshape(1, -1)).astype(int), average="micro", zero_division=0)
            print(f"[val metrics per-label] micro-F1: {micro:.4f} | macro-F1: {macro_local:.4f}")
        else:
            best_t, macro_local = threshold_search_global(val_logits, val_labels, sweep=cfg["threshold_sweep"])
            thresholds = np.array([best_t] * new_num_labels)
            print(f"[threshold search] best global threshold: {best_t:.3f} (macro-F1={macro_local:.4f})")
            micro, macro_local, _, _ = evaluate_preds(normalize_logits(val_logits), val_labels, best_t)
            print(f"[val metrics @t={best_t:.3f}] micro-F1: {micro:.4f} | macro-F1: {macro_local:.4f}")

        if macro_local > best_macro:
            best_macro = macro_local
            best_thresholds_per_label = thresholds.copy()
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer": cfg["model_name"],
                "thresholds_per_label": best_thresholds_per_label.tolist(),
                "label_names": new_label_names
            }, cfg["checkpoint"])
            print("[checkpoint] saved", cfg["checkpoint"])
            patience = 0
        else:
            patience += 1
            print(f"[patience] {patience}/{cfg['patience']}")
            if patience >= cfg["patience"]:
                print("[early stop] stopping training")
                break

    # Final evaluation after loading best checkpoint
    ckpt = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    saved_thresholds = ckpt.get("thresholds_per_label", None)
    if saved_thresholds is not None:
        final_thresholds = np.array(saved_thresholds)
    else:
        final_thresholds = np.array([0.5] * new_num_labels)

    val_loss, val_logits, val_labels = eval_full(model, dev_loader, device, criterion=criterion)
    if val_logits.size == 0:
        print("[final eval] no logits collected from validation set.")
        return

    probs = sigmoid_np(normalize_logits(val_logits))
    preds = (probs >= final_thresholds.reshape(1, -1)).astype(int)
    micro = f1_score(val_labels, preds, average="micro", zero_division=0)
    macro = f1_score(val_labels, preds, average="macro", zero_division=0)
    print(f"[final eval] micro-F1: {micro:.4f} | macro-F1: {macro:.4f}")

    p, r, f, s = precision_recall_fscore_support(val_labels, preds, average=None, zero_division=0)
    print("\nPer-label F1 scores:")
    for i, nm in enumerate(new_label_names):
        sup_i = int(s[i]) if i < len(s) else 0
        print(f"{i:02d} {nm:15s} | prec {p[i]:.3f} rec {r[i]:.3f} f1 {f[i]:.3f} sup {sup_i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=DEFAULTS["train_file"])
    parser.add_argument("--dev_file", default=DEFAULTS["dev_file"])
    parser.add_argument("--labels_file", default=DEFAULTS["labels_file"])
    parser.add_argument("--model_name", default=DEFAULTS["model_name"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--max_len", type=int, default=DEFAULTS["max_len"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--num_labels", type=int, default=DEFAULTS["num_labels"])  # final labels after merge
    parser.add_argument("--original_num_labels", type=int, default=DEFAULTS["original_num_labels"])  # source labels
    parser.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--use_oversample", type=bool, default=DEFAULTS["use_oversample"])
    parser.add_argument("--rare_label_factor", type=float, default=DEFAULTS["rare_label_factor"])
    parser.add_argument("--use_asl", type=bool, default=DEFAULTS["use_asl"])
    parser.add_argument("--per_label_threshold", type=bool, default=DEFAULTS["per_label_threshold"])
    parser.add_argument("--augmentation_level", type=int, default=DEFAULTS["augmentation_level"])
    parser.add_argument("--neutral_keep_frac", type=float, default=DEFAULTS["neutral_keep_frac"])
    parser.add_argument("--grad_accum_steps", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULTS["max_grad_norm"])
    parser.add_argument("--pos_weight_cap", type=float, default=DEFAULTS["pos_weight_cap"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    args = parser.parse_args()
    main(args)
