import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score


EPS = 1e-12


@dataclass
class DatasetDefaults:
    dataset: str
    leads: int
    hz: int
    duration: int
    data_dir_name: str
    label_file_name: str
    unseen_classes: List[int]

    @property
    def length(self) -> int:
        return self.hz * self.duration


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {v}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_dataset_defaults(dataset: str) -> DatasetDefaults:
    dataset = str(dataset)
    presets: Dict[str, DatasetDefaults] = {
        "CPSC18": DatasetDefaults("CPSC18", 12, 500, 30, "CPSC18_szhou_all", "CPSC18_label_all.csv", [4]),
        "PTB": DatasetDefaults("PTB", 12, 500, 10, "PTB_szhou_all", "PTB_label_all.csv", [3]),
        "Georgia": DatasetDefaults("Georgia", 12, 500, 10, "Georgia_szhou_all", "Georgia_label_all.csv", [3]),
        "CPSC18_U3": DatasetDefaults("CPSC18_U3", 12, 500, 30, "CPSC18_szhou_all", "CPSC18_U3_label_all.csv", [4]),
        "Georgia_U3": DatasetDefaults("Georgia_U3", 12, 500, 10, "Georgia_szhou_all", "Georgia_U3_label_all.csv", [3]),
        "CPSC18-STE": DatasetDefaults("CPSC18-STE", 12, 500, 30, "CPSC18_szhou_all", "CPSC18-STE_label_all.csv", [4]),
    }
    if dataset in presets:
        return presets[dataset]
    return DatasetDefaults(dataset, 12, 500, 10, f"{dataset}_szhou_all", f"{dataset}_label_all.csv", [0])


def parse_unseen_classes(unseen_classes: Optional[str], default_unseen: Sequence[int]) -> List[int]:
    if unseen_classes is None or str(unseen_classes).strip() == "":
        return sorted(list({int(x) for x in default_unseen}))
    raw = str(unseen_classes).replace(";", ",").replace(" ", "")
    values = [x for x in raw.split(",") if x != ""]
    return sorted(list({int(v) for v in values}))


def _infer_recording_column(df: pd.DataFrame) -> str:
    candidates = [
        "Recording",
        "recording",
        "filename",
        "file_name",
        "file",
        "path",
        "Path",
        "ecg_path",
        "record",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "Unable to find recording column in label file. Expected one of: " + ", ".join(candidates)
    )


def _infer_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "org_label",
        "label",
        "Label",
        "class",
        "target",
        "y",
        "diagnosis",
        "diagnostic_class",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "Unable to find label column in label file. Expected one of: " + ", ".join(candidates)
    )


def load_base_label_dataframe(label_file: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(label_file)
    rec_col = _infer_recording_column(df)
    label_col = _infer_label_column(df)
    out = pd.DataFrame({
        "Recording": df[rec_col].astype(str).str.strip(),
        "org_label": pd.to_numeric(df[label_col], errors="coerce"),
    })
    out = out.dropna(subset=["Recording", "org_label"]).copy()
    out["org_label"] = out["org_label"].astype(int)
    if data_dir is not None:
        out["exists"] = out["Recording"].apply(lambda x: resolve_recording_path(data_dir, x) is not None)
        out = out[out["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def resolve_recording_path(data_dir: str, recording: str) -> Optional[str]:
    recording = str(recording)
    candidates = []
    if os.path.isabs(recording):
        candidates.append(recording)
    else:
        candidates.append(os.path.join(data_dir, recording))
        root, ext = os.path.splitext(recording)
        if ext == "":
            for suffix in [".csv", ".txt", ".npy"]:
                candidates.append(os.path.join(data_dir, recording + suffix))
        if root != recording:
            candidates.append(os.path.join(data_dir, root))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _split_class_indices(indices: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.array(indices, dtype=int)
    rng.shuffle(indices)
    n = len(indices)
    if n == 0:
        return indices, indices, indices

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    elif n == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0

    n_test = n - n_train - n_val
    if n_test < 0:
        overflow = -n_test
        if n_val >= overflow:
            n_val -= overflow
        else:
            overflow -= n_val
            n_val = 0
            n_train = max(1, n_train - overflow)
        n_test = n - n_train - n_val

    if n >= 3 and n_test == 0:
        if n_val > 1:
            n_val -= 1
            n_test = 1
        elif n_train > 1:
            n_train -= 1
            n_test = 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_open_world_split_frames(
    label_file: str,
    data_dir: str,
    unseen_classes: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = load_base_label_dataframe(label_file, data_dir=data_dir)
    all_labels = sorted(df["org_label"].unique().tolist())
    unseen_set = {int(x) for x in unseen_classes}
    seen_classes = [int(x) for x in all_labels if int(x) not in unseen_set]
    if len(seen_classes) == 0:
        raise ValueError("No seen classes left after removing unseen classes.")

    known_map = {orig: idx for idx, orig in enumerate(seen_classes)}
    unknown_open_label = len(seen_classes)

    base_split = np.array([None] * len(df), dtype=object)
    for label in all_labels:
        class_idx = np.where(df["org_label"].values == label)[0]
        train_idx, val_idx, test_idx = _split_class_indices(class_idx, train_ratio, val_ratio, seed + int(label) * 13)
        base_split[train_idx] = "train"
        base_split[val_idx] = "valid"
        base_split[test_idx] = "test"

    df = df.copy()
    df["base_split"] = base_split
    df["is_seen"] = df["org_label"].apply(lambda x: int(x) in known_map)
    df["label"] = df["org_label"].apply(lambda x: known_map.get(int(x), -1))
    df["open_label"] = df["org_label"].apply(lambda x: known_map.get(int(x), unknown_open_label))

    stage1 = df.copy()
    stage1["split"] = stage1.apply(
        lambda row: row["base_split"] if bool(row["is_seen"]) else "test",
        axis=1,
    )
    stage1 = stage1[["Recording", "org_label", "label", "open_label", "is_seen", "base_split", "split"]]
    stage1 = stage1.reset_index(drop=True)

    stage2 = df.copy()
    stage2["split"] = stage2["base_split"].map({"train": "train_valid", "valid": "train_valid", "test": "test"})
    stage2 = stage2[["Recording", "org_label", "label", "open_label", "is_seen", "base_split", "split"]]
    stage2 = stage2.reset_index(drop=True)

    meta = {
        "num_known_classes": len(seen_classes),
        "unknown_open_label": unknown_open_label,
        "seen_classes": seen_classes,
        "unseen_classes": sorted(list(unseen_set)),
        "known_map": {str(k): int(v) for k, v in known_map.items()},
        "train_count": int((stage1["split"] == "train").sum()),
        "valid_count": int((stage1["split"] == "valid").sum()),
        "discover_count": int((stage2["split"] == "train_valid").sum()),
        "test_count": int((stage2["split"] == "test").sum()),
    }
    return stage1, stage2, meta


def gen_label_csv_unseen_setting(
    data_dir: str,
    label_file: str,
    output_csv: str,
    unseen_class_name,
    trn_ratio: float,
    val_ratio: float,
    seed: int,
    transform_type: str = "hardneg",
):
    unseen_classes = unseen_class_name if isinstance(unseen_class_name, (list, tuple, set)) else [int(unseen_class_name)]
    stage1, _, _ = build_open_world_split_frames(label_file, data_dir, unseen_classes, trn_ratio, val_ratio, seed)
    ensure_dir(os.path.dirname(output_csv) or ".")
    stage1.to_csv(output_csv, index=False)
    return stage1


def gen_label_csv_unseen_setting_2_MHL(
    label_file: str,
    output_csv: str,
    unseen_class_name,
    trn_ratio: float,
    val_ratio: float,
    seed: int,
    data_dir: Optional[str] = None,
):
    unseen_classes = unseen_class_name if isinstance(unseen_class_name, (list, tuple, set)) else [int(unseen_class_name)]
    if data_dir is None:
        data_dir = os.path.dirname(label_file)
    _, stage2, meta = build_open_world_split_frames(label_file, data_dir, unseen_classes, trn_ratio, val_ratio, seed)
    ensure_dir(os.path.dirname(output_csv) or ".")
    stage2.to_csv(output_csv, index=False)
    counts = []
    df_seen = stage2[(stage2["split"] == "train_valid") & (stage2["is_seen"])]
    for idx in range(meta["num_known_classes"]):
        counts.append(int((df_seen["label"] == idx).sum()))
    return counts


def save_json(obj: Dict[str, object], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_hot_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(int)
    return np.argmax(y, axis=1).astype(int)


def cal_f1s_naive(y_trues, y_scores, args=None):
    y_true = one_hot_to_int(y_trues)
    y_pred = one_hot_to_int(y_scores)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report)
    class_keys = [k for k in report_df.columns if k not in {"accuracy", "macro avg", "weighted avg"}]
    class_keys = sorted(class_keys, key=lambda x: int(x) if str(x).isdigit() else str(x))
    f1_classes = np.array([float(report_df.loc["f1-score", str(k)]) for k in class_keys], dtype=np.float32)
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_all = report_df.loc["f1-score"].values.astype(np.float32)
    return f1_classes, f1_micro, f1_all, report_df


def compute_open_set_metrics(
    y_true_open: Sequence[int],
    y_pred_open: Sequence[int],
    unknown_label: int,
    novel_scores: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true_open = np.asarray(y_true_open, dtype=int)
    y_pred_open = np.asarray(y_pred_open, dtype=int)
    report = classification_report(y_true_open, y_pred_open, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report)

    metrics = {
        "accuracy": float((y_true_open == y_pred_open).mean()),
        "macro_f1": float(f1_score(y_true_open, y_pred_open, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true_open, y_pred_open, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_open, y_pred_open, average="weighted", zero_division=0)),
        "old_macro_f1": float(f1_score(y_true_open[y_true_open != unknown_label], y_pred_open[y_true_open != unknown_label], average="macro", zero_division=0))
        if np.any(y_true_open != unknown_label)
        else 0.0,
        "new_f1": float(f1_score((y_true_open == unknown_label).astype(int), (y_pred_open == unknown_label).astype(int), zero_division=0)),
    }

    if novel_scores is not None and len(np.unique((y_true_open == unknown_label).astype(int))) > 1:
        try:
            metrics["auroc_known_vs_unknown"] = float(
                roc_auc_score((y_true_open == unknown_label).astype(int), np.asarray(novel_scores, dtype=float))
            )
        except Exception:
            metrics["auroc_known_vs_unknown"] = float("nan")
    else:
        metrics["auroc_known_vs_unknown"] = float("nan")

    return metrics, report_df


def collapse_unknown_predictions(y_pred: Sequence[int], unknown_label: int) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=int)
    y_pred = np.where(y_pred >= unknown_label, unknown_label, y_pred)
    return y_pred


def format_metric_row(metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    if prefix == "":
        return {k: float(v) if isinstance(v, (int, float, np.number)) and not (isinstance(v, float) and math.isnan(v)) else v for k, v in metrics.items()}
    return {f"{prefix}{k}": float(v) if isinstance(v, (int, float, np.number)) and not (isinstance(v, float) and math.isnan(v)) else v for k, v in metrics.items()}


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)


def save_embedding_csv(path: str, embeddings: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    pd.DataFrame(embeddings).to_csv(path, index=False, header=False)


def pin_memory_for_device(device: torch.device) -> bool:
    return device.type == "cuda"
