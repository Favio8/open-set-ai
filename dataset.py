import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import wfdb
except Exception:  # pragma: no cover
    wfdb = None

from utils import resolve_recording_path


EPS = 1e-8


def _safe_to_numeric_frame(df: pd.DataFrame) -> np.ndarray:
    arr = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").values
    if arr.size == 0:
        raise ValueError("No numeric values found in ECG file.")
    return arr.astype(np.float32)


def _read_csv_signal(file_path: str) -> np.ndarray:
    tried = []
    for header in [None, "infer"]:
        try:
            df = pd.read_csv(file_path, header=header)
            arr = _safe_to_numeric_frame(df)
            return arr
        except Exception as exc:
            tried.append(str(exc))
    raise ValueError(f"Unable to read ECG csv file: {file_path}. Errors: {tried}")



def load_ecg_signal(file_path: str, nleads: int, length: int) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".csv", ".txt"}:
        signal = _read_csv_signal(file_path)
    elif ext == ".npy":
        signal = np.load(file_path).astype(np.float32)
    else:
        if wfdb is None:
            raise ValueError(
                f"Unsupported ECG file format for {file_path}. Install wfdb or convert files to csv/npy."
            )
        base = os.path.splitext(file_path)[0]
        signal, _ = wfdb.rdsamp(base)
        signal = signal.astype(np.float32)

    if signal.ndim == 1:
        signal = signal[:, None]

    if signal.shape[0] == nleads and signal.shape[1] != nleads:
        signal = signal.T
    if signal.shape[1] < nleads:
        pad = np.zeros((signal.shape[0], nleads - signal.shape[1]), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=1)
    elif signal.shape[1] > nleads:
        signal = signal[:, :nleads]

    signal = signal[-length:, :]
    result = np.zeros((length, nleads), dtype=np.float32)
    result[-signal.shape[0]:, :] = signal
    return result.T  # [C, L]



def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (x - mean) / std



def _random_crop_or_shift(x: np.ndarray, max_shift_ratio: float = 0.1) -> np.ndarray:
    x = x.copy()
    max_shift = max(1, int(x.shape[1] * max_shift_ratio))
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift, axis=1)



def _amplitude_scale(x: np.ndarray, low: float, high: float) -> np.ndarray:
    scale = np.random.uniform(low, high)
    return x * scale



def _gaussian_noise(x: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, std, size=x.shape).astype(np.float32)
    return x + noise



def _lead_dropout(x: np.ndarray, p: float, max_drop_ratio: float = 0.25) -> np.ndarray:
    x = x.copy()
    if np.random.rand() >= p:
        return x
    max_drop = max(1, int(round(x.shape[0] * max_drop_ratio)))
    n_drop = np.random.randint(1, max_drop + 1)
    leads = np.random.choice(x.shape[0], n_drop, replace=False)
    x[leads] = 0.0
    return x



def _temporal_mask(x: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    x = x.copy()
    width = max(1, int(round(x.shape[1] * ratio)))
    if width >= x.shape[1]:
        return x
    start = np.random.randint(0, x.shape[1] - width)
    x[:, start:start + width] = 0.0
    return x



def augment_ecg(x: np.ndarray, augmentation_type: str = "default") -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    aug = str(augmentation_type or "default").lower()

    if aug in {"none", "raw"}:
        return x

    if aug in {"weak", "default"}:
        x = _amplitude_scale(x, 0.95, 1.05)
        x = _gaussian_noise(x, 0.005)
        x = _random_crop_or_shift(x, 0.03)
        return x

    if aug == "hardneg":
        x = _amplitude_scale(x, 0.80, 1.20)
        x = _gaussian_noise(x, 0.02)
        x = _random_crop_or_shift(x, 0.12)
        x = _lead_dropout(x, 0.5, 0.35)
        x = _temporal_mask(x, 0.10)
        return x

    if aug == "strong":
        x = _amplitude_scale(x, 0.75, 1.25)
        x = _gaussian_noise(x, 0.03)
        x = _random_crop_or_shift(x, 0.15)
        x = _lead_dropout(x, 0.6, 0.40)
        x = _temporal_mask(x, 0.12)
        return x

    if aug == "reverse":
        x = x[:, ::-1].copy()
        x = _gaussian_noise(x, 0.01)
        return x

    if aug == "scaling_up":
        return _amplitude_scale(x, 1.05, 1.25)

    if aug == "scaling_down":
        return _amplitude_scale(x, 0.75, 0.95)

    if aug == "jitter":
        return _gaussian_noise(x, 0.02)

    if aug == "drop_lead":
        return _lead_dropout(x, 1.0, 0.35)

    return augment_ecg(x, "default")



def make_multiview_pair(x: np.ndarray, transform_type: str = "hardneg") -> Tuple[np.ndarray, np.ndarray]:
    view1 = augment_ecg(x, "weak")
    transform_type = str(transform_type or "hardneg").lower()
    if transform_type in {"weak", "default", "none", "raw"}:
        view2 = augment_ecg(x, "default")
    elif transform_type in {"hardneg", "strong", "reverse", "scaling_up", "scaling_down", "jitter", "drop_lead"}:
        view2 = augment_ecg(x, transform_type)
    else:
        view2 = augment_ecg(x, "hardneg")
    return view1.astype(np.float32), view2.astype(np.float32)


class ECGDataset_unseen(Dataset):
    def __init__(
        self,
        phase: str,
        data_dir: str,
        label_csv: str,
        leads: int,
        length: int,
        transform_type: str = "hardneg",
        dual_view: Optional[bool] = None,
        label_key: str = "label",
        return_index: bool = False,
        normalize: bool = False,
    ):
        super().__init__()
        self.phase = phase
        self.data_dir = data_dir
        self.leads = int(leads)
        self.length = int(length)
        self.transform_type = transform_type
        self.label_key = label_key
        self.return_index = return_index
        self.normalize = normalize

        df = pd.read_csv(label_csv)
        if "split" not in df.columns:
            raise KeyError(f"Column 'split' not found in split file: {label_csv}")
        df = df[df["split"] == phase].copy().reset_index(drop=True)

        resolved_paths = []
        keep_rows = []
        for i, rec in enumerate(df["Recording"].tolist()):
            path = resolve_recording_path(data_dir, rec)
            if path is not None:
                resolved_paths.append(path)
                keep_rows.append(i)
        df = df.iloc[keep_rows].reset_index(drop=True)
        self.file_paths = resolved_paths
        self.df = df

        if dual_view is None:
            dual_view = phase in {"train", "train_valid", "discover"}
        self.dual_view = bool(dual_view)
        self.pseudo_labels: Optional[np.ndarray] = None

    def set_pseudo_labels(self, pseudo_labels: Optional[Sequence[int]]) -> None:
        if pseudo_labels is None:
            self.pseudo_labels = None
            return
        pseudo_labels = np.asarray(pseudo_labels, dtype=np.int64)
        if len(pseudo_labels) != len(self.df):
            raise ValueError(
                f"Pseudo label length mismatch: expected {len(self.df)}, got {len(pseudo_labels)}"
            )
        self.pseudo_labels = pseudo_labels

    def _get_label(self, index: int) -> int:
        if self.pseudo_labels is not None:
            return int(self.pseudo_labels[index])
        if self.label_key in self.df.columns:
            return int(self.df.iloc[index][self.label_key])
        return -1

    def _load_signal(self, index: int) -> np.ndarray:
        x = load_ecg_signal(self.file_paths[index], self.leads, self.length)
        if self.normalize:
            x = zscore_per_lead(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return x

    def __getitem__(self, index: int):
        x = self._load_signal(index)
        label = self._get_label(index)

        if self.dual_view:
            view1, view2 = make_multiview_pair(x, self.transform_type)
            item = [torch.from_numpy(view1), torch.from_numpy(view2), torch.tensor(label, dtype=torch.long)]
        else:
            item = [torch.from_numpy(x), torch.tensor(label, dtype=torch.long)]

        if self.return_index:
            item.append(torch.tensor(index, dtype=torch.long))
        return tuple(item)

    def __len__(self) -> int:
        return len(self.df)


class ECGDataset_unseen_MHL_stage2(ECGDataset_unseen):
    def __init__(
        self,
        phase: str,
        data_dir: str,
        label_csv: str,
        leads: int,
        length: int,
        transform_type: str = "hardneg",
        dual_view: Optional[bool] = None,
        label_key: str = "open_label",
        return_index: bool = False,
        normalize: bool = False,
    ):
        super().__init__(
            phase=phase,
            data_dir=data_dir,
            label_csv=label_csv,
            leads=leads,
            length=length,
            transform_type=transform_type,
            dual_view=dual_view,
            label_key=label_key,
            return_index=return_index,
            normalize=normalize,
        )
