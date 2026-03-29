from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


EPS = 1e-12



def _as_int_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 2:
        return torch.argmax(labels, dim=1)
    return labels.long()



def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape [B, V, D], got {tuple(features.shape)}")

    device = features.device
    batch_size, n_views, _ = features.shape
    labels = _as_int_labels(labels).view(-1)
    if labels.numel() != batch_size:
        raise ValueError("Label length does not match batch size in supervised contrastive loss.")

    features = F.normalize(features, dim=-1)
    contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*V, D]

    logits = torch.matmul(contrast_features, contrast_features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(device)
    mask = mask.repeat(n_views, n_views)

    logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views, device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + EPS)

    positive_count = mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return contrast_features.new_tensor(0.0)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
    loss = -mean_log_prob_pos.mean()
    return loss



def mutual_information_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    p_y = probs.mean(dim=0)
    h_y = -(p_y * torch.log(p_y + EPS)).sum()
    h_y_given_x = -(probs * torch.log(probs + EPS)).sum(dim=1).mean()
    mi = h_y - h_y_given_x
    return -mi


class MCMILoss(nn.Module):
    def __init__(self, temp: float = 0.07, alpha: float = 1.0, beta: float = 0.4):
        super().__init__()
        self.temp = float(temp)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ce = nn.CrossEntropyLoss()

    def convert_vec_to_int(self, label_vec: torch.Tensor) -> torch.Tensor:
        return _as_int_labels(label_vec)

    def forward_initial_pair(
        self,
        outputs_v1: Dict[str, torch.Tensor],
        outputs_v2: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        known_classes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        targets = self.convert_vec_to_int(targets)
        features = torch.stack([outputs_v1["instance_emb"], outputs_v2["instance_emb"]], dim=1)
        sup_ins = supervised_contrastive_loss(features, targets, temperature=self.temp)

        logits_v1 = outputs_v1["logits"]
        logits_v2 = outputs_v2["logits"]
        if known_classes is not None:
            logits_v1 = logits_v1[:, :known_classes]
            logits_v2 = logits_v2[:, :known_classes]
        ce_loss = 0.5 * (self.ce(logits_v1, targets) + self.ce(logits_v2, targets))
        total = sup_ins + self.alpha * ce_loss
        return {"loss": total, "sup_ins": sup_ins, "ce": ce_loss}

    def forward_continuous_pair(
        self,
        outputs_v1: Dict[str, torch.Tensor],
        outputs_v2: Dict[str, torch.Tensor],
        pseudo_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pseudo_labels = self.convert_vec_to_int(pseudo_labels)
        ins_features = torch.stack([outputs_v1["instance_emb"], outputs_v2["instance_emb"]], dim=1)
        cls_features = torch.stack([outputs_v1["class_emb"], outputs_v2["class_emb"]], dim=1)

        psup_ins = supervised_contrastive_loss(ins_features, pseudo_labels, temperature=self.temp)
        psup_cls = supervised_contrastive_loss(cls_features, pseudo_labels, temperature=self.temp)
        avg_logits = 0.5 * (outputs_v1["logits"] + outputs_v2["logits"])
        mi = mutual_information_loss(avg_logits)
        total = psup_ins + self.alpha * psup_cls + self.beta * mi
        return {"loss": total, "psup_ins": psup_ins, "psup_cls": psup_cls, "mi": mi}

    def forward_initial_stage(self, emb_ins: torch.Tensor, out_cls: torch.Tensor, targets: torch.Tensor):
        targets = self.convert_vec_to_int(targets)
        if emb_ins.ndim == 2:
            emb_ins = torch.stack([emb_ins, emb_ins], dim=1)
        sup_ins = supervised_contrastive_loss(emb_ins, targets, temperature=self.temp)
        ce = self.ce(out_cls, targets)
        total = sup_ins + self.alpha * ce
        return total, sup_ins, ce

    def forward_continuous_stage(
        self,
        emb_ins: torch.Tensor,
        emb_cls: torch.Tensor,
        out_cls: torch.Tensor,
        pseudo_labels_ins: torch.Tensor,
        pseudo_labels_cls: Optional[torch.Tensor] = None,
    ):
        pseudo_labels_ins = self.convert_vec_to_int(pseudo_labels_ins)
        pseudo_labels_cls = pseudo_labels_ins if pseudo_labels_cls is None else self.convert_vec_to_int(pseudo_labels_cls)
        if emb_ins.ndim == 2:
            emb_ins = torch.stack([emb_ins, emb_ins], dim=1)
        if emb_cls.ndim == 2:
            emb_cls = torch.stack([emb_cls, emb_cls], dim=1)
        psup_ins = supervised_contrastive_loss(emb_ins, pseudo_labels_ins, temperature=self.temp)
        psup_cls = supervised_contrastive_loss(emb_cls, pseudo_labels_cls, temperature=self.temp)
        mi = mutual_information_loss(out_cls)
        total = psup_ins + self.alpha * psup_cls + self.beta * mi
        return total, psup_ins, psup_cls, mi

    def forward(self, *args, stage: str = "initial", **kwargs):
        if stage == "initial":
            return self.forward_initial_stage(*args, **kwargs)
        if stage == "continuous":
            return self.forward_continuous_stage(*args, **kwargs)
        raise ValueError(f"Unsupported stage: {stage}")


class OpenMaxSeparation(nn.Module):
    def __init__(self, num_known_classes: Optional[int] = None, distance_scale: float = 2.0, min_known_score: float = 0.35):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.distance_scale = float(distance_scale)
        self.min_known_score = float(min_known_score)
        self.register_buffer("centers", torch.empty(0))
        self.register_buffer("dist_mean", torch.empty(0))
        self.register_buffer("dist_std", torch.empty(0))

    @torch.no_grad()
    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        labels = _as_int_labels(labels).view(-1)
        features = features.detach().float()
        unique_labels = sorted(int(x) for x in labels.unique().tolist() if int(x) >= 0)
        if self.num_known_classes is None:
            self.num_known_classes = len(unique_labels)
        if len(unique_labels) == 0:
            raise ValueError("OpenMaxSeparation.fit received no valid known labels.")

        centers = []
        means = []
        stds = []
        device = features.device
        feat_dim = features.shape[1]
        for c in range(self.num_known_classes):
            class_feat = features[labels == c]
            if class_feat.numel() == 0:
                centers.append(torch.zeros(feat_dim, device=device, dtype=features.dtype))
                means.append(torch.tensor(1.0, device=device, dtype=features.dtype))
                stds.append(torch.tensor(1.0, device=device, dtype=features.dtype))
                continue
            center = class_feat.mean(dim=0)
            dist = torch.norm(class_feat - center, dim=1)
            centers.append(center)
            means.append(dist.mean())
            stds.append(dist.std(unbiased=False) + 1e-6)
        self.centers = torch.stack(centers, dim=0)
        self.dist_mean = torch.stack(means, dim=0)
        self.dist_std = torch.stack(stds, dim=0)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "num_known_classes": self.num_known_classes,
            "distance_scale": self.distance_scale,
            "min_known_score": self.min_known_score,
            "centers": self.centers.detach().cpu(),
            "dist_mean": self.dist_mean.detach().cpu(),
            "dist_std": self.dist_std.detach().cpu(),
        }

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        self.num_known_classes = int(state_dict["num_known_classes"])
        self.distance_scale = float(state_dict["distance_scale"])
        self.min_known_score = float(state_dict["min_known_score"])
        self.centers = state_dict["centers"].detach().clone()
        self.dist_mean = state_dict["dist_mean"].detach().clone()
        self.dist_std = state_dict["dist_std"].detach().clone()
        return self

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls(
            num_known_classes=int(state_dict["num_known_classes"]),
            distance_scale=float(state_dict["distance_scale"]),
            min_known_score=float(state_dict["min_known_score"]),
        )
        obj.load_state_dict(state_dict)
        return obj

    def _check_fitted(self):
        if self.centers.numel() == 0:
            raise RuntimeError("OpenMaxSeparation has not been fitted yet.")

    @torch.no_grad()
    def predict(self, features: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._check_fitted()
        device = features.device
        centers = self.centers.to(device)
        dist_mean = self.dist_mean.to(device)
        dist_std = self.dist_std.to(device)

        dist_matrix = torch.cdist(features.float(), centers.float())
        nearest_dist, nearest_known = dist_matrix.min(dim=1)
        nearest_mean = dist_mean[nearest_known]
        nearest_std = dist_std[nearest_known]
        thresholds = nearest_mean + self.distance_scale * nearest_std
        proto_score = torch.exp(-torch.clamp((nearest_dist - nearest_mean) / (nearest_std + EPS), min=0.0))

        probs_known = F.softmax(logits[:, : self.num_known_classes], dim=1)
        cls_conf, cls_pred = probs_known.max(dim=1)
        agreement = (cls_pred == nearest_known).float()

        known_score = 0.55 * proto_score + 0.35 * cls_conf + 0.10 * agreement
        novel_score = 1.0 - known_score
        novel_mask = (nearest_dist > thresholds) | (known_score < self.min_known_score)

        corrected_probs = probs_known * proto_score.unsqueeze(1)
        corrected_probs = corrected_probs / corrected_probs.sum(dim=1, keepdim=True).clamp_min(EPS)

        return {
            "novel_mask": novel_mask,
            "known_score": known_score,
            "novel_score": novel_score,
            "corrected_probs": corrected_probs,
            "nearest_known": nearest_known,
            "nearest_dist": nearest_dist,
            "threshold": thresholds,
        }

    @torch.no_grad()
    def predict_open_labels(self, features: torch.Tensor, logits: torch.Tensor, unknown_label: Optional[int] = None) -> Dict[str, torch.Tensor]:
        out = self.predict(features, logits)
        pred_known = out["corrected_probs"].argmax(dim=1)
        if unknown_label is None:
            unknown_label = self.num_known_classes
        pred_open = pred_known.clone()
        pred_open[out["novel_mask"]] = int(unknown_label)
        out["pred_known"] = pred_known
        out["pred_open"] = pred_open
        return out


class PseudoLabelGenerator(nn.Module):
    def __init__(self, num_known_classes: int, n_clusters: int = 1, random_state: int = 42):
        super().__init__()
        self.num_known_classes = int(num_known_classes)
        self.n_clusters = int(max(1, n_clusters))
        self.random_state = int(random_state)

    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        features: torch.Tensor,
        mask_novel: torch.Tensor,
        predictions: torch.Tensor,
        num_known_classes: Optional[int] = None,
        known_predictions: Optional[Dict[str, torch.Tensor]] = None,
        cluster_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_known_classes is None:
            num_known_classes = self.num_known_classes
        features = features.detach()
        device = features.device
        mask_novel = mask_novel.bool().view(-1)
        batch_size = features.shape[0]
        pseudo_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        if known_predictions is None:
            probs = F.softmax(predictions[:, :num_known_classes], dim=1)
            pseudo_labels[~mask_novel] = probs[~mask_novel].argmax(dim=1)
        else:
            pseudo_known = known_predictions["corrected_probs"].argmax(dim=1)
            nearest_known = known_predictions["nearest_known"]
            agree = pseudo_known == nearest_known
            pseudo_known = torch.where(agree, pseudo_known, nearest_known)
            pseudo_labels[~mask_novel] = pseudo_known[~mask_novel]

        if mask_novel.any():
            novel_feat = features[mask_novel] if cluster_features is None else cluster_features[mask_novel]
            n_novel = novel_feat.shape[0]
            n_clusters = min(self.n_clusters, n_novel)
            if n_clusters <= 1:
                novel_assign = torch.zeros(n_novel, dtype=torch.long, device=device)
            else:
                feat_np = F.normalize(novel_feat, dim=1).cpu().numpy()
                km = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
                novel_assign = torch.from_numpy(km.fit_predict(feat_np)).to(device=device, dtype=torch.long)
            pseudo_labels[mask_novel] = novel_assign + num_known_classes
        return pseudo_labels


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def convert_vec_to_int(self, label_vec: torch.Tensor) -> torch.Tensor:
        return _as_int_labels(label_vec)

    def forward(self, embeddings: torch.Tensor, output_vec: torch.Tensor, targets: torch.Tensor):
        del embeddings
        targets = self.convert_vec_to_int(targets)
        return self.xent_loss(output_vec, targets)


class SupConLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temp: float = 0.1):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = float(alpha)
        self.temp = float(temp)

    def convert_vec_to_int(self, label_vec: torch.Tensor) -> torch.Tensor:
        return _as_int_labels(label_vec)

    def forward(self, embeddings: torch.Tensor, output_vec: torch.Tensor, targets: torch.Tensor):
        targets = self.convert_vec_to_int(targets)
        if embeddings.ndim == 2:
            features = torch.stack([embeddings, embeddings], dim=1)
        elif embeddings.ndim == 3:
            features = embeddings
        else:
            raise ValueError(f"Unsupported embedding shape for SupConLoss: {embeddings.shape}")
        cl = supervised_contrastive_loss(features, targets, temperature=self.temp)
        ce = self.xent_loss(output_vec, targets)
        return self.alpha * cl + (1.0 - self.alpha) * ce
