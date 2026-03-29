from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetOpenWorld1D(nn.Module):
    def __init__(
        self,
        layers,
        input_channels: int = 12,
        num_classes: int = 5,
        known_classes: Optional[int] = None,
        feature_dim: int = 256,
        instance_dim: int = 128,
        class_dim: int = 128,
        dropout: float = 0.1,
        use_lstm: bool = False,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.known_classes = int(known_classes if known_classes is not None else num_classes)
        self.feature_dim = int(feature_dim)
        self.instance_dim = int(instance_dim)
        self.class_dim = int(class_dim)
        self.use_lstm = bool(use_lstm)

        self.stem = nn.Sequential(
            nn.Conv1d(self.input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.inplanes = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(128, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(512, layers[3], stride=2, dropout=dropout)

        if self.use_lstm:
            self.temporal_lstm = nn.LSTM(
                input_size=512,
                hidden_size=256,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            pooled_dim = 512 * 2
        else:
            self.temporal_lstm = None
            pooled_dim = 512 * 2

        self.feature_neck = nn.Sequential(
            nn.Linear(pooled_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.instance_head = MLPHead(self.feature_dim, self.feature_dim, self.instance_dim, dropout=dropout)
        self.class_head = MLPHead(self.feature_dim, self.feature_dim, self.class_dim, dropout=dropout)
        self.classifier = nn.Linear(self.class_dim, self.num_classes)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = [BasicBlock1d(self.inplanes, planes, stride=stride, dropout=dropout)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(self.inplanes, planes, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward_backbone_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.forward_backbone_map(x)
        if self.temporal_lstm is not None:
            seq = fmap.transpose(1, 2)
            seq, _ = self.temporal_lstm(seq)
            pooled = torch.cat([seq.mean(dim=1), seq.max(dim=1).values], dim=1)
        else:
            avg = self.avgpool(fmap).flatten(1)
            mx = self.maxpool(fmap).flatten(1)
            pooled = torch.cat([avg, mx], dim=1)
        feat = self.feature_neck(pooled)
        return feat

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(x)
        ins_raw = self.instance_head(feat)
        cls_raw = self.class_head(feat)
        instance_emb = F.normalize(ins_raw, dim=1)
        class_emb = F.normalize(cls_raw, dim=1)
        logits = self.classifier(cls_raw)
        return {
            "feat": feat,
            "instance_emb": instance_emb,
            "class_emb": class_emb,
            "logits": logits,
        }

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        outputs = self.forward_dict(x)
        if return_dict:
            return outputs
        return outputs["instance_emb"], outputs["class_emb"], outputs["logits"]


class ResNet1d(ResNetOpenWorld1D):
    def __init__(self, block, layers, **kwargs):
        del block
        super().__init__(layers=layers, **kwargs)


class ResNet1d_LSTM(ResNetOpenWorld1D):
    def __init__(self, block, layers, **kwargs):
        del block
        kwargs = dict(kwargs)
        kwargs["use_lstm"] = True
        super().__init__(layers=layers, **kwargs)


class ResNet1d_MHL(ResNetOpenWorld1D):
    def __init__(self, block, layers, **kwargs):
        del block
        super().__init__(layers=layers, **kwargs)



def resnet18(**kwargs):
    return ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)



def resnet34(**kwargs):
    return ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)



def resnet34_LSTM(**kwargs):
    return ResNet1d_LSTM(BasicBlock1d, [3, 4, 6, 3], **kwargs)



def resnet34_MHL(**kwargs):
    return ResNet1d_MHL(BasicBlock1d, [3, 4, 6, 3], **kwargs)



def build_model(model_name: str, **kwargs):
    name = str(model_name).lower()
    if name in {"resnet18", "resnet_18"}:
        return resnet18(**kwargs)
    if name in {"resnet34", "resnet_34", "resnet34_mhl", "resnet34-mhl"}:
        return resnet34(**kwargs)
    if name in {"resnet34_lstm", "resnet34lstm", "resnet_34_lstm"}:
        return resnet34_LSTM(**kwargs)
    raise ValueError(f"Unsupported model name: {model_name}")
