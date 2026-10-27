# -*- coding: utf-8 -*-
"""
Paper 数据集上的 RGIFormer 模型

模块级创新点：
1. 多尺度表征恢复：结构感知掩码、加权特征重建与一致性约束协同驱动的残差表征学习
2. 关系结构建模：边级稀疏融合的全局关联与样本自适应关联双重耦合机制
3. 序列依赖编码：图偏置增强的特征序列化稀疏注意力编码与 token 压缩机制
4. 可靠性引导推理：由恢复误差估计传感器可信度，并以保守残差方式校正后续关系与序列建模
5. 原型证据决策：类别原型距离与判别分类器双证据融合，增强小样本故障类别的可解释判别边界
6. 串行协同融合：表示恢复、关系建模与序列依赖学习的逐阶段增量式增强框架
"""

from pathlib import Path as _Path
import sys as _sys
_REPOSITORY_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


import math
import os
import argparse
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ROOT = SCRIPT_PATH.parents[4] / "data"
DATA_PATH = DATA_ROOT / "Paper" / "prediction_dataset_full_11000.csv"
MODEL_DIR = REPO_ROOT / "model"
RUN_NAME = "RGIFormer_Paper"
MODEL_PATH = MODEL_DIR / f"{RUN_NAME}.pth"
DEFAULT_MULTI_OUTPUT_DIR = REPO_ROOT / "results" / "01_model_comparison/neural" / "Paper" / "RGIFormer"
EPS = 1e-6
MODEL_CHOICES = [
    "rgiformer",
    "resnet1d",
    "transformer",
    "informer",
    "patchtst_group",
    "timesnet",
    "itransformer",
    "mtgnn",
    "gat_informer",
]
MODEL_DISPLAY_NAMES = {
    "rgiformer": "RGIFormer",
    "resnet1d": "ResNet1D",
    "transformer": "Transformer Encoder",
    "informer": "Informer",
    "patchtst_group": "PatchTST-Group-Cls",
    "timesnet": "TimesNet",
    "itransformer": "iTransformer",
    "mtgnn": "MTGNN",
    "gat_informer": "GAT-Informer",
}
MODEL_OUTPUT_NAMES = {
    "rgiformer": "RGIFormer",
    "resnet1d": "ResNet1D",
    "transformer": "TransformerEncoder",
    "informer": "Informer",
    "patchtst_group": "PatchTSTGroupCls",
    "timesnet": "TimesNet",
    "itransformer": "iTransformer",
    "mtgnn": "MTGNN",
    "gat_informer": "GATInformer",
}


class PaperDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss


def create_class_weights(labels, boost_classes=(0, 5), boost_factor=2.0):
    class_counts = np.bincount(labels)
    weights = 1.0 / np.maximum(class_counts, 1)
    weights = weights / weights.sum() * len(weights)
    for cls in boost_classes:
        weights[cls] *= boost_factor
    return torch.FloatTensor(weights)


def create_sample_weights(labels, boost_classes=(0, 5), boost_factor=2.0):
    weights = np.ones(len(labels), dtype=np.float32)
    for cls in boost_classes:
        weights[labels == cls] *= boost_factor
    return weights


def parse_seeds(seed_text):
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("至少需要提供一个 seed")
    return seeds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_choice):
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 cuda，但当前环境不可用")
    return torch.device(device_choice)


def load_paper_data(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件未找到: {data_path}")
    data = pd.read_csv(data_path)
    labels = data["fault_id"].values
    feature_data = data.drop(["fault_id", "fault_name"], axis=1).values
    return feature_data, labels, len(np.unique(labels))


def prepare_split(feature_data, labels, seed):
    X_train, X_temp, y_train, y_temp = train_test_split(
        feature_data, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model_display_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def get_model_output_name(model_name):
    return MODEL_OUTPUT_NAMES.get(model_name, model_name)


def build_model(input_dim, num_classes, device, model_name="rgiformer"):
    if model_name == "rgiformer":
        model = RGIFormer(
            input_dim=input_dim,
            hidden_dim=512,
            node_dim=64,
            num_groups=14,
            sequence_layers=2,
            sequence_heads=8,
            dropout=0.2,
            num_classes=num_classes,
            lambda_recovery=0.05,
            lambda_graph=1e-3,
        )
    elif model_name == "resnet1d":
        model = ResNet1DBaseline(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "transformer":
        model = TransformerEncoderBaseline(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "informer":
        model = InformerClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "patchtst_group":
        model = PatchTSTGroupClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "timesnet":
        model = TimesNetClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "itransformer":
        model = ITransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "mtgnn":
        model = MTGNNClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_name == "gat_informer":
        model = GATInformerClassifier(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")
    return model.to(device)


class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = self.ffn(self.norm(x))
        gate = self.gate(x)
        return x + gate * residual


class GroupFeatureMixin:
    def _init_grouping(self, input_dim, num_groups):
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_size = math.ceil(input_dim / num_groups)
        self.padded_dim = self.num_groups * self.group_size

    def _group_features(self, x):
        if self.padded_dim > x.size(1):
            x = F.pad(x, (0, self.padded_dim - x.size(1)))
        return x.view(x.size(0), self.num_groups, self.group_size)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_blocks=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([GatedResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


class CNN1DBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(self.branches), 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        pooled = [branch(x).mean(dim=-1) for branch in self.branches]
        return self.classifier(torch.cat(pooled, dim=1))


class BiLSTMBaseline(nn.Module, GroupFeatureMixin):
    def __init__(self, input_dim, num_classes, num_groups=14, hidden_dim=256, dropout=0.2):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.token_proj = nn.Sequential(
            nn.Linear(self.group_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.token_proj(self._group_features(x))
        out, _ = self.lstm(tokens)
        weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class GRUBaseline(nn.Module, GroupFeatureMixin):
    def __init__(self, input_dim, num_classes, num_groups=14, hidden_dim=192, dropout=0.3):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.token_proj = nn.Sequential(
            nn.Linear(self.group_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.token_proj(self._group_features(x))
        out, _ = self.gru(tokens)
        weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim, dilation=1, dropout=0.3):
        super().__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class TCNBaseline(nn.Module, GroupFeatureMixin):
    def __init__(self, input_dim, num_classes, num_groups=14, hidden_dim=192, dropout=0.3):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.token_proj = nn.Sequential(
            nn.Linear(self.group_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            TemporalConvBlock(hidden_dim, dilation=1, dropout=dropout),
            TemporalConvBlock(hidden_dim, dilation=2, dropout=dropout),
            TemporalConvBlock(hidden_dim, dilation=4, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.token_proj(self._group_features(x)).transpose(1, 2)
        features = self.blocks(tokens).transpose(1, 2)
        pooled = torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=1)
        return self.classifier(pooled)


class ResNet1DBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.25):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResNet1DBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, channels=96, dropout=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResNet1DBlock(channels, kernel_size=7, dropout=dropout),
            ResNet1DBlock(channels, kernel_size=5, dropout=dropout),
            ResNet1DBlock(channels, kernel_size=3, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        features = self.blocks(self.stem(x.unsqueeze(1)))
        pooled = torch.cat([features.mean(dim=-1), features.max(dim=-1).values], dim=1)
        return self.classifier(pooled)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.25):
        super().__init__()
        branch_channels = out_channels // 4
        self.bottleneck = nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(branch_channels, branch_channels, kernel_size=kernel_size, padding=kernel_size // 2)
                for kernel_size in (9, 19, 39)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(branch_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_channels, branch_channels * 4, kernel_size=1)

    def forward(self, x):
        z = self.bottleneck(x)
        branches = [branch(z) for branch in self.branches]
        branches.append(self.pool_branch(x))
        return self.out(torch.cat(branches, dim=1) + self.shortcut(x))


class InceptionTimeBaseline(nn.Module):
    def __init__(self, input_dim, num_classes, channels=128, dropout=0.25):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionBlock1D(1, channels, dropout=dropout),
            InceptionBlock1D(channels, channels, dropout=dropout),
            InceptionBlock1D(channels, channels, dropout=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        features = self.blocks(x.unsqueeze(1))
        return self.classifier(features.mean(dim=-1))


class TransformerEncoderBaseline(nn.Module, GroupFeatureMixin):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_groups=14,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.token_proj = nn.Linear(self.group_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_groups + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        tokens = self.token_proj(self._group_features(x))
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
        tokens = self.encoder(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


class PatchTSTGroupClassifier(TransformerEncoderBaseline):
    def __init__(self, input_dim, num_classes, num_groups=14, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.2):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            num_groups=num_groups,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )


class InformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_groups=14, num_heads=8, dropout=0.25):
        super().__init__()
        self.encoder = SparseSequenceBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout,
            factor=5,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


class TimesBlockLite(nn.Module):
    def __init__(self, hidden_dim, dropout=0.25):
        super().__init__()
        self.period_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.weight_proj = nn.Linear(hidden_dim, len(self.period_branches))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        weights = torch.softmax(self.weight_proj(x.mean(dim=1)), dim=-1)
        branch_out = torch.stack([branch(x.transpose(1, 2)).transpose(1, 2) for branch in self.period_branches], dim=1)
        out = torch.sum(branch_out * weights[:, :, None, None], dim=1)
        return self.norm(x + out)


class TimesNetClassifier(nn.Module, GroupFeatureMixin):
    def __init__(self, input_dim, num_classes, num_groups=14, hidden_dim=192, dropout=0.25):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.token_proj = nn.Sequential(
            nn.Linear(self.group_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([TimesBlockLite(hidden_dim, dropout=dropout) for _ in range(3)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.token_proj(self._group_features(x))
        for block in self.blocks:
            tokens = block(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=1)
        return self.classifier(pooled)


class ITransformerClassifier(nn.Module, GroupFeatureMixin):
    def __init__(self, input_dim, num_classes, num_groups=14, hidden_dim=192, num_layers=3, num_heads=6, dropout=0.25):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.variable_proj = nn.Linear(self.group_size, hidden_dim)
        self.variable_embedding = nn.Parameter(torch.zeros(1, num_groups, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.variable_proj(self._group_features(x)) + self.variable_embedding
        tokens = self.norm(self.encoder(tokens))
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=1)
        return self.classifier(pooled)


class DGCformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        cluster_tokens=32,
        node_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cluster_tokens = cluster_tokens
        self.node_embedding = nn.Linear(1, node_dim)
        self.assignment_logits = nn.Parameter(torch.randn(input_dim, cluster_tokens) * 0.02)
        self.cluster_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, cluster_tokens + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def graph_regularization(self):
        assignment = torch.softmax(self.assignment_logits, dim=1)
        entropy = -(assignment * torch.log(assignment + EPS)).sum(dim=1).mean()
        balance = assignment.mean(dim=0)
        balance_loss = torch.mean((balance - 1.0 / self.cluster_tokens) ** 2)
        return 1e-4 * entropy + balance_loss

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        assignment = torch.softmax(self.assignment_logits, dim=1)
        cluster_sum = torch.einsum("bnd,nk->bkd", nodes, assignment)
        cluster_den = assignment.sum(dim=0).clamp_min(EPS).view(1, self.cluster_tokens, 1)
        clusters = self.cluster_proj(cluster_sum / cluster_den)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, clusters], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
        tokens = self.encoder(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


class DynamicEdgeConv(nn.Module):
    def __init__(self, node_dim, k=8, dropout=0.2):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x):
        dist = torch.cdist(x, x)
        k = min(self.k + 1, x.size(1))
        knn_idx = dist.topk(k=k, largest=False, dim=-1).indices[:, :, 1:]
        batch_idx = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
        neighbors = x[batch_idx, knn_idx]
        center = x.unsqueeze(2).expand(-1, -1, neighbors.size(2), -1)
        edge_features = torch.cat([center, neighbors - center], dim=-1)
        out = self.edge_mlp(edge_features).max(dim=2).values
        return self.norm(x + out)


class EdgeConvFormerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        node_dim=64,
        k=8,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()
        self.node_embedding = nn.Linear(1, node_dim)
        self.edge_layers = nn.ModuleList([DynamicEdgeConv(node_dim=node_dim, k=k, dropout=dropout) for _ in range(2)])
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        for edge_layer in self.edge_layers:
            nodes = edge_layer(nodes)
        tokens = self.node_proj(nodes)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.encoder(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


class LocalGraphSAGEBlock(nn.Module):
    def __init__(self, node_dim, dropout=0.3):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x):
        neighbors = F.avg_pool1d(x.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
        return self.norm(x + self.update(torch.cat([x, neighbors], dim=-1)))


class StaticGCNBlock(nn.Module):
    def __init__(self, node_dim, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(node_dim, node_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.ffn = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x):
        graph_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + self.ffn(graph_out))


class GroupGraphTransformerBase(nn.Module, GroupFeatureMixin):
    def _init_graph_transformer(
        self,
        input_dim,
        num_classes,
        num_groups=14,
        node_dim=64,
        hidden_dim=192,
        num_layers=2,
        num_heads=6,
        dropout=0.3,
    ):
        self._init_grouping(input_dim, num_groups)
        self.node_embedding = nn.Linear(1, node_dim)
        self.group_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_groups + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _nodes_to_group_tokens(self, nodes):
        if self.padded_dim > nodes.size(1):
            nodes = F.pad(nodes, (0, 0, 0, self.padded_dim - nodes.size(1)))
        nodes = nodes.view(nodes.size(0), self.num_groups, self.group_size, nodes.size(-1))
        return self.group_proj(nodes.mean(dim=2))

    def _classify_tokens(self, tokens):
        cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
        tokens = self.encoder(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


class GraphSAGETransformerClassifier(GroupGraphTransformerBase):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self._init_graph_transformer(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
        self.sage1 = LocalGraphSAGEBlock(node_dim=64, dropout=dropout)
        self.sage2 = LocalGraphSAGEBlock(node_dim=64, dropout=dropout)

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        nodes = self.sage1(nodes)
        nodes = self.sage2(nodes)
        return self._classify_tokens(self._nodes_to_group_tokens(nodes))


class GCNTransformerClassifier(GroupGraphTransformerBase):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self._init_graph_transformer(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
        self.gcn1 = StaticGCNBlock(node_dim=64, kernel_size=5, dropout=dropout)
        self.gcn2 = StaticGCNBlock(node_dim=64, kernel_size=7, dropout=dropout)

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        nodes = self.gcn1(nodes)
        nodes = self.gcn2(nodes)
        return self._classify_tokens(self._nodes_to_group_tokens(nodes))


class MixPropLayer(nn.Module):
    def __init__(self, node_dim, depth=2, dropout=0.25):
        super().__init__()
        self.depth = depth
        self.proj = nn.Sequential(
            nn.Linear(node_dim * (depth + 1), node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _normalize_adj(adj):
        degree = adj.sum(dim=-1).clamp_min(EPS)
        return adj / degree.unsqueeze(-1)

    def forward(self, x, adj):
        adj = self._normalize_adj(adj)
        states = [x]
        current = x
        for _ in range(self.depth):
            current = torch.bmm(adj, current)
            states.append(current)
        return self.proj(torch.cat(states, dim=-1))


class MTGNNClassifier(nn.Module, GroupFeatureMixin):
    def __init__(self, input_dim, num_classes, num_groups=14, node_dim=64, hidden_dim=192, dropout=0.25):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.input_dim = input_dim
        self.node_embedding = nn.Linear(1, node_dim)
        self.source_embedding = nn.Parameter(torch.randn(input_dim, node_dim) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(input_dim, node_dim) * 0.02)
        self.mixprop1 = MixPropLayer(node_dim=node_dim, depth=2, dropout=dropout)
        self.mixprop2 = MixPropLayer(node_dim=node_dim, depth=2, dropout=dropout)
        self.group_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _adaptive_adj(self, batch_size):
        scores = torch.matmul(self.source_embedding, self.target_embedding.t()) / math.sqrt(self.source_embedding.size(-1))
        adj = torch.softmax(F.relu(scores), dim=-1)
        adj = adj + torch.eye(self.input_dim, device=adj.device, dtype=adj.dtype)
        return adj.unsqueeze(0).expand(batch_size, -1, -1)

    def _nodes_to_groups(self, nodes):
        if self.padded_dim > nodes.size(1):
            nodes = F.pad(nodes, (0, 0, 0, self.padded_dim - nodes.size(1)))
        nodes = nodes.view(nodes.size(0), self.num_groups, self.group_size, nodes.size(-1))
        return self.group_proj(nodes.mean(dim=2))

    def graph_regularization(self):
        adj = self._adaptive_adj(1).squeeze(0)
        off_diag = adj - torch.diag(torch.diag(adj))
        return 1e-4 * off_diag.abs().mean()

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        adj = self._adaptive_adj(x.size(0))
        nodes = nodes + self.mixprop1(nodes, adj)
        nodes = nodes + self.mixprop2(nodes, adj.transpose(1, 2))
        group_tokens = self._nodes_to_groups(nodes)
        group_tokens = group_tokens + self.temporal_conv(group_tokens.transpose(1, 2)).transpose(1, 2)
        pooled = torch.cat([group_tokens.mean(dim=1), group_tokens.max(dim=1).values], dim=1)
        return self.classifier(pooled)


class SparseGATLayer(nn.Module):
    def __init__(self, node_dim, top_k=16, dropout=0.2):
        super().__init__()
        self.top_k = top_k
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(x.size(-1))
        top_k = min(self.top_k, x.size(1))
        top_idx = scores.topk(top_k, dim=-1).indices
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(2, top_idx, torch.gather(scores, 2, top_idx))
        attn = torch.softmax(mask, dim=-1)
        attn = self.dropout(attn)
        out = self.out_proj(torch.matmul(attn, v))
        return self.norm(x + out)


class GATInformerClassifier(nn.Module, GroupFeatureMixin):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_groups=14,
        node_dim=64,
        hidden_dim=256,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()
        self._init_grouping(input_dim, num_groups)
        self.node_embedding = nn.Linear(1, node_dim)
        self.gat1 = SparseGATLayer(node_dim=node_dim, top_k=16, dropout=dropout)
        self.gat2 = SparseGATLayer(node_dim=node_dim, top_k=16, dropout=dropout)
        self.graph_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.informer_branch = SparseSequenceBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        nodes = self.gat1(nodes)
        nodes = self.gat2(nodes)
        graph_feature = self.graph_proj(nodes.mean(dim=1))
        sequence_feature = self.informer_branch(x)
        return self.classifier(torch.cat([graph_feature, sequence_feature], dim=1))


class MultiScaleRecoveryBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_groups=14,
        latent_local_dim=256,
        latent_global_dim=256,
        mask_ratio=0.2,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_size = math.ceil(input_dim / num_groups)
        self.padded_dim = self.num_groups * self.group_size
        self.mask_ratio = mask_ratio
        self.latent_local_dim = latent_local_dim
        self.latent_global_dim = latent_global_dim

        self.local_encoder = nn.Sequential(
            nn.Linear(self.group_size, latent_local_dim),
            nn.LayerNorm(latent_local_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            GatedResidualBlock(latent_local_dim, dropout=dropout),
            GatedResidualBlock(latent_local_dim, dropout=dropout),
        )
        self.local_align = nn.Sequential(
            nn.Linear(latent_local_dim, latent_global_dim),
            nn.LayerNorm(latent_global_dim),
            nn.GELU(),
        )
        self.global_input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_blocks = nn.Sequential(
            GatedResidualBlock(hidden_dim, dropout=dropout),
            GatedResidualBlock(hidden_dim, dropout=dropout),
            GatedResidualBlock(hidden_dim, dropout=dropout),
        )
        self.global_out = nn.Sequential(
            nn.Linear(hidden_dim, latent_global_dim),
            nn.LayerNorm(latent_global_dim),
            nn.GELU(),
        )
        self.cross_scale_gate = nn.Sequential(
            nn.Linear(latent_global_dim * 2, latent_global_dim),
            nn.GELU(),
            nn.Linear(latent_global_dim, latent_global_dim),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(latent_global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.feature_decoder = nn.Sequential(
            nn.Linear(latent_global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.group_decoder = nn.Sequential(
            nn.Linear(latent_local_dim, latent_local_dim // 2),
            nn.GELU(),
            nn.Linear(latent_local_dim // 2, 2),
        )
        self.scale_project = nn.Linear(latent_local_dim, latent_global_dim)
        self.mask_temperature = nn.Parameter(torch.tensor(1.0))
        self.mask_balance = nn.Parameter(torch.tensor(0.5))
        self.consistency_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _group_features(self, x):
        if self.padded_dim > x.size(1):
            x = F.pad(x, (0, self.padded_dim - x.size(1)))
        return x.view(x.size(0), self.num_groups, self.group_size)

    @staticmethod
    def _normalize_score(score):
        score = score - score.mean(dim=-1, keepdim=True)
        return score / (score.std(dim=-1, keepdim=True) + 1e-6)

    def _structure_scores(self, x):
        batch_var = x.var(dim=0, unbiased=False, keepdim=True).expand_as(x)
        amplitude = x.abs()
        grouped = self._group_features(amplitude)
        group_energy = grouped.mean(dim=-1, keepdim=True).expand_as(grouped)
        group_energy = group_energy.reshape(x.size(0), self.padded_dim)[:, : self.input_dim]
        blend = torch.sigmoid(self.mask_balance)
        feature_score = blend * self._normalize_score(amplitude) + (1.0 - blend) * self._normalize_score(batch_var)
        feature_score = feature_score + 0.5 * self._normalize_score(group_energy)
        return feature_score

    def _sample_group_mask(self, group_scores):
        mask_groups = max(1, int(round(self.num_groups * self.mask_ratio * 0.6)))
        noisy_score = group_scores + torch.rand_like(group_scores) * torch.sigmoid(self.mask_temperature)
        top_idx = noisy_score.topk(mask_groups, dim=1).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, top_idx, 1.0)
        return group_mask

    def _apply_mask(self, x):
        feature_score = self._structure_scores(x)
        recon_weight = 1.0 + torch.sigmoid(feature_score)
        if not self.training or self.mask_ratio <= 0:
            return x, recon_weight

        grouped_score = self._group_features(feature_score).mean(dim=-1)
        group_mask = self._sample_group_mask(grouped_score)
        group_mask = group_mask.unsqueeze(-1).expand(-1, -1, self.group_size).reshape(x.size(0), self.padded_dim)
        group_mask = group_mask[:, : self.input_dim]

        point_prob = torch.sigmoid(feature_score / (self.mask_temperature.abs() + 1e-6))
        point_threshold = 1.0 - self.mask_ratio * point_prob
        point_keep = (torch.rand_like(x) < point_threshold.clamp(0.05, 0.95)).float()
        struct_keep = 1.0 - group_mask
        mask = struct_keep * point_keep
        return x * mask, recon_weight

    def _encode_impl(self, x, original_x=None, recon_weight=None):
        if original_x is None:
            original_x = x
        if recon_weight is None:
            recon_weight = torch.ones_like(original_x)
        original_groups = self._group_features(original_x)
        masked_groups = self._group_features(x)

        local_tokens = self.local_encoder(masked_groups)
        local_pool = local_tokens.mean(dim=1)
        local_aligned = self.local_align(local_pool)

        global_feature = self.global_input(x)
        global_feature = self.global_blocks(global_feature)
        global_latent = self.global_out(global_feature)

        gate = self.cross_scale_gate(torch.cat([local_aligned, global_latent], dim=1))
        fused_latent = gate * local_aligned + (1.0 - gate) * global_latent
        branch_feature = self.output_proj(fused_latent)

        x_recon = self.feature_decoder(fused_latent)
        group_recon = self.group_decoder(local_tokens)
        group_target = torch.stack(
            [
                original_groups.mean(dim=-1),
                original_groups.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        scale_source = self.scale_project(local_pool)
        return branch_feature, {
            "x_recon": x_recon,
            "group_recon": group_recon,
            "group_target": group_target,
            "scale_source": scale_source,
            "scale_target": global_latent.detach(),
            "recon_weight": recon_weight,
        }

    def forward(self, x):
        masked_x, recon_weight = self._apply_mask(x)
        branch_feature, aux = self._encode_impl(masked_x, original_x=x, recon_weight=recon_weight)

        if self.training and self.mask_ratio > 0:
            masked_x_alt, _ = self._apply_mask(x)
            alt_feature, _ = self._encode_impl(masked_x_alt, original_x=x, recon_weight=recon_weight)
            aux["consistency_anchor"] = self.consistency_proj(branch_feature)
            aux["consistency_target"] = self.consistency_proj(alt_feature).detach()
        else:
            aux["consistency_anchor"] = self.consistency_proj(branch_feature)
            aux["consistency_target"] = aux["consistency_anchor"].detach()

        return branch_feature, aux


class BatchGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.bmm(adj, support)


class DualRelationGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        node_dim=64,
        hidden_dim=512,
        num_groups=14,
        dropout=0.2,
        adj_l1_weight=1e-6,
        adj_diag_weight=1e-3,
        self_loop_weight=0.2,
        adj_init_scale=0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.num_groups = num_groups
        self.group_size = math.ceil(input_dim / num_groups)
        self.adj_l1_weight = adj_l1_weight
        self.adj_diag_weight = adj_diag_weight
        self.self_loop_weight = self_loop_weight

        init = torch.eye(input_dim) + adj_init_scale * torch.randn(input_dim, input_dim)
        self.global_adj_param = nn.Parameter(init)
        self.node_embedding = nn.Linear(1, node_dim)
        self.dynamic_q = nn.Linear(node_dim, node_dim)
        self.dynamic_k = nn.Linear(node_dim, node_dim)
        self.fusion_q = nn.Linear(node_dim, node_dim)
        self.fusion_k = nn.Linear(node_dim, node_dim)
        self.conv1 = BatchGraphConv(node_dim, node_dim)
        self.conv2 = BatchGraphConv(node_dim, node_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.sparsity_ratio = 0.15
        group_assign = torch.zeros(input_dim, num_groups)
        for idx in range(input_dim):
            group_assign[idx, min(idx // self.group_size, num_groups - 1)] = 1.0
        self.register_buffer("group_assign", group_assign, persistent=False)

    def _normalize_adj(self, adj):
        degree = torch.sum(adj, dim=-1)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        if adj.dim() == 2:
            return degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        return degree_inv_sqrt.unsqueeze(-1) * adj * degree_inv_sqrt.unsqueeze(-2)

    def _global_adj(self):
        adj = 0.5 * (self.global_adj_param + self.global_adj_param.t())
        adj = F.softplus(adj)
        adj = adj + self.self_loop_weight * torch.eye(self.input_dim, device=adj.device)
        return self._normalize_adj(adj)

    def graph_regularization(self):
        adj = 0.5 * (self.global_adj_param + self.global_adj_param.t())
        adj = F.softplus(adj)
        off_diag = adj - torch.diag(torch.diag(adj))
        loss = self.adj_l1_weight * torch.mean(torch.abs(off_diag))
        loss = loss + self.adj_diag_weight * torch.mean(torch.diag(adj) ** 2)
        symmetry_penalty = torch.mean((self.global_adj_param - self.global_adj_param.t()) ** 2)
        return loss + 1e-4 * symmetry_penalty

    def _dynamic_adj(self, node_features):
        q = self.dynamic_q(node_features)
        k = self.dynamic_k(node_features)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.node_dim)
        scores = 0.5 * (scores + scores.transpose(1, 2))
        adj = torch.softmax(scores, dim=-1)
        adj = 0.5 * (adj + adj.transpose(1, 2))
        eye = torch.eye(self.input_dim, device=adj.device).unsqueeze(0)
        adj = adj + self.self_loop_weight * eye
        return self._normalize_adj(adj)

    def _fusion_gate(self, node_features):
        q = self.fusion_q(node_features)
        k = self.fusion_k(node_features)
        gate = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.node_dim)
        gate = 0.5 * (gate + gate.transpose(1, 2))
        return torch.sigmoid(gate)

    def _sparsify_adj(self, adj):
        top_k = max(4, int(round(self.input_dim * self.sparsity_ratio)))
        top_idx = adj.topk(top_k, dim=-1).indices
        mask = torch.zeros_like(adj)
        mask.scatter_(2, top_idx, 1.0)
        adj = adj * mask
        adj = 0.5 * (adj + adj.transpose(1, 2))
        eye = torch.eye(self.input_dim, device=adj.device).unsqueeze(0)
        adj = adj + self.self_loop_weight * eye
        return self._normalize_adj(adj)

    def _group_graph_bias(self, adj):
        assign = self.group_assign.to(adj.dtype)
        denom = torch.matmul(assign.t(), assign).clamp_min(1.0)
        grouped = torch.matmul(adj, assign.unsqueeze(0).expand(adj.size(0), -1, -1))
        group_adj = torch.matmul(assign.t().unsqueeze(0).expand(adj.size(0), -1, -1), grouped)
        group_adj = group_adj / denom.unsqueeze(0)
        group_adj = torch.log1p(group_adj)
        return group_adj - group_adj.mean(dim=(-1, -2), keepdim=True)

    def forward(self, x):
        nodes = self.node_embedding(x.unsqueeze(-1))
        dynamic_adj = self._dynamic_adj(nodes)
        global_adj = self._global_adj()
        edge_gate = self._fusion_gate(nodes)
        fused_adj = edge_gate * global_adj.unsqueeze(0) + (1.0 - edge_gate) * dynamic_adj
        fused_adj = self._sparsify_adj(fused_adj)

        out = self.conv1(nodes, fused_adj)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out + nodes

        out2 = self.conv2(out, fused_adj)
        out2 = self.norm2(out2)
        out2 = F.gelu(out2)
        out2 = self.dropout(out2)
        out = out + out2

        pooled = out.mean(dim=1)
        branch_feature = self.output_proj(pooled)
        return branch_feature, {
            "graph_alpha": edge_gate.mean(dim=(1, 2)),
            "graph_reg": self.graph_regularization(),
            "graph_group_bias": self._group_graph_bias(fused_adj),
        }


class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.2, factor=5):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, attn_bias=None):
        batch_size, seq_len, _ = x.size()
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        full_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            full_scores = full_scores + attn_bias.unsqueeze(1)
        sparsity = full_scores.max(dim=-1).values - full_scores.mean(dim=-1)
        top_u = min(seq_len, max(1, int(self.factor * math.log(seq_len + 1))))
        top_idx = sparsity.topk(top_u, dim=-1).indices

        gather_index = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        q_top = torch.gather(q, 2, gather_index)

        top_scores = torch.matmul(q_top, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        top_attn = torch.softmax(top_scores, dim=-1)
        top_attn = self.dropout(top_attn)
        top_context = torch.matmul(top_attn, v)

        context = v.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        context.scatter_(2, gather_index, top_context)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class SparseSequenceEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2, factor=5):
        super().__init__()
        self.attention = ProbSparseSelfAttention(d_model, num_heads=num_heads, dropout=dropout, factor=factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        x = x + self.attention(self.norm1(x), attn_bias=attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class TokenDistill(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x, attn_bias=None):
        cls_token = x[:, :1, :]
        tokens = x[:, 1:, :]
        if tokens.size(1) <= 1:
            return x, attn_bias
        tokens = self.conv(tokens.transpose(1, 2))
        tokens = F.gelu(tokens)
        tokens = tokens.transpose(1, 2)
        keep_tokens = max(1, math.ceil(tokens.size(1) / 2))
        scores = self.score_proj(tokens).squeeze(-1) + tokens.norm(dim=-1)
        if attn_bias is not None:
            scores = scores + attn_bias[:, 1:, 1:].mean(dim=-1)
        top_idx = scores.topk(keep_tokens, dim=1).indices.sort(dim=1).values
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        tokens = torch.gather(tokens, 1, gather_idx)
        new_tokens = torch.cat([cls_token, tokens], dim=1)

        if attn_bias is None:
            return new_tokens, None

        batch_index = torch.arange(attn_bias.size(0), device=attn_bias.device).unsqueeze(-1)
        node_idx = top_idx + 1
        keep_idx = torch.cat(
            [torch.zeros(attn_bias.size(0), 1, dtype=torch.long, device=attn_bias.device), node_idx],
            dim=1,
        )
        reduced_bias = attn_bias[batch_index[:, :, None], keep_idx[:, :, None], keep_idx[:, None, :]]
        return new_tokens, reduced_bias


class SparseSequenceBranch(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_groups=14,
        num_layers=2,
        num_heads=8,
        dropout=0.2,
        factor=5,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = math.ceil(input_dim / num_groups)
        self.padded_dim = self.num_groups * self.group_size

        self.token_proj = nn.Linear(self.group_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_groups + 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [SparseSequenceEncoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout, factor=factor) for _ in range(num_layers)]
        )
        self.distillers = nn.ModuleList([TokenDistill(hidden_dim) for _ in range(max(0, num_layers - 1))])
        self.norm = nn.LayerNorm(hidden_dim)
        self.graph_bias_scale = nn.Parameter(torch.tensor(1.0))

    def _group_features(self, x):
        if self.padded_dim > x.size(1):
            x = F.pad(x, (0, self.padded_dim - x.size(1)))
        return x.view(x.size(0), self.num_groups, self.group_size)

    def forward(self, x, graph_bias=None):
        tokens = self._group_features(x)
        tokens = self.token_proj(tokens)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = self.dropout(tokens + self.pos_embedding[:, :tokens.size(1), :])
        attn_bias = None
        if graph_bias is not None:
            attn_bias = torch.zeros(
                x.size(0),
                tokens.size(1),
                tokens.size(1),
                device=x.device,
                dtype=tokens.dtype,
            )
            attn_bias[:, 1:, 1:] = self.graph_bias_scale * graph_bias

        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, attn_bias=attn_bias)
            if idx < len(self.distillers):
                tokens, attn_bias = self.distillers[idx](tokens, attn_bias=attn_bias)

        tokens = self.norm(tokens)
        return tokens[:, 0, :]


class SequentialStageFusion(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.2):
        super().__init__()
        self.recovery_to_input = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.relation_to_input = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.stage1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stage3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, raw_x, shared_feature, recovery_feature, relation_branch, sequence_branch):
        recovery_delta = 0.1 * self.recovery_to_input(recovery_feature)
        relation_input = raw_x + recovery_delta
        relation_feature, relation_aux = relation_branch(relation_input)

        relation_delta = 0.1 * self.relation_to_input(relation_feature)
        sequence_input = relation_input + relation_delta
        sequence_feature = sequence_branch(sequence_input, graph_bias=relation_aux["graph_group_bias"])

        stage_feature = self.stage1(torch.cat([shared_feature, recovery_feature], dim=1))
        stage_feature = self.stage_feature_update(self.stage2, stage_feature, relation_feature)
        stage_feature = self.stage_feature_update(self.stage3, stage_feature, sequence_feature)

        stage_scales = torch.stack(
            [
                recovery_delta.abs().mean(dim=1),
                relation_delta.abs().mean(dim=1),
                sequence_feature.abs().mean(dim=1),
            ],
            dim=1,
        )
        return stage_feature, relation_aux, {
            "recovery_delta": recovery_delta,
            "relation_delta": relation_delta,
            "relation_input": relation_input,
            "sequence_input": sequence_input,
            "relation_feature": relation_feature,
            "sequence_feature": sequence_feature,
            "stage_scales": stage_scales,
        }

    @staticmethod
    def stage_feature_update(stage_block, prev_feature, new_feature):
        return stage_block(torch.cat([prev_feature, new_feature], dim=1))


class RecoveryReliabilityGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, correction_scale=0.15):
        super().__init__()
        self.correction_scale = correction_scale
        self.score_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, raw_x, recon_x):
        recon_error = torch.abs(raw_x - recon_x)
        gate_input = torch.stack([raw_x, recon_x, recon_error], dim=-1)
        reliability = torch.sigmoid(self.score_net(gate_input)).squeeze(-1)
        correction = (1.0 - reliability) * (recon_x - raw_x)
        refined_x = raw_x + self.correction_scale * correction
        reliability_entropy = -(
            reliability * torch.log(reliability + EPS)
            + (1.0 - reliability) * torch.log(1.0 - reliability + EPS)
        ).mean()
        return refined_x, {
            "sensor_reliability": reliability,
            "reliability_mean": reliability.mean(dim=1),
            "reliability_entropy": reliability_entropy,
            "reliability_correction": correction.abs().mean(dim=1),
        }


class PrototypeEvidenceHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, prototype_dim=256, dropout=0.2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, prototype_dim),
            nn.LayerNorm(prototype_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prototype_dim, prototype_dim),
            nn.LayerNorm(prototype_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, prototype_dim) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, feature):
        embedding = F.normalize(self.embedding(feature), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        distances = torch.cdist(embedding, prototypes, p=2).pow(2)
        scale = F.softplus(self.temperature) + EPS
        prototype_logits = -distances / scale
        nearest_proto = distances.argmin(dim=1)
        return {
            "prototype_embedding": embedding,
            "prototype_logits": prototype_logits,
            "prototype_distances": distances,
            "nearest_proto": nearest_proto,
            "prototype_temperature": scale.detach(),
        }


class RGIFormer(nn.Module):
    def __init__(
        self,
        input_dim=462,
        hidden_dim=512,
        node_dim=64,
        num_groups=14,
        sequence_layers=2,
        sequence_heads=8,
        dropout=0.2,
        num_classes=11,
        lambda_feat=1.0,
        lambda_group=0.5,
        lambda_scale=0.1,
        lambda_consistency=0.2,
        lambda_recovery=0.05,
        lambda_graph=1e-3,
        lambda_proto_ce=0.2,
        lambda_proto_pull=0.05,
        lambda_proto_sep=0.01,
        prototype_logit_scale=0.35,
    ):
        super().__init__()
        self.lambda_feat = lambda_feat
        self.lambda_group = lambda_group
        self.lambda_scale = lambda_scale
        self.lambda_consistency = lambda_consistency
        self.lambda_recovery = lambda_recovery
        self.lambda_graph = lambda_graph
        self.lambda_proto_ce = lambda_proto_ce
        self.lambda_proto_pull = lambda_proto_pull
        self.lambda_proto_sep = lambda_proto_sep
        self.prototype_logit_scale = prototype_logit_scale

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.recovery_branch = MultiScaleRecoveryBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            latent_local_dim=256,
            latent_global_dim=256,
            mask_ratio=0.2,
            dropout=dropout,
        )
        self.relation_branch = DualRelationGraphEncoder(
            input_dim=input_dim,
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.sequence_branch = SparseSequenceBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_layers=sequence_layers,
            num_heads=sequence_heads,
            dropout=dropout,
        )
        self.reliability_gate = RecoveryReliabilityGate(input_dim=input_dim)
        self.sequential_fusion = SequentialStageFusion(hidden_dim=hidden_dim, input_dim=input_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.prototype_head = PrototypeEvidenceHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prototype_dim=max(128, hidden_dim // 2),
            dropout=dropout,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def compute_auxiliary_losses(self, aux_outputs, inputs, targets=None):
        recon_weight = aux_outputs["recon_weight"]
        feat_diff = F.smooth_l1_loss(aux_outputs["x_recon"], inputs, reduction="none")
        recon_feat = (feat_diff * recon_weight).sum(dim=1).mean() / (recon_weight.sum(dim=1).mean() + EPS)
        recon_group = F.mse_loss(aux_outputs["group_recon"], aux_outputs["group_target"])
        scale_loss = F.mse_loss(aux_outputs["scale_source"], aux_outputs["scale_target"])
        consistency_loss = 1.0 - F.cosine_similarity(
            aux_outputs["consistency_anchor"],
            aux_outputs["consistency_target"],
            dim=1,
        ).mean()
        recovery_total = (
            self.lambda_feat * recon_feat
            + self.lambda_group * recon_group
            + self.lambda_scale * scale_loss
            + self.lambda_consistency * consistency_loss
        )

        if targets is None:
            zero = inputs.new_tensor(0.0)
            proto_ce = zero
            proto_pull = zero
        else:
            proto_logits = aux_outputs["prototype_logits"]
            proto_distances = aux_outputs["prototype_distances"]
            proto_ce = F.cross_entropy(proto_logits, targets)
            proto_pull = proto_distances.gather(1, targets.unsqueeze(1)).mean()

        proto_similarity = torch.matmul(
            F.normalize(self.prototype_head.prototypes, dim=1),
            F.normalize(self.prototype_head.prototypes, dim=1).t(),
        )
        eye = torch.eye(proto_similarity.size(0), device=proto_similarity.device, dtype=torch.bool)
        proto_sep = F.relu(proto_similarity.masked_fill(eye, -1.0)).pow(2).mean()
        prototype_total = (
            self.lambda_proto_ce * proto_ce
            + self.lambda_proto_pull * proto_pull
            + self.lambda_proto_sep * proto_sep
        )

        return {
            "recon_feat": recon_feat,
            "recon_group": recon_group,
            "scale": scale_loss,
            "consistency": consistency_loss,
            "recovery_total": recovery_total,
            "graph_reg": aux_outputs["graph_reg"],
            "proto_ce": proto_ce,
            "proto_pull": proto_pull,
            "proto_sep": proto_sep,
            "prototype_total": prototype_total,
        }

    def forward(self, x, return_aux=False):
        shared_feature = self.shared_encoder(x)
        recovery_feature, recovery_aux = self.recovery_branch(x)
        refined_x, reliability_aux = self.reliability_gate(x, recovery_aux["x_recon"])
        sequential_feature, relation_aux, sequence_aux = self.sequential_fusion(
            raw_x=refined_x,
            shared_feature=shared_feature,
            recovery_feature=recovery_feature,
            relation_branch=self.relation_branch,
            sequence_branch=self.sequence_branch,
        )
        classifier_logits = self.classifier(sequential_feature)
        prototype_aux = self.prototype_head(sequential_feature)
        logits = classifier_logits + self.prototype_logit_scale * prototype_aux["prototype_logits"]

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "classifier_logits": classifier_logits,
            "x_recon": recovery_aux["x_recon"],
            "group_recon": recovery_aux["group_recon"],
            "group_target": recovery_aux["group_target"],
            "scale_source": recovery_aux["scale_source"],
            "scale_target": recovery_aux["scale_target"],
            "recon_weight": recovery_aux["recon_weight"],
            "consistency_anchor": recovery_aux["consistency_anchor"],
            "consistency_target": recovery_aux["consistency_target"],
            "graph_reg": relation_aux["graph_reg"],
            "graph_alpha": relation_aux["graph_alpha"],
            "stage_scales": sequence_aux["stage_scales"],
            "refined_input": refined_x,
            "sensor_reliability": reliability_aux["sensor_reliability"],
            "reliability_mean": reliability_aux["reliability_mean"],
            "reliability_entropy": reliability_aux["reliability_entropy"],
            "reliability_correction": reliability_aux["reliability_correction"],
            "prototype_logits": prototype_aux["prototype_logits"],
            "prototype_distances": prototype_aux["prototype_distances"],
            "nearest_proto": prototype_aux["nearest_proto"],
            "prototype_temperature": prototype_aux["prototype_temperature"],
        }


def compute_model_loss(model, x, y, criterion):
    if hasattr(model, "compute_auxiliary_losses"):
        outputs = model(x, return_aux=True)
        logits = outputs["logits"]
        cls_loss = criterion(logits, y)
        aux_losses = model.compute_auxiliary_losses(outputs, x, targets=y)
        loss = cls_loss
        loss = loss + model.lambda_recovery * aux_losses["recovery_total"]
        loss = loss + model.lambda_graph * aux_losses["graph_reg"]
        loss = loss + aux_losses["prototype_total"]
        metrics = {
            "recovery_loss": aux_losses["recovery_total"].item(),
            "graph_loss": aux_losses["graph_reg"].item(),
            "prototype_loss": aux_losses["prototype_total"].item(),
            "stage_scales": outputs["stage_scales"].mean(dim=0),
            "reliability_mean": outputs["reliability_mean"].mean().item(),
            "reliability_correction": outputs["reliability_correction"].mean().item(),
            "nearest_proto": outputs["nearest_proto"],
        }
        return loss, cls_loss, logits, metrics

    logits = model(x)
    cls_loss = criterion(logits, y)
    graph_loss = x.new_tensor(0.0)
    if hasattr(model, "graph_regularization"):
        graph_loss = model.graph_regularization()
    loss = cls_loss + 1e-3 * graph_loss
    metrics = {
        "recovery_loss": 0.0,
        "graph_loss": graph_loss.item(),
        "prototype_loss": 0.0,
        "stage_scales": torch.zeros(3, device=x.device),
        "reliability_mean": 0.0,
        "reliability_correction": 0.0,
        "nearest_proto": None,
    }
    return loss, cls_loss, logits, metrics


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recovery_loss = 0.0
    total_graph_loss = 0.0
    total_prototype_loss = 0.0
    correct = 0
    total = 0
    stage_scale_sum = torch.zeros(3, device=device)
    reliability_sum = 0.0
    reliability_correction_sum = 0.0
    proto_agree = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss, cls_loss, logits, batch_metrics = compute_model_loss(model, x, y, criterion)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_recovery_loss += batch_metrics["recovery_loss"]
        total_graph_loss += batch_metrics["graph_loss"]
        total_prototype_loss += batch_metrics["prototype_loss"]
        stage_scale_sum += batch_metrics["stage_scales"]
        reliability_sum += batch_metrics["reliability_mean"]
        reliability_correction_sum += batch_metrics["reliability_correction"]

        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
        if batch_metrics["nearest_proto"] is not None:
            proto_agree += (batch_metrics["nearest_proto"] == y).sum().item()

    num_batches = max(len(train_loader), 1)
    return {
        "loss": total_loss / num_batches,
        "acc": correct / max(total, 1),
        "cls_loss": total_cls_loss / num_batches,
        "recovery_loss": total_recovery_loss / num_batches,
        "graph_loss": total_graph_loss / num_batches,
        "prototype_loss": total_prototype_loss / num_batches,
        "stage_scales": (stage_scale_sum / num_batches).detach().cpu().numpy(),
        "reliability_mean": reliability_sum / num_batches,
        "reliability_correction": reliability_correction_sum / num_batches,
        "prototype_agreement": proto_agree / max(total, 1),
    }


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recovery_loss = 0.0
    total_graph_loss = 0.0
    total_prototype_loss = 0.0
    correct = 0
    total = 0
    stage_scale_sum = torch.zeros(3, device=device)
    reliability_sum = 0.0
    reliability_correction_sum = 0.0
    proto_agree = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss, cls_loss, logits, batch_metrics = compute_model_loss(model, x, y, criterion)

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_recovery_loss += batch_metrics["recovery_loss"]
            total_graph_loss += batch_metrics["graph_loss"]
            total_prototype_loss += batch_metrics["prototype_loss"]
            stage_scale_sum += batch_metrics["stage_scales"]
            reliability_sum += batch_metrics["reliability_mean"]
            reliability_correction_sum += batch_metrics["reliability_correction"]

            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            if batch_metrics["nearest_proto"] is not None:
                proto_agree += (batch_metrics["nearest_proto"] == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    num_batches = max(len(data_loader), 1)
    return {
        "loss": total_loss / num_batches,
        "acc": correct / max(total, 1),
        "cls_loss": total_cls_loss / num_batches,
        "recovery_loss": total_recovery_loss / num_batches,
        "graph_loss": total_graph_loss / num_batches,
        "prototype_loss": total_prototype_loss / num_batches,
        "stage_scales": (stage_scale_sum / num_batches).detach().cpu().numpy(),
        "reliability_mean": reliability_sum / num_batches,
        "reliability_correction": reliability_correction_sum / num_batches,
        "prototype_agreement": proto_agree / max(total, 1),
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def plot_training_curves(train_history, val_history, best_val_acc, output_path=None, model_display_name="RGIFormer"):
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.plot(train_history["loss"], label="Train Loss", alpha=0.8)
    plt.plot(val_history["loss"], label="Val Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_display_name} Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(train_history["acc"], label="Train Acc", alpha=0.8)
    plt.plot(val_history["acc"], label="Val Acc", alpha=0.8)
    plt.axhline(y=best_val_acc, color="g", linestyle="--", label=f"Best {best_val_acc:.3f}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_display_name} Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    train_stage = np.array(train_history["stage_scales"])
    for idx, name in enumerate(["Recovery to Relation", "Relation to Sequence", "Sequence Magnitude"]):
        plt.plot(train_stage[:, idx], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Scale")
    plt.title("Sequential Stage Gains")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(train_history["reliability_mean"], label="Train Reliability", alpha=0.8)
    plt.plot(val_history["reliability_mean"], label="Val Reliability", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Reliability")
    plt.title("Sensor Reliability")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(train_history["reliability_correction"], label="Train Correction", alpha=0.8)
    plt.plot(val_history["reliability_correction"], label="Val Correction", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Correction")
    plt.title("Reliability-Guided Correction")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    plt.plot(train_history["prototype_agreement"], label="Train Prototype", alpha=0.8)
    plt.plot(val_history["prototype_agreement"], label="Val Prototype", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Agreement")
    plt.title("Prototype Evidence Agreement")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is None:
        output_path = MODEL_DIR / f"training_curves_{RUN_NAME}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, test_acc, output_path=None, model_display_name="RGIFormer"):
    plt.figure(figsize=(12, 10))
    cm_percent = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=[str(i) for i in range(cm.shape[0])],
        yticklabels=[str(i) for i in range(cm.shape[0])],
        cbar_kws={"label": "Percent (%)"},
        vmin=0,
        vmax=100,
        linewidths=0.8,
        linecolor="gray",
    )
    plt.title(f"{model_display_name} Confusion Matrix (Test Acc: {test_acc * 100:.2f}%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    if output_path is None:
        output_path = MODEL_DIR / f"confusion_matrix_{RUN_NAME}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_report(test_metrics, best_val_acc, report_path=None, model_display_name="RGIFormer"):
    preds = test_metrics["preds"]
    labels = test_metrics["labels"]
    cm = confusion_matrix(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    if report_path is None:
        report_path = MODEL_DIR / f"classification_report_{RUN_NAME}.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("=" * 100 + "\n")
        file.write(f"{model_display_name} 性能报告\n")
        file.write("=" * 100 + "\n\n")
        file.write(f"最佳验证准确率: {best_val_acc * 100:.2f}%\n")
        file.write(f"测试准确率: {accuracy_score(labels, preds) * 100:.2f}%\n")
        file.write(f"宏平均精确率: {precision_macro * 100:.2f}%\n")
        file.write(f"宏平均召回率: {recall_macro * 100:.2f}%\n")
        file.write(f"宏平均 F1 值: {f1_macro * 100:.2f}%\n")
        file.write(
            "平均阶段增益: "
            f"恢复到关系={test_metrics['stage_scales'][0]:.4f}, "
            f"关系到序列={test_metrics['stage_scales'][1]:.4f}, "
            f"序列表征幅值={test_metrics['stage_scales'][2]:.4f}\n\n"
        )
        file.write(
            "可靠性与原型证据: "
            f"平均传感器可靠性={test_metrics['reliability_mean']:.4f}, "
            f"平均可靠性校正幅值={test_metrics['reliability_correction']:.4f}, "
            f"原型最近邻一致率={test_metrics['prototype_agreement'] * 100:.2f}%\n\n"
        )
        file.write(classification_report(labels, preds, digits=4, zero_division=0))
        file.write("\n\n各类别详细指标:\n")
        for idx in range(len(precision)):
            class_acc = cm[idx, idx] / max(cm[idx, :].sum(), 1) * 100
            file.write(
                f"类别{idx:2d}: 准确率={class_acc:6.2f}%, 召回率={recall[idx] * 100:6.2f}%, "
                f"精确率={precision[idx] * 100:6.2f}%, F1={f1[idx] * 100:6.2f}%\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train RGIFormer on the Paper dataset")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="rgiformer", help="Model or baseline to train")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single mode")
    parser.add_argument("--seeds", default="42,3407,2024,2025,2026", help="Comma-separated seeds for multi mode")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MULTI_OUTPUT_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def save_history_npz(output_path, train_history, val_history):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        train_loss=np.array(train_history["loss"]),
        val_loss=np.array(val_history["loss"]),
        train_acc=np.array(train_history["acc"]),
        val_acc=np.array(val_history["acc"]),
        train_stage_scales=np.array(train_history["stage_scales"]),
        val_stage_scales=np.array(val_history["stage_scales"]),
        train_reliability_mean=np.array(train_history["reliability_mean"]),
        val_reliability_mean=np.array(val_history["reliability_mean"]),
        train_reliability_correction=np.array(train_history["reliability_correction"]),
        val_reliability_correction=np.array(val_history["reliability_correction"]),
        train_prototype_agreement=np.array(train_history["prototype_agreement"]),
        val_prototype_agreement=np.array(val_history["prototype_agreement"]),
    )


def save_legacy_history(train_history, val_history):
    np.save(MODEL_DIR / f"loss_train_{RUN_NAME}.npy", np.array(train_history["loss"]))
    np.save(MODEL_DIR / f"loss_val_{RUN_NAME}.npy", np.array(val_history["loss"]))
    np.save(MODEL_DIR / f"acc_train_{RUN_NAME}.npy", np.array(train_history["acc"]))
    np.save(MODEL_DIR / f"acc_val_{RUN_NAME}.npy", np.array(val_history["acc"]))
    np.save(MODEL_DIR / f"stage_scales_train_{RUN_NAME}.npy", np.array(train_history["stage_scales"]))
    np.save(MODEL_DIR / f"stage_scales_val_{RUN_NAME}.npy", np.array(val_history["stage_scales"]))
    np.save(MODEL_DIR / f"reliability_train_{RUN_NAME}.npy", np.array(train_history["reliability_mean"]))
    np.save(MODEL_DIR / f"reliability_val_{RUN_NAME}.npy", np.array(val_history["reliability_mean"]))
    np.save(MODEL_DIR / f"prototype_agreement_train_{RUN_NAME}.npy", np.array(train_history["prototype_agreement"]))
    np.save(MODEL_DIR / f"prototype_agreement_val_{RUN_NAME}.npy", np.array(val_history["prototype_agreement"]))


def initialize_history():
    return {
        "loss": [],
        "acc": [],
        "stage_scales": [],
        "reliability_mean": [],
        "reliability_correction": [],
        "prototype_agreement": [],
    }


def run_single_experiment(config, seed, output_dir, legacy_outputs=False):
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_display_name = get_model_display_name(config.model)
    model_path = MODEL_PATH if legacy_outputs else output_dir / "model.pth"
    start_time = time.time()

    device = resolve_device(config.device)
    print(f"使用设备: {device}")
    print("=" * 100)
    print(f"{model_display_name} 模型 | seed={seed}")
    print("=" * 100)

    feature_data_raw, labels, num_classes = load_paper_data(config.data_path)
    print(f"数据形状: {feature_data_raw.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_split(feature_data_raw, labels, seed)
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    class_weights = create_class_weights(y_train, boost_classes=(0, 5), boost_factor=2.0)
    sample_weights = create_sample_weights(y_train, boost_classes=(0, 5), boost_factor=2.0)

    train_dataset = PaperDataset(X_train, y_train)
    val_dataset = PaperDataset(X_val, y_val)
    test_dataset = PaperDataset(X_test, y_test)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 1009)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = build_model(input_dim=X_train.shape[1], num_classes=num_classes, device=device, model_name=config.model)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.5, label_smoothing=0.05)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    epochs_trained = 0
    train_history = initialize_history()
    val_history = initialize_history()

    for epoch in range(config.max_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epochs_trained = epoch + 1

        for key in train_history:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])

        scheduler.step()

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(
                f"第 {epoch + 1:3d} 轮: [最佳] "
                f"训练准确率={train_metrics['acc']:.4f}, 验证准确率={val_metrics['acc']:.4f}, "
                f"验证损失={val_metrics['loss']:.4f}, "
                f"原型一致率={val_metrics['prototype_agreement']:.4f}, "
                f"阶段增益={val_metrics['stage_scales']}"
            )
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(
                    f"第 {epoch + 1:3d} 轮: 训练准确率={train_metrics['acc']:.4f}, "
                    f"验证准确率={val_metrics['acc']:.4f}, 验证损失={val_metrics['loss']:.4f}, "
                    f"原型一致率={val_metrics['prototype_agreement']:.4f}"
                )

        if patience_counter >= config.patience:
            print(f"早停触发: {config.patience} 个 epoch 无提升")
            break

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    test_acc = accuracy_score(test_metrics["labels"], test_metrics["preds"])
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])

    if legacy_outputs:
        save_legacy_history(train_history, val_history)
        plot_training_curves(train_history, val_history, best_val_acc, model_display_name=model_display_name)
        plot_confusion_matrix(cm, test_acc, model_display_name=model_display_name)
        save_report(test_metrics, best_val_acc, model_display_name=model_display_name)
    else:
        save_history_npz(output_dir / "history.npz", train_history, val_history)
        np.savez(output_dir / "predictions.npz", labels=test_metrics["labels"], preds=test_metrics["preds"])
        plot_training_curves(
            train_history,
            val_history,
            best_val_acc,
            output_dir / "training_curves.png",
            model_display_name=model_display_name,
        )
        plot_confusion_matrix(cm, test_acc, output_dir / "confusion_matrix.png", model_display_name=model_display_name)
        save_report(test_metrics, best_val_acc, output_dir / "classification_report.txt", model_display_name=model_display_name)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        test_metrics["labels"], test_metrics["preds"], average="macro", zero_division=0
    )
    elapsed_seconds = time.time() - start_time

    row = {
        "model": config.model,
        "model_display_name": model_display_name,
        "seed": seed,
        "accuracy": test_acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "stage_recovery_to_relation": test_metrics["stage_scales"][0],
        "stage_relation_to_sequence": test_metrics["stage_scales"][1],
        "stage_sequence_magnitude": test_metrics["stage_scales"][2],
        "reliability_mean": test_metrics["reliability_mean"],
        "reliability_correction": test_metrics["reliability_correction"],
        "prototype_agreement": test_metrics["prototype_agreement"],
        "elapsed_seconds": elapsed_seconds,
    }

    print("=" * 100)
    print("最终测试集性能")
    print("=" * 100)
    print(f"最佳验证准确率: {best_val_acc * 100:.2f}%")
    print(f"测试准确率: {test_acc * 100:.2f}%")
    print(f"宏平均精确率: {precision_macro * 100:.2f}%")
    print(f"宏平均召回率: {recall_macro * 100:.2f}%")
    print(f"宏平均 F1 值: {f1_macro * 100:.2f}%")
    print(
        "测试平均阶段增益: "
        f"恢复到关系={test_metrics['stage_scales'][0]:.4f}, "
        f"关系到序列={test_metrics['stage_scales'][1]:.4f}, "
        f"序列表征幅值={test_metrics['stage_scales'][2]:.4f}"
    )
    print(
        "测试可靠性与原型证据: "
        f"平均可靠性={test_metrics['reliability_mean']:.4f}, "
        f"校正幅值={test_metrics['reliability_correction']:.4f}, "
        f"原型最近邻一致率={test_metrics['prototype_agreement'] * 100:.2f}%"
    )
    print(f"模型保存至: {model_path}")
    return row


def summarize_multi_seed(run_rows, output_dir):
    output_dir = Path(output_dir)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(output_dir / "runs.csv", index=False, encoding="utf-8-sig")

    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "best_val_acc",
        "total_params",
        "trainable_params",
        "stage_recovery_to_relation",
        "stage_relation_to_sequence",
        "stage_sequence_magnitude",
        "reliability_mean",
        "reliability_correction",
        "prototype_agreement",
        "elapsed_seconds",
    ]
    n = len(runs_df)
    t_value = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    summary_rows = []
    for metric in metric_columns:
        values = runs_df[metric].astype(float).to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        ci_half_width = float(t_value * std / math.sqrt(n)) if n > 1 else 0.0
        is_percent_metric = metric not in {"elapsed_seconds", "total_params", "trainable_params"}
        summary_rows.append(
            {
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci_half_width,
                "ci95_high": mean + ci_half_width,
                "ci95_half_width": ci_half_width,
                "mean_percent": mean * 100 if is_percent_metric else np.nan,
                "std_percent": std * 100 if is_percent_metric else np.nan,
                "ci95_half_width_percent": ci_half_width * 100 if is_percent_metric else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return runs_df, summary_df


def write_protocol(config, seeds, output_dir, summary_df):
    output_dir = Path(output_dir)
    model_display_name = get_model_display_name(config.model)
    lines = [
        f"# {model_display_name} Paper Multi-Seed Protocol",
        "",
        f"- Model: `{config.model}` ({model_display_name})",
        f"- Data path: `{Path(config.data_path)}`",
        "- Dataset split: stratified 70/15/15 train/validation/test split per seed.",
        "- Scaler: `StandardScaler` is fitted on the training split only, then applied to validation/test.",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "- Reproducibility: Python, NumPy, PyTorch, and CUDA seeds are set; cuDNN deterministic mode is enabled.",
        "- Sampler: `WeightedRandomSampler` with seed-controlled `torch.Generator`; classes 0 and 5 use boost factor 2.0.",
        f"- Batch size: {config.batch_size}",
        f"- Max epochs: {config.max_epochs}",
        f"- Early stopping patience: {config.patience}",
        "- Optimizer: AdamW, lr=8e-4, weight_decay=1e-4.",
        "- Scheduler: CosineAnnealingWarmRestarts, T_0=15, T_mult=2, eta_min=1e-6.",
        "- Loss: FocalLoss with gamma=2.5, label_smoothing=0.05, class weights from the training split.",
        "- Model architecture is selected by `--model`; all baselines reuse the same training protocol.",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Mean | Std | 95% CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        metric = row["metric"]
        if metric == "elapsed_seconds":
            lines.append(
                f"| {metric} | {row['mean']:.2f} | {row['std']:.2f} | "
                f"[{row['ci95_low']:.2f}, {row['ci95_high']:.2f}] |"
            )
        elif metric in {"total_params", "trainable_params"}:
            lines.append(
                f"| {metric} | {row['mean']:.0f} | {row['std']:.0f} | "
                f"[{row['ci95_low']:.0f}, {row['ci95_high']:.0f}] |"
            )
        else:
            lines.append(
                f"| {metric} | {row['mean'] * 100:.2f}% | {row['std'] * 100:.2f}% | "
                f"[{row['ci95_low'] * 100:.2f}%, {row['ci95_high'] * 100:.2f}%] |"
            )
    lines.append("")
    (output_dir / "protocol.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_dir == DEFAULT_MULTI_OUTPUT_DIR:
        args.output_dir = REPO_ROOT / "results" / "01_model_comparison/neural" / "Paper" / get_model_output_name(args.model)

    if args.mode == "single":
        single_output_dir = MODEL_DIR if args.model == "rgiformer" else args.output_dir / f"seed_{args.seed}"
        run_single_experiment(args, seed=args.seed, output_dir=single_output_dir, legacy_outputs=args.model == "rgiformer")
        return

    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = []
    for seed in seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        row = run_single_experiment(args, seed=seed, output_dir=seed_dir, legacy_outputs=False)
        run_rows.append(row)
        pd.DataFrame(run_rows).to_csv(args.output_dir / "runs.csv", index=False, encoding="utf-8-sig")

    _, summary_df = summarize_multi_seed(run_rows, args.output_dir)
    write_protocol(args, seeds, args.output_dir, summary_df)
    print(f"多 seed 实验结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
