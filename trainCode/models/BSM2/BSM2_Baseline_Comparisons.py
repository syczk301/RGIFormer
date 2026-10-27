# -*- coding: utf-8 -*-
"""
BSM2 数据集上的 RGIFormer 与对比模型统一训练脚本

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

from trainCode.models.BSM2.bsm2_temporal_features import build_bsm2_temporal_features, describe_temporal_feature_config, parse_history_steps

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ROOT = SCRIPT_PATH.parents[4] / "data"
TRAIN_DATA_PATH = DATA_ROOT / "BSM2" / "bsm2_train_data.csv"
TEST_DATA_PATH = DATA_ROOT / "BSM2" / "bsm2_test_data.csv"
MODEL_DIR = REPO_ROOT / "model"
RUN_NAME = "RGIFormer_BSM2"
MODEL_PATH = MODEL_DIR / f"{RUN_NAME}.pth"
DEFAULT_BSM2_OUTPUT_ROOT = REPO_ROOT / "results" / "01_model_comparison/neural" / "BSM2_temporal"
DEFAULT_MULTI_OUTPUT_DIR = DEFAULT_BSM2_OUTPUT_ROOT / "RGIFormer_Temporal"
EPS = 1e-6
BASE_MODEL_CHOICES = [
    "rgiformer",
    "rgiformer_graph_effective",
    "mlp",
    "cnn1d",
    "bilstm",
    "resnet1d",
    "transformer",
    "informer",
    "patchtst_group",
    "timesnet",
    "itransformer",
    "mtgnn",
    "gat_informer",
    "dgcformer",
    "edgeconvformer",
]
PAPER_COMPARISON_MODEL_CHOICES = [
    "patchtst_group",
    "itransformer",
    "gat_informer",
    "informer",
    "mtgnn",
    "dgcformer",
    "edgeconvformer",
]
MODEL_GROUPS = {
    "paper_comparison": PAPER_COMPARISON_MODEL_CHOICES,
    # Keep the archived baseline suite stable; the graph-effective model is an
    # explicit experimental opt-in rather than an automatic extra baseline.
    "all": [
        model_name
        for model_name in BASE_MODEL_CHOICES
        if model_name != "rgiformer_graph_effective"
    ],
}
MODEL_CHOICES = [*MODEL_GROUPS.keys(), *BASE_MODEL_CHOICES]
MODEL_DISPLAY_NAMES = {
    "rgiformer": "RGIFormer",
    "rgiformer_graph_effective": "RGIFormer-GraphEffective",
    "mlp": "MLP",
    "cnn1d": "CNN1D",
    "bilstm": "BiLSTM",
    "resnet1d": "ResNet1D",
    "transformer": "Transformer Encoder",
    "informer": "Informer",
    "patchtst_group": "PatchTST-Group-Cls",
    "timesnet": "TimesNet",
    "itransformer": "iTransformer",
    "mtgnn": "MTGNN",
    "gat_informer": "GAT-Informer",
    "dgcformer": "DGCformer-Cls",
    "edgeconvformer": "EdgeConvFormer-Cls",
}
MODEL_OUTPUT_NAMES = {
    "rgiformer": "RGIFormer",
    "rgiformer_graph_effective": "RGIFormerGraphEffective",
    "mlp": "MLP",
    "cnn1d": "CNN1D",
    "bilstm": "BiLSTM",
    "resnet1d": "ResNet1D",
    "transformer": "TransformerEncoder",
    "informer": "Informer",
    "patchtst_group": "PatchTSTGroupCls",
    "timesnet": "TimesNet",
    "itransformer": "iTransformer",
    "mtgnn": "MTGNN",
    "gat_informer": "GATInformer",
    "dgcformer": "DGCformerCls",
    "edgeconvformer": "EdgeConvFormerCls",
}

TRAINING_CONFIG_FIELDS = (
    "max_epochs",
    "patience",
    "batch_size",
    "lr",
    "weight_decay",
    "scheduler_t0",
    "scheduler_tmult",
    "eta_min",
    "focal_gamma",
    "label_smoothing",
    "graph_reg_weight",
    "grad_clip",
)
LEGACY_TRAINING_CONFIG = {
    "max_epochs": 200,
    "patience": 50,
    "batch_size": 128,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "scheduler_t0": 15,
    "scheduler_tmult": 2,
    "eta_min": 1e-6,
    "focal_gamma": 2.5,
    "label_smoothing": 0.05,
    "graph_reg_weight": 1e-3,
    "grad_clip": 1.0,
}
TUNED_TRAINING_CONFIG = {
    "max_epochs": 240,
    "patience": 60,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 8e-5,
    "scheduler_t0": 20,
    "scheduler_tmult": 2,
    "eta_min": 1e-6,
    "focal_gamma": 2.0,
    "label_smoothing": 0.03,
    "graph_reg_weight": 7e-4,
    "grad_clip": 1.0,
}
MODEL_TUNING_PRESETS = {
    "mlp": {
        "model_kwargs": {"hidden_dim": 768, "num_blocks": 5, "dropout": 0.25},
        "lr": 4e-4,
        "weight_decay": 1e-4,
    },
    "cnn1d": {
        "model_kwargs": {"hidden_dim": 160, "dropout": 0.20},
        "lr": 6e-4,
        "weight_decay": 8e-5,
    },
    "bilstm": {
        "model_kwargs": {"hidden_dim": 320, "dropout": 0.25},
        "lr": 4e-4,
        "weight_decay": 8e-5,
    },
    "resnet1d": {
        "model_kwargs": {"channels": 128, "dropout": 0.20},
        "lr": 6e-4,
        "weight_decay": 8e-5,
    },
    "transformer": {
        "model_kwargs": {"hidden_dim": 256, "num_layers": 4, "num_heads": 8, "dropout": 0.18},
        "lr": 4e-4,
        "weight_decay": 1e-4,
    },
    "informer": {
        "model_kwargs": {"hidden_dim": 256, "num_groups": 14, "num_heads": 8, "dropout": 0.20},
        "lr": 4e-4,
        "weight_decay": 1e-4,
    },
    "patchtst_group": {
        "model_kwargs": {"hidden_dim": 256, "num_layers": 4, "num_heads": 8, "dropout": 0.18},
        "lr": 4e-4,
        "weight_decay": 1e-4,
    },
    "timesnet": {
        "model_kwargs": {"hidden_dim": 256, "dropout": 0.20},
        "lr": 5e-4,
        "weight_decay": 8e-5,
    },
    "itransformer": {
        "model_kwargs": {"hidden_dim": 256, "num_layers": 4, "num_heads": 8, "dropout": 0.20},
        "lr": 4e-4,
        "weight_decay": 1e-4,
    },
    "mtgnn": {
        "model_kwargs": {"node_dim": 80, "hidden_dim": 256, "dropout": 0.20},
        "lr": 5e-4,
        "weight_decay": 6e-5,
        "max_epochs": 260,
        "patience": 70,
    },
    "gat_informer": {
        "model_kwargs": {"node_dim": 80, "hidden_dim": 256, "num_heads": 8, "dropout": 0.18},
        "lr": 4e-4,
        "weight_decay": 8e-5,
    },
    "dgcformer": {
        "model_kwargs": {
            "cluster_tokens": 16,
            "node_dim": 64,
            "hidden_dim": 192,
            "num_layers": 2,
            "num_heads": 6,
            "dropout": 0.25,
        },
        "lr": 3e-4,
        "weight_decay": 5e-5,
        "max_epochs": 280,
        "patience": 80,
        "graph_reg_weight": 5e-4,
        "grad_clip": 0.8,
    },
    "edgeconvformer": {
        "model_kwargs": {
            "node_dim": 96,
            "k": 6,
            "hidden_dim": 192,
            "num_layers": 2,
            "num_heads": 6,
            "dropout": 0.25,
            "num_edge_layers": 2,
        },
        "lr": 3e-4,
        "weight_decay": 5e-5,
        "max_epochs": 280,
        "patience": 80,
        "grad_clip": 0.8,
    },
}


class BSM2Dataset(Dataset):
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


REALISTIC_BSM2_FEATURES = (
    "in_Q",
    "in_TEMP",
    "in_TSS",
    "in_SNH",
    "r5_SO",
    "r5_SNH",
    "r5_SNO",
    "DO_sensor",
    "eff_Q",
    "eff_TEMP",
    "eff_TSS",
    "eff_SNH",
    "eff_SNO",
)


def create_class_weights(labels, boost_classes=(), boost_factor=1.0):
    class_counts = np.bincount(labels)
    weights = 1.0 / np.maximum(class_counts, 1)
    weights = weights / weights.sum() * len(weights)
    for cls in boost_classes:
        if cls < len(weights):
            weights[cls] *= boost_factor
    return torch.FloatTensor(weights)


def create_sample_weights(labels, boost_classes=(), boost_factor=1.0):
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


def select_bsm2_feature_names(data, use_realistic_features=False):
    if use_realistic_features:
        return [col for col in REALISTIC_BSM2_FEATURES if col in data.columns]
    return [col for col in data.columns if col not in ("time_day", "label")]


def load_bsm2_data(
    train_path,
    test_path,
    use_realistic_features=False,
    temporal_features="lagdiff",
    history_steps=(1, 2, 4),
    rolling_window=0,
):
    train_path = Path(train_path)
    test_path = Path(test_path)
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据文件未找到: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"测试数据文件未找到: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    feature_names = select_bsm2_feature_names(train_data, use_realistic_features=use_realistic_features)
    if not feature_names:
        raise ValueError("未选择到任何 BSM2 特征列")

    X_train_full, expanded_feature_names = build_bsm2_temporal_features(
        train_data,
        feature_names,
        mode=temporal_features,
        history_steps=history_steps,
        rolling_window=rolling_window,
    )
    y_train_full = train_data["label"].values.astype(np.int64)
    X_test, _ = build_bsm2_temporal_features(
        test_data,
        feature_names,
        mode=temporal_features,
        history_steps=history_steps,
        rolling_window=rolling_window,
    )
    y_test = test_data["label"].values.astype(np.int64)
    num_classes = int(max(y_train_full.max(), y_test.max()) + 1)
    return X_train_full, y_train_full, X_test, y_test, num_classes, expanded_feature_names


def prepare_split(train_features, train_labels, test_features, test_labels, seed, val_ratio=0.2):
    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test_features)
    return X_train, X_val, X_test, y_train, y_val, test_labels


def assert_feature_vector_input(X_train, X_val, X_test, feature_names):
    """Keep every baseline on the same B x F engineered-feature input protocol."""
    feature_dim = len(feature_names)
    arrays = {"train": X_train, "val": X_val, "test": X_test}
    for split_name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(
                f"{split_name} input must be temporal-engineered [batch, feature_dim], got shape={array.shape}."
            )
        if array.shape[1] != feature_dim:
            raise ValueError(
                f"{split_name} feature dimension mismatch: got {array.shape[1]}, expected {feature_dim}."
            )


def get_model_display_name(model_name):
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def get_model_output_name(model_name):
    return f"{MODEL_OUTPUT_NAMES.get(model_name, model_name)}_Temporal"


def resolve_effective_config(args):
    resolved = argparse.Namespace(**vars(args))
    if resolved.tuning_profile == "legacy":
        base_config = LEGACY_TRAINING_CONFIG
        model_preset = {}
        preset_name = "legacy"
    else:
        base_config = TUNED_TRAINING_CONFIG
        model_preset = MODEL_TUNING_PRESETS.get(resolved.model, {})
        preset_name = "tuned"

    for field in TRAINING_CONFIG_FIELDS:
        if getattr(resolved, field) is None:
            setattr(resolved, field, model_preset.get(field, base_config[field]))

    resolved.model_kwargs = dict(model_preset.get("model_kwargs", {}))
    resolved.resolved_tuning_profile = preset_name
    return resolved


def format_model_kwargs(model_kwargs):
    if not model_kwargs:
        return "default constructor parameters"
    return ", ".join(f"{key}={value}" for key, value in sorted(model_kwargs.items()))


def build_model(input_dim, num_classes, device, model_name="rgiformer", model_kwargs=None):
    model_kwargs = dict(model_kwargs or {})
    if model_name == "rgiformer":
        kwargs = {
            "input_dim": input_dim,
            "hidden_dim": 512,
            "node_dim": 64,
            "num_groups": 14,
            "sequence_layers": 2,
            "sequence_heads": 8,
            "dropout": 0.2,
            "num_classes": num_classes,
            "lambda_recovery": 0.05,
            "lambda_graph": 1e-3,
        }
        kwargs.update(model_kwargs)
        model = RGIFormer(**kwargs)
    elif model_name == "rgiformer_graph_effective":
        kwargs = {
            "input_dim": input_dim,
            "hidden_dim": 512,
            "node_dim": 64,
            "num_groups": 14,
            "sequence_layers": 2,
            "sequence_heads": 8,
            "dropout": 0.2,
            "num_classes": num_classes,
            "lambda_recovery": 0.05,
            "lambda_graph": 1e-2,
            "lambda_graph_reconstruction": 0.03,
        }
        kwargs.update(model_kwargs)
        model = GraphEffectiveRGIFormer(**kwargs)
    elif model_name == "mlp":
        model = MLPBaseline(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "cnn1d":
        model = CNN1DBaseline(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "bilstm":
        model = BiLSTMBaseline(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "resnet1d":
        model = ResNet1DBaseline(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "transformer":
        model = TransformerEncoderBaseline(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "informer":
        model = InformerClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "patchtst_group":
        model = PatchTSTGroupClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "timesnet":
        model = TimesNetClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "itransformer":
        model = ITransformerClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "mtgnn":
        model = MTGNNClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "gat_informer":
        model = GATInformerClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "dgcformer":
        model = DGCformerClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
    elif model_name == "edgeconvformer":
        model = EdgeConvFormerClassifier(input_dim=input_dim, num_classes=num_classes, **model_kwargs)
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
        num_edge_layers=2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.node_embedding = nn.Linear(1, node_dim)
        self.feature_embedding = nn.Parameter(torch.randn(1, input_dim, node_dim) * 0.02)
        self.edge_layers = nn.ModuleList(
            [DynamicEdgeConv(node_dim=node_dim, k=k, dropout=dropout) for _ in range(num_edge_layers)]
        )
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, input_dim + 1, hidden_dim))
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
        nodes = self.node_embedding(x.unsqueeze(-1)) + self.feature_embedding[:, : x.size(1), :]
        for edge_layer in self.edge_layers:
            nodes = edge_layer(nodes)
        tokens = self.node_proj(nodes)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]
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


class GraphEffectiveDualRelationGraphEncoder(DualRelationGraphEncoder):
    """Experimental graph encoder whose output depends explicitly on topology."""

    def __init__(
        self,
        input_dim,
        node_dim=64,
        hidden_dim=512,
        num_groups=14,
        dropout=0.2,
        mask_ratio=0.15,
        graph_mode="full",
        random_graph_seed=0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            dropout=dropout,
            **kwargs,
        )
        valid_graph_modes = {
            "full",
            "global_only",
            "dynamic_only",
            "identity",
            "random",
        }
        if graph_mode not in valid_graph_modes:
            raise ValueError(
                f"Unsupported graph_mode={graph_mode!r}; "
                f"expected one of {sorted(valid_graph_modes)}"
            )
        self.mask_ratio = mask_ratio
        self.graph_mode = graph_mode
        self.node_identity = nn.Parameter(torch.randn(1, input_dim, node_dim) * 0.02)
        self.message_gate_logits = nn.Parameter(torch.tensor([-1.39, -1.39]))
        self.graph_input_gate_logit = nn.Parameter(torch.tensor(-1.73))
        self.node_reconstruction = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 1),
        )
        self.node_to_input = nn.Linear(node_dim, 1, bias=False)

        # softplus(0) is dense and large. Start the experimental global graph
        # with weak off-diagonal edges and a stronger diagonal instead.
        with torch.no_grad():
            self.global_adj_param.normal_(mean=-2.5, std=0.1)
            self.global_adj_param.copy_(
                0.5 * (self.global_adj_param + self.global_adj_param.t())
            )
            self.global_adj_param.fill_diagonal_(0.0)

        random_generator = torch.Generator()
        random_generator.manual_seed(int(random_graph_seed))
        random_scores = torch.rand(
            input_dim,
            input_dim,
            generator=random_generator,
        )
        random_scores = 0.5 * (random_scores + random_scores.t())
        with torch.no_grad():
            fixed_random_adj = self._sparsify_adj(
                random_scores.unsqueeze(0)
            ).squeeze(0)
        self.register_buffer("fixed_random_adj", fixed_random_adj, persistent=False)

    def _select_adjacency(self, relation_nodes):
        batch_size = relation_nodes.size(0)
        if self.graph_mode == "identity":
            adjacency = torch.eye(
                self.input_dim,
                device=relation_nodes.device,
                dtype=relation_nodes.dtype,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            edge_gate = relation_nodes.new_ones(
                batch_size,
                self.input_dim,
                self.input_dim,
            )
            return adjacency, edge_gate
        if self.graph_mode == "random":
            adjacency = self.fixed_random_adj.to(
                device=relation_nodes.device,
                dtype=relation_nodes.dtype,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            edge_gate = relation_nodes.new_zeros(
                batch_size,
                self.input_dim,
                self.input_dim,
            )
            return adjacency, edge_gate

        dynamic_adj = self._dynamic_adj(relation_nodes)
        global_adj = self._global_adj().unsqueeze(0).expand(batch_size, -1, -1)
        if self.graph_mode == "global_only":
            return self._sparsify_adj(global_adj), torch.ones_like(global_adj)
        if self.graph_mode == "dynamic_only":
            return self._sparsify_adj(dynamic_adj), torch.zeros_like(dynamic_adj)

        edge_gate = self._fusion_gate(relation_nodes)
        fused_adj = edge_gate * global_adj + (1.0 - edge_gate) * dynamic_adj
        return self._sparsify_adj(fused_adj), edge_gate

    def _encode_graph(self, x):
        # Identity is used to learn sensor-specific edges, but is kept out of
        # the value/residual path so it cannot become a graph-free classifier
        # shortcut.
        nodes = self.node_embedding(x.unsqueeze(-1))
        relation_nodes = nodes + self.node_identity
        fused_adj, edge_gate = self._select_adjacency(relation_nodes)

        # Subtract the center transform so an identity graph produces almost no
        # cross-node message. This prevents the graph block from degenerating
        # into an ordinary shared per-node MLP.
        neighbor1 = self.conv1(nodes, fused_adj)
        center1 = self.conv1.linear(nodes)
        signal1 = neighbor1 - center1
        normalized1 = self.norm1(signal1) - self.norm1(torch.zeros_like(signal1))
        message1 = self.dropout(F.gelu(normalized1))
        gate1 = torch.sigmoid(self.message_gate_logits[0])
        update1 = gate1 * message1
        out = nodes + update1

        neighbor2 = self.conv2(out, fused_adj)
        center2 = self.conv2.linear(out)
        signal2 = neighbor2 - center2
        normalized2 = self.norm2(signal2) - self.norm2(torch.zeros_like(signal2))
        message2 = self.dropout(F.gelu(normalized2))
        gate2 = torch.sigmoid(self.message_gate_logits[1])
        update2 = gate2 * message2
        topology_signal = update1 + update2
        out = nodes + topology_signal
        return nodes, out, topology_signal, fused_adj, edge_gate

    def _final_graph_regularization(self, fused_adj):
        eye = torch.eye(
            self.input_dim,
            device=fused_adj.device,
            dtype=fused_adj.dtype,
        ).unsqueeze(0)
        off_diagonal_mass = (fused_adj * (1.0 - eye)).mean()
        row_prob = fused_adj / fused_adj.sum(dim=-1, keepdim=True).clamp_min(EPS)
        normalized_entropy = -(
            row_prob * torch.log(row_prob.clamp_min(EPS))
        ).sum(dim=-1).mean() / math.log(self.input_dim)
        return (
            self.graph_regularization()
            + 0.05 * off_diagonal_mass
            + 0.01 * normalized_entropy
        )

    def _sample_reconstruction_mask(self, x):
        mask = torch.rand_like(x) < self.mask_ratio
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask[empty_rows, 0] = True
        return mask

    def forward(self, x):
        nodes, graph_nodes, topology_signal, fused_adj, edge_gate = self._encode_graph(x)
        pooled = topology_signal.mean(dim=1)
        # Remove the projection's learned baseline so an identity graph yields
        # a strictly zero topology feature rather than an MLP bias shortcut.
        branch_feature = self.output_proj(pooled) - self.output_proj(
            torch.zeros_like(pooled)
        )

        graph_delta = torch.tanh(
            self.node_to_input(topology_signal).squeeze(-1)
        )
        graph_delta = torch.sigmoid(self.graph_input_gate_logit) * graph_delta

        if self.training and self.mask_ratio > 0:
            reconstruction_mask = self._sample_reconstruction_mask(x)
            masked_x = x.masked_fill(reconstruction_mask, 0.0)
            _, masked_graph_nodes, _, _, _ = self._encode_graph(masked_x)
            graph_reconstruction = self.node_reconstruction(
                masked_graph_nodes
            ).squeeze(-1)
        else:
            reconstruction_mask = torch.zeros_like(x, dtype=torch.bool)
            graph_reconstruction = self.node_reconstruction(graph_nodes).squeeze(-1)

        return branch_feature, {
            "graph_alpha": edge_gate.mean(dim=(1, 2)),
            "graph_reg": self._final_graph_regularization(fused_adj),
            "graph_group_bias": self._group_graph_bias(fused_adj),
            "graph_input_delta": graph_delta,
            "graph_reconstruction": graph_reconstruction,
            "graph_reconstruction_mask": reconstruction_mask,
            "graph_message_scale": torch.sigmoid(self.message_gate_logits),
            "graph_input_scale": torch.sigmoid(self.graph_input_gate_logit),
        }


class ProbSparseSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads=8,
        dropout=0.2,
        factor=5,
        apply_bias_to_attention=False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.factor = factor
        self.apply_bias_to_attention = apply_bias_to_attention
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

        if self.apply_bias_to_attention:
            # full_scores already contains the graph bias. Gathering from it
            # makes the graph affect both sparse-query selection and the actual
            # attention probabilities.
            top_scores = torch.gather(
                full_scores,
                dim=2,
                index=top_idx.unsqueeze(-1).expand(-1, -1, -1, seq_len),
            )
        else:
            # Preserve the archived RGIFormer/baseline behavior.
            top_scores = torch.matmul(q_top, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        top_attn = torch.softmax(top_scores, dim=-1)
        top_attn = self.dropout(top_attn)
        top_context = torch.matmul(top_attn, v)

        context = v.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        context.scatter_(2, gather_index, top_context)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class SparseSequenceEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.2,
        factor=5,
        apply_bias_to_attention=False,
    ):
        super().__init__()
        self.attention = ProbSparseSelfAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            factor=factor,
            apply_bias_to_attention=apply_bias_to_attention,
        )
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
        apply_bias_to_attention=False,
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
            [
                SparseSequenceEncoderLayer(
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    factor=factor,
                    apply_bias_to_attention=apply_bias_to_attention,
                )
                for _ in range(num_layers)
            ]
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
        if "graph_input_delta" in relation_aux:
            relation_delta = relation_delta + relation_aux["graph_input_delta"]
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

        outputs = {
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
        for key in (
            "graph_reconstruction",
            "graph_reconstruction_mask",
            "graph_message_scale",
            "graph_input_scale",
        ):
            if key in relation_aux:
                outputs[key] = relation_aux[key]
        return outputs


class GraphEffectiveRGIFormer(RGIFormer):
    """RGIFormer variant designed to make learned topology testable."""

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
        lambda_graph_reconstruction=0.03,
        graph_mask_ratio=0.15,
        graph_mode="full",
        random_graph_seed=0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_groups=num_groups,
            sequence_layers=sequence_layers,
            sequence_heads=sequence_heads,
            dropout=dropout,
            num_classes=num_classes,
            **kwargs,
        )
        self.lambda_graph_reconstruction = lambda_graph_reconstruction
        self.relation_branch = GraphEffectiveDualRelationGraphEncoder(
            input_dim=input_dim,
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            dropout=dropout,
            mask_ratio=graph_mask_ratio,
            graph_mode=graph_mode,
            random_graph_seed=random_graph_seed,
        )
        self.sequence_branch = SparseSequenceBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_layers=sequence_layers,
            num_heads=sequence_heads,
            dropout=dropout,
            apply_bias_to_attention=True,
        )
        self._initialize_weights()

    def compute_auxiliary_losses(self, aux_outputs, inputs, targets=None):
        losses = super().compute_auxiliary_losses(
            aux_outputs,
            inputs,
            targets=targets,
        )
        reconstruction_mask = aux_outputs["graph_reconstruction_mask"]
        if reconstruction_mask.any():
            graph_reconstruction = F.smooth_l1_loss(
                aux_outputs["graph_reconstruction"][reconstruction_mask],
                inputs[reconstruction_mask],
            )
        else:
            graph_reconstruction = inputs.new_tensor(0.0)
        losses["graph_reconstruction"] = graph_reconstruction
        return losses


class TaskSupervisedDualRelationGraphEncoder(
    GraphEffectiveDualRelationGraphEncoder
):
    """Stable dual graph with a task-supervised adjacency representation."""

    def __init__(
        self,
        input_dim,
        node_dim=64,
        hidden_dim=512,
        num_groups=14,
        dropout=0.2,
        mask_ratio=0.15,
        graph_mode="full",
        random_graph_seed=0,
        num_classes=13,
        graph_embedding_dim=128,
        dynamic_residual_init=0.08,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            dropout=dropout,
            mask_ratio=mask_ratio,
            graph_mode=graph_mode,
            random_graph_seed=random_graph_seed,
            **kwargs,
        )
        dynamic_residual_init = float(
            min(max(dynamic_residual_init, 1e-4), 1.0 - 1e-4)
        )
        self.dynamic_residual_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    dynamic_residual_init
                    / (1.0 - dynamic_residual_init)
                )
            )
        )
        self.node_readout_score = nn.Sequential(
            nn.Linear(node_dim, max(16, node_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, node_dim // 2), 1),
        )
        self.centrality_readout_scale = nn.Parameter(torch.tensor(0.5))
        descriptor_dim = num_groups * num_groups
        self.graph_descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, graph_embedding_dim),
            nn.LayerNorm(graph_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.LayerNorm(graph_embedding_dim),
            nn.GELU(),
        )
        self.graph_classifier = nn.Linear(graph_embedding_dim, num_classes)

    def _select_adjacency(self, relation_nodes):
        if self.graph_mode in {"identity", "random"}:
            return super()._select_adjacency(relation_nodes)

        batch_size = relation_nodes.size(0)
        dynamic_adj = self._dynamic_adj(relation_nodes)
        global_adj = self._global_adj().unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        if self.graph_mode == "global_only":
            return self._sparsify_adj(global_adj), torch.ones_like(global_adj)
        if self.graph_mode == "dynamic_only":
            return self._sparsify_adj(dynamic_adj), torch.zeros_like(dynamic_adj)

        # The sample-adaptive relation is a bounded residual around the stable
        # global graph. This avoids allowing one snapshot to replace the entire
        # topology with a high-variance adjacency.
        dynamic_scale = torch.sigmoid(self.dynamic_residual_logit)
        fused_adj = global_adj + dynamic_scale * (dynamic_adj - global_adj)
        global_weight = torch.ones_like(global_adj) * (1.0 - dynamic_scale)
        return self._sparsify_adj(fused_adj), global_weight

    def _centrality_features(self, fused_adj):
        if self.graph_mode == "identity":
            node_centrality = fused_adj.new_zeros(
                fused_adj.size(0),
                self.input_dim,
            )
            group_centrality = fused_adj.new_zeros(
                fused_adj.size(0),
                self.num_groups,
            )
            return node_centrality, group_centrality

        node_centrality = 0.5 * (
            fused_adj.sum(dim=-1) + fused_adj.sum(dim=-2)
        )
        node_centrality = node_centrality - node_centrality.mean(
            dim=1,
            keepdim=True,
        )
        node_centrality = node_centrality / node_centrality.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)

        assign = self.group_assign.to(fused_adj.dtype)
        group_counts = assign.sum(dim=0).clamp_min(1.0)
        group_centrality = torch.matmul(node_centrality, assign)
        group_centrality = group_centrality / group_counts.unsqueeze(0)
        group_centrality = group_centrality - group_centrality.mean(
            dim=1,
            keepdim=True,
        )
        group_centrality = group_centrality / group_centrality.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)
        return node_centrality, group_centrality

    def forward(self, x):
        (
            nodes,
            graph_nodes,
            topology_signal,
            fused_adj,
            edge_gate,
        ) = self._encode_graph(x)
        node_centrality, group_centrality = self._centrality_features(
            fused_adj
        )

        readout_score = self.node_readout_score(
            topology_signal
        ).squeeze(-1)
        readout_score = (
            readout_score
            + self.centrality_readout_scale * node_centrality
        )
        readout_weight = torch.softmax(readout_score, dim=1)
        pooled = torch.sum(
            readout_weight.unsqueeze(-1) * topology_signal,
            dim=1,
        )
        branch_feature = self.output_proj(pooled) - self.output_proj(
            torch.zeros_like(pooled)
        )

        graph_delta = torch.tanh(
            self.node_to_input(topology_signal).squeeze(-1)
        )
        graph_delta = (
            torch.sigmoid(self.graph_input_gate_logit) * graph_delta
        )

        graph_group_bias = self._group_graph_bias(fused_adj)
        graph_embedding = self.graph_descriptor_encoder(
            graph_group_bias.flatten(start_dim=1)
        )
        graph_logits = self.graph_classifier(graph_embedding)

        if self.training and self.mask_ratio > 0:
            reconstruction_mask = self._sample_reconstruction_mask(x)
            masked_x = x.masked_fill(reconstruction_mask, 0.0)
            _, masked_graph_nodes, _, _, _ = self._encode_graph(masked_x)
            graph_reconstruction = self.node_reconstruction(
                masked_graph_nodes
            ).squeeze(-1)
        else:
            reconstruction_mask = torch.zeros_like(x, dtype=torch.bool)
            graph_reconstruction = self.node_reconstruction(
                graph_nodes
            ).squeeze(-1)

        return branch_feature, {
            "graph_alpha": edge_gate.mean(dim=(1, 2)),
            "graph_reg": self._final_graph_regularization(fused_adj),
            "graph_group_bias": graph_group_bias,
            "graph_group_centrality": group_centrality,
            "graph_input_delta": graph_delta,
            "graph_reconstruction": graph_reconstruction,
            "graph_reconstruction_mask": reconstruction_mask,
            "graph_message_scale": torch.sigmoid(
                self.message_gate_logits
            ),
            "graph_input_scale": torch.sigmoid(
                self.graph_input_gate_logit
            ),
            "graph_embedding": graph_embedding,
            "graph_logits": graph_logits,
            "dynamic_residual_scale": torch.sigmoid(
                self.dynamic_residual_logit
            ),
        }


class MessageSupervisedDualRelationGraphEncoder(
    TaskSupervisedDualRelationGraphEncoder
):
    """Supervise sample-dependent messages on a semantically supported graph.

    The V2 graph classifier encoded only the adjacency matrix.  A static
    global graph therefore produced the same graph embedding for every
    sample, even though its propagated node values were sample dependent.
    This encoder instead classifies grouped topology messages and optionally
    constrains the global relation to a supplied semantic support.
    """

    def __init__(
        self,
        *args,
        global_prior=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        descriptor_dim = (
            self.num_groups * self.node_dim + self.num_groups
        )
        graph_embedding_dim = self.graph_classifier.in_features
        self.graph_descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, graph_embedding_dim * 2),
            nn.LayerNorm(graph_embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(graph_embedding_dim * 2, graph_embedding_dim),
            nn.LayerNorm(graph_embedding_dim),
            nn.GELU(),
        )

        if global_prior is None:
            global_prior = torch.ones(
                self.input_dim,
                self.input_dim,
            ) - torch.eye(self.input_dim)
        global_prior = torch.as_tensor(
            global_prior,
            dtype=torch.float32,
        )
        if global_prior.shape != (self.input_dim, self.input_dim):
            raise ValueError(
                "global_prior must have shape "
                f"({self.input_dim}, {self.input_dim}), got "
                f"{tuple(global_prior.shape)}"
            )
        global_prior = 0.5 * (
            global_prior + global_prior.t()
        )
        global_prior = global_prior.clamp_min(0.0)
        global_prior.fill_diagonal_(0.0)
        support = global_prior.gt(0.0).to(torch.float32)
        self.register_buffer(
            "global_support_mask",
            support,
            persistent=True,
        )
        self._initialize_global_prior(global_prior)

    def _initialize_global_prior(self, global_prior):
        positive = global_prior[self.global_support_mask.bool()]
        if positive.numel() > 0:
            scale = positive.max().clamp_min(1e-6)
            normalized = global_prior / scale
        else:
            normalized = global_prior
        # softplus(-6) keeps unsupported entries numerically quiet.  Supported
        # edges start between softplus(-1) and softplus(0.5), while preserving
        # the supplied relative edge strengths.
        initialized = -1.0 + 1.5 * normalized
        initialized = torch.where(
            self.global_support_mask.bool(),
            initialized,
            torch.full_like(initialized, -6.0),
        )
        initialized.fill_diagonal_(0.0)
        with torch.no_grad():
            self.global_adj_param.copy_(initialized)

    def _global_adj(self):
        logits = 0.5 * (
            self.global_adj_param + self.global_adj_param.t()
        )
        adjacency = (
            F.softplus(logits)
            * self.global_support_mask.to(logits.dtype)
        )
        adjacency = adjacency + self.self_loop_weight * torch.eye(
            self.input_dim,
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
        return self._normalize_adj(adjacency)

    def graph_regularization(self):
        logits = 0.5 * (
            self.global_adj_param + self.global_adj_param.t()
        )
        edge_weights = (
            F.softplus(logits)
            * self.global_support_mask.to(logits.dtype)
        )
        edge_count = self.global_support_mask.sum().clamp_min(1.0)
        magnitude = edge_weights.abs().sum() / edge_count
        symmetry = torch.mean(
            (
                self.global_adj_param
                - self.global_adj_param.t()
            )
            ** 2
        )
        return self.adj_l1_weight * magnitude + 1e-4 * symmetry

    def _message_embedding(
        self,
        topology_signal,
        group_centrality,
    ):
        assign = self.group_assign.to(topology_signal.dtype)
        group_counts = assign.sum(dim=0).clamp_min(1.0)
        grouped_messages = torch.einsum(
            "ng,bnd->bgd",
            assign,
            topology_signal,
        )
        grouped_messages = (
            grouped_messages / group_counts.view(1, -1, 1)
        )
        descriptor = torch.cat(
            [
                grouped_messages.flatten(start_dim=1),
                group_centrality,
            ],
            dim=1,
        )
        return self.graph_descriptor_encoder(descriptor)

    def forward(self, x):
        (
            nodes,
            graph_nodes,
            topology_signal,
            fused_adj,
            edge_gate,
        ) = self._encode_graph(x)
        node_centrality, group_centrality = (
            self._centrality_features(fused_adj)
        )

        readout_score = self.node_readout_score(
            topology_signal
        ).squeeze(-1)
        readout_score = (
            readout_score
            + self.centrality_readout_scale * node_centrality
        )
        readout_weight = torch.softmax(readout_score, dim=1)
        pooled = torch.sum(
            readout_weight.unsqueeze(-1) * topology_signal,
            dim=1,
        )
        branch_feature = self.output_proj(pooled) - self.output_proj(
            torch.zeros_like(pooled)
        )

        graph_delta = torch.tanh(
            self.node_to_input(topology_signal).squeeze(-1)
        )
        graph_delta = (
            torch.sigmoid(self.graph_input_gate_logit) * graph_delta
        )
        graph_group_bias = self._group_graph_bias(fused_adj)
        graph_embedding = self._message_embedding(
            topology_signal,
            group_centrality,
        )
        graph_logits = self.graph_classifier(graph_embedding)

        if self.training and self.mask_ratio > 0:
            reconstruction_mask = self._sample_reconstruction_mask(x)
            masked_x = x.masked_fill(reconstruction_mask, 0.0)
            _, masked_graph_nodes, _, _, _ = self._encode_graph(
                masked_x
            )
            graph_reconstruction = self.node_reconstruction(
                masked_graph_nodes
            ).squeeze(-1)
        else:
            reconstruction_mask = torch.zeros_like(
                x,
                dtype=torch.bool,
            )
            graph_reconstruction = self.node_reconstruction(
                graph_nodes
            ).squeeze(-1)

        return branch_feature, {
            "graph_alpha": edge_gate.mean(dim=(1, 2)),
            "graph_reg": self._final_graph_regularization(fused_adj),
            "graph_group_bias": graph_group_bias,
            "graph_group_centrality": group_centrality,
            "graph_input_delta": graph_delta,
            "graph_reconstruction": graph_reconstruction,
            "graph_reconstruction_mask": reconstruction_mask,
            "graph_message_scale": torch.sigmoid(
                self.message_gate_logits
            ),
            "graph_input_scale": torch.sigmoid(
                self.graph_input_gate_logit
            ),
            "graph_embedding": graph_embedding,
            "graph_logits": graph_logits,
            "dynamic_residual_scale": torch.sigmoid(
                self.dynamic_residual_logit
            ),
        }


class GraphConditionedSparseSequenceBranch(SparseSequenceBranch):
    """Give graph centrality a direct path into CLS attention and pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.token_proj.out_features
        self.cls_graph_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.graph_pool_gate_logit = nn.Parameter(torch.tensor(-1.39))
        self.graph_pool_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

    def forward(
        self,
        x,
        graph_bias=None,
        graph_centrality=None,
    ):
        group_values = self._group_features(x)
        group_tokens = self.token_proj(group_values)
        if graph_centrality is None:
            graph_centrality = group_tokens.new_zeros(
                group_tokens.size(0),
                self.num_groups,
            )
        graph_centrality = graph_centrality - graph_centrality.mean(
            dim=1,
            keepdim=True,
        )

        graph_weight = torch.softmax(graph_centrality, dim=1)
        uniform_weight = 1.0 / float(self.num_groups)
        graph_pool = torch.sum(
            (graph_weight - uniform_weight).unsqueeze(-1)
            * group_tokens,
            dim=1,
        )

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_token, group_tokens], dim=1)
        tokens = self.dropout(
            tokens + self.pos_embedding[:, :tokens.size(1), :]
        )

        attn_bias = torch.zeros(
            x.size(0),
            tokens.size(1),
            tokens.size(1),
            device=x.device,
            dtype=tokens.dtype,
        )
        if graph_bias is not None:
            attn_bias[:, 1:, 1:] = (
                self.graph_bias_scale * graph_bias
            )
        cls_bias = self.cls_graph_bias_scale * graph_centrality
        attn_bias[:, 0, 1:] = cls_bias
        attn_bias[:, 1:, 0] = cls_bias

        for idx, layer in enumerate(self.layers):
            tokens = layer(tokens, attn_bias=attn_bias)
            if idx < len(self.distillers):
                tokens, attn_bias = self.distillers[idx](
                    tokens,
                    attn_bias=attn_bias,
                )

        cls_feature = self.norm(tokens)[:, 0, :]
        graph_pool = self.graph_pool_proj(graph_pool)
        return cls_feature + torch.sigmoid(
            self.graph_pool_gate_logit
        ) * graph_pool


class GraphUsefulSequentialStageFusion(SequentialStageFusion):
    """Sequential fusion that forwards graph centrality to the CLS token."""

    def forward(
        self,
        raw_x,
        shared_feature,
        recovery_feature,
        relation_branch,
        sequence_branch,
    ):
        recovery_delta = 0.1 * self.recovery_to_input(recovery_feature)
        relation_input = raw_x + recovery_delta
        relation_feature, relation_aux = relation_branch(relation_input)

        relation_delta = 0.1 * self.relation_to_input(relation_feature)
        if "graph_input_delta" in relation_aux:
            relation_delta = (
                relation_delta + relation_aux["graph_input_delta"]
            )
        sequence_input = relation_input + relation_delta
        sequence_feature = sequence_branch(
            sequence_input,
            graph_bias=relation_aux["graph_group_bias"],
            graph_centrality=relation_aux.get(
                "graph_group_centrality"
            ),
        )

        stage_feature = self.stage1(
            torch.cat([shared_feature, recovery_feature], dim=1)
        )
        stage_feature = self.stage_feature_update(
            self.stage2,
            stage_feature,
            relation_feature,
        )
        stage_feature = self.stage_feature_update(
            self.stage3,
            stage_feature,
            sequence_feature,
        )
        return stage_feature, relation_aux, {
            "stage_scales": torch.stack(
                [
                    recovery_delta.abs().mean(dim=1),
                    relation_delta.abs().mean(dim=1),
                    sequence_feature.abs().mean(dim=1),
                ],
                dim=1,
            )
        }


class GraphUsefulRGIFormer(GraphEffectiveRGIFormer):
    """Graph-effective model with task-supervised topology and direct readout."""

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
        lambda_graph_reconstruction=0.02,
        graph_mask_ratio=0.15,
        graph_mode="full",
        random_graph_seed=0,
        lambda_graph_ce=0.08,
        lambda_graph_contrastive=0.02,
        lambda_graph_corruption=0.04,
        graph_corruption_margin=0.25,
        dynamic_residual_init=0.08,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_groups=num_groups,
            sequence_layers=sequence_layers,
            sequence_heads=sequence_heads,
            dropout=dropout,
            num_classes=num_classes,
            lambda_graph_reconstruction=lambda_graph_reconstruction,
            graph_mask_ratio=graph_mask_ratio,
            graph_mode=graph_mode,
            random_graph_seed=random_graph_seed,
            **kwargs,
        )
        self.lambda_graph_ce = lambda_graph_ce
        self.lambda_graph_contrastive = lambda_graph_contrastive
        self.lambda_graph_corruption = lambda_graph_corruption
        self.graph_corruption_margin = graph_corruption_margin
        self.relation_branch = TaskSupervisedDualRelationGraphEncoder(
            input_dim=input_dim,
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            dropout=dropout,
            mask_ratio=graph_mask_ratio,
            graph_mode=graph_mode,
            random_graph_seed=random_graph_seed,
            num_classes=num_classes,
            dynamic_residual_init=dynamic_residual_init,
        )
        self.sequence_branch = GraphConditionedSparseSequenceBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_layers=sequence_layers,
            num_heads=sequence_heads,
            dropout=dropout,
            apply_bias_to_attention=True,
        )
        self.sequential_fusion = GraphUsefulSequentialStageFusion(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            dropout=dropout,
        )
        self.graph_logit_gate_logit = nn.Parameter(torch.tensor(-1.73))
        self._initialize_weights()

    @staticmethod
    def _supervised_graph_contrastive_loss(
        embeddings,
        targets,
        temperature=0.15,
    ):
        if embeddings.size(0) < 2:
            return embeddings.new_tensor(0.0)
        embeddings = F.normalize(embeddings, dim=1)
        similarities = torch.matmul(
            embeddings,
            embeddings.t(),
        ) / temperature
        self_mask = torch.eye(
            embeddings.size(0),
            device=embeddings.device,
            dtype=torch.bool,
        )
        positive_mask = targets.unsqueeze(0).eq(
            targets.unsqueeze(1)
        ) & ~self_mask
        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            return embeddings.new_tensor(0.0)

        similarities = similarities.masked_fill(self_mask, -1e9)
        log_prob = similarities - torch.logsumexp(
            similarities,
            dim=1,
            keepdim=True,
        )
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        mean_positive_log_prob = (
            log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count
        )
        return -mean_positive_log_prob[valid_anchor].mean()

    def compute_auxiliary_losses(
        self,
        aux_outputs,
        inputs,
        targets=None,
    ):
        losses = super().compute_auxiliary_losses(
            aux_outputs,
            inputs,
            targets=targets,
        )
        zero = inputs.new_tensor(0.0)
        graph_supervision_enabled = (
            targets is not None
            and hasattr(self.relation_branch, "graph_classifier")
            and (
                self.lambda_graph_ce > 0
                or self.lambda_graph_contrastive > 0
                or self.lambda_graph_corruption > 0
            )
        )
        if not graph_supervision_enabled:
            graph_ce = zero
            graph_contrastive = zero
            graph_corruption = zero
        else:
            graph_logits = aux_outputs["graph_logits"]
            graph_embedding = aux_outputs["graph_embedding"]
            graph_ce_per_sample = F.cross_entropy(
                graph_logits,
                targets,
                reduction="none",
            )
            graph_ce = graph_ce_per_sample.mean()
            graph_contrastive = (
                self._supervised_graph_contrastive_loss(
                    graph_embedding,
                    targets,
                )
            )

            permutation = torch.randperm(
                graph_embedding.size(0),
                device=graph_embedding.device,
            )
            corrupted_embedding = graph_embedding[permutation]
            corrupted_logits = (
                self.relation_branch.graph_classifier(
                    corrupted_embedding
                )
            )
            corrupted_ce = F.cross_entropy(
                corrupted_logits,
                targets,
                reduction="none",
            )
            valid_corruption = targets.ne(targets[permutation])
            if valid_corruption.any():
                graph_corruption = F.relu(
                    self.graph_corruption_margin
                    + graph_ce_per_sample
                    - corrupted_ce
                )[valid_corruption].mean()
            else:
                graph_corruption = zero

        losses["graph_ce"] = graph_ce
        losses["graph_contrastive"] = graph_contrastive
        losses["graph_corruption"] = graph_corruption
        losses["graph_supervision_total"] = (
            self.lambda_graph_ce * graph_ce
            + self.lambda_graph_contrastive * graph_contrastive
            + self.lambda_graph_corruption * graph_corruption
        )
        return losses

    def forward(self, x, return_aux=False):
        shared_feature = self.shared_encoder(x)
        recovery_feature, recovery_aux = self.recovery_branch(x)
        refined_x, reliability_aux = self.reliability_gate(
            x,
            recovery_aux["x_recon"],
        )
        (
            sequential_feature,
            relation_aux,
            sequence_aux,
        ) = self.sequential_fusion(
            raw_x=refined_x,
            shared_feature=shared_feature,
            recovery_feature=recovery_feature,
            relation_branch=self.relation_branch,
            sequence_branch=self.sequence_branch,
        )
        classifier_logits = self.classifier(sequential_feature)
        prototype_aux = self.prototype_head(sequential_feature)
        graph_logit_scale = torch.sigmoid(
            self.graph_logit_gate_logit
        )
        logits = (
            classifier_logits
            + self.prototype_logit_scale
            * prototype_aux["prototype_logits"]
            + graph_logit_scale * relation_aux["graph_logits"]
        )

        if not return_aux:
            return logits

        outputs = {
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
            "graph_group_centrality": relation_aux[
                "graph_group_centrality"
            ],
            "graph_embedding": relation_aux["graph_embedding"],
            "graph_logits": relation_aux["graph_logits"],
            "graph_logit_scale": graph_logit_scale,
            "dynamic_residual_scale": relation_aux[
                "dynamic_residual_scale"
            ],
            "stage_scales": sequence_aux["stage_scales"],
            "refined_input": refined_x,
            "sensor_reliability": reliability_aux[
                "sensor_reliability"
            ],
            "reliability_mean": reliability_aux[
                "reliability_mean"
            ],
            "reliability_entropy": reliability_aux[
                "reliability_entropy"
            ],
            "reliability_correction": reliability_aux[
                "reliability_correction"
            ],
            "prototype_logits": prototype_aux["prototype_logits"],
            "prototype_distances": prototype_aux[
                "prototype_distances"
            ],
            "nearest_proto": prototype_aux["nearest_proto"],
            "prototype_temperature": prototype_aux[
                "prototype_temperature"
            ],
        }
        for key in (
            "graph_reconstruction",
            "graph_reconstruction_mask",
            "graph_message_scale",
            "graph_input_scale",
        ):
            outputs[key] = relation_aux[key]
        return outputs


def compute_model_loss(model, x, y, criterion, graph_reg_weight=1e-3):
    if hasattr(model, "compute_auxiliary_losses"):
        outputs = model(x, return_aux=True)
        logits = outputs["logits"]
        cls_loss = criterion(logits, y)
        aux_losses = model.compute_auxiliary_losses(outputs, x, targets=y)
        loss = cls_loss
        loss = loss + model.lambda_recovery * aux_losses["recovery_total"]
        loss = loss + model.lambda_graph * aux_losses["graph_reg"]
        if "graph_reconstruction" in aux_losses:
            loss = loss + (
                model.lambda_graph_reconstruction
                * aux_losses["graph_reconstruction"]
            )
        if "graph_supervision_total" in aux_losses:
            loss = loss + aux_losses["graph_supervision_total"]
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
        if "graph_supervision_total" in aux_losses:
            metrics["graph_supervision_loss"] = aux_losses[
                "graph_supervision_total"
            ].item()
        return loss, cls_loss, logits, metrics

    logits = model(x)
    cls_loss = criterion(logits, y)
    graph_loss = x.new_tensor(0.0)
    if hasattr(model, "graph_regularization"):
        graph_loss = model.graph_regularization()
    loss = cls_loss + graph_reg_weight * graph_loss
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


def train_epoch(model, train_loader, optimizer, criterion, device, graph_reg_weight=1e-3, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recovery_loss = 0.0
    total_graph_loss = 0.0
    total_graph_supervision_loss = 0.0
    total_prototype_loss = 0.0
    correct = 0
    total = 0
    stage_scale_sum = torch.zeros(3, device=device)
    reliability_sum = 0.0
    reliability_correction_sum = 0.0
    proto_agree = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss, cls_loss, logits, batch_metrics = compute_model_loss(
            model,
            x,
            y,
            criterion,
            graph_reg_weight=graph_reg_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_recovery_loss += batch_metrics["recovery_loss"]
        total_graph_loss += batch_metrics["graph_loss"]
        total_graph_supervision_loss += batch_metrics.get(
            "graph_supervision_loss",
            0.0,
        )
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
        "graph_supervision_loss": (
            total_graph_supervision_loss / num_batches
        ),
        "prototype_loss": total_prototype_loss / num_batches,
        "stage_scales": (stage_scale_sum / num_batches).detach().cpu().numpy(),
        "reliability_mean": reliability_sum / num_batches,
        "reliability_correction": reliability_correction_sum / num_batches,
        "prototype_agreement": proto_agree / max(total, 1),
    }


def evaluate(model, data_loader, criterion, device, graph_reg_weight=1e-3):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recovery_loss = 0.0
    total_graph_loss = 0.0
    total_graph_supervision_loss = 0.0
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
            loss, cls_loss, logits, batch_metrics = compute_model_loss(
                model,
                x,
                y,
                criterion,
                graph_reg_weight=graph_reg_weight,
            )

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_recovery_loss += batch_metrics["recovery_loss"]
            total_graph_loss += batch_metrics["graph_loss"]
            total_graph_supervision_loss += batch_metrics.get(
                "graph_supervision_loss",
                0.0,
            )
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
        "graph_supervision_loss": (
            total_graph_supervision_loss / num_batches
        ),
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
    parser = argparse.ArgumentParser(description="Train RGIFormer and comparison baselines on the BSM2 dataset")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="paper_comparison",
        help="Model, baseline group, or all candidate baselines to train",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for single mode")
    parser.add_argument("--seeds", default="42,3407,2024,2025,2026", help="Comma-separated seeds for multi mode")
    parser.add_argument("--train-path", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MULTI_OUTPUT_DIR)
    parser.add_argument("--tuning-profile", choices=["tuned", "legacy"], default="tuned")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler-t0", type=int, default=None)
    parser.add_argument("--scheduler-tmult", type=int, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--graph-reg-weight", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--boost-factor", type=float, default=1.0)
    parser.add_argument("--realistic-features", action="store_true")
    parser.add_argument(
        "--temporal-features",
        choices=["none", "lag", "diff", "lagdiff"],
        default="lagdiff",
        help="Expand BSM2 rows with per-label time_day-sorted lag/difference features",
    )
    parser.add_argument("--history-steps", default="1,2,4", help="Comma-separated lag steps for temporal features")
    parser.add_argument("--rolling-window", type=int, default=0, help="Optional rolling mean/std window; 0 disables it")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded model/seed plan without training")
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
    print(f"调参方案: {config.resolved_tuning_profile}")
    print(f"模型构造参数: {format_model_kwargs(config.model_kwargs)}")
    print(
        "训练参数: "
        f"batch_size={config.batch_size}, max_epochs={config.max_epochs}, patience={config.patience}, "
        f"lr={config.lr:g}, weight_decay={config.weight_decay:g}, "
        f"focal_gamma={config.focal_gamma:g}, label_smoothing={config.label_smoothing:g}"
    )
    history_steps = parse_history_steps(config.history_steps)
    temporal_desc = describe_temporal_feature_config(
        config.temporal_features,
        history_steps,
        config.rolling_window,
    )
    print(f"时序特征: {temporal_desc}")

    train_features, train_labels, test_features, test_labels, num_classes, feature_names = load_bsm2_data(
        config.train_path,
        config.test_path,
        use_realistic_features=config.realistic_features,
        temporal_features=config.temporal_features,
        history_steps=history_steps,
        rolling_window=config.rolling_window,
    )
    print(f"训练源数据形状: {train_features.shape}")
    print(f"测试源数据形状: {test_features.shape}")
    print(f"特征数: {len(feature_names)}")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_split(
        train_features,
        train_labels,
        test_features,
        test_labels,
        seed,
        val_ratio=config.val_ratio,
    )
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    assert_feature_vector_input(X_train, X_val, X_test, feature_names)
    print(f"输入协议: temporal-engineered feature vector [batch, {X_train.shape[1]}]，所有 BSM2 模型统一使用")

    class_weights = create_class_weights(y_train, boost_factor=config.boost_factor)
    sample_weights = create_sample_weights(y_train, boost_factor=config.boost_factor)

    train_dataset = BSM2Dataset(X_train, y_train)
    val_dataset = BSM2Dataset(X_val, y_val)
    test_dataset = BSM2Dataset(X_test, y_test)

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

    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        device=device,
        model_name=config.model,
        model_kwargs=config.model_kwargs,
    )

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_tmult,
        eta_min=config.eta_min,
    )
    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    epochs_trained = 0
    train_history = initialize_history()
    val_history = initialize_history()

    for epoch in range(config.max_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            graph_reg_weight=config.graph_reg_weight,
            grad_clip=config.grad_clip,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            graph_reg_weight=config.graph_reg_weight,
        )
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
    test_metrics = evaluate(model, test_loader, criterion, device, graph_reg_weight=config.graph_reg_weight)
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
        "tuning_profile": config.resolved_tuning_profile,
        "model_kwargs": format_model_kwargs(config.model_kwargs),
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "focal_gamma": config.focal_gamma,
        "label_smoothing": config.label_smoothing,
        "graph_reg_weight": config.graph_reg_weight,
        "grad_clip": config.grad_clip,
        "accuracy": test_acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "input_protocol": "temporal_engineered_BxF",
        "temporal_features": config.temporal_features,
        "history_steps": ",".join(str(step) for step in history_steps),
        "rolling_window": config.rolling_window,
        "input_dim": X_train.shape[1],
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
    history_steps = parse_history_steps(config.history_steps)
    lines = [
        f"# {model_display_name} BSM2 Multi-Seed Protocol",
        "",
        f"- Model: `{config.model}` ({model_display_name})",
        f"- Train data path: `{Path(config.train_path)}`",
        f"- Test data path: `{Path(config.test_path)}`",
        f"- Feature set: `{'realistic online sensors' if config.realistic_features else 'all BSM2 variables except time_day/label'}`.",
        "- Input protocol: all BSM2 models use the same temporal-engineered `[batch, feature_dim]` vector.",
        f"- Temporal features: `{describe_temporal_feature_config(config.temporal_features, history_steps, config.rolling_window)}`; history is reconstructed within each label by sorting `time_day`, then rows are restored to CSV order.",
        f"- Dataset split: official BSM2 train/test split; validation is a stratified {config.val_ratio:.0%} split from training per seed.",
        "- Scaler: `StandardScaler` is fitted on the training subset only, then applied to validation/test.",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "- Reproducibility: Python, NumPy, PyTorch, and CUDA seeds are set; cuDNN deterministic mode is enabled.",
        f"- Sampler: `WeightedRandomSampler` with seed-controlled `torch.Generator`; boost factor is {config.boost_factor}.",
        f"- Tuning profile: `{config.resolved_tuning_profile}`.",
        f"- Model constructor overrides: `{format_model_kwargs(config.model_kwargs)}`.",
        f"- Batch size: {config.batch_size}",
        f"- Max epochs: {config.max_epochs}",
        f"- Early stopping patience: {config.patience}",
        f"- Optimizer: AdamW, lr={config.lr:g}, weight_decay={config.weight_decay:g}.",
        f"- Scheduler: CosineAnnealingWarmRestarts, T_0={config.scheduler_t0}, T_mult={config.scheduler_tmult}, eta_min={config.eta_min:g}.",
        f"- Loss: FocalLoss with gamma={config.focal_gamma:g}, label_smoothing={config.label_smoothing:g}, class weights from the training split.",
        f"- Baseline graph regularization weight: {config.graph_reg_weight:g}; gradient clipping max_norm={config.grad_clip:g}.",
        "- Model architecture is selected by `--model`; every baseline keeps the same input protocol, while tuned presets can use model-specific capacity and optimizer settings.",
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


def expand_requested_models(model_name):
    if model_name in MODEL_GROUPS:
        return MODEL_GROUPS[model_name]
    return [model_name]


def resolve_output_dir(args, model_name, default_output_requested, root_output=None):
    if default_output_requested:
        return DEFAULT_BSM2_OUTPUT_ROOT / get_model_output_name(model_name)
    if root_output is not None:
        return Path(root_output) / get_model_output_name(model_name)
    return Path(args.output_dir)


def print_run_plan(args, model_names, default_output_requested):
    seeds = [args.seed] if args.mode == "single" else parse_seeds(args.seeds)
    print("BSM2 baseline run plan")
    print(f"  mode: {args.mode}")
    print(f"  seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"  models ({len(model_names)}): {', '.join(get_model_output_name(name) for name in model_names)}")
    print(
        "  temporal features: "
        f"{describe_temporal_feature_config(args.temporal_features, parse_history_steps(args.history_steps), args.rolling_window)}"
    )
    for model_name in model_names:
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        resolved_args = resolve_effective_config(model_args)
        out_dir = resolve_output_dir(
            args,
            model_name,
            default_output_requested,
            root_output=args.output_dir if len(model_names) > 1 else None,
        )
        print(
            f"  - {model_name} -> {out_dir} | "
            f"profile={resolved_args.resolved_tuning_profile}, "
            f"batch={resolved_args.batch_size}, lr={resolved_args.lr:g}, "
            f"epochs={resolved_args.max_epochs}, patience={resolved_args.patience}, "
            f"model_kwargs=({format_model_kwargs(resolved_args.model_kwargs)})"
        )


def run_model_suite(args, default_output_requested):
    args = resolve_effective_config(args)
    if default_output_requested:
        args.output_dir = DEFAULT_BSM2_OUTPUT_ROOT / get_model_output_name(args.model)

    if args.mode == "single":
        single_output_dir = args.output_dir / f"seed_{args.seed}"
        run_single_experiment(args, seed=args.seed, output_dir=single_output_dir, legacy_outputs=False)
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


def main():
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    default_output_requested = args.output_dir == DEFAULT_MULTI_OUTPUT_DIR
    model_names = expand_requested_models(args.model)

    if args.dry_run:
        print_run_plan(args, model_names, default_output_requested)
        return

    if len(model_names) > 1:
        root_output = args.output_dir
        for model_name in model_names:
            model_args = argparse.Namespace(**vars(args))
            model_args.model = model_name
            model_args.output_dir = resolve_output_dir(
                model_args,
                model_name,
                default_output_requested=default_output_requested,
                root_output=None if default_output_requested else root_output,
            )
            run_model_suite(model_args, default_output_requested=False)
        return

    run_model_suite(args, default_output_requested=default_output_requested)


if __name__ == "__main__":
    main()
