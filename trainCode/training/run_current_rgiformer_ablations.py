# -*- coding: utf-8 -*-
"""Matched five-seed ablations for the current Paper RGIFormer implementation.

This runner deliberately imports the model that generates the manuscript's
current Paper results and changes only one active mechanism at a time.  It does
not reuse the archived causal/uncertainty ablation model family.

Outputs are resumable and self-auditing.  Each completed run stores its split
indices, best checkpoint, predictions, history, and metrics before the next run
starts.
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPOSITORY_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainCode.models.Paper import RAE_GCN_Informer_Paper_ModularInnov as base  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "02_ablation/original" / "Paper"
DEFAULT_SEEDS = (42, 3407, 2024, 2025, 2026)
EPS = 1e-12


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    description: str


VARIANTS = (
    VariantSpec(
        "full",
        "Full model",
        "Unmodified current RGIFormer implementation.",
    ),
    VariantSpec(
        "without_amsr",
        "w/o AMSR",
        "Remove the recovery representation, recovery-to-input injection, "
        "reconstruction-guided correction, and recovery auxiliary loss.",
    ),
    VariantSpec(
        "without_recovery_correction",
        "w/o recovery correction",
        "Retain AMSR and its losses, but bypass the reconstruction-guided "
        "input correction before relation modeling.",
    ),
    VariantSpec(
        "global_relation_only",
        "Global relation only",
        "Retain the symmetric global relation and graph convolutions, but "
        "remove the sample-adaptive relation and edge-wise fusion gate.",
    ),
    VariantSpec(
        "without_graph_guidance",
        "w/o graph guidance",
        "Retain DRGC and the sparse grouped encoder, but remove the DRGC "
        "group bias from sparse query/token selection.",
    ),
    VariantSpec(
        "without_prototype_decision",
        "w/o prototype decision",
        "Use classifier logits only and remove prototype logits and prototype "
        "auxiliary losses from training.",
    ),
)
VARIANT_BY_NAME = {spec.name: spec for spec in VARIANTS}


def parse_csv_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_variants(text: str) -> list[VariantSpec]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(VARIANTS)
    unknown = sorted(set(names) - set(VARIANT_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return [VARIANT_BY_NAME[name] for name in names]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class CurrentRGIFormerAblation(base.RGIFormer):
    """Current RGIFormer with one explicitly bounded forward-path ablation."""

    def __init__(self, variant: str, **kwargs):
        if variant not in VARIANT_BY_NAME:
            raise ValueError(f"Unsupported variant: {variant}")
        super().__init__(**kwargs)
        self.variant = variant
        self.hidden_dim = int(kwargs.get("hidden_dim", 512))
        self.num_groups = int(kwargs.get("num_groups", 14))
        self.num_classes = int(kwargs.get("num_classes", 11))

        if variant == "without_amsr":
            freeze_module(self.recovery_branch)
            freeze_module(self.reliability_gate)
            freeze_module(self.sequential_fusion.recovery_to_input)
        elif variant == "without_recovery_correction":
            freeze_module(self.reliability_gate)
        elif variant == "global_relation_only":
            freeze_module(self.relation_branch.dynamic_q)
            freeze_module(self.relation_branch.dynamic_k)
            freeze_module(self.relation_branch.fusion_q)
            freeze_module(self.relation_branch.fusion_k)
        elif variant == "without_graph_guidance":
            self.sequence_branch.graph_bias_scale.requires_grad_(False)
        elif variant == "without_prototype_decision":
            freeze_module(self.prototype_head)

    def _dummy_recovery(self, x: torch.Tensor):
        grouped = self.recovery_branch._group_features(x)
        group_target = torch.stack(
            [
                grouped.mean(dim=-1),
                grouped.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        recovery_feature = x.new_zeros(x.size(0), self.hidden_dim)
        anchor = x.new_ones(x.size(0), 1)
        return recovery_feature, {
            "x_recon": x,
            "group_recon": group_target,
            "group_target": group_target,
            "scale_source": anchor,
            "scale_target": anchor,
            "recon_weight": torch.ones_like(x),
            "consistency_anchor": anchor,
            "consistency_target": anchor,
        }

    @staticmethod
    def _identity_reliability(x: torch.Tensor):
        ones = torch.ones_like(x)
        zeros = x.new_zeros(x.size(0))
        return x, {
            "sensor_reliability": ones,
            "reliability_mean": torch.ones_like(zeros),
            "reliability_entropy": x.new_tensor(0.0),
            "reliability_correction": zeros,
        }

    def _relation_forward(self, x: torch.Tensor):
        if self.variant != "global_relation_only":
            return self.relation_branch(x)

        branch = self.relation_branch
        nodes = branch.node_embedding(x.unsqueeze(-1))
        global_adj = branch._global_adj().unsqueeze(0).expand(x.size(0), -1, -1)
        fused_adj = branch._sparsify_adj(global_adj)

        out = branch.conv1(nodes, fused_adj)
        out = F.gelu(branch.norm1(out))
        out = branch.dropout(out) + nodes
        out2 = branch.conv2(out, fused_adj)
        out2 = F.gelu(branch.norm2(out2))
        out2 = branch.dropout(out2)
        out = out + out2

        pooled = out.mean(dim=1)
        relation_feature = branch.output_proj(pooled)
        return relation_feature, {
            "graph_alpha": x.new_ones(x.size(0)),
            "graph_reg": branch.graph_regularization(),
            "graph_group_bias": branch._group_graph_bias(fused_adj),
        }

    def _sequential_forward(
        self,
        raw_x: torch.Tensor,
        shared_feature: torch.Tensor,
        recovery_feature: torch.Tensor,
    ):
        fusion = self.sequential_fusion
        if self.variant == "without_amsr":
            recovery_delta = torch.zeros_like(raw_x)
        else:
            recovery_delta = 0.1 * fusion.recovery_to_input(recovery_feature)
        relation_input = raw_x + recovery_delta
        relation_feature, relation_aux = self._relation_forward(relation_input)

        relation_delta = 0.1 * fusion.relation_to_input(relation_feature)
        sequence_input = relation_input + relation_delta
        graph_bias = None
        if self.variant != "without_graph_guidance":
            graph_bias = relation_aux["graph_group_bias"]
        sequence_feature = self.sequence_branch(sequence_input, graph_bias=graph_bias)

        stage_feature = fusion.stage1(torch.cat([shared_feature, recovery_feature], dim=1))
        stage_feature = fusion.stage_feature_update(fusion.stage2, stage_feature, relation_feature)
        stage_feature = fusion.stage_feature_update(fusion.stage3, stage_feature, sequence_feature)
        stage_scales = torch.stack(
            [
                recovery_delta.abs().mean(dim=1),
                relation_delta.abs().mean(dim=1),
                sequence_feature.abs().mean(dim=1),
            ],
            dim=1,
        )
        return stage_feature, relation_aux, {
            "stage_scales": stage_scales,
            "relation_input": relation_input,
            "sequence_input": sequence_input,
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        shared_feature = self.shared_encoder(x)
        if self.variant == "without_amsr":
            recovery_feature, recovery_aux = self._dummy_recovery(x)
            refined_x, reliability_aux = self._identity_reliability(x)
        else:
            recovery_feature, recovery_aux = self.recovery_branch(x)
            if self.variant == "without_recovery_correction":
                refined_x, reliability_aux = self._identity_reliability(x)
            else:
                refined_x, reliability_aux = self.reliability_gate(x, recovery_aux["x_recon"])

        sequential_feature, relation_aux, sequence_aux = self._sequential_forward(
            raw_x=refined_x,
            shared_feature=shared_feature,
            recovery_feature=recovery_feature,
        )
        classifier_logits = self.classifier(sequential_feature)

        if self.variant == "without_prototype_decision":
            prototype_aux = {
                "prototype_logits": torch.zeros_like(classifier_logits),
                "prototype_distances": torch.zeros_like(classifier_logits),
                "nearest_proto": classifier_logits.argmax(dim=1),
                "prototype_temperature": classifier_logits.new_tensor(1.0),
            }
            logits = classifier_logits
        else:
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

    def compute_auxiliary_losses(self, aux_outputs, inputs, targets=None):
        losses = super().compute_auxiliary_losses(aux_outputs, inputs, targets=targets)
        if self.variant == "without_amsr":
            losses["recovery_total"] = losses["recovery_total"] * 0.0
        if self.variant == "without_prototype_decision":
            losses["proto_ce"] = losses["proto_ce"] * 0.0
            losses["proto_pull"] = losses["proto_pull"] * 0.0
            losses["proto_sep"] = losses["proto_sep"] * 0.0
            losses["prototype_total"] = losses["prototype_total"] * 0.0
        return losses


def prepare_split_with_indices(
    feature_data: np.ndarray,
    labels: np.ndarray,
    seed: int,
):
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=labels[temp_idx],
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(feature_data[train_idx])
    x_val = scaler.transform(feature_data[val_idx])
    x_test = scaler.transform(feature_data[test_idx])
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": labels[train_idx],
        "y_val": labels[val_idx],
        "y_test": labels[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }


def make_loaders(split: dict, seed: int, batch_size: int):
    sample_weights = base.create_sample_weights(
        split["y_train"],
        boost_classes=(0, 5),
        boost_factor=2.0,
    )
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
        base.PaperDataset(split["x_train"], split["y_train"]),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        base.PaperDataset(split["x_val"], split["y_val"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        base.PaperDataset(split["x_test"], split["y_test"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def json_dump(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_progress(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_one(
    spec: VariantSpec,
    seed: int,
    split: dict,
    num_classes: int,
    args,
    device: torch.device,
):
    run_dir = args.output_dir / spec.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "model.pth"
    predictions_path = run_dir / "predictions.npz"
    progress_path = run_dir / "progress.log"

    if (
        not args.overwrite
        and metrics_path.exists()
        and checkpoint_path.exists()
        and predictions_path.exists()
    ):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            append_progress(progress_path, "Complete artifacts found; skipping.")
            return payload

    base.set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(split, seed, args.batch_size)
    model = CurrentRGIFormerAblation(
        variant=spec.name,
        input_dim=split["x_train"].shape[1],
        hidden_dim=512,
        node_dim=64,
        num_groups=14,
        sequence_layers=2,
        sequence_heads=8,
        dropout=0.2,
        num_classes=num_classes,
        lambda_recovery=0.05,
        lambda_graph=1e-3,
    ).to(device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=8e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6,
    )
    class_weights = base.create_class_weights(
        split["y_train"],
        boost_classes=(0, 5),
        boost_factor=2.0,
    )
    criterion = base.FocalLoss(
        alpha=class_weights.to(device),
        gamma=2.5,
        label_smoothing=0.05,
    )

    split_path = args.output_dir / "splits" / f"seed_{seed}.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if not split_path.exists() or args.overwrite:
        np.savez_compressed(
            split_path,
            train_indices=split["train_idx"],
            val_indices=split["val_idx"],
            test_indices=split["test_idx"],
            scaler_mean=split["scaler_mean"],
            scaler_scale=split["scaler_scale"],
        )

    best_val_acc = -math.inf
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    started = time.time()
    append_progress(
        progress_path,
        f"START variant={spec.name} seed={seed} device={device} "
        f"trainable={trainable_params:,}/{total_params:,}",
    )

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = base.train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = base.evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": spec.name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            append_progress(
                progress_path,
                f"epoch={epoch:03d} train_acc={train_metrics['acc']:.6f} "
                f"val_acc={val_metrics['acc']:.6f} best={best_val_acc:.6f} "
                f"patience={patience_counter}/{args.patience}",
            )
        if patience_counter >= args.patience:
            append_progress(progress_path, f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = base.evaluate(model, test_loader, criterion, device)
    labels = test_metrics["labels"].astype(np.int64)
    preds = test_metrics["preds"].astype(np.int64)
    accuracy = accuracy_score(labels, preds)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    elapsed_seconds = time.time() - started

    np.savez_compressed(
        predictions_path,
        labels=labels,
        preds=preds,
        test_indices=split["test_idx"],
    )
    np.savez_compressed(
        run_dir / "history.npz",
        **{key: np.asarray(values, dtype=np.float64) for key, values in history.items()},
    )
    payload = {
        "status": "complete",
        "variant": spec.name,
        "variant_label": spec.label,
        "seed": seed,
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(history["train_loss"])),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "train_samples": int(len(split["y_train"])),
        "val_samples": int(len(split["y_val"])),
        "test_samples": int(len(split["y_test"])),
        "elapsed_seconds": float(elapsed_seconds),
        "test_index_sha256": hashlib.sha256(
            np.asarray(split["test_idx"], dtype=np.int64).tobytes()
        ).hexdigest(),
        "label_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(preds.tobytes()).hexdigest(),
    }
    json_dump(metrics_path, payload)
    append_progress(
        progress_path,
        f"DONE accuracy={accuracy * 100:.4f}% macro_f1={macro_f1 * 100:.4f}% "
        f"best_epoch={best_epoch} elapsed={elapsed_seconds:.1f}s",
    )
    return payload


def scan_completed(output_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(output_dir.glob("*/seed_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    order = {spec.name: index for index, spec in enumerate(VARIANTS)}
    frame = pd.DataFrame(rows)
    frame["_variant_order"] = frame["variant"].map(order)
    frame = frame.sort_values(["_variant_order", "seed"]).drop(columns="_variant_order")
    return frame.reset_index(drop=True)


def aggregate_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = scan_completed(output_dir)
    if runs.empty:
        return runs, pd.DataFrame(), pd.DataFrame()
    runs.to_csv(output_dir / "all_runs.csv", index=False, encoding="utf-8-sig")

    rows = []
    for spec in VARIANTS:
        subset = runs[runs["variant"] == spec.name]
        if subset.empty:
            continue
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            values = subset[metric].astype(float).to_numpy()
            n = len(values)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else 0.0
            t_value = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
            half = t_value * sd / math.sqrt(n) if n > 1 else 0.0
            rows.append(
                {
                    "variant": spec.name,
                    "variant_label": spec.label,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "mean_percent": mean * 100,
                    "sd_percent": sd * 100,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    paired_rows = []
    full = runs[runs["variant"] == "full"].set_index("seed")
    for spec in VARIANTS:
        if spec.name == "full":
            continue
        variant = runs[runs["variant"] == spec.name].set_index("seed")
        shared = sorted(set(full.index) & set(variant.index))
        for seed in shared:
            for metric in ("accuracy", "macro_f1"):
                paired_rows.append(
                    {
                        "variant": spec.name,
                        "variant_label": spec.label,
                        "seed": int(seed),
                        "metric": metric,
                        "delta_vs_full": float(variant.loc[seed, metric] - full.loc[seed, metric]),
                        "delta_percentage_points": float(
                            100 * (variant.loc[seed, metric] - full.loc[seed, metric])
                        ),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_deltas.csv", index=False, encoding="utf-8-sig")
    return runs, summary, paired


def write_protocol(args, selected: list[VariantSpec], dataset_hash: str) -> None:
    lines = [
        "# Current-model matched ablation protocol",
        "",
        "## Fixed evaluation protocol",
        "",
        "- Dataset: Paper, 462 measured features and 11 classes.",
        "- Split: seed-specific stratified 70/15/15 train/validation/test split.",
        "- Standardization: fitted on the training split only.",
        f"- Seeds: {args.seeds}.",
        "- Sampling: seed-controlled weighted random sampler; classes 0 and 5 "
        "receive factor 2.0.",
        f"- Training: max {args.max_epochs} epochs, patience {args.patience}, "
        f"batch size {args.batch_size}.",
        "- Optimizer: AdamW (lr 8e-4, weight decay 1e-4).",
        "- Scheduler: cosine warm restarts (T0=15, Tmult=2, eta_min=1e-6).",
        "- Objective: focal loss (gamma 2.5, smoothing 0.05), with the current "
        "model's recovery, graph, and prototype auxiliary terms except when the "
        "corresponding mechanism is removed.",
        "- Model selection: highest validation accuracy; the test split is used once "
        "after checkpoint selection.",
        "- Reported uncertainty: sample standard deviation across the five seeds.",
        f"- Dataset SHA-256: `{dataset_hash}`.",
        f"- Base-model source SHA-256: `{sha256_file(Path(base.__file__).resolve())}`.",
        f"- Runner source SHA-256: `{sha256_file(SCRIPT_PATH)}`.",
        "",
        "## Ablation definitions",
        "",
    ]
    for spec in selected:
        lines.append(f"- **{spec.label}** (`{spec.name}`): {spec.description}")
    lines.append("")
    (args.output_dir / "protocol.md").write_text("\n".join(lines), encoding="utf-8")


def audit_results(
    output_dir: Path,
    expected_variants: list[VariantSpec],
    expected_seeds: list[int],
) -> dict:
    runs, summary, paired = aggregate_results(output_dir)
    expected_pairs = {
        (spec.name, int(seed)) for spec in expected_variants for seed in expected_seeds
    }
    observed_pairs = (
        set(zip(runs["variant"], runs["seed"].astype(int))) if not runs.empty else set()
    )
    errors = []
    if expected_pairs != observed_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        if missing:
            errors.append(f"Missing completed runs: {missing}")
        if extra:
            errors.append(f"Unexpected completed runs: {extra}")

    for seed in expected_seeds:
        reference_indices = None
        reference_labels = None
        for spec in expected_variants:
            pred_path = output_dir / spec.name / f"seed_{seed}" / "predictions.npz"
            checkpoint_path = output_dir / spec.name / f"seed_{seed}" / "model.pth"
            if not pred_path.exists() or not checkpoint_path.exists():
                continue
            artifact = np.load(pred_path)
            indices = artifact["test_indices"]
            labels = artifact["labels"]
            preds = artifact["preds"]
            if len(indices) != 1650 or len(labels) != 1650 or len(preds) != 1650:
                errors.append(
                    f"{spec.name}/seed_{seed}: expected 1,650 test samples, "
                    f"found {len(indices)}/{len(labels)}/{len(preds)}."
                )
            if reference_indices is None:
                reference_indices = indices
                reference_labels = labels
            elif not np.array_equal(indices, reference_indices):
                errors.append(f"seed_{seed}: test indices differ across variants.")
            elif not np.array_equal(labels, reference_labels):
                errors.append(f"seed_{seed}: test labels differ across variants.")

            matching = runs[
                (runs["variant"] == spec.name) & (runs["seed"].astype(int) == seed)
            ]
            if len(matching) == 1:
                recomputed_acc = accuracy_score(labels, preds)
                _, _, recomputed_f1, _ = precision_recall_fscore_support(
                    labels,
                    preds,
                    average="macro",
                    zero_division=0,
                )
                if not math.isclose(
                    float(matching.iloc[0]["accuracy"]),
                    float(recomputed_acc),
                    abs_tol=1e-12,
                ):
                    errors.append(f"{spec.name}/seed_{seed}: accuracy mismatch.")
                if not math.isclose(
                    float(matching.iloc[0]["macro_f1"]),
                    float(recomputed_f1),
                    abs_tol=1e-12,
                ):
                    errors.append(f"{spec.name}/seed_{seed}: macro-F1 mismatch.")

    payload = {
        "status": "pass" if not errors else "fail",
        "expected_run_count": len(expected_pairs),
        "completed_run_count": len(observed_pairs),
        "summary_row_count": int(len(summary)),
        "paired_delta_row_count": int(len(paired)),
        "errors": errors,
    }
    json_dump(output_dir / "audit.json", payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched ablations of the current Paper RGIFormer."
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names or 'all'.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--data-path", type=Path, default=base.DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Do not train; aggregate and audit existing outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected = parse_variants(args.variants)
    seeds = parse_csv_ints(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.audit_only:
        audit = audit_results(args.output_dir, selected, seeds)
        print(json.dumps(audit, indent=2), flush=True)
        if audit["status"] != "pass":
            raise SystemExit(1)
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_path = args.data_path.resolve()
    feature_data, labels, num_classes = base.load_paper_data(data_path)
    dataset_hash = sha256_file(data_path)
    split_cache = {
        seed: prepare_split_with_indices(feature_data, labels, seed) for seed in seeds
    }
    source_manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "data_path": str(data_path),
        "data_sha256": dataset_hash,
        "data_shape": list(feature_data.shape),
        "num_classes": int(num_classes),
        "base_model_source": str(Path(base.__file__).resolve()),
        "base_model_sha256": sha256_file(Path(base.__file__).resolve()),
        "runner_source": str(SCRIPT_PATH),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "seeds": seeds,
        "variants": [asdict(spec) for spec in selected],
    }
    json_dump(args.output_dir / "source_manifest.json", source_manifest)
    write_protocol(args, selected, dataset_hash)

    for spec in selected:
        for seed in seeds:
            run_one(
                spec=spec,
                seed=seed,
                split=split_cache[seed],
                num_classes=num_classes,
                args=args,
                device=device,
            )
            aggregate_results(args.output_dir)

    audit = audit_results(args.output_dir, selected, seeds)
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
