# -*- coding: utf-8 -*-
"""Temporal feature expansion shared by the BSM2 training scripts."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPOSITORY_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


from collections.abc import Iterable

import numpy as np
import pandas as pd


VALID_TEMPORAL_MODES = ("none", "lag", "diff", "lagdiff")


def parse_history_steps(step_text):
    if step_text is None:
        return ()
    if isinstance(step_text, str):
        steps = [int(item.strip()) for item in step_text.split(",") if item.strip()]
    elif isinstance(step_text, Iterable):
        steps = [int(item) for item in step_text]
    else:
        steps = [int(step_text)]
    steps = sorted({step for step in steps if step > 0})
    return tuple(steps)


def parse_time_periods(period_text):
    if period_text is None:
        return ()
    if isinstance(period_text, str):
        periods = [float(item.strip()) for item in period_text.split(",") if item.strip()]
    elif isinstance(period_text, Iterable):
        periods = [float(item) for item in period_text]
    else:
        periods = [float(period_text)]
    periods = sorted({period for period in periods if period > 0})
    return tuple(periods)


def describe_temporal_feature_config(
    mode,
    history_steps,
    rolling_window=0,
    include_time_features=False,
    time_periods=(),
    future_steps=(),
):
    history_text = ",".join(str(step) for step in history_steps) if history_steps else "none"
    future_text = ",".join(str(step) for step in parse_history_steps(future_steps)) if future_steps else "none"
    time_text = ",".join(f"{period:g}" for period in parse_time_periods(time_periods)) or "none"
    return (
        f"mode={mode}, history_steps={history_text}, rolling_window={rolling_window}, "
        f"future_steps={future_text}, include_time_features={include_time_features}, time_periods={time_text}"
    )


def build_time_features(data, time_col="time_day", periods=(1, 7, 30, 365)):
    if time_col not in data.columns:
        raise ValueError(f"Time column not found: {time_col}")

    periods = parse_time_periods(periods)
    time_values = data[time_col].astype(np.float32).to_numpy()[:, None]
    parts = [time_values]
    names = [time_col]

    for period in periods:
        angle = (2.0 * np.pi * time_values) / period
        parts.append(np.sin(angle).astype(np.float32))
        parts.append(np.cos(angle).astype(np.float32))
        period_name = f"{period:g}".replace(".", "p")
        names.extend([f"{time_col}_sin{period_name}", f"{time_col}_cos{period_name}"])

    return np.concatenate(parts, axis=1).astype(np.float32), names


def build_bsm2_temporal_features(
    data,
    base_feature_names,
    mode="lagdiff",
    history_steps=(1, 2, 4),
    rolling_window=0,
    include_time_features=False,
    time_periods=(1, 7, 30, 365),
    future_steps=(),
    label_col="label",
    time_col="time_day",
):
    """Return a 2-D feature matrix with per-class temporal context.

    BSM2 CSV rows are often shuffled, so history is reconstructed by sorting each
    label group by `time_day`. The expanded rows are then restored to the CSV's
    original row order. This keeps every downstream model on the same B x F input
    protocol while adding temporal information.
    """

    if mode not in VALID_TEMPORAL_MODES:
        raise ValueError(f"Unsupported temporal feature mode: {mode}")

    history_steps = parse_history_steps(history_steps)
    future_steps = parse_history_steps(future_steps)
    base_frame = data.loc[:, base_feature_names].astype(np.float32)
    parts = [base_frame]
    names = list(base_feature_names)

    if include_time_features:
        time_values, time_names = build_time_features(data, time_col=time_col, periods=time_periods)
        time_frame = pd.DataFrame(time_values, index=data.index, columns=time_names, dtype=np.float32)
        parts.append(time_frame)
        names.extend(time_names)

    if mode == "none" or (not history_steps and not future_steps):
        expanded = pd.concat(parts, axis=1)
        expanded = expanded.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return expanded.to_numpy(dtype=np.float32), names

    group_keys = [label_col] if label_col in data.columns else None
    grouped = data.groupby(group_keys, sort=False) if group_keys else [(None, data)]

    def append_shift_features(steps, shift_sign, value_suffix, diff_suffix):
        nonlocal parts, names
        for step in steps:
            shifted_frame = pd.DataFrame(index=data.index, columns=base_feature_names, dtype=np.float32)
            diff_frame = pd.DataFrame(index=data.index, columns=base_feature_names, dtype=np.float32)

            for _, group in grouped:
                if time_col in group.columns:
                    group = group.sort_values(time_col, kind="mergesort")
                group_values = group.loc[:, base_feature_names].astype(np.float32)
                shifted_values = group_values.shift(shift_sign * step)
                shifted_values = shifted_values.where(~shifted_values.isna(), group_values)
                diff_values = group_values - shifted_values
                shifted_frame.loc[group.index, :] = shifted_values.to_numpy(dtype=np.float32)
                diff_frame.loc[group.index, :] = diff_values.to_numpy(dtype=np.float32)

            if mode in ("lag", "lagdiff"):
                shifted_frame.columns = [f"{name}_{value_suffix}{step}" for name in base_feature_names]
                parts.append(shifted_frame)
                names.extend(shifted_frame.columns.tolist())
            if mode in ("diff", "lagdiff"):
                diff_frame.columns = [f"{name}_{diff_suffix}{step}" for name in base_feature_names]
                parts.append(diff_frame)
                names.extend(diff_frame.columns.tolist())

    append_shift_features(history_steps, 1, "lag", "diff")
    append_shift_features(future_steps, -1, "lead", "difflead")

    if rolling_window and rolling_window > 1:
        rolling_mean = pd.DataFrame(index=data.index, columns=base_feature_names, dtype=np.float32)
        rolling_std = pd.DataFrame(index=data.index, columns=base_feature_names, dtype=np.float32)
        grouped = data.groupby(group_keys, sort=False) if group_keys else [(None, data)]

        for _, group in grouped:
            if time_col in group.columns:
                group = group.sort_values(time_col, kind="mergesort")
            group_values = group.loc[:, base_feature_names].astype(np.float32)
            mean_values = group_values.rolling(rolling_window, min_periods=1).mean()
            std_values = group_values.rolling(rolling_window, min_periods=1).std().fillna(0.0)
            rolling_mean.loc[group.index, :] = mean_values.to_numpy(dtype=np.float32)
            rolling_std.loc[group.index, :] = std_values.to_numpy(dtype=np.float32)

        rolling_mean.columns = [f"{name}_rollmean{rolling_window}" for name in base_feature_names]
        rolling_std.columns = [f"{name}_rollstd{rolling_window}" for name in base_feature_names]
        parts.extend([rolling_mean, rolling_std])
        names.extend(rolling_mean.columns.tolist())
        names.extend(rolling_std.columns.tolist())

    expanded = pd.concat(parts, axis=1)
    expanded = expanded.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return expanded.to_numpy(dtype=np.float32), names


def build_bsm2_windows(
    data,
    base_feature_names,
    window_size=16,
    label_col="label",
    time_col="time_day",
):
    """Return [sample, window, feature] windows reconstructed from BSM2 rows.

    The CSV rows are restored to per-label chronological order before windowing,
    then written back to the original row order. Early rows are left-padded by
    repeating the earliest available sample in that label group.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    values = data.loc[:, base_feature_names].astype(np.float32)
    windows = np.zeros((len(data), window_size, len(base_feature_names)), dtype=np.float32)
    group_keys = [label_col] if label_col in data.columns else None
    grouped = data.groupby(group_keys, sort=False) if group_keys else [(None, data)]

    for _, group in grouped:
        if time_col in group.columns:
            group = group.sort_values(time_col, kind="mergesort")
        group_values = values.loc[group.index].to_numpy(dtype=np.float32)
        group_indices = group.index.to_numpy()

        for pos, row_index in enumerate(group_indices):
            start = max(0, pos - window_size + 1)
            window = group_values[start : pos + 1]
            if len(window) < window_size:
                pad = np.repeat(window[:1], window_size - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            windows[row_index] = window

    return windows, list(base_feature_names)
