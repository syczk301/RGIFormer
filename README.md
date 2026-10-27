# RGIFormer core implementation

Core model and training code for industrial fault classification on BSM2 and papermaking data. The implementation combines feature recovery, graph relation learning, a graph-guided sequence encoder, and prototype-assisted classification.

This source-only distribution includes model definitions, preprocessing, neural and traditional baselines, and matched ablation training. Datasets, pretrained weights, saved predictions, experiment results, manuscript files, plotting scripts, and internal audit materials are not distributed.

## Contents

| Path | Purpose |
|---|---|
| `trainCode/models/BSM2/BSM2_Baseline_Comparisons.py` | BSM2 RGIFormer and neural baselines; training CLI |
| `trainCode/models/BSM2/bsm2_temporal_features.py` | Optional lag and rolling feature preprocessing |
| `trainCode/models/Paper/RAE_GCN_Informer_Paper_ModularInnov.py` | Papermaking RGIFormer; training and evaluation |
| `trainCode/models/Paper/Paper_Baseline_Comparisons.py` | Papermaking neural baseline comparisons |
| `trainCode/training/run_current_rgiformer_ablations.py` | Matched papermaking ablations |
| `trainCode/training/run_traditional_baselines.py` | Traditional classification baselines |

## Environment

Use Python 3.10 or newer in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

Dependency versions were checked against the development environment. Select the PyTorch build appropriate for your CPU or CUDA installation. A fresh dependency installation and full retraining have not been validated as part of this source-only export.

## Input data

Supply your own compatible CSV files. By default, data are read from a `data/` directory next to the repository:

```text
parent/
  RGIFormer/
  data/
    BSM2/bsm2_train_data.csv
    BSM2/bsm2_test_data.csv
    Paper/prediction_dataset_full_11000.csv
```

BSM2 uses numeric process variables and an integer `label` column. The optional temporal feature pipeline uses `time_day` to recover within-label order. The main papermaking loader requires `fault_id` and `fault_name`, drops both, and uses every remaining column as a numeric feature in CSV order. The provided loaders define the precise input contract. Dataset files and permission to redistribute them are not included.

## Running

Run commands from the repository root. All five training entry points support `--help`.

```bash
# BSM2: explicit settings for the static-feature RGIFormer configuration.
python -m trainCode.models.BSM2.BSM2_Baseline_Comparisons --model rgiformer --mode single --seed 42 --temporal-features none --tuning-profile legacy --device auto

# Papermaking main model.
python -m trainCode.models.Paper.RAE_GCN_Informer_Paper_ModularInnov --mode single --seed 42 --device auto

# Neural baselines: use --help to inspect model choices.
python -m trainCode.models.Paper.Paper_Baseline_Comparisons --help

# Matched ablations.
python -m trainCode.training.run_current_rgiformer_ablations --help

# Traditional baselines.
python -m trainCode.training.run_traditional_baselines --help
```

Use the available input/output options to select your own files and output directories. Training generates weights and results locally; the repository allowlist excludes these artifacts from new commits. Only the complete model and training definitions are supplied: reproducing reported results requires the original inputs, split protocols, seeds, and suitable compute.

Papermaking sample-level classification does not establish temporal forecasting or prospective plant validation. BSM2 lag construction within labels should not be interpreted as independently observed industrial event chronology.
