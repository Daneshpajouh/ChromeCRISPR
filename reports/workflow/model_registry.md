# ChromeCRISPR Model Registry

Generated at: `2026-04-17T01:37:30Z`

## Summary

- Canonical model root: `models`
- Hyperparameter root: `docs/hyperparameters`
- Canonical model count: `20`

## Category Counts

| Category | Count | Expected |
|---|---:|---:|
| Base models | 5 | 5 |
| Base models + GC | 4 | 4 |
| Deep models | 4 | 4 |
| Deep models + GC | 4 | 4 |
| ChromeCRISPR hybrid models | 3 | 3 |

## Models

| Model | Category | Spearman | MSE | Checkpoint | Hyperparameters |
|---|---|---:|---:|---|---|
| CNN_GRU+GC | ChromeCRISPR hybrid models | 0.876 | 0.0093 | `models/chromecrispr_hybrid_models/CNN_GRU+GC.pth` | `docs/hyperparameters/CNN_GRU+GC_hyperparameters.json` |
| deepCNN+GC | Deep models + GC | 0.873 | 0.0093 | `models/deep_models_with_gc/deepCNN+GC.pth` | `docs/hyperparameters/deepCNN+GC_hyperparameters.json` |
| CNN_BiLSTM+GC | ChromeCRISPR hybrid models | 0.870 | 0.0096 | `models/chromecrispr_hybrid_models/CNN_BiLSTM+GC.pth` | `docs/hyperparameters/CNN_BiLSTM+GC_hyperparameters.json` |
| deepCNN | Deep models | 0.869 | 0.0098 | `models/deep_models/deepCNN.pth` | `docs/hyperparameters/deepCNN_hyperparameters.json` |
| deepGRU | Deep models | 0.868 | 0.0099 | `models/deep_models/deepGRU.pth` | `docs/hyperparameters/deepGRU_hyperparameters.json` |
| deepBiLSTM+GC | Deep models + GC | 0.867 | 0.0098 | `models/deep_models_with_gc/deepBiLSTM+GC.pth` | `docs/hyperparameters/deepBiLSTM+GC_hyperparameters.json` |
| CNN_LSTM+GC | ChromeCRISPR hybrid models | 0.867 | 0.0115 | `models/chromecrispr_hybrid_models/CNN_LSTM+GC.pth` | `docs/hyperparameters/CNN_LSTM+GC_hyperparameters.json` |
| deepGRU+GC | Deep models + GC | 0.867 | 0.0098 | `models/deep_models_with_gc/deepGRU+GC.pth` | `docs/hyperparameters/deepGRU+GC_hyperparameters.json` |
| deepLSTM | Deep models | 0.862 | 0.0103 | `models/deep_models/deepLSTM.pth` | `docs/hyperparameters/deepLSTM_hyperparameters.json` |
| deepBiLSTM | Deep models | 0.862 | 0.0104 | `models/deep_models/deepBiLSTM.pth` | `docs/hyperparameters/deepBiLSTM_hyperparameters.json` |
| deepLSTM+GC | Deep models + GC | 0.860 | 0.0104 | `models/deep_models_with_gc/deepLSTM+GC.pth` | `docs/hyperparameters/deepLSTM+GC_hyperparameters.json` |
| LSTM+GC | Base models + GC | 0.856 | 0.0112 | `models/base_models_with_gc/LSTM+GC.pth` | `docs/hyperparameters/LSTM+GC_hyperparameters.json` |
| BiLSTM+GC | Base models + GC | 0.855 | 0.0110 | `models/base_models_with_gc/BiLSTM+GC.pth` | `docs/hyperparameters/BiLSTM+GC_hyperparameters.json` |
| BiLSTM | Base models | 0.843 | 0.0120 | `models/base_models/BiLSTM.pth` | `docs/hyperparameters/BiLSTM_hyperparameters.json` |
| GRU+GC | Base models + GC | 0.840 | 0.0122 | `models/base_models_with_gc/GRU+GC.pth` | `docs/hyperparameters/GRU+GC_hyperparameters.json` |
| LSTM | Base models | 0.837 | 0.0122 | `models/base_models/LSTM.pth` | `docs/hyperparameters/LSTM_hyperparameters.json` |
| GRU | Base models | 0.837 | 0.0121 | `models/base_models/GRU.pth` | `docs/hyperparameters/GRU_hyperparameters.json` |
| CNN | Base models | 0.792 | 0.0161 | `models/base_models/CNN.pth` | `docs/hyperparameters/CNN_hyperparameters.json` |
| RF | Base models | 0.789 | 0.0161 | `models/base_models/RF.joblib` | `docs/hyperparameters/RF_hyperparameters.json` |
| CNN+GC | Base models + GC | 0.781 | 0.0170 | `models/base_models_with_gc/CNN+GC.pth` | `docs/hyperparameters/CNN+GC_hyperparameters.json` |
