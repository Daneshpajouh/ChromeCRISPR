# ChromeCRISPR: Exact Model Architecture Documentation

## Best Performing Model: CNN-GRU+GC (Trial 62)

### Performance Metrics (From Manuscript)
- **Spearman Correlation**: 0.876
- **Mean Squared Error**: 0.0093
- **Standard Deviation**: 0.0006

### Model File
- **File**: BestModel_Bio_53054713.9_trial_62_epoch_73_Sp_0.8765.pth
- **Total Parameters**: 369,087

### Exact Architecture (Extracted from Model File)

#### 1. CNN Layers
- **cnn_layer_0**: Bias shape: [128]
- **cnn_layer_1**: Bias shape: [128]

#### 2. RNN Layer (GRU)
- **rnn.weight_ih_l0**: Shape [384, 128]
- **rnn.weight_hh_l0**: Shape [384, 128]
- **rnn.bias_ih_l0**: Shape [384]
- **rnn.bias_hh_l0**: Shape [384]
- **rnn.weight_ih_l1**: Shape [384, 128]
- **rnn.weight_hh_l1**: Shape [384, 128]
- **rnn.bias_ih_l1**: Shape [384]
- **rnn.bias_hh_l1**: Shape [384]

#### 3. MLP Mixer
- **mlp_mixer.0.mlp_token.0.weight**: Shape [172, 21]
- **mlp_mixer.0.mlp_token.0.bias**: Shape [172]
- **mlp_mixer.0.mlp_token.3.weight**: Shape [21, 172]
- **mlp_mixer.0.mlp_token.3.bias**: Shape [21]
- **mlp_mixer.0.mlp_channel.0.weight**: Shape [197, 128]
- **mlp_mixer.0.mlp_channel.0.bias**: Shape [197]
- **mlp_mixer.0.mlp_channel.3.weight**: Shape [128, 197]
- **mlp_mixer.0.mlp_channel.3.bias**: Shape [128]

#### 4. Fully Connected Layers
- **fc_layer_0**: Weight shape: [128, 133], Bias shape: [128]
- **fc_layer_3**: Weight shape: [64, 128], Bias shape: [64]
- **fc_layer_6**: Weight shape: [32, 64], Bias shape: [32]
- **fc_layer_9**: Weight shape: [1, 32], Bias shape: [1]


### Trial Configuration (From Logs)
```json
{
  "lr": 0.00020972671691680056,
  "batch_size": 64,
  "weight_decay": 1.882255599576252e-05,
  "num_epochs": 84,
  "use_biofeature_GC_Content": true,
  "use_biofeature_Td": true,
  "use_biofeature_Free_Energy_Duplex": false,
  "use_biofeature_Enthalpy_Duplex": true,
  "use_biofeature_Entropy_Duplex": true,
  "use_biofeature_Free_Energy_RNA": true,
  "use_amp": true,
  "optimizer": "adam",
  "use_grad_clip": true,
  "grad_clip_norm": 3.848456063450914,
  "use_lookahead": true,
  "scheduler": "plateau",
  "dropout_rate": 0.14201131516203347,
  "activation_func": "relu",
  "normalization": "batchnorm",
  "use_cnn": true,
  "use_inception": false,
  "num_cnn_layers": 2,
  "cnn_kernel_size": 5,
  "cnn_pooling": null,
  "rnn_type": "gru",
  "num_rnn_layers": 2,
  "use_transformer": false,
  "use_mlp_mixer": true,
  "mlp_mixer_depth": 1,
  "mlp_mixer_token_dim": 216,
  "mlp_mixer_channel_dim": 179,
  "residual_block_needed": false,
  "se_block_needed": false,
  "attention_pooling": true,
  "num_fc_layers": 3
}
```

### Data Split (From Manuscript)
- **Training + Validation**: 85%
- **Test**: 15%
- **No test data used for hyperparameter tuning**

### Input/Output
- **Input**: 21-mer sgRNA sequences (one-hot encoded)
- **Output**: Single value (efficiency score)
- **Biological Features**: GC Content, Enthalpy Duplex, Entropy Duplex, Free Energy RNA

### Training Details
- **Optimizer**: Adam
- **Learning Rate**: 0.00020972671691680056
- **Batch Size**: 64
- **Epochs**: 84
- **Dropout Rate**: 0.14201131516203347
