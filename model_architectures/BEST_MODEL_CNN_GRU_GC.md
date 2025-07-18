# Best Model: CNN-GRU + GC Content (Trial 74)

**Spearman Correlation:** 0.87698

## Architecture
- 2D Convolutional Neural Network (CNN)
- Gated Recurrent Unit (GRU)
- MLP Mixer
- GC Content as biological feature
- No Transformer, no Inception, no BiLSTM

## Hyperparameters
- Learning rate: 0.0011062631134035848
- Batch size: 64
- Weight decay: 1.614789571500063e-05
- Epochs: 113
- Optimizer: Adam
- Scheduler: Plateau
- Use AMP: True
- Use Lookahead: True
- Use Grad Clip: True (norm: 8.701070756488479)
- Dropout rate: 0.1603840537295789
- Activation: ReLU
- Normalization: BatchNorm
- Residual block: False
- SE block: False
- Attention pooling: True

### CNN
- Use CNN: True
- Num CNN layers: 2
- Kernel size: 5
- Pooling: avg
- Use Inception: False

### RNN
- RNN type: GRU
- Num RNN layers: 2

### MLP Mixer
- Use MLP Mixer: True
- Depth: 1
- Token dim: 150
- Channel dim: 171

### Biological Features
- GC Content: True
- Td: True
- Free Energy Duplex: False
- Enthalpy Duplex: True
- Entropy Duplex: True
- Free Energy RNA: True

## Full Parameter Dict
```
{'lr': 0.0011062631134035848, 'batch_size': 64, 'weight_decay': 1.614789571500063e-05, 'num_epochs': 113, 'use_biofeature_GC_Content': True, 'use_biofeature_Td': True, 'use_biofeature_Free_Energy_Duplex': False, 'use_biofeature_Enthalpy_Duplex': True, 'use_biofeature_Entropy_Duplex': True, 'use_biofeature_Free_Energy_RNA': True, 'use_amp': True, 'optimizer': 'adam', 'use_grad_clip': True, 'grad_clip_norm': 8.701070756488479, 'use_lookahead': True, 'scheduler': 'plateau', 'dropout_rate': 0.1603840537295789, 'activation_func': 'relu', 'normalization': 'batchnorm', 'use_cnn': True, 'use_inception': False, 'num_cnn_layers': 2, 'cnn_kernel_size': 5, 'cnn_pooling': 'avg', 'rnn_type': 'gru', 'num_rnn_layers': 2, 'use_transformer': False, 'use_mlp_mixer': True, 'mlp_mixer_depth': 1, 'mlp_mixer_token_dim': 150, 'mlp_mixer_channel_dim': 171, 'residual_block_needed': False, 'se_block_needed': False, 'attention_pooling': True, 'num_fc_layers': 3}
```
