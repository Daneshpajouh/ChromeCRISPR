# Paper-specification verification

75 of 75 checks passed.

| model | property asserted from ChromeCRISPR | expected | observed | |
|---|---|---|---|---|
| CNN | two convolutional layers () | `2` | `2` | pass |
| CNN | each with 128 filters | `128` | `128` | pass |
| CNN | a kernel size of 3 | `3` | `3` | pass |
| CNN | stride of 1 | `1` | `1` | pass |
| CNN | padding of 1 | `1` | `1` | pass |
| CNN | first dense layer has 64 units | `64` | `64` | pass |
| CNN | second has one output unit | `1` | `1` | pass |
| CNN | fully connected layers with batch normalization | `True` | `True` | pass |
| GRU | two recurrent layers (Sections 2.5.3 to 2.5.5) | `2` | `2` | pass |
| GRU | each with 128 hidden units | `128` | `128` | pass |
| GRU | bidirectional only for BiLSTM | `False` | `False` | pass |
| GRU | first dense layer has 64 units | `64` | `64` | pass |
| GRU | second has one output unit | `1` | `1` | pass |
| GRU | fully connected layers with batch normalization | `True` | `True` | pass |
| LSTM | two recurrent layers (Sections 2.5.3 to 2.5.5) | `2` | `2` | pass |
| LSTM | each with 128 hidden units | `128` | `128` | pass |
| LSTM | bidirectional only for BiLSTM | `False` | `False` | pass |
| LSTM | first dense layer has 64 units | `64` | `64` | pass |
| LSTM | second has one output unit | `1` | `1` | pass |
| LSTM | fully connected layers with batch normalization | `True` | `True` | pass |
| BiLSTM | two recurrent layers (Sections 2.5.3 to 2.5.5) | `2` | `2` | pass |
| BiLSTM | each with 128 hidden units | `128` | `128` | pass |
| BiLSTM | bidirectional only for BiLSTM | `True` | `True` | pass |
| BiLSTM | first dense layer has 64 units | `64` | `64` | pass |
| BiLSTM | second has one output unit | `1` | `1` | pass |
| BiLSTM | fully connected layers with batch normalization | `True` | `True` | pass |
| deepCNN | three convolutional layers with 128 filters each (2.5.6) | `3` | `3` | pass |
| deepCNN | 128 filters each | `128` | `128` | pass |
| deepCNN | dense layers of 128, 64 and 32, then the output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| deepGRU | three recurrent layers () | `3` | `3` | pass |
| deepGRU | with 128 hidden units each | `128` | `128` | pass |
| deepGRU | dense layers of 128, 64 and 32, then the output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| deepLSTM | three recurrent layers () | `3` | `3` | pass |
| deepLSTM | with 128 hidden units each | `128` | `128` | pass |
| deepLSTM | dense layers of 128, 64 and 32, then the output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| deepBiLSTM | three recurrent layers () | `3` | `3` | pass |
| deepBiLSTM | with 128 hidden units each | `128` | `128` | pass |
| deepBiLSTM | dense layers of 128, 64 and 32, then the output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| CNN_GRU+GC | CNN branch includes three convolutional layers (2.5.7) | `3` | `3` | pass |
| CNN_GRU+GC | with 128 filters each | `128` | `128` | pass |
| CNN_GRU+GC | kernel size of 3 | `3` | `3` | pass |
| CNN_GRU+GC | RNN branch includes three recurrent layers | `3` | `3` | pass |
| CNN_GRU+GC | with 128 hidden units each | `128` | `128` | pass |
| CNN_GRU+GC | CNN followed by RNN: the RNN reads the CNN features | `128` | `128` | pass |
| CNN_GRU+GC | 256 features concatenated, combined with GC content (257 features) | `257` | `257` | pass |
| CNN_GRU+GC | three dense layers (128, 64, 32 units) before the final output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| CNN_GRU+GC | a single output unit | `(2, 1)` | `(2, 1)` | pass |
| CNN_LSTM+GC | CNN branch includes three convolutional layers (2.5.7) | `3` | `3` | pass |
| CNN_LSTM+GC | with 128 filters each | `128` | `128` | pass |
| CNN_LSTM+GC | kernel size of 3 | `3` | `3` | pass |
| CNN_LSTM+GC | RNN branch includes three recurrent layers | `3` | `3` | pass |
| CNN_LSTM+GC | with 128 hidden units each | `128` | `128` | pass |
| CNN_LSTM+GC | CNN followed by RNN: the RNN reads the CNN features | `128` | `128` | pass |
| CNN_LSTM+GC | 256 features concatenated, combined with GC content (257 features) | `257` | `257` | pass |
| CNN_LSTM+GC | three dense layers (128, 64, 32 units) before the final output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| CNN_LSTM+GC | a single output unit | `(2, 1)` | `(2, 1)` | pass |
| CNN_BiLSTM+GC | CNN branch includes three convolutional layers (2.5.7) | `3` | `3` | pass |
| CNN_BiLSTM+GC | with 128 filters each | `128` | `128` | pass |
| CNN_BiLSTM+GC | kernel size of 3 | `3` | `3` | pass |
| CNN_BiLSTM+GC | RNN branch includes three recurrent layers | `3` | `3` | pass |
| CNN_BiLSTM+GC | with 128 hidden units each | `128` | `128` | pass |
| CNN_BiLSTM+GC | CNN followed by RNN: the RNN reads the CNN features | `128` | `128` | pass |
| CNN_BiLSTM+GC | 256 features concatenated, combined with GC content (257 features) | `257` | `257` | pass |
| CNN_BiLSTM+GC | three dense layers (128, 64, 32 units) before the final output layer | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| CNN_BiLSTM+GC | a single output unit | `(2, 1)` | `(2, 1)` | pass |
| Transformer | three transformer layers () | `3` | `3` | pass |
| Transformer | multi-head self-attention consisting of 8 attention heads | `8` | `8` | pass |
| Transformer | with 128 hidden units each | `128` | `128` | pass |
| Transformer | uses layer normalization | `True` | `True` | pass |
| Transformer | includes positional encoding to maintain sequence order information | `True` | `True` | pass |
| Transformer | followed by dense layers with batch normalization | `[128, 64, 32, 1]` | `[128, 64, 32, 1]` | pass |
| Transformer | a single output unit | `(2, 1)` | `(2, 1)` | pass |
| CNN_GRU+GC[flatten_proj] | every faithful reading of the unspecified CNN readout runs | `(2, 1)` | `(2, 1)` | pass |
| CNN_GRU+GC[max] | every faithful reading of the unspecified CNN readout runs | `(2, 1)` | `(2, 1)` | pass |
| CNN_GRU+GC[mean] | every faithful reading of the unspecified CNN readout runs | `(2, 1)` | `(2, 1)` | pass |
