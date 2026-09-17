# Tests

    pip install -r requirements-dev.txt
    pytest

| file | covers |
|---|---|
| `test_architectures.py` | each architecture against ChromeCRISPR's description, and that every model builds and runs |
| `test_records.py` | the per-model records, and that regenerating them from ChromeCRISPR is reproducible |
| `test_training_protocol.py` | the training and validation protocol of  |
