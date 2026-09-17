"""The training protocol .

 sets aside 15% of the data for testing and uses the remaining 85% for
hyperparameter tuning and training, with model selection on a validation split. These tests
check that the training code follows that protocol.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAINER = ROOT / "src" / "training" / "trainer.py"
TRAINING_CODE = sorted((ROOT / "src").rglob("*.py"))


def test_training_loop_uses_a_validation_split():
    """: hyperparameters are selected on a validation split."""
    trainer = TRAINER.read_text()
    assert "val_loader" in trainer
    assert re.search(r"best_val|val_loss|val_spearman", trainer)


def test_cross_validation_splits_the_training_portion():
    """: nested cross-validation over the 85% used for tuning and training."""
    trainer = TRAINER.read_text()
    assert re.search(r"KFold|kf\.split", trainer)


def test_checkpoints_follow_the_validation_metric():
    """Where a checkpoint is written because a metric improved, the metric in that condition is
    the validation one. Unconditional saves, such as writing the finished model, are not
    selection and are not examined."""
    lines = TRAINER.read_text().splitlines()
    guard = re.compile(r"^\s*if\s+(?P<expr>[^:]+):\s*$")
    metric = re.compile(r"\b\w*(loss|spearman|score|metric)\w*\b")
    checked = 0
    for i, line in enumerate(lines):
        if "torch.save(" not in line:
            continue
        for j in range(i - 1, max(-1, i - 12), -1):       # nearest enclosing condition
            m = guard.match(lines[j])
            if not m:
                continue
            expr = m.group("expr")
            names = metric.findall(expr)
            if not names:
                break                                     # not a metric-guarded save
            checked += 1
            assert re.search(r"\bval_|_val\b|best_val", expr), (
                f"checkpoint at line {i+1} is selected on `{expr.strip()}`, "
                "which is not a validation metric")
            break
    assert checked >= 1, "expected at least one validation-selected checkpoint"


@pytest.mark.parametrize("name", ["train", "val"])
def test_data_loaders_are_named_by_their_role(name):
    """The loaders the training code builds are the training and validation loaders."""
    assert f"{name}_loader" in TRAINER.read_text()


def test_held_out_partition_does_not_reach_the_training_loop():
    """The held-out partition is evaluated separately, never inside the training loop."""
    partition_names = (r"\btest_loader\b", r"\bX_test\b", r"\by_test\b", r"\btest_data\b")
    hits = [f"{p.relative_to(ROOT)}:{i}"
            for p in TRAINING_CODE
            for i, line in enumerate(p.read_text().splitlines(), 1)
            for pattern in partition_names
            if re.search(pattern, line)]
    assert hits == []


def test_evaluation_reads_the_held_out_set_after_training():
    """scripts/train_all_models.py evaluates once, after the epoch loop has finished."""
    source = (ROOT / "scripts" / "train_all_models.py").read_text()
    lines = source.splitlines()
    epoch_loop = next(i for i, l in enumerate(lines) if re.search(r"for ep in range\(epochs\)", l))
    loop_indent = len(lines[epoch_loop]) - len(lines[epoch_loop].lstrip())
    end = next((i for i in range(epoch_loop + 1, len(lines))
                if lines[i].strip() and
                (len(lines[i]) - len(lines[i].lstrip())) <= loop_indent), len(lines))
    inside = "\n".join(lines[epoch_loop:end])
    assert "Xte" not in inside and "yte" not in inside, \
        "the held-out set must not be read inside the epoch loop"
    assert "Xte" in "\n".join(lines[end:]), "the held-out set is evaluated after training"
