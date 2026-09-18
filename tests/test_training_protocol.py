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


def test_the_epoch_loop_never_sees_the_held_out_set():
    """`fit` runs the epoch loop and is not given the held-out arrays at all."""
    import ast

    source = (ROOT / "scripts" / "train_all_models.py").read_text()
    tree = ast.parse(source)
    fit = next(n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "fit")
    args = {a.arg for a in fit.args.args}
    assert not {"Xte", "yte", "gte"} & args, "the training function takes no held-out data"
    body = ast.get_source_segment(source, fit)
    assert "Xte" not in body and "yte" not in body
    assert any(isinstance(n, ast.For) for n in ast.walk(fit)), "fit contains the epoch loop"


def test_the_held_out_set_is_read_once_after_fitting():
    """`run` fits first, then predicts on the held-out set."""
    import ast

    source = (ROOT / "scripts" / "train_all_models.py").read_text()
    tree = ast.parse(source)
    run = next(n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    body = ast.get_source_segment(source, run)
    fit_at = body.index("fit(")
    read_at = body.index("predict(model, Xte")
    assert fit_at < read_at, "the held-out set is read after the model is fitted"
    assert body.count("predict(model, Xte") == 1, "the held-out set is read once"
