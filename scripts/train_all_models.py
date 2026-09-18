#!/usr/bin/env python3
"""Train every model, save its weights, and record its test performance.

Each model gets its own hyperparameter search over the learning rate, batch size, dropout,
weight decay and schedule. Every choice is made on a validation split drawn from the training
portion, and is scored with the same rank correlation the results report, so the quantity being
selected on is the quantity being reported.

Search and the validation fit run in the same regime, with the same epoch budget, patience and
schedule horizon, so a setting that wins the search is judged under the conditions the final
model trains in.

Once the hyperparameters and the epoch count are fixed, the model is refitted on the training
and validation rows together for that many epochs. Selection never sees those extra rows and
the refit never sees a validation signal, so the split still does its job while the fitted model
keeps all the data the split was carved out of.

The held-out test set is read once per model, after the final model is fitted, and never
influences the search, the epoch count or the selection.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.architectures import MODELS, create_random_forest


def load(data_dir):
    tr = np.load(os.path.join(data_dir, "train_data.npz"))
    te = np.load(os.path.join(data_dir, "test_data.npz"))
    return (tr["X_train"].astype(np.float32), tr["y_train"].astype(np.float32),
            tr["gc_train"].astype(np.float32),
            te["X_test"].astype(np.float32), te["y_test"].astype(np.float32),
            te["gc_test"].astype(np.float32))


def run(name, Xtr, ytr, gtr, Xva, yva, gva, Xte, yte, gte, device, epochs, patience,
        trials, refit=True):
    Xall = np.concatenate([Xtr, Xva]) if refit else Xtr
    yall = np.concatenate([ytr, yva]) if refit else ytr
    gall = np.concatenate([gtr, gva]) if refit else gtr

    if name == "RF":
        model = create_random_forest(random_state=SEED)
        model.fit(np.concatenate([Xall.reshape(len(Xall), -1), gall[:, None]], 1), yall)
        p = model.predict(np.concatenate([Xte.reshape(len(Xte), -1), gte[:, None]], 1))
        return (model, float(spearmanr(p, yte).correlation),
                float(mean_squared_error(yte, p)), {}, {"rows": len(Xall)})

    cfg = search(name, Xtr, ytr, gtr, Xva, yva, gva, device, trials, epochs, patience)
    model, vsp, best_epoch = fit(name, Xtr, ytr, gtr, Xva, yva, gva, device,
                                 epochs=epochs, patience=patience, **cfg)
    note = {"validation_spearman": round(vsp, 4), "selected_epoch": best_epoch + 1,
            "rows": len(Xall)}
    if refit:
        # The epoch count and every hyperparameter are already fixed. Refitting on the
        # validation rows as well cannot feed back into either, and the schedule horizon
        # stays at the search budget so the learning-rate path is the one that was selected.
        model = fit(name, Xall, yall, gall, None, None, None, device,
                    epochs=best_epoch + 1, patience=None, schedule_horizon=epochs, **cfg)[0]

    uses_gc = getattr(model, "use_gc_content", False)
    model.eval()
    with torch.no_grad():                                  # the test set, read once
        pt = predict(model, Xte, gte, uses_gc, device)
    return (model, float(spearmanr(pt, yte).correlation),
            float(mean_squared_error(yte, pt)), cfg, note)


def predict(model, X, g, uses_gc, device, chunk=4096):
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.as_tensor(X[i:i + chunk], device=device)
        gb = torch.as_tensor(g[i:i + chunk], device=device)
        out.append((model(xb, gb) if uses_gc else model(xb)).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


SEED = 0

SEARCH_SPACE = {
    "learning_rate": [2e-4, 3e-4, 5e-4, 1e-3, 2e-3],
    "batch_size": [64, 128, 256],
    "dropout": [0.0, 0.1, 0.15, 0.2, 0.3],
    # AdamW applies decoupled weight decay, so its effect scales with the learning rate: at
    # these rates anything below about 1e-2 is indistinguishable from none.
    "weight_decay": [0.0, 0.01, 0.1, 0.3],
    "schedule": ["none", "cosine"],
}


def search(name, Xtr, ytr, gtr, Xva, yva, gva, device, trials, epochs, patience, seed=0):
    """Pick hyperparameters on the validation split. The test set is not touched here.

    Trials run at the same epoch budget, patience and schedule horizon as the final fit, so a
    setting is scored under the conditions it will actually train in. They are ranked by
    validation rank correlation, which is what the results report.
    """
    rng = np.random.default_rng(seed)
    best, best_cfg = -np.inf, None
    seen = set()
    for _ in range(trials):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        _, vsp, _ = fit(name, Xtr, ytr, gtr, Xva, yva, gva, device,
                        epochs=epochs, patience=patience, **cfg)
        if vsp > best:
            best, best_cfg = vsp, cfg
    return best_cfg or {"learning_rate": 1e-3, "batch_size": 64, "dropout": 0.0,
                        "weight_decay": 0.0, "schedule": "none"}


def fit(name, Xtr, ytr, gtr, Xva, yva, gva, device, epochs, patience,
        learning_rate, batch_size, dropout, weight_decay=0.0, schedule="none",
        schedule_horizon=None, seed=SEED):
    """Fit one model.

    With a validation split, returns the model restored to its best epoch, that epoch's
    validation rank correlation, and its index. Without one, trains for exactly ``epochs``
    and returns the final model; nothing is monitored and nothing is restored, because the
    epoch count was already chosen.

    ``schedule_horizon`` sets the cosine period. It defaults to ``epochs`` and is passed
    explicitly on a refit so the learning rate follows the same path it followed when the
    setting was selected.

    Seeded, so the same inputs and settings give the same weights on a rerun.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[name](dropout=dropout).to(device)
    uses_gc = getattr(model, "use_gc_content", False)
    opt = (torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
           if weight_decay else torch.optim.Adam(model.parameters(), lr=learning_rate))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=schedule_horizon or epochs)
             if schedule == "cosine" else None)
    lossf = nn.MSELoss()
    X, y, g = (torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device),
               torch.as_tensor(gtr, device=device))
    scored = Xva is not None
    best, best_state, best_epoch, bad = -np.inf, None, 0, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), device=device, generator=generator)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = (model(X[idx], g[idx]) if uses_gc else model(X[idx])).squeeze(-1)
            lossf(out, y[idx]).backward()
            opt.step()
        if sched is not None:
            sched.step()
        if not scored:
            continue
        model.eval()
        with torch.no_grad():
            rho = spearmanr(predict(model, Xva, gva, uses_gc, device), yva).correlation
        rho = -np.inf if rho is None or np.isnan(rho) else float(rho)
        if rho > best + 1e-6:
            best, best_epoch, bad = rho, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (best if scored else float("nan")), best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--results", default="docs/training_results.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-refit", action="store_true",
                    help="stop after the validation fit instead of refitting on all rows")
    a = ap.parse_args()

    globals()["SEED"] = a.seed
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    X, y, g, Xte, yte, gte = load(a.data_dir)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    cut = int(0.2 * len(X))
    va, tr = order[:cut], order[cut:]

    os.makedirs(a.out_dir, exist_ok=True)
    names = a.only or (["RF"] + sorted(MODELS))
    results = {}
    for name in names:
        t0 = time.time()
        model, sp, mse, cfg, note = run(
            name, X[tr], y[tr], g[tr], X[va], y[va], g[va],
            Xte, yte, gte, device, a.epochs, a.patience, a.trials,
            refit=not a.no_refit)
        path = os.path.join(a.out_dir, f"{name}.pt" if name != "RF" else "RF.joblib")
        if name == "RF":
            import joblib
            joblib.dump(model, path, compress=3)   # 328 MB uncompressed, 70 MB compressed
        else:
            torch.save(model.state_dict(), path)
        results[name] = {"spearman_correlation": round(sp, 4),
                         "mean_squared_error": round(mse, 4),
                         "hyperparameters": cfg,
                         "selection": note,
                         "checkpoint": os.path.relpath(path),
                         "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{name:16s} spearman {sp:.4f}  mse {mse:.4f}  "
              f"({results[name]['minutes']} min)", flush=True)
        os.makedirs(os.path.dirname(a.results), exist_ok=True)
        json.dump(results, open(a.results, "w"), indent=2)
    print(f"trained {len(results)} models on {device}")


if __name__ == "__main__":
    main()
