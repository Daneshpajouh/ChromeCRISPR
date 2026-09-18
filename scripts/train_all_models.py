#!/usr/bin/env python3
"""Train every model, save its weights, and record its test performance.

Each model gets its own hyperparameter search over the learning rate, batch size and dropout.
Every choice is made on a validation split drawn from the training portion. The held-out test
set is read once per model, after the final model is fitted, and never influences the search,
the epoch count or the selection.
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


def run(name, Xtr, ytr, gtr, Xva, yva, gva, Xte, yte, gte, device, epochs, patience, trials):
    if name == "RF":
        model = create_random_forest(random_state=SEED)
        model.fit(np.concatenate([Xtr.reshape(len(Xtr), -1), gtr[:, None]], 1), ytr)
        p = model.predict(np.concatenate([Xte.reshape(len(Xte), -1), gte[:, None]], 1))
        return model, float(spearmanr(p, yte).correlation), float(mean_squared_error(yte, p))

    cfg = search(name, Xtr, ytr, gtr, Xva, yva, gva, device, trials)
    model, _ = fit(name, Xtr, ytr, gtr, Xva, yva, gva, device,
                   epochs=epochs, patience=patience, **cfg)
    uses_gc = getattr(model, "use_gc_content", False)
    model.eval()
    with torch.no_grad():                                  # the test set, read once
        pt = predict(model, Xte, gte, uses_gc, device)
    return model, float(spearmanr(pt, yte).correlation), float(mean_squared_error(yte, pt)), cfg


def predict(model, X, g, uses_gc, device, chunk=4096):
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.as_tensor(X[i:i + chunk], device=device)
        gb = torch.as_tensor(g[i:i + chunk], device=device)
        out.append((model(xb, gb) if uses_gc else model(xb)).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


SEED = 0

SEARCH_SPACE = {
    "learning_rate": [3e-4, 5e-4, 1e-3, 2e-3],
    "batch_size": [32, 64, 128],
    "dropout": [0.0, 0.1, 0.2],
}


def search(name, Xtr, ytr, gtr, Xva, yva, gva, device, trials, seed=0):
    """Pick hyperparameters on the validation split. The test set is not touched here."""
    rng = np.random.default_rng(seed)
    best, best_cfg = np.inf, None
    seen = set()
    for _ in range(trials):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        _, vl = fit(name, Xtr, ytr, gtr, Xva, yva, gva, device,
                    epochs=30, patience=6, **cfg)
        if vl < best:
            best, best_cfg = vl, cfg
    return best_cfg or {"learning_rate": 1e-3, "batch_size": 64, "dropout": 0.0}


def fit(name, Xtr, ytr, gtr, Xva, yva, gva, device, epochs, patience,
        learning_rate, batch_size, dropout, seed=SEED):
    """Fit one model and return it with its best validation loss.

    Seeded, so the same inputs and settings give the same weights on a rerun.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[name](dropout=dropout).to(device)
    uses_gc = getattr(model, "use_gc_content", False)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossf = nn.MSELoss()
    X, y, g = (torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device),
               torch.as_tensor(gtr, device=device))
    best, best_state, bad = np.inf, None, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
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
        model.eval()
        with torch.no_grad():
            vl = float(mean_squared_error(yva, predict(model, Xva, gva, uses_gc, device)))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


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
        out = run(name, X[tr], y[tr], g[tr], X[va], y[va], g[va],
                  Xte, yte, gte, device, a.epochs, a.patience, a.trials)
        model, sp, mse = out[0], out[1], out[2]
        cfg = out[3] if len(out) > 3 else {}
        path = os.path.join(a.out_dir, f"{name}.pt" if name != "RF" else "RF.joblib")
        if name == "RF":
            import joblib
            joblib.dump(model, path, compress=3)   # 328 MB uncompressed, 70 MB compressed
        else:
            torch.save(model.state_dict(), path)
        results[name] = {"spearman_correlation": round(sp, 4),
                         "mean_squared_error": round(mse, 4),
                         "hyperparameters": cfg,
                         "checkpoint": os.path.relpath(path),
                         "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{name:16s} spearman {sp:.4f}  mse {mse:.4f}  "
              f"({results[name]['minutes']} min)", flush=True)
        os.makedirs(os.path.dirname(a.results), exist_ok=True)
        json.dump(results, open(a.results, "w"), indent=2)
    print(f"trained {len(results)} models on {device}")


if __name__ == "__main__":
    main()
