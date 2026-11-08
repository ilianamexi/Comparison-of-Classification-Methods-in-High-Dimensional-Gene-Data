"""Compare Random Forest, SMV-MRPV and NNV-MRPV on high-dimensional gene-expression data.

Protocol
- Datasets: leukemia, colon, prostate (OpenML, version 1).
- Repeated stratified k-fold cross-validation (default 5 folds x 3 repeats).
- Standardisation is fitted on the training folds only.
- Every random element is seeded (projections, RF, inner folds, neural network).
- Metrics: accuracy and balanced accuracy (colon is imbalanced, 40 vs 22).
- NNV-MRPV is trained on out-of-fold LDA probabilities. The earlier in-sample variant is
  reported as an ablation ("NNV in-sample") to show why that matters.

Usage:
    python run_experiments.py                       # full run, writes results/
    python run_experiments.py --repeats 1 --epochs 200   # quicker
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from mrpv import RandomProjectionEnsemble, majority_vote, nnv_predict

METHODS = ["RF", "SMV-MRPV", "NNV-MRPV", "NNV in-sample (ablation)"]


def load_dataset(name: str):
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
    X = data.data.select_dtypes(include="number").to_numpy(dtype=float)
    classes, y = np.unique(data.target.astype(str).to_numpy(), return_inverse=True)
    return X, y, [str(c) for c in classes]


def evaluate_dataset(X, y, args, name="data"):
    n_classes = len(np.unique(y))
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    rows = []
    for split, (tr, te) in enumerate(cv.split(X, y)):
        seed = args.seed + split
        t0 = time.time()
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        preds = {}
        rf = RandomForestClassifier(n_estimators=args.rf_trees, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        preds["RF"] = rf.predict(Xte)

        ens = RandomProjectionEnsemble(args.K, args.d_prime, seed).fit(Xtr, ytr)
        preds["SMV-MRPV"] = majority_vote(ens.predict_each(Xte))

        P_test = ens.proba_features(Xte)
        P_oof = ens.out_of_fold_proba_features(Xtr, ytr, n_inner=args.inner_folds)
        preds["NNV-MRPV"] = nnv_predict(P_oof, ytr, P_test, n_classes, seed, args.epochs)
        if not args.skip_ablation:
            P_in = ens.proba_features(Xtr)
            preds["NNV in-sample (ablation)"] = nnv_predict(P_in, ytr, P_test, n_classes, seed, args.epochs)

        for method, p in preds.items():
            rows.append({"dataset": name, "split": split, "repeat": split // args.folds, "fold": split % args.folds,
                         "method": method, "accuracy": accuracy_score(yte, p),
                         "balanced_accuracy": balanced_accuracy_score(yte, p)})
        print(f"  {name} split {split + 1}/{args.folds * args.repeats} "
              + "  ".join(f"{m}={accuracy_score(yte, p):.3f}" for m, p in preds.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "method"])[["accuracy", "balanced_accuracy"]]
    out = g.agg(["mean", "std"]).reset_index()
    out.columns = ["dataset", "method", "acc_mean", "acc_std", "bacc_mean", "bacc_std"]
    order = {m: i for i, m in enumerate(METHODS)}
    return out.sort_values(["dataset", "method"], key=lambda s: s.map(order) if s.name == "method" else s)


def to_markdown(summary: pd.DataFrame, info: dict, args) -> str:
    lines = [f"Repeated stratified {args.folds}-fold CV x {args.repeats} repeats "
             f"(K = {args.K} projections, d' = {args.d_prime}, NNV epochs = {args.epochs}, seed = {args.seed}). "
             "Mean ± standard deviation over all splits, in %.", ""]
    for ds, d in summary.groupby("dataset", sort=False):
        i = info[ds]
        lines += [f"**{ds}** ({i['n_samples']} samples, {i['n_features']:,} genes, classes {i['class_counts']})", "",
                  "| Method | Accuracy | Balanced accuracy |", "|---|---|---|"]
        for _, r in d.iterrows():
            lines.append(f"| {r.method} | {100 * r.acc_mean:.1f} ± {100 * r.acc_std:.1f} "
                         f"| {100 * r.bacc_mean:.1f} ± {100 * r.bacc_std:.1f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["leukemia", "colon", "prostate"])
    p.add_argument("--K", type=int, default=100, help="number of random projections")
    p.add_argument("--d-prime", dest="d_prime", type=int, default=100, help="projected dimension")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inner-folds", dest="inner_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--rf-trees", dest="rf_trees", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    rows, info = [], {}
    for name in args.datasets:
        X, y, classes = load_dataset(name)
        counts = {c: int(n) for c, n in zip(classes, np.bincount(y))}
        info[name] = {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "class_counts": counts}
        print(f"\n>>> {name}: {X.shape[0]} samples x {X.shape[1]} features, classes {counts}", flush=True)
        rows += evaluate_dataset(X, y, args, name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "fold_results.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary, info, args), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({"args": vars(args), "datasets": info}, indent=2), encoding="utf-8")
    print("\n" + to_markdown(summary, info, args))


if __name__ == "__main__":
    main()
