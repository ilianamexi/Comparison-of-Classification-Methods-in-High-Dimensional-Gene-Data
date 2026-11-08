"""Tests on synthetic data (no download needed)."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrpv import RandomProjectionEnsemble, majority_vote  # noqa: E402


def make_data(n=60, p=500, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.standard_normal((n, p))
    X[y == 1, :20] += 1.0  # 20 informative "genes"
    return X, y


def test_projections_are_reproducible():
    X, y = make_data()
    a = RandomProjectionEnsemble(5, 10, seed=3).fit(X, y).proba_features(X)
    b = RandomProjectionEnsemble(5, 10, seed=3).fit(X, y).proba_features(X)
    c = RandomProjectionEnsemble(5, 10, seed=4).fit(X, y).proba_features(X)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_feature_shapes():
    X, y = make_data()
    ens = RandomProjectionEnsemble(7, 10, seed=0).fit(X, y)
    assert ens.predict_each(X).shape == (7, len(y))
    assert ens.proba_features(X).shape == (len(y), 14)
    assert ens.out_of_fold_proba_features(X, y, n_inner=3).shape == (len(y), 14)


def test_in_sample_votes_look_perfect_but_out_of_fold_votes_do_not():
    # d' = 60 > ~48 training samples: an LDA fitted on a sample almost always "gets it right",
    # which is not what happens on unseen samples. OOF votes show the realistic accuracy.
    X, y = make_data(n=60, p=500)
    ens = RandomProjectionEnsemble(10, 60, seed=0).fit(X, y)
    vote_acc = lambda P: ((P[:, 1::2] > 0.5) == y[:, None]).mean()  # P(class 1) > 0.5 per base learner
    in_sample = vote_acc(ens.proba_features(X))
    oof = vote_acc(ens.out_of_fold_proba_features(X, y, n_inner=5))
    assert in_sample > 0.95
    assert oof < in_sample - 0.2


def test_majority_vote():
    preds = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]])
    assert majority_vote(preds).tolist() == [0, 1, 0]


def test_full_pipeline_runs_on_synthetic_data():
    from run_experiments import evaluate_dataset

    X, y = make_data(n=40, p=200)
    args = argparse.Namespace(folds=2, repeats=1, seed=0, rf_trees=10, K=4, d_prime=8,
                              inner_folds=2, epochs=2, skip_ablation=False)
    rows = evaluate_dataset(X, y, args, "synthetic")
    methods = {r["method"] for r in rows}
    assert methods == {"RF", "SMV-MRPV", "NNV-MRPV", "NNV in-sample (ablation)"}
    assert all(0 <= r["accuracy"] <= 1 for r in rows)
