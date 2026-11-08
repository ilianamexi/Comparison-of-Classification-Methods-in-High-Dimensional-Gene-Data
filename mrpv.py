"""Multiple Random Projection Voting (MRPV) classifiers for high-dimensional data.

Each of K base learners is a Linear Discriminant Analysis (LDA) model trained on the data
projected by its own Gaussian random matrix R_i (n_features x d').

- SMV-MRPV: the K LDA predictions are combined by simple majority vote.
- NNV-MRPV: the K LDA class-probability vectors are concatenated and fed to a small neural
  network (meta-learner) that learns how to combine them.

Stacking detail that matters: the meta-learner must be trained on probabilities that the
base learners produce for samples they did NOT see during fitting (out-of-fold predictions).
With d' close to or above the number of training samples, an LDA almost always classifies its
own training samples correctly, so in-sample probabilities show the meta-learner near-perfect
votes, while on new samples each vote is right far less often. The in-sample setting is
still run in run_experiments.py as an ablation ("NNV in-sample") to show the difference.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mode
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold


class RandomProjectionEnsemble:
    """K Gaussian random projections, each followed by an LDA classifier."""

    def __init__(self, n_projections: int = 100, d_prime: int = 100, seed: int = 0):
        self.n_projections = n_projections
        self.d_prime = d_prime
        self.seed = seed

    def _draw_projections(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.projections_ = [rng.standard_normal((n_features, self.d_prime)) for _ in range(self.n_projections)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomProjectionEnsemble":
        self.classes_ = np.unique(y)
        self._draw_projections(X.shape[1])
        self.models_ = [LinearDiscriminantAnalysis().fit(X @ R, y) for R in self.projections_]
        return self

    def predict_each(self, X: np.ndarray) -> np.ndarray:
        """Labels from every base learner, shape (K, n_samples)."""
        return np.stack([m.predict(X @ R) for m, R in zip(self.models_, self.projections_)])

    def proba_features(self, X: np.ndarray) -> np.ndarray:
        """Concatenated class probabilities, shape (n_samples, K * n_classes)."""
        return np.hstack([m.predict_proba(X @ R) for m, R in zip(self.models_, self.projections_)])

    def out_of_fold_proba_features(self, X: np.ndarray, y: np.ndarray, n_inner: int = 5) -> np.ndarray:
        """Meta-features for the training set built without letting any LDA see the sample it scores.

        The same K projection matrices are reused, so the meta-learner sees the same feature layout
        at train and test time; only the LDA fits change per inner fold.
        """
        n_classes = len(self.classes_)
        oof = np.zeros((X.shape[0], self.n_projections * n_classes))
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=self.seed)
        for tr, va in inner.split(X, y):
            for k, R in enumerate(self.projections_):
                lda = LinearDiscriminantAnalysis().fit(X[tr] @ R, y[tr])
                # predict_proba columns follow lda.classes_; map them onto the full class list
                cols = np.searchsorted(self.classes_, lda.classes_)
                block = np.zeros((len(va), n_classes))
                block[:, cols] = lda.predict_proba(X[va] @ R)
                oof[va, k * n_classes:(k + 1) * n_classes] = block
        return oof


def majority_vote(predictions: np.ndarray) -> np.ndarray:
    """Most frequent label per column of a (K, n_samples) array (ties -> smallest label)."""
    return mode(predictions, axis=0, keepdims=False).mode


def build_meta_learner(n_inputs: int, n_classes: int, seed: int):
    """The MLP meta-learner of NNV-MRPV: Dense(32) -> Dropout(0.5) -> Dense(128) -> Dropout(0.2) -> softmax."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def nnv_predict(P_train: np.ndarray, y_train: np.ndarray, P_test: np.ndarray, n_classes: int,
                seed: int, epochs: int = 500, batch_size: int = 16) -> np.ndarray:
    model = build_meta_learner(P_train.shape[1], n_classes, seed)
    model.fit(P_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return np.argmax(model.predict(P_test, verbose=0), axis=1)
