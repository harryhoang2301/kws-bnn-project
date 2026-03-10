"""Continual-learning algorithms for head-only adaptation."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from cl.model_utils import logits_from_features


def _sparse_ce_from_logits(y_true: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=logits, from_logits=True)
    )


class CLAlgorithm(ABC):
    """Base class for CL head-training algorithms."""

    def __init__(
        self,
        feature_extractor: tf.keras.Model,
        head_w: tf.Variable,
        head_b: tf.Variable,
        old_class_indices: np.ndarray,
        new_class_indices: np.ndarray,
        lr: float,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.head_w = head_w
        self.head_b = head_b    
        self.old_class_indices = np.array(old_class_indices, dtype=np.int32)
        self.new_class_indices = np.array(new_class_indices, dtype=np.int32)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.train_vars = [self.head_w, self.head_b]

    def _features(self, x: tf.Tensor) -> tf.Tensor:
        return self.feature_extractor(x, training=False)

    def _logits(self, feats: tf.Tensor) -> tf.Tensor:
        return logits_from_features(feats, self.head_w, self.head_b)

    @abstractmethod
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        pass

    def get_eval_head(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.head_w.numpy(), self.head_b.numpy()

    def sanity_check_train_vars(self) -> None:
        extractor_var_ids = {id(v) for v in self.feature_extractor.variables}
        if any(id(v) in extractor_var_ids for v in self.train_vars):
            raise AssertionError("Backbone variables leaked into train_vars.")

    def post_train_sanity(self) -> None:
        """Optional algorithm-specific checks."""
        return


class TinyOL(CLAlgorithm):
    """TinyOL: online SGD on the last layer only."""

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            feats = self._features(x)
            logits = self._logits(feats)
            loss = _sparse_ce_from_logits(y, logits)
        grads = tape.gradient(loss, self.train_vars)
        self.optimizer.apply_gradients(zip(grads, self.train_vars))
        return {"loss": float(loss.numpy())}


class TinyOLv2(CLAlgorithm):
    """TinyOL v2: only new-class head parameters are updated via gradient masking."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        num_classes = int(self.head_b.shape[0])
        mask = np.zeros((num_classes,), dtype=np.float32)
        mask[self.new_class_indices] = 1.0
        self.class_mask = tf.constant(mask, dtype=tf.float32)

        self.w_old_snapshot = self.head_w.numpy()[:, self.old_class_indices].copy()
        self.b_old_snapshot = self.head_b.numpy()[self.old_class_indices].copy()

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            feats = self._features(x)
            logits = self._logits(feats)
            loss = _sparse_ce_from_logits(y, logits)
        g_w, g_b = tape.gradient(loss, self.train_vars)
        g_w = g_w * self.class_mask[None, :]
        g_b = g_b * self.class_mask
        self.optimizer.apply_gradients([(g_w, self.head_w), (g_b, self.head_b)])
        return {"loss": float(loss.numpy())}

    def post_train_sanity(self) -> None:
        w_old_now = self.head_w.numpy()[:, self.old_class_indices]
        b_old_now = self.head_b.numpy()[self.old_class_indices]
        w_diff = float(np.max(np.abs(w_old_now - self.w_old_snapshot)))
        b_diff = float(np.max(np.abs(b_old_now - self.b_old_snapshot)))
        if w_diff != 0.0 or b_diff != 0.0:
            raise AssertionError(f"TinyOLv2 old-class params changed (w_diff={w_diff}, b_diff={b_diff}).")


class LwF(CLAlgorithm):
    """Learning without Forgetting for head-only CL.

    Loss = CE(y_true, logits_train) + lambda_kd * KD(logits_copy_old, logits_train_old).
    """

    def __init__(
        self,
        *args,
        temperature: float = 2.0,
        lambda_kd: float = 1.0,
        refresh_copy_each_step: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)
        self.lambda_kd = float(lambda_kd)
        self.refresh_copy_each_step = bool(refresh_copy_each_step)
        self.batch_size = int(batch_size)
        self.prediction_counter = 0

        self.copy_w = tf.Variable(self.head_w.numpy(), trainable=False, name="lwf_copy_w")
        self.copy_b = tf.Variable(self.head_b.numpy(), trainable=False, name="lwf_copy_b")

    def _ce_to_copy_logits(self, logits_train: tf.Tensor, logits_copy: tf.Tensor) -> tf.Tensor:
        # Paper-style LwF variant uses CE between current logits and copy-layer predictions.
        copy_probs = tf.nn.softmax(logits_copy, axis=1)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=copy_probs, logits=logits_train))

    def _current_lambda(self) -> float:
        # Match equations reported in the paper:
        #  - LwF:   lambda = 100 / (100 + prediction_counter)
        #  - LwF_b: lambda = batch_size / prediction_counter
        p = max(1, int(self.prediction_counter))
        if self.refresh_copy_each_step:
            lam = float(self.batch_size) / float(p)
        else:
            lam = 100.0 / (100.0 + float(p))
        return float(np.clip(lam, 0.0, 1.0))

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        self.prediction_counter += int(y.shape[0])
        lam = tf.constant(self._current_lambda(), dtype=tf.float32)
        with tf.GradientTape() as tape:
            feats = self._features(x)
            logits = self._logits(feats)
            logits_copy = logits_from_features(feats, self.copy_w, self.copy_b)
            ce_true = _sparse_ce_from_logits(y, logits)
            ce_copy = self._ce_to_copy_logits(logits, logits_copy)
            loss = (1.0 - lam) * ce_true + lam * ce_copy

        grads = tape.gradient(loss, self.train_vars)
        self.optimizer.apply_gradients(zip(grads, self.train_vars))

        if self.refresh_copy_each_step:
            self.copy_w.assign(self.head_w)
            self.copy_b.assign(self.head_b)

        return {
            "loss": float(loss.numpy()),
            "ce_true": float(ce_true.numpy()),
            "ce_copy": float(ce_copy.numpy()),
            "lambda": float(lam.numpy()),
        }


class CWR(CLAlgorithm):
    """Copy Weight with Reinit.

    - Train head is optimized each step.
    - Consolidated head is updated after each batch for classes present in the batch.
    - Updated class columns in train head are reinitialized to zero after consolidation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_w = tf.Variable(self.head_w.numpy(), trainable=True, name="cwr_train_w")
        self.train_b = tf.Variable(self.head_b.numpy(), trainable=True, name="cwr_train_b")
        self.cons_w = tf.Variable(self.head_w.numpy(), trainable=False, name="cwr_cons_w")
        self.cons_b = tf.Variable(self.head_b.numpy(), trainable=False, name="cwr_cons_b")
        self.train_vars = [self.train_w, self.train_b]
        self.seen_counts = np.zeros((int(self.cons_b.shape[0]),), dtype=np.int64)

    def _set_class_column(self, var: tf.Variable, class_idx: int, value: np.ndarray) -> None:
        arr = var.numpy()
        if arr.ndim == 2:
            arr[:, class_idx] = value
        else:
            arr[class_idx] = value
        var.assign(arr)

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, float]:
        # Start batch from consolidated parameters.
        self.train_w.assign(self.cons_w)
        self.train_b.assign(self.cons_b)

        with tf.GradientTape() as tape:
            feats = self._features(x)
            logits = logits_from_features(feats, self.train_w, self.train_b)
            loss = _sparse_ce_from_logits(y, logits)
        grads = tape.gradient(loss, self.train_vars)
        self.optimizer.apply_gradients(zip(grads, self.train_vars))

        batch_classes = np.unique(y.numpy())
        for cls_idx in batch_classes:
            c = int(cls_idx)
            count = int(self.seen_counts[c])
            w_cons_prev = self.cons_w.numpy()[:, c]
            b_cons_prev = self.cons_b.numpy()[c]
            w_train_c = self.train_w.numpy()[:, c]
            b_train_c = self.train_b.numpy()[c]

            # Running average consolidation per class.
            w_cons_new = (w_cons_prev * count + w_train_c) / float(count + 1)
            b_cons_new = (b_cons_prev * count + b_train_c) / float(count + 1)
            self._set_class_column(self.cons_w, c, w_cons_new)
            self._set_class_column(self.cons_b, c, np.array(b_cons_new, dtype=np.float32))
            self.seen_counts[c] += 1

        # Paper-style CWR keeps training-layer synced to consolidated layer at batch boundaries.
        self.train_w.assign(self.cons_w)
        self.train_b.assign(self.cons_b)

        return {"loss": float(loss.numpy())}

    def get_eval_head(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.cons_w.numpy(), self.cons_b.numpy()


def build_algorithm(
    algo_name: str,
    feature_extractor: tf.keras.Model,
    head_w: tf.Variable,
    head_b: tf.Variable,
    old_class_indices: np.ndarray,
    new_class_indices: np.ndarray,
    lr: float,
    temperature: float,
    lambda_kd: float,
    batch_size: int,
) -> CLAlgorithm:
    """Factory for CL algorithms. Batch variants share the same implementation."""
    name = algo_name.lower()
    if name in {"tinyol", "tinyol_b"}:
        return TinyOL(feature_extractor, head_w, head_b, old_class_indices, new_class_indices, lr)
    if name in {"tinyolv2", "tinyolv2_b"}:
        return TinyOLv2(feature_extractor, head_w, head_b, old_class_indices, new_class_indices, lr)
    if name in {"lwf", "lwf_b"}:
        return LwF(
            feature_extractor,
            head_w,
            head_b,
            old_class_indices,
            new_class_indices,
            lr,
            temperature=temperature,
            lambda_kd=lambda_kd,
            refresh_copy_each_step=(name == "lwf_b"),
            batch_size=batch_size,
        )
    if name == "cwr":
        return CWR(feature_extractor, head_w, head_b, old_class_indices, new_class_indices, lr)
    raise ValueError(f"Unsupported algo: {algo_name}")
