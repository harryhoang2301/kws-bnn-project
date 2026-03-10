"""Evaluation + diagnostics utilities for continual-learning KWS."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


def predict_with_head(
    feature_extractor: tf.keras.Model,
    head_w: np.ndarray,
    head_b: np.ndarray,
    x: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    preds = []
    w = tf.convert_to_tensor(head_w, dtype=tf.float32)
    b = tf.convert_to_tensor(head_b, dtype=tf.float32)
    for start in range(0, x.shape[0], batch_size):
        xb = tf.convert_to_tensor(x[start : start + batch_size], dtype=tf.float32)
        feats = feature_extractor(xb, training=False)
        logits = tf.matmul(feats, w) + b
        preds.append(tf.argmax(logits, axis=1).numpy())
    return np.concatenate(preds, axis=0)


def forgetting_metric(base_acc_before: float, base_acc_after: float) -> float:
    return float(base_acc_before - base_acc_after)


def _safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _class_counts(y: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    counts = {}
    for lbl in labels.tolist():
        counts[int(lbl)] = int(np.sum(y == int(lbl)))
    return counts


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    out = {}
    for lbl in labels.tolist():
        m = y_true == int(lbl)
        out[int(lbl)] = _safe_accuracy(y_true[m], y_pred[m]) if np.any(m) else float("nan")
    return out


def _top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, int]]:
    conf = []
    for t in labels.tolist():
        mt = y_true == int(t)
        if not np.any(mt):
            continue
        pvals, pcounts = np.unique(y_pred[mt], return_counts=True)
        for p, c in zip(pvals.tolist(), pcounts.tolist()):
            if int(p) != int(t):
                conf.append({"true_id": int(t), "pred_id": int(p), "count": int(c)})
    conf.sort(key=lambda x: x["count"], reverse=True)
    return conf[:top_k]


def _balanced_subset(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluation-only balanced subset: exactly k samples per class."""
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for lbl in labels.tolist():
        idx = np.where(y == int(lbl))[0]
        if idx.size < k:
            raise ValueError(f"balanced_eval_per_class={k} but class {lbl} has only {idx.size} samples.")
        pick = rng.choice(idx, size=k, replace=False)
        xs.append(x[pick])
        ys.append(y[pick])
    x_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    perm = rng.permutation(y_out.shape[0])
    return x_out[perm], y_out[perm]


def evaluate_cl_accuracy(
    feature_extractor: tf.keras.Model,
    head_w: np.ndarray,
    head_b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    old_class_indices: np.ndarray,
    new_class_indices: np.ndarray,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Compatibility wrapper used by existing code paths.

    Fix applied:
    - acc_all is computed on exactly old∪new labels, not all dataset labels.
    """
    report = evaluate_cl_with_diagnostics(
        feature_extractor=feature_extractor,
        head_w=head_w,
        head_b=head_b,
        x=x,
        y=y,
        old_class_indices=old_class_indices,
        new_class_indices=new_class_indices,
        class_names=None,
        batch_size=batch_size,
        debug=False,
    )
    return {
        "acc_base": report["metrics"]["acc_base_micro"],
        "acc_new": report["metrics"]["acc_new_micro"],
        "acc_all": report["metrics"]["acc_all_micro"],
    }


def evaluate_cl_with_diagnostics(
    feature_extractor: tf.keras.Model,
    head_w: np.ndarray,
    head_b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    old_class_indices: np.ndarray,
    new_class_indices: np.ndarray,
    class_names: Optional[np.ndarray],
    batch_size: int = 256,
    debug: bool = False,
    balanced_eval_per_class: Optional[int] = None,
    balanced_seed: int = 0,
) -> Dict[str, object]:
    """Compute metrics and full diagnostics for base/new/all label subsets."""
    old_labels = np.array(sorted(set(old_class_indices.tolist())), dtype=np.int32)
    new_labels = np.array(sorted(set(new_class_indices.tolist())), dtype=np.int32)
    union_labels = np.array(sorted(set(old_labels.tolist()) | set(new_labels.tolist())), dtype=np.int32)

    # HARD ASSERT: acc_all label set must be exactly base∪new.
    if set(union_labels.tolist()) != set(old_labels.tolist()) | set(new_labels.tolist()):
        raise AssertionError("acc_all label set mismatch; expected exact base∪new labels.")

    y_pred_all = predict_with_head(feature_extractor, head_w, head_b, x, batch_size=batch_size)

    mask_base = np.isin(y, old_labels)
    mask_new = np.isin(y, new_labels)
    mask_all = np.isin(y, union_labels)

    present_all = set(np.unique(y[mask_all]).tolist())
    expected_all = set(union_labels.tolist())
    extra = sorted(present_all - expected_all)
    missing = sorted(expected_all - present_all)
    # This checks what the acc_all subset includes from ground-truth y.
    # Any extra/missing indicates a label-set mismatch in evaluation prep.
    if extra or missing:
        raise AssertionError(
            f"acc_all label-set ERROR. extra_labels={extra}, missing_labels={missing}, expected={sorted(expected_all)}"
        )

    yb, pb = y[mask_base], y_pred_all[mask_base]
    yn, pn = y[mask_new], y_pred_all[mask_new]
    ya, pa = y[mask_all], y_pred_all[mask_all]

    per_class_all = _per_class_accuracy(ya, pa, union_labels)
    all_class_acc_vals = [v for v in per_class_all.values() if not np.isnan(v)]
    acc_all_macro = float(np.mean(all_class_acc_vals)) if all_class_acc_vals else float("nan")

    metrics = {
        "acc_base_micro": _safe_accuracy(yb, pb),
        "acc_new_micro": _safe_accuracy(yn, pn),
        "acc_all_micro": _safe_accuracy(ya, pa),
        "acc_all_macro": acc_all_macro,
        "acc_all_balanced": acc_all_macro,  # Balanced accuracy == macro over per-class recalls here.
    }

    n_base = int(yb.size)
    n_new = int(yn.size)
    expected_weighted = (
        (n_base * metrics["acc_base_micro"] + n_new * metrics["acc_new_micro"]) / float(n_base + n_new)
        if (n_base + n_new) > 0
        else float("nan")
    )
    expected_macro = acc_all_macro
    metrics["expected_acc_all_weighted"] = float(expected_weighted)
    metrics["expected_acc_all_macro"] = float(expected_macro)
    metrics["acc_all_minus_expected_weighted"] = float(metrics["acc_all_micro"] - expected_weighted)

    balanced_metrics = None
    if balanced_eval_per_class is not None and balanced_eval_per_class > 0:
        xb, yb_bal = _balanced_subset(x, y, union_labels, balanced_eval_per_class, balanced_seed)
        pb_bal = predict_with_head(feature_extractor, head_w, head_b, xb, batch_size=batch_size)
        per_cls_bal = _per_class_accuracy(yb_bal, pb_bal, union_labels)
        vals_bal = [v for v in per_cls_bal.values() if not np.isnan(v)]
        macro_bal = float(np.mean(vals_bal)) if vals_bal else float("nan")
        balanced_metrics = {
            "k_per_class": int(balanced_eval_per_class),
            "acc_micro_balanced_subset": _safe_accuracy(yb_bal, pb_bal),
            "acc_macro_balanced_subset": macro_bal,
            "acc_balanced_balanced_subset": macro_bal,
            "counts": _class_counts(yb_bal, union_labels),
        }

    def lbl_name(lbl: int) -> str:
        if class_names is None:
            return str(lbl)
        return str(class_names[int(lbl)])

    subsets = {
        "base": {
            "label_ids": old_labels.tolist(),
            "label_names": [lbl_name(x) for x in old_labels.tolist()],
            "counts": _class_counts(y, old_labels),
        },
        "new": {
            "label_ids": new_labels.tolist(),
            "label_names": [lbl_name(x) for x in new_labels.tolist()],
            "counts": _class_counts(y, new_labels),
        },
        "all": {
            "label_ids": union_labels.tolist(),
            "label_names": [lbl_name(x) for x in union_labels.tolist()],
            "counts": _class_counts(y, union_labels),
        },
    }

    per_class_table = []
    for lbl in union_labels.tolist():
        c = int(np.sum(ya == int(lbl)))
        p = (100.0 * c / max(1, ya.size))
        per_class_table.append(
            {
                "label_id": int(lbl),
                "label_name": lbl_name(int(lbl)),
                "count": c,
                "pct": p,
                "acc": float(per_class_all[int(lbl)]),
            }
        )

    report = {
        "summary": {
            "num_classes_in_model": int(head_b.shape[0]),
            "num_labels_in_acc_base": int(len(old_labels)),
            "num_labels_in_acc_new": int(len(new_labels)),
            "num_labels_in_acc_all": int(len(union_labels)),
            "included_labels_all": union_labels.tolist(),
            "excluded_labels_from_dataset_for_acc_all": sorted(
                set(np.unique(y).tolist()) - set(union_labels.tolist())
            ),
        },
        "metrics": metrics,
        "subsets": subsets,
        "per_class_all": per_class_table,
        "top_confusions_all": _top_confusions(ya, pa, union_labels, top_k=10),
        "balanced_eval": balanced_metrics,
    }

    if debug:
        print("[EVAL DEBUG] Label subsets:")
        for key in ("base", "new", "all"):
            info = subsets[key]
            print(f"  - {key}: ids={info['label_ids']}, names={info['label_names']}")
            print(f"    counts={info['counts']}")
        print("[EVAL DEBUG] Summary:")
        print(f"  num_classes_in_model={report['summary']['num_classes_in_model']}")
        print(f"  excluded_labels_from_dataset_for_acc_all={report['summary']['excluded_labels_from_dataset_for_acc_all']}")
        print("[EVAL DEBUG] Metrics:")
        print(
            "  "
            f"micro_all={metrics['acc_all_micro']:.4f} "
            f"macro_all={metrics['acc_all_macro']:.4f} "
            f"expected_weighted={metrics['expected_acc_all_weighted']:.4f} "
            f"delta={metrics['acc_all_minus_expected_weighted']:.4f}"
        )
        if abs(metrics["acc_all_minus_expected_weighted"]) > 1e-3:
            print(
                "  WARNING: measured acc_all differs from weighted expectation. "
                "Check label set inclusion/mapping/subset mismatch."
            )
        if balanced_metrics is not None:
            print(
                "[EVAL DEBUG] Balanced eval: "
                f"k={balanced_metrics['k_per_class']} "
                f"micro={balanced_metrics['acc_micro_balanced_subset']:.4f} "
                f"macro={balanced_metrics['acc_macro_balanced_subset']:.4f}"
            )

    return report


def mismatch_rate_vs_model(
    feature_extractor: tf.keras.Model,
    head_w: np.ndarray,
    head_b: np.ndarray,
    x: np.ndarray,
    reference_model: Optional[tf.keras.Model],
    batch_size: int = 256,
) -> float:
    """Optional mismatch against an external TF model."""
    if reference_model is None:
        return float("nan")
    pred_head = predict_with_head(feature_extractor, head_w, head_b, x, batch_size=batch_size)
    pred_ref = []
    for start in range(0, x.shape[0], batch_size):
        xb = tf.convert_to_tensor(x[start : start + batch_size], dtype=tf.float32)
        logits = reference_model(xb, training=False)
        pred_ref.append(tf.argmax(logits, axis=1).numpy())
    pred_ref = np.concatenate(pred_ref, axis=0)
    return float(np.mean(pred_head != pred_ref))
