"""Model helpers for CL on top of a frozen BNN backbone."""

from typing import Dict, List

import numpy as np
import tensorflow as tf

from train_bnn import build_bnn_model


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_keyword_list(raw: str) -> List[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Keyword list is empty.")
    return items


def keyword_names_to_indices(class_names: np.ndarray, keywords: List[str]) -> np.ndarray:
    class_names_list = [str(x) for x in class_names.tolist()]
    idx = []
    for kw in keywords:
        if kw not in class_names_list:
            raise ValueError(f"Keyword '{kw}' not found in class_names.")
        idx.append(class_names_list.index(kw))
    return np.array(idx, dtype=np.int32)


def load_frozen_backbone_and_expand_head(
    base_weights_path: str,
    base_class_indices: np.ndarray,
    total_num_classes: int,
) -> Dict[str, object]:
    """Load pretrained base model, freeze backbone, and expand final head.

    Assumption:
    - Pretrained base head output ordering matches the provided base_class_indices order.
    """
    num_base = int(len(base_class_indices))
    base_model = build_bnn_model(num_base)
    loaded_from_full_model = False
    full_model = None
    try:
        base_model.load_weights(base_weights_path)
    except ValueError as exc:
        # Fallback: load a full-class checkpoint and slice the base-class columns.
        full_model = build_bnn_model(total_num_classes)
        full_model.load_weights(base_weights_path)
        loaded_from_full_model = True
        print(
            "[WARN] base checkpoint head shape did not match num_base; "
            "loaded as full-class checkpoint and sliced base columns."
        )

    # Backbone output is penultimate layer (input to final softmax Dense).
    source_model = full_model if loaded_from_full_model else base_model
    feature_extractor = tf.keras.Model(
        inputs=source_model.input,
        outputs=source_model.layers[-2].output,
        name="frozen_backbone",
    )
    feature_extractor.trainable = False
    for layer in feature_extractor.layers:
        layer.trainable = False

    # Base classifier weights.
    if loaded_from_full_model:
        w_full_src, b_full_src = source_model.layers[-1].get_weights()
        w_base = w_full_src[:, base_class_indices]
        b_base = b_full_src[base_class_indices]
    else:
        w_base, b_base = source_model.layers[-1].get_weights()  # (feat_dim, num_base), (num_base,)
    feat_dim = int(w_base.shape[0])

    # Expanded head aligned to global class indices.
    w_full = np.zeros((feat_dim, total_num_classes), dtype=np.float32)
    b_full = np.zeros((total_num_classes,), dtype=np.float32)
    for base_col, cls_idx in enumerate(base_class_indices):
        w_full[:, int(cls_idx)] = w_base[:, base_col]
        b_full[int(cls_idx)] = b_base[base_col]

    w_var = tf.Variable(w_full, trainable=True, name="cl_head_kernel")
    b_var = tf.Variable(b_full, trainable=True, name="cl_head_bias")

    return {
        "feature_extractor": feature_extractor,
        "head_w": w_var,
        "head_b": b_var,
        "feat_dim": feat_dim,
    }


def logits_from_features(features: tf.Tensor, head_w: tf.Variable, head_b: tf.Variable) -> tf.Tensor:
    return tf.matmul(features, head_w) + head_b


def sanity_check_only_last_layer_trainable(feature_extractor: tf.keras.Model) -> None:
    if feature_extractor.trainable:
        raise AssertionError("Feature extractor must be frozen.")
    trainable_count = sum(int(v.shape.num_elements()) for v in feature_extractor.trainable_variables)
    if trainable_count != 0:
        raise AssertionError("Frozen backbone has trainable variables; CL should update head only.")
